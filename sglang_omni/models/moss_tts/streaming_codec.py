# SPDX-License-Identifier: Apache-2.0
"""Indexed streaming state contract for the MOSS audio tokenizer codec.

The codec state capacity is deliberately independent from the execution batch
width.  The adapter keeps a compatibility path for the current remote codec,
whose only streaming entry point is a fixed-width ``streaming(B)`` context.
That path is temporary: it lets callers migrate to the indexed contract while
the repository-owned decoder gains native compact execution support.
"""

from __future__ import annotations

import logging
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any

import torch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StreamingExecutionContext:
    """Map execution rows to persistent decoder state slots.

    ``valid_rows=False`` is reserved for graph padding/scratch rows.  Real
    request rows must use slots below ``real_state_capacity``. Graph runners
    may mark scratch rows disposable when every invalid row is guaranteed to
    address a non-leaseable scratch slot; the default preserves invalid state
    for general indexed callers.
    """

    state_slot_ids: torch.Tensor
    valid_rows: torch.Tensor
    scratch_rows_are_disposable: bool = False

    def validate(
        self,
        *,
        batch_size: int,
        state_capacity: int,
        device: torch.device,
        real_state_capacity: int | None = None,
        check_values: bool = False,
    ) -> None:
        if self.state_slot_ids.shape != (batch_size,):
            raise ValueError(
                f"Expected state_slot_ids shape ({batch_size},), got "
                f"{tuple(self.state_slot_ids.shape)}"
            )
        if self.valid_rows.shape != (batch_size,):
            raise ValueError(
                f"Expected valid_rows shape ({batch_size},), got "
                f"{tuple(self.valid_rows.shape)}"
            )
        if self.state_slot_ids.device != device or self.valid_rows.device != device:
            raise ValueError(
                "Streaming execution metadata must be on the same device as "
                "decoder inputs."
            )
        if self.state_slot_ids.dtype != torch.long:
            raise TypeError("state_slot_ids must have dtype torch.long")
        if self.valid_rows.dtype != torch.bool:
            raise TypeError("valid_rows must have dtype torch.bool")
        if state_capacity <= 0:
            raise ValueError(f"state_capacity must be positive, got {state_capacity}")
        if real_state_capacity is None:
            real_state_capacity = state_capacity
        if not 0 < real_state_capacity <= state_capacity:
            raise ValueError(
                "real_state_capacity must be in [1, state_capacity], got "
                f"{real_state_capacity} for state_capacity={state_capacity}"
            )
        if not check_values or batch_size == 0:
            return

        slots = self.state_slot_ids
        if bool(torch.any(slots < 0)) or bool(torch.any(slots >= state_capacity)):
            raise ValueError(
                f"state_slot_ids must be in [0, {state_capacity}), got "
                f"{slots.detach().to('cpu').tolist()}"
            )
        valid_slots = slots[self.valid_rows]
        if valid_slots.numel():
            if bool(torch.any(valid_slots >= real_state_capacity)):
                raise ValueError(
                    "valid execution rows may only use real decoder state slots "
                    f"below {real_state_capacity}"
                )
            if torch.unique(valid_slots).numel() != valid_slots.numel():
                raise ValueError("valid execution rows must use unique state slots")


@dataclass
class StreamingDecodeOutput:
    """Waveform and per-row sample lengths returned by streaming decode."""

    audio: torch.Tensor
    audio_lengths: torch.Tensor


_NATIVE_STATE_API = (
    "initialize_decoder_state_pool",
    "reset_decoder_state_slots",
    "close_decoder_state_pool",
    "decode_streaming_batch",
    "decode_streaming_tensors",
)


def _has_native_state_api(codec: Any) -> bool:
    return all(callable(getattr(codec, name, None)) for name in _NATIVE_STATE_API)


def _unpack_decode_output(result: Any) -> StreamingDecodeOutput:
    if hasattr(result, "audio") and hasattr(result, "audio_lengths"):
        audio = result.audio
        audio_lengths = result.audio_lengths
    elif isinstance(result, tuple) and len(result) == 2:
        audio, audio_lengths = result
    else:
        raise TypeError(
            "MOSS streaming codec decode must return (audio, audio_lengths) "
            "or an object with those attributes"
        )
    if audio is None or audio_lengths is None:
        raise RuntimeError("MOSS streaming codec returned empty audio output")
    return StreamingDecodeOutput(audio=audio, audio_lengths=audio_lengths)


class MossAudioTokenizerStreamingStatePool:
    """Own the lifetime and indexed access to decoder streaming state.

    Native repository codecs are called directly.  Older codecs are entered
    once in a fixed-width context and receive a full-width compatibility batch;
    the public contract remains indexed in both cases.
    """

    def __init__(self, codec: Any, *, n_vq: int | None = None) -> None:
        self.codec = codec
        self.n_vq = None if n_vq is None else int(n_vq)
        try:
            self.device = next(codec.parameters()).device
        except (AttributeError, StopIteration) as exc:
            raise RuntimeError(
                "MOSS streaming codec must expose at least one parameter so "
                "its execution device can be resolved"
            ) from exc
        self._native = _has_native_state_api(codec)
        self._exit_stack: ExitStack | None = None
        self._legacy_exec_mask: torch.Tensor | None = None
        self._state_capacity = 0
        self._real_state_capacity = 0
        self._scratch_capacity = 0
        self._slot_ids: torch.Tensor | None = None
        self._closed = True

    @property
    def is_initialized(self) -> bool:
        return not self._closed

    @property
    def uses_native_api(self) -> bool:
        """Whether the wrapped codec owns the indexed implementation."""
        return self._native

    @property
    def state_capacity(self) -> int:
        return self._state_capacity

    @property
    def real_state_capacity(self) -> int:
        return self._real_state_capacity

    @property
    def scratch_capacity(self) -> int:
        return self._scratch_capacity

    @property
    def slot_ids(self) -> torch.Tensor:
        if self._slot_ids is None:
            raise RuntimeError("MOSS decoder state pool is not initialized")
        return self._slot_ids

    def initialize_decoder_state_pool(
        self,
        state_capacity: int,
        scratch_capacity: int = 0,
    ) -> None:
        """Allocate persistent state; execution width is chosen per decode call."""
        if not isinstance(state_capacity, int) or isinstance(state_capacity, bool):
            raise TypeError("state_capacity must be an int")
        if not isinstance(scratch_capacity, int) or isinstance(scratch_capacity, bool):
            raise TypeError("scratch_capacity must be an int")
        if state_capacity <= 0 or scratch_capacity < 0:
            raise ValueError(
                "state_capacity must be > 0 and scratch_capacity must be >= 0; "
                f"got state_capacity={state_capacity}, "
                f"scratch_capacity={scratch_capacity}"
            )
        if self.is_initialized:
            raise RuntimeError("MOSS decoder state pool is already initialized")

        total_capacity = state_capacity + scratch_capacity
        if self._native:
            self.codec.initialize_decoder_state_pool(
                state_capacity,
                scratch_capacity,
            )
        else:
            streaming = getattr(self.codec, "streaming", None)
            set_exec_mask = getattr(self.codec, "_set_streaming_exec_mask", None)
            decode_frame = getattr(self.codec, "_decode_frame", None)
            if not callable(streaming) or not callable(set_exec_mask) or not callable(
                decode_frame
            ):
                raise RuntimeError(
                    "MOSS codec implements neither the indexed streaming API nor "
                    "the legacy streaming()/_decode_frame() surface"
                )
            self._exit_stack = ExitStack()
            try:
                self._exit_stack.enter_context(streaming(total_capacity))
            except Exception:
                self._exit_stack.close()
                self._exit_stack = None
                raise
            self._legacy_exec_mask = torch.ones(
                total_capacity,
                dtype=torch.bool,
                device=self.device,
            )
            logger.debug(
                "Using fixed-width compatibility adapter for MOSS codec "
                "(state_capacity=%d, scratch_capacity=%d)",
                state_capacity,
                scratch_capacity,
            )

        self._state_capacity = total_capacity
        self._real_state_capacity = state_capacity
        self._scratch_capacity = scratch_capacity
        self._slot_ids = torch.arange(
            total_capacity,
            dtype=torch.long,
            device=self.device,
        )
        self._closed = False

    def _require_initialized(self) -> None:
        if self._closed:
            raise RuntimeError("MOSS decoder state pool is not initialized")

    def _validate_slot_ids(self, state_slot_ids: torch.Tensor) -> None:
        if state_slot_ids.ndim != 1:
            raise ValueError(
                f"state_slot_ids must be rank 1, got {tuple(state_slot_ids.shape)}"
            )
        if state_slot_ids.dtype != torch.long:
            raise TypeError("state_slot_ids must have dtype torch.long")
        if state_slot_ids.device != self.device:
            raise ValueError(
                f"state_slot_ids must be on {self.device}, got {state_slot_ids.device}"
            )
        # Value checks on CUDA would synchronize the scheduler thread.  The
        # scheduler owns GPU slot tensors; keep eager value validation for CPU
        # callers/tests and let CUDA indexing report malformed trusted inputs.
        if self.device.type != "cuda" and state_slot_ids.numel() and (
            bool(torch.any(state_slot_ids < 0))
            or bool(torch.any(state_slot_ids >= self._state_capacity))
        ):
            raise ValueError(
                f"state_slot_ids must be in [0, {self._state_capacity}), got "
                f"{state_slot_ids.detach().to('cpu').tolist()}"
            )

    def reset_decoder_state_slots(self, state_slot_ids: torch.Tensor) -> None:
        """Reset only the supplied persistent state slots."""
        self._require_initialized()
        self._validate_slot_ids(state_slot_ids)
        if state_slot_ids.numel() == 0:
            return
        if self._native:
            self.codec.reset_decoder_state_slots(state_slot_ids)
        else:
            reset_streaming_slots = getattr(self.codec, "_reset_streaming_slots", None)
            if callable(reset_streaming_slots):
                reset_mask = torch.zeros(
                    self._state_capacity,
                    dtype=torch.bool,
                    device=self.device,
                )
                reset_mask.index_fill_(0, state_slot_ids, True)
                reset_streaming_slots(reset_mask)
            else:
                self._reset_legacy_by_traversal(state_slot_ids)
            assert self._legacy_exec_mask is not None
            self._legacy_exec_mask.index_fill_(0, state_slot_ids, True)

    def _reset_legacy_by_traversal(self, state_slot_ids: torch.Tensor) -> None:
        apply = getattr(self.codec, "apply", None)
        if not callable(apply):
            raise RuntimeError("legacy MOSS codec has no resettable module traversal")
        reset_mask = torch.zeros(
            self._state_capacity,
            dtype=torch.bool,
            device=self.device,
        )
        reset_mask.index_fill_(0, state_slot_ids, True)
        reset_states = 0

        def reset_module(module: Any) -> None:
            nonlocal reset_states
            state = getattr(module, "_streaming_state", None)
            if state is None or not callable(getattr(state, "reset", None)):
                return
            state.reset(reset_mask.to(getattr(state, "device", self.device)))
            reset_states += 1

        with torch.no_grad():
            apply(reset_module)
        if reset_states == 0:
            raise RuntimeError(
                "MOSS legacy codec has no resettable streaming state; "
                "implement reset_decoder_state_slots on the repository codec"
            )

    def _validate_decode_inputs(
        self,
        codes: torch.Tensor,
        codes_lengths: torch.Tensor,
        state_slot_ids: torch.Tensor,
        valid_rows: torch.Tensor,
    ) -> StreamingExecutionContext:
        self._require_initialized()
        if codes.ndim != 3:
            raise ValueError(
                f"codes must have shape [NQ, B, T], got {tuple(codes.shape)}"
            )
        if codes.dtype != torch.long:
            raise TypeError("codes must have dtype torch.long")
        if codes.device != self.device:
            raise ValueError(f"codes must be on {self.device}, got {codes.device}")
        _, batch_size, frame_count = codes.shape
        if batch_size <= 0 or frame_count <= 0:
            raise ValueError(
                f"codes must have positive B and T, got B={batch_size}, T={frame_count}"
            )
        if self.n_vq is not None and int(codes.shape[0]) != self.n_vq:
            raise ValueError(
                f"codes has {int(codes.shape[0])} quantizers, expected {self.n_vq}"
            )
        if codes_lengths.shape != (batch_size,):
            raise ValueError(
                f"codes_lengths must have shape ({batch_size},), got "
                f"{tuple(codes_lengths.shape)}"
            )
        if codes_lengths.dtype != torch.long:
            raise TypeError("codes_lengths must have dtype torch.long")
        if codes_lengths.device != self.device:
            raise ValueError(
                f"codes_lengths must be on {self.device}, got {codes_lengths.device}"
            )
        if state_slot_ids.shape != (batch_size,):
            raise ValueError(
                f"state_slot_ids must have shape ({batch_size},), got "
                f"{tuple(state_slot_ids.shape)}"
            )
        if valid_rows.shape != (batch_size,):
            raise ValueError(
                f"valid_rows must have shape ({batch_size},), got "
                f"{tuple(valid_rows.shape)}"
            )
        context = StreamingExecutionContext(state_slot_ids, valid_rows)
        canonical = (
            not self._native
            and batch_size == self._state_capacity
            and torch.equal(state_slot_ids, self.slot_ids)
        )
        context.validate(
            batch_size=batch_size,
            state_capacity=self._state_capacity,
            real_state_capacity=self._real_state_capacity,
            device=self.device,
            check_values=self.device.type != "cuda" and not canonical,
        )
        if self.device.type != "cuda" and not canonical:
            if bool(torch.any(codes_lengths < 0)) or bool(
                torch.any(codes_lengths > frame_count)
            ):
                raise ValueError(
                    f"codes_lengths must be in [0, {frame_count}]"
                )
        return context

    def decode_streaming_batch(
        self,
        codes: torch.Tensor,
        codes_lengths: torch.Tensor,
        state_slot_ids: torch.Tensor,
        valid_rows: torch.Tensor,
    ) -> StreamingDecodeOutput:
        """Decode one execution batch against indexed persistent state."""
        context = self._validate_decode_inputs(
            codes,
            codes_lengths,
            state_slot_ids,
            valid_rows,
        )
        if self._native:
            result = self.codec.decode_streaming_batch(
                codes,
                codes_lengths,
                state_slot_ids,
                valid_rows,
            )
            return _unpack_decode_output(result)

        assert self._legacy_exec_mask is not None
        canonical = int(codes.shape[1]) == self._state_capacity and torch.equal(
            state_slot_ids, self.slot_ids
        )
        if canonical:
            full_codes = codes
            full_lengths = codes_lengths
            full_mask = valid_rows
        else:
            full_codes = torch.zeros(
                codes.shape[0],
                self._state_capacity,
                codes.shape[2],
                dtype=codes.dtype,
                device=self.device,
            )
            full_lengths = torch.zeros(
                self._state_capacity,
                dtype=torch.long,
                device=self.device,
            )
            full_mask = torch.zeros(
                self._state_capacity,
                dtype=torch.bool,
                device=self.device,
            )
            valid_indices = torch.nonzero(valid_rows, as_tuple=False).flatten()
            if valid_indices.numel():
                slots = state_slot_ids.index_select(0, valid_indices)
                full_codes.index_copy_(
                    1,
                    slots,
                    codes.index_select(1, valid_indices),
                )
                full_lengths.index_copy_(
                    0,
                    slots,
                    codes_lengths.index_select(0, valid_indices),
                )
                full_mask.index_fill_(0, slots, True)
        set_exec_mask = getattr(self.codec, "_set_streaming_exec_mask", None)
        decode_frame = getattr(self.codec, "_decode_frame", None)
        assert callable(set_exec_mask) and callable(decode_frame)
        set_exec_mask(full_mask)
        result = _unpack_decode_output(decode_frame(full_codes, full_lengths))
        if canonical:
            return result

        row_audio = result.audio.index_select(0, state_slot_ids)
        row_lengths = result.audio_lengths.index_select(0, state_slot_ids)
        invalid = ~context.valid_rows
        if bool(torch.any(invalid)):
            row_audio = row_audio.masked_fill(invalid.view(-1, 1, 1), 0)
            row_lengths = row_lengths.masked_fill(invalid, 0)
        return StreamingDecodeOutput(row_audio, row_lengths)

    def decode_streaming_tensors(
        self,
        codes: torch.Tensor,
        codes_lengths: torch.Tensor,
        state_slot_ids: torch.Tensor,
        valid_rows: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Tensor-only form used by graph/compile adapters."""
        if self._native:
            context = self._validate_decode_inputs(
                codes,
                codes_lengths,
                state_slot_ids,
                valid_rows,
            )
            del context
            result = self.codec.decode_streaming_tensors(
                codes,
                codes_lengths,
                state_slot_ids,
                valid_rows,
            )
            output = _unpack_decode_output(result)
            return output.audio, output.audio_lengths
        output = self.decode_streaming_batch(
            codes,
            codes_lengths,
            state_slot_ids,
            valid_rows,
        )
        return output.audio, output.audio_lengths

    def close_decoder_state_pool(self) -> None:
        """Close the pool and release any legacy streaming context."""
        if self._closed:
            return
        error: BaseException | None = None
        try:
            if self._native:
                self.codec.close_decoder_state_pool()
        except BaseException as exc:  # pragma: no cover - cleanup error path
            error = exc
        finally:
            if self._exit_stack is not None:
                try:
                    self._exit_stack.close()
                except BaseException as exc:  # pragma: no cover - cleanup error path
                    if error is None:
                        error = exc
                self._exit_stack = None
            self._legacy_exec_mask = None
            self._slot_ids = None
            self._state_capacity = 0
            self._real_state_capacity = 0
            self._scratch_capacity = 0
            self._closed = True
        if error is not None:
            raise error


__all__ = [
    "MossAudioTokenizerStreamingStatePool",
    "StreamingDecodeOutput",
    "StreamingExecutionContext",
]
