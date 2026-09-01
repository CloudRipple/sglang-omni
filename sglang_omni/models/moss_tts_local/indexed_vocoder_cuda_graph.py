# SPDX-License-Identifier: Apache-2.0
"""CUDA-graph replay for the indexed MOSS streaming decoder.

The repository-owned codec keeps decoder state in a persistent slot pool.  A
graph therefore captures a fixed execution width while replay supplies the
real state slot ids for the live rows.  Padding rows address scratch slots and
are marked invalid, so they cannot advance a request's decoder state.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import nn

logger = logging.getLogger(__name__)


class _MossStreamingDecodeCompileAdapter(nn.Module):
    """Expose only the tensor streaming boundary to TorchInductor.

    Slot leasing, graph padding, and replay stay in the caller.  Keeping the
    adapter at this boundary also means the compiled callable never observes
    Python dictionaries or host-side audio materialization.
    """

    def __init__(self, codec: nn.Module) -> None:
        super().__init__()
        self.codec = codec

    def forward(
        self,
        codes: torch.Tensor,
        codes_lengths: torch.Tensor,
        state_slot_ids: torch.Tensor,
        valid_rows: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.codec.decode_streaming_tensors(
            codes,
            codes_lengths,
            state_slot_ids,
            valid_rows,
            scratch_rows_are_disposable=True,
        )


@dataclass
class _CapturedIndexedVocoderGraph:
    graph: torch.cuda.CUDAGraph
    static_codes: torch.Tensor
    static_lengths: torch.Tensor
    static_state_slot_ids: torch.Tensor
    scratch_state_slot_ids: torch.Tensor
    static_valid_rows: torch.Tensor
    static_audio: torch.Tensor
    static_audio_lengths: torch.Tensor
    compiled: bool = False
    active_batch_size: int = 0


class MossIndexedVocoderCudaGraphRunner:
    """Replay native indexed streaming decode graphs keyed by ``(B, T)``.

    ``B`` is a graph bucket, while ``T`` is exact.  Frame padding is
    intentionally unsupported: the codec's causal state advances per frame,
    so padding a valid row would change the waveform.  Batch padding is safe
    because scratch rows are passed with ``valid_rows=False``.
    """

    def __init__(
        self,
        codec,
        *,
        real_state_capacity: int,
        scratch_capacity: int,
        batch_sizes: Iterable[int],
        frame_sizes: Iterable[int],
        num_quantizers: int,
        warmup_iters: int = 3,
        min_free_gb: float = 3.0,
        compile_decode: bool = False,
        compile_decode_shapes: Iterable[tuple[int, int]] | None = None,
        compile_mode: str | None = None,
    ) -> None:
        self._codec = codec
        self._real_state_capacity = int(real_state_capacity)
        self._scratch_capacity = int(scratch_capacity)
        self._device = next(codec.parameters()).device
        self._num_quantizers = int(num_quantizers)
        self._warmup_iters = max(int(warmup_iters), 1)
        self._min_free_bytes = int(float(min_free_gb) * (1024**3))
        self._batch_sizes = sorted(
            {
                int(size)
                for size in batch_sizes
                if 0 < int(size) <= self._real_state_capacity
            }
        )
        self._frame_sizes = sorted({int(size) for size in frame_sizes if int(size) > 0})
        self._graphs: dict[tuple[int, int], _CapturedIndexedVocoderGraph] = {}
        self._pool = None
        self._sealed = False
        self._warmup_attempted = False
        self._compile_requested = bool(compile_decode)
        self._compile_shapes = (
            None
            if compile_decode_shapes is None
            else {
                (int(batch_size), int(frame_size))
                for batch_size, frame_size in compile_decode_shapes
            }
        )
        if self._compile_shapes is not None:
            invalid_compile_shapes = sorted(
                shape
                for shape in self._compile_shapes
                if shape[0] < 1 or shape[1] < 1
            )
            if invalid_compile_shapes:
                raise ValueError(
                    "compile_decode_shapes must contain positive (B,T) pairs; "
                    f"got {invalid_compile_shapes}"
                )
            capture_shapes = {
                (batch_size, frame_size)
                for batch_size in self._batch_sizes
                for frame_size in self._frame_sizes
            }
            unavailable_compile_shapes = sorted(
                self._compile_shapes - capture_shapes
            )
            if unavailable_compile_shapes:
                raise ValueError(
                    "compile_decode_shapes must be present in the configured "
                    f"capture grid; unavailable={unavailable_compile_shapes}"
                )
        self._compile_mode = compile_mode or os.environ.get(
            "SGLANG_TORCH_COMPILE_MODE", "default"
        )
        self._compiled_decode: nn.Module | None = None
        self._graph_decode = _MossStreamingDecodeCompileAdapter(self._codec)
        self._compile_disabled = False
        self._compiled_capture_sizes: set[tuple[int, int]] = set()

        if self._real_state_capacity <= 0:
            raise ValueError("real_state_capacity must be positive")
        if self._scratch_capacity < max(self._batch_sizes, default=0):
            raise ValueError(
                "scratch_capacity must cover the largest indexed graph bucket; "
                f"got scratch_capacity={self._scratch_capacity}, "
                f"largest_bucket={max(self._batch_sizes, default=0)}"
            )
        if self._num_quantizers <= 0:
            raise ValueError("num_quantizers must be positive")

    @property
    def is_ready(self) -> bool:
        return bool(self._graphs)

    @property
    def capture_sizes(self) -> list[tuple[int, int]]:
        return sorted(self._graphs)

    @property
    def batch_sizes(self) -> list[int]:
        return list(self._batch_sizes)

    @property
    def frame_sizes(self) -> list[int]:
        return list(self._frame_sizes)

    @property
    def scratch_capacity(self) -> int:
        return self._scratch_capacity

    @property
    def compile_requested(self) -> bool:
        return self._compile_requested

    @property
    def compiled_capture_sizes(self) -> list[tuple[int, int]]:
        return sorted(self._compiled_capture_sizes)

    @property
    def compile_shapes(self) -> list[tuple[int, int]] | None:
        return None if self._compile_shapes is None else sorted(self._compile_shapes)

    def _compile_shape_enabled(self, batch_size: int, frame_size: int) -> bool:
        if not self._compile_requested or self._compile_disabled:
            return False
        return self._compile_shapes is None or (
            int(batch_size), int(frame_size)
        ) in self._compile_shapes

    def _ensure_compiled_decode(self) -> nn.Module | None:
        if not self._compile_requested or self._compile_disabled:
            return None
        if self._compiled_decode is not None:
            return self._compiled_decode
        compiler = getattr(torch, "compile", None)
        if compiler is None:
            logger.warning(
                "MOSS indexed vocoder compile requested but torch.compile is unavailable; "
                "using plain CUDA graphs"
            )
            self._compile_disabled = True
            return None
        try:
            # Keep Inductor's own CUDA-graph mode off: this runner owns the
            # outer graph and must not capture nested cudagraph trees.  The
            # shared SGLang helper enables coordinate-descent autotuning, which
            # is useful for large AR kernels but makes this 12-stage codec's
            # startup prohibitively expensive, so configure this small adapter
            # conservatively and independently.
            import torch._inductor.config as inductor_config

            inductor_config.coordinate_descent_tuning = False
            inductor_config.fx_graph_cache = True
            if hasattr(inductor_config, "triton"):
                inductor_config.triton.unique_kernel_names = True
                # The outer CUDA graph already fixes the hot shapes. Triton
                # pointwise/cuBLAS autotune would otherwise benchmark many
                # variants during every cold service start.
                inductor_config.triton.autotune_pointwise = False
                inductor_config.triton.autotune_cublasLt = False
                if hasattr(inductor_config.triton, "autotune_at_compile_time"):
                    inductor_config.triton.autotune_at_compile_time = False
        except Exception:
            logger.debug(
                "MOSS indexed vocoder could not install local Inductor config",
                exc_info=True,
            )
        try:
            self._compiled_decode = compiler(
                self._graph_decode,
                dynamic=self._compile_shapes is None,
                mode=self._compile_mode,
            )
        except Exception:
            logger.warning(
                "MOSS indexed vocoder compile adapter creation failed; using plain "
                "CUDA graphs",
                exc_info=True,
            )
            self._compile_disabled = True
            return None
        logger.info(
            "MOSS indexed vocoder streaming decode compile adapter enabled "
            "(dynamic=%s, shapes=%s, mode=%s)",
            self._compile_shapes is None,
            self.compile_shapes,
            self._compile_mode,
        )
        return self._compiled_decode

    def _reset_capture_slots(self, batch_size: int, device: torch.device) -> None:
        """Drain failed warmup work before reusing scratch state rows."""
        try:
            torch.cuda.synchronize(device)
            self._codec.reset_decoder_state_slots(self._scratch_slots(batch_size, device=device))
        except Exception:
            logger.debug(
                "MOSS indexed vocoder scratch reset failed after capture probe",
                exc_info=True,
            )

    def _scratch_slots(self, batch_size: int, *, device: torch.device) -> torch.Tensor:
        return self._real_state_capacity + torch.arange(
            batch_size,
            dtype=torch.long,
            device=device,
        )

    def _enough_free_vram(self) -> tuple[bool, int]:
        free, _ = torch.cuda.mem_get_info(self._device)
        return free >= self._min_free_bytes, free

    @torch.no_grad()
    def _capture(self, batch_size: int, frame_size: int) -> bool:
        compiled_decode = (
            self._ensure_compiled_decode()
            if self._compile_shape_enabled(batch_size, frame_size)
            else None
        )
        if compiled_decode is not None:
            try:
                self._capture_with_decode(
                    batch_size,
                    frame_size,
                    compiled_decode,
                    compiled=True,
                )
                self._compiled_capture_sizes.add((batch_size, frame_size))
                return True
            except Exception:
                # A failed Inductor trace or nested graph capture must not
                # poison the serving path.  Disable the candidate and retry
                # this exact shape with the known-good plain graph.
                self._reset_capture_slots(batch_size, self._device)
                self._compiled_decode = None
                self._compile_disabled = True
                logger.warning(
                    "MOSS indexed vocoder compile/capture failed for (B,T)=(%d,%d); "
                    "retrying a plain CUDA graph",
                    batch_size,
                    frame_size,
                    exc_info=True,
                )

        self._capture_with_decode(
            batch_size,
            frame_size,
            self._graph_decode,
            compiled=False,
        )
        return False

    @torch.no_grad()
    def _reserve_streaming_cache_shapes(
        self, frame_sizes: Iterable[int] | None = None
    ) -> None:
        """Grow indexed codec caches before capturing any graph.

        Different input frame sizes can take different local-context branches
        after the codec's patch stages.  A cache that grows while a previous
        graph already holds its old address makes that graph unsafe to replay.
        One eager probe per unique frame size (at the largest batch bucket) is
        enough to establish the maximum physical width for every decoder layer
        before capture begins.
        """
        requested_frame_sizes = self._frame_sizes if frame_sizes is None else sorted(
            {int(frame) for frame in frame_sizes if int(frame) > 0}
        )
        if self._device.type != "cuda" or not requested_frame_sizes:
            return
        batch_size = max(self._batch_sizes, default=0)
        if batch_size <= 0:
            return
        device = self._device
        slots = self._scratch_slots(batch_size, device=device)
        for frame_size in requested_frame_sizes:
            codes = torch.zeros(
                self._num_quantizers,
                batch_size,
                frame_size,
                dtype=torch.long,
                device=device,
            )
            lengths = torch.full(
                (batch_size,),
                frame_size,
                dtype=torch.long,
                device=device,
            )
            valid_rows = torch.zeros(batch_size, dtype=torch.bool, device=device)
            # This probe is intentionally eager.  It may allocate a larger
            # ring cache, but no graph has captured a pointer yet.
            self._codec.decode_streaming_tensors(
                codes,
                lengths,
                slots,
                valid_rows,
            )
            torch.cuda.synchronize(device)
            self._codec.reset_decoder_state_slots(slots)
        logger.info(
            "MOSS indexed vocoder reserved streaming cache shapes for frame sizes=%s",
            requested_frame_sizes,
        )

    @torch.no_grad()
    def _capture_with_decode(
        self,
        batch_size: int,
        frame_size: int,
        decode,
        *,
        compiled: bool,
    ) -> None:
        device = self._device
        codes = torch.zeros(
            self._num_quantizers,
            batch_size,
            frame_size,
            dtype=torch.long,
            device=device,
        )
        # Every replay for this exact-T graph uses the same length.  Keeping
        # it populated for scratch rows avoids a per-replay fill; invalid rows
        # are masked by ``valid_rows`` before output/state updates.
        lengths = torch.full(
            (batch_size,), frame_size, dtype=torch.long, device=device
        )
        state_slot_ids = self._scratch_slots(batch_size, device=device)
        valid_rows = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # Warm up outside capture so lazy kernel/workspace allocations and the
        # first decoder cache initialization do not become capture failures.
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            for _ in range(self._warmup_iters):
                decode(
                    codes,
                    lengths,
                    state_slot_ids,
                    valid_rows,
                )
        torch.cuda.current_stream(device).wait_stream(stream)
        torch.cuda.synchronize(device)
        self._codec.reset_decoder_state_slots(state_slot_ids)

        if self._pool is None:
            self._pool = torch.cuda.graph_pool_handle()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(
            graph,
            pool=self._pool,
            capture_error_mode="thread_local",
        ):
            static_audio, static_audio_lengths = decode(
                codes,
                lengths,
                state_slot_ids,
                valid_rows,
            )
        self._graphs[(batch_size, frame_size)] = _CapturedIndexedVocoderGraph(
            graph=graph,
            static_codes=codes,
            static_lengths=lengths,
            static_state_slot_ids=state_slot_ids,
            scratch_state_slot_ids=state_slot_ids.clone(),
            static_valid_rows=valid_rows,
            static_audio=static_audio,
            static_audio_lengths=static_audio_lengths,
            compiled=compiled,
        )

    @torch.no_grad()
    def warmup(self, frames: Iterable[int] | None = None) -> list[tuple[int, int]]:
        """Capture configured ``(batch_bucket, exact_frame_count)`` graphs.

        Capture is best effort.  A low-VRAM device or an individual capture
        error leaves that key on eager execution; other keys may still be
        captured.  The runner is sealed after one attempt to prevent capture
        work from happening on the serving hot path.
        """
        if self._sealed:
            return self.capture_sizes
        self._warmup_attempted = True
        self._sealed = True
        if self._device.type != "cuda" or not torch.cuda.is_available():
            return []
        frame_sizes = self._frame_sizes if frames is None else sorted(
            {int(frame) for frame in frames if int(frame) > 0}
        )
        if not self._batch_sizes or not frame_sizes:
            return []

        # Capture the largest allocations first when sharing one graph pool.
        keys = sorted(
            (
                (batch_size, frame_size)
                for batch_size in self._batch_sizes
                for frame_size in frame_sizes
            ),
            reverse=True,
        )
        self._reserve_streaming_cache_shapes(frame_sizes)
        with torch.cuda.device(self._device):
            for batch_size, frame_size in keys:
                key = (batch_size, frame_size)
                enough, free = self._enough_free_vram()
                if not enough:
                    logger.warning(
                        "MOSS indexed vocoder CG: free VRAM %.1fGB < %.1fGB; "
                        "skipping remaining captures",
                        free / 1024**3,
                        self._min_free_bytes / 1024**3,
                    )
                    break
                try:
                    compiled = self._capture(batch_size, frame_size)
                    logger.info(
                        "MOSS indexed vocoder captured %s graph for (B,T)=%s",
                        "compiled" if compiled else "plain",
                        key,
                    )
                except Exception:
                    self._graphs.pop(key, None)
                    # A failed warmup may have advanced scratch state before
                    # capture.  Reset is best effort; eager serving remains
                    # available even if a broken graph poisoned this key.
                    try:
                        self._reset_capture_slots(batch_size, self._device)
                    except Exception:
                        logger.exception(
                            "failed to reset indexed vocoder scratch slots after "
                            "capture failure for (B,T)=%s",
                            key,
                        )
                    logger.warning(
                        "MOSS indexed vocoder CG capture failed for (B,T)=%s; "
                        "using eager",
                        key,
                        exc_info=True,
                    )
        logger.info(
            "MOSS indexed vocoder CUDA graphs sealed: %d/%d captured %s; "
            "compiled=%s",
            len(self._graphs),
            len(keys),
            self.capture_sizes,
            self.compiled_capture_sizes,
        )
        return self.capture_sizes

    def captured_frames(self) -> list[int]:
        return sorted({frame_size for _, frame_size in self._graphs})

    @staticmethod
    def _stage_active_rows(
        entry: _CapturedIndexedVocoderGraph,
        state_slot_ids: torch.Tensor,
    ) -> int:
        """Stage live rows and restore retired rows to disposable scratch slots."""
        actual_batch_size = int(state_slot_ids.shape[0])
        entry.static_state_slot_ids[:actual_batch_size].copy_(
            state_slot_ids,
            non_blocking=True,
        )
        previous_batch_size = entry.active_batch_size
        if actual_batch_size > previous_batch_size:
            entry.static_valid_rows[previous_batch_size:actual_batch_size].fill_(True)
        elif actual_batch_size < previous_batch_size:
            entry.static_state_slot_ids[actual_batch_size:previous_batch_size].copy_(
                entry.scratch_state_slot_ids[actual_batch_size:previous_batch_size],
                non_blocking=True,
            )
            entry.static_valid_rows[actual_batch_size:previous_batch_size].zero_()
        entry.active_batch_size = actual_batch_size
        return actual_batch_size

    @torch.no_grad()
    def decode_step(
        self,
        codes: torch.Tensor,
        state_slot_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Replay a graph for active indexed rows, or return ``None`` for eager."""
        if not codes.is_cuda or torch.cuda.is_current_stream_capturing():
            return None
        if codes.ndim != 3 or state_slot_ids.ndim != 1:
            return None
        num_quantizers, actual_batch_size, frame_size = map(int, codes.shape)
        if (
            num_quantizers != self._num_quantizers
            or int(state_slot_ids.shape[0]) != actual_batch_size
            or actual_batch_size <= 0
        ):
            return None
        batch_size = next(
            (size for size in self._batch_sizes if size >= actual_batch_size),
            None,
        )
        if batch_size is None:
            return None
        entry = self._graphs.get((batch_size, frame_size))
        if entry is None:
            return None

        entry.static_codes[:, :actual_batch_size, :].copy_(codes, non_blocking=True)
        self._stage_active_rows(entry, state_slot_ids)
        entry.graph.replay()
        return entry.static_audio, entry.static_audio_lengths


__all__ = ["MossIndexedVocoderCudaGraphRunner"]
