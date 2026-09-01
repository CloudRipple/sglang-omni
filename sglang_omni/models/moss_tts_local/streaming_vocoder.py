# SPDX-License-Identifier: Apache-2.0
"""Streaming vocoder scheduler for MOSS-TTS Local.

Streaming requests share one persistent indexed codec-state session.
Pure non-streaming traffic uses the MOSS decoder with packed SGLang FlashAttention
when no live streaming session owns the codec state.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Mapping

import torch

from sglang_omni.models.moss_tts.audio_tokenizer import MossAudioTokenizerVocoderDecoder
from sglang_omni.models.moss_tts.streaming_codec import (
    MossAudioTokenizerStreamingStatePool,
)
from sglang_omni.models.moss_tts_local.payload_types import MossTTSLocalState
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.pipeline_state import build_usage
from sglang_omni.scheduling.streaming_vocoder import (
    StreamingVocoderBase,
    resolve_initial_codec_chunk_frames,
)
from sglang_omni.utils.audio_payload import audio_waveform_payload

logger = logging.getLogger(__name__)

_SOURCE_HINT = "MOSS-TTS Local"
_SESSION_RESERVED_OFFLINE_SLOTS = 1
_OUTPUT_OVERLAP_POOL_SIZE = 2


@dataclass(eq=False)
class _PinnedOutputSlot:
    """Reusable pinned host staging slot for one in-flight codec step."""

    audio: torch.Tensor | None = None
    lengths: torch.Tensor | None = None
    event: Any | None = None


@dataclass
class _PendingOutputStep:
    """One asynchronous D2H copy and the request owners of each row."""

    slot: _PinnedOutputSlot
    owners: tuple[object, ...]


class _CodecStreamSession:
    """Persistent codec state session with slot bookkeeping.

    Live requests hold stream slots for their lifetime. Offline work always has
    reserved slots and can also borrow currently idle stream slots. All methods
    run on the scheduler-loop thread.
    """

    def __init__(
        self,
        codec: Any,
        *,
        stream_slots: int,
        offline_slots: int,
        n_vq: int,
        compact_streaming: bool = False,
        compile_streaming_decode: bool = False,
        compile_streaming_decode_shapes: list[tuple[int, int]] | None = None,
        output_overlap: bool = False,
    ) -> None:
        self._codec = codec
        self._stream_slots = int(stream_slots)
        self._offline_slots = int(offline_slots)
        self._batch_size = self._stream_slots + self._offline_slots
        self._n_vq = int(n_vq)
        self._compact_streaming = bool(compact_streaming)
        self._compile_streaming_decode = bool(compile_streaming_decode)
        self._compile_streaming_decode_shapes = (
            None
            if compile_streaming_decode_shapes is None
            else sorted(
                {
                    (int(batch_size), int(frame_size))
                    for batch_size, frame_size in compile_streaming_decode_shapes
                }
            )
        )
        self._device = next(codec.parameters()).device
        self._output_overlap_requested = bool(output_overlap)
        # The overlap candidate is CUDA-only in production.  Keeping this as a
        # separate runtime flag makes a failed allocator/event operation
        # fail-safe to the existing synchronous path.
        self._output_overlap_active = bool(
            self._output_overlap_requested and self._device.type == "cuda"
        )
        self._output_free: list[_PinnedOutputSlot] = []
        self._output_retired: list[_PinnedOutputSlot] = []
        self._output_created = 0
        self._pending_output: _PendingOutputStep | None = None
        self._output_overlap_started = False
        self._state_pool = MossAudioTokenizerStreamingStatePool(
            codec,
            n_vq=self._n_vq,
        )
        self._native_indexed = self._state_pool.uses_native_api
        # Native compact graphs pad the execution batch with scratch state
        # rows.  Legacy compatibility execution remains exactly fixed-width
        # and therefore has no scratch allocation.
        self._graph_scratch_capacity = (
            self._stream_slots if self._native_indexed and self._compact_streaming else 0
        )
        self._state_pool.initialize_decoder_state_pool(
            self._batch_size,
            scratch_capacity=self._graph_scratch_capacity,
        )
        self._all_slot_ids = self._state_pool.slot_ids[: self._batch_size]
        self._free_stream_slots = list(range(self._stream_slots))
        self._stream_slots_in_use: set[int] = set()
        self._closed = False
        self._cg_runner: Any | None = None
        # Capture is attempted at most once per session; a low-VRAM skip must not re-probe per step.
        self.warmup_attempted = False
        # Per-T graph-vs-eager step counts for capture-hit-rate reporting (host-side, no GPU sync).
        self._cg_graph_t: Counter = Counter()
        self._cg_eager_t: Counter = Counter()
        self._cg_total_steps = 0
        self._compact_batch_sizes: Counter = Counter()
        self._compact_shape_sizes: Counter = Counter()
        # The state pool owns the streaming context and reset traversal. Both
        # legacy fixed-width and native indexed runners use the same replay
        # failure accounting below.

    def warmup_cuda_graph(
        self, frames: list[int], *, min_free_gb: float = 3.0
    ) -> list[int]:
        """Capture per-T graphs then reset all slots; returns the captured T list (rest fall back to
        eager). Attempted at most once per session; never captures during ``step``."""
        self.warmup_attempted = True
        if self._closed:
            return []
        if self._native_indexed and self._compact_streaming:
            from sglang_omni.models.moss_tts_local.indexed_vocoder_cuda_graph import (
                MossIndexedVocoderCudaGraphRunner,
            )

            if self._cg_runner is None:
                self._cg_runner = MossIndexedVocoderCudaGraphRunner(
                    self._codec,
                    real_state_capacity=self._batch_size,
                    scratch_capacity=self._graph_scratch_capacity,
                    batch_sizes=self._graph_batch_sizes(),
                    frame_sizes=frames,
                    num_quantizers=self._n_vq,
                    min_free_gb=min_free_gb,
                    compile_decode=self._compile_streaming_decode,
                    compile_decode_shapes=self._compile_streaming_decode_shapes,
                )
            try:
                self._cg_runner.warmup(frames)
            except Exception:
                self._cg_runner = None
                raise
            captured = self._cg_runner.captured_frames()
            if not captured:
                self._cg_runner = None
            return captured
        if self._native_indexed:
            logger.info(
                "Skipping indexed MOSS vocoder CUDA graphs because compact_streaming "
                "is disabled; using the fixed-width eager native path"
            )
            return []
        from sglang_omni.models.moss_tts_local.vocoder_cuda_graph import (
            MossVocoderCudaGraphRunner,
            patch_codec_attention_cache_for_cuda_graph,
        )

        # Patch the codec attention cache to an in-place write so the graph can capture it
        # (bit-identical to eager).
        patch_codec_attention_cache_for_cuda_graph(self._codec)
        if self._cg_runner is None:
            # Scheduler owns the capture shape range (max_frames = the largest T it asks for), rather
            # than the runner keeping an independent default limit.
            self._cg_runner = MossVocoderCudaGraphRunner(
                self._codec,
                batch_size=self._batch_size,
                n_vq=self._n_vq,
                max_frames=max(frames) if frames else 1,
                min_free_gb=min_free_gb,
            )
        try:
            self._cg_runner.warmup(frames)
        except Exception:
            # Drop a half-built runner on probe failure so serving stays on the eager path.
            self._cg_runner = None
            raise
        self._reset_slots(list(range(self._batch_size)))
        captured = self._cg_runner.captured_frames()
        if not captured:
            # Nothing captured (low VRAM / all failed): drop the runner so serving does not pay a
            # wasted decode_step probe every step only to fall back to eager.
            self._cg_runner = None
        return captured

    def has_cuda_graph_runner(self) -> bool:
        # True only if the runner exists AND captured at least one graph.
        return bool(self._cg_runner and self._cg_runner.captured_frames())

    def captured_frames(self) -> list[int]:
        return self._cg_runner.captured_frames() if self._cg_runner else []

    def _graph_batch_sizes(self) -> list[int]:
        """Compact graph buckets bounded by the number of live stream slots."""
        buckets = [1, 2, 4, 8, 12, 16, self._stream_slots]
        return sorted(
            {
                bucket
                for bucket in buckets
                if 0 < int(bucket) <= self._stream_slots
            }
        )

    def acquire(self) -> int | None:
        if not self._free_stream_slots:
            return None
        slot = self._free_stream_slots.pop()
        self._stream_slots_in_use.add(slot)
        return slot

    def release(self, slot: int) -> None:
        if self._closed:
            return
        if slot not in self._stream_slots_in_use:
            raise RuntimeError(f"MOSS vocoder stream slot {slot} is not leased")
        self._reset_slots([slot])
        self._stream_slots_in_use.remove(slot)
        self._free_stream_slots.append(slot)

    def close(self) -> None:
        if self._closed:
            return
        if self._pending_output is not None:
            try:
                # Shutdown is the one lifecycle point where waiting for the
                # last copy is intentional; no client is waiting on this
                # scheduler anymore.
                self.flush_pending(wait=True)
            except Exception:
                logger.exception(
                    "MOSS vocoder failed to drain an asynchronous output copy "
                    "during shutdown"
                )
        if self._cg_runner is not None:
            self._log_cg_stats()
        if self._compact_streaming and self._compact_batch_sizes:
            logger.info(
                "MOSS vocoder compact streaming batches: B=%s",
                dict(sorted(self._compact_batch_sizes.items())),
            )
        if self._compact_streaming and self._compact_shape_sizes:
            logger.info(
                "MOSS vocoder compact streaming shapes: BT=%s",
                dict(sorted(self._compact_shape_sizes.items())),
            )
        with torch.no_grad():
            self._state_pool.close_decoder_state_pool()
        self._cg_runner = None
        self._closed = True

    def _log_cg_stats(self) -> None:
        graph = sum(self._cg_graph_t.values())
        eager = sum(self._cg_eager_t.values())
        total = graph + eager
        if not total:
            return
        logger.info(
            "MOSS vocoder CG stats: %d/%d steps graphed (%.1f%%); graph T=%s eager T=%s",
            graph,
            total,
            100.0 * graph / total,
            dict(sorted(self._cg_graph_t.items())),
            dict(sorted(self._cg_eager_t.items())),
        )

    def _reset_slots(self, slots: list[int]) -> None:
        if not slots:
            return
        slot_ids = torch.as_tensor(slots, dtype=torch.long, device=self._device)
        with torch.no_grad():
            self._state_pool.reset_decoder_state_slots(slot_ids)

    # ------------------------------------------------------------------
    # Optional asynchronous output materialization
    # ------------------------------------------------------------------

    def _allocate_output_tensor(
        self,
        shape: tuple[int, ...],
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Allocate a pinned host tensor, overridable by CPU-only tests."""
        return torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)

    def _new_output_event(self) -> Any:
        if self._device.type == "cuda":
            with torch.cuda.device(self._device):
                return torch.cuda.Event()
        # This branch is only used by CPU fakes that explicitly force the
        # candidate on; normal CPU serving never requests output overlap.
        return torch.cuda.Event()

    def _reap_retired_output_slots(self) -> None:
        if not self._output_retired:
            return
        still_retired: list[_PinnedOutputSlot] = []
        for slot in self._output_retired:
            try:
                complete = self._event_query(slot)
            except Exception:
                logger.exception("MOSS vocoder failed to query a retired output copy")
                continue
            if complete:
                self._output_free.append(slot)
            else:
                still_retired.append(slot)
        self._output_retired = still_retired

    def _acquire_output_slot(self) -> _PinnedOutputSlot | None:
        self._reap_retired_output_slots()
        if self._output_free:
            return self._output_free.pop()
        if self._output_created >= _OUTPUT_OVERLAP_POOL_SIZE:
            return None
        try:
            slot = _PinnedOutputSlot(event=self._new_output_event())
        except Exception:
            logger.exception(
                "MOSS vocoder asynchronous output event allocation failed; "
                "falling back to synchronous materialization"
            )
            self._output_overlap_active = False
            return None
        self._output_created += 1
        return slot

    def _release_output_slot(self, slot: _PinnedOutputSlot) -> None:
        self._output_free.append(slot)

    def _retire_output_slot(self, slot: _PinnedOutputSlot) -> None:
        # A failed event/copy cannot be safely returned to the pool.  Retain
        # it until its event is known complete, then reuse it on a later step.
        self._output_retired.append(slot)

    def _event_query(self, slot: _PinnedOutputSlot) -> bool:
        if slot.event is None:
            return True
        if self._device.type == "cuda":
            with torch.cuda.device(self._device):
                return bool(slot.event.query())
        return bool(slot.event.query())

    def _event_synchronize(self, slot: _PinnedOutputSlot) -> None:
        if slot.event is None:
            return
        if self._device.type == "cuda":
            with torch.cuda.device(self._device):
                slot.event.synchronize()
        else:
            slot.event.synchronize()

    @staticmethod
    def _owner_key(slot: int, owner_ids: Mapping[int, object] | None) -> object:
        return slot if owner_ids is None else owner_ids.get(slot, slot)

    def _select_output_rows(
        self,
        audio: torch.Tensor,
        audio_lengths: torch.Tensor,
        slots: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._compact_streaming:
            return audio, audio_lengths
        index = torch.as_tensor(slots, dtype=torch.long, device=audio.device)
        return audio.index_select(0, index), audio_lengths.index_select(0, index)

    def _launch_async_output(
        self,
        slot: _PinnedOutputSlot,
        audio: torch.Tensor,
        audio_lengths: torch.Tensor,
        slots: list[int],
        owner_ids: Mapping[int, object] | None,
    ) -> _PendingOutputStep:
        selected_audio, selected_lengths = self._select_output_rows(
            audio, audio_lengths, slots
        )
        rows, channels, samples = map(int, selected_audio.shape)
        if (
            slot.audio is None
            or tuple(slot.audio.shape) != (rows, channels, samples)
            or slot.audio.dtype != torch.float32
        ):
            slot.audio = self._allocate_output_tensor(
                (rows, channels, samples), dtype=torch.float32
            )
        if (
            slot.lengths is None
            or tuple(slot.lengths.shape) != (rows,)
            or slot.lengths.dtype != torch.long
        ):
            slot.lengths = self._allocate_output_tensor(
                (rows,), dtype=torch.long
            )
        # The copy and event are enqueued on the current codec stream.  This
        # preserves CUDA-graph static-output lifetime: the next replay cannot
        # overwrite a graph buffer until this copy has been ordered before it.
        slot.audio.copy_(selected_audio.detach().to(torch.float32), non_blocking=True)
        slot.lengths.copy_(selected_lengths.detach(), non_blocking=True)
        assert slot.event is not None
        slot.event.record()
        return _PendingOutputStep(
            slot=slot,
            owners=tuple(self._owner_key(item, owner_ids) for item in slots),
        )

    def _materialize_sync_output(
        self,
        audio: torch.Tensor,
        audio_lengths: torch.Tensor,
        slots: list[int],
        owner_ids: Mapping[int, object] | None,
    ) -> dict[object, torch.Tensor]:
        selected_audio, selected_lengths = self._select_output_rows(
            audio, audio_lengths, slots
        )
        audio_cpu = selected_audio.detach().to("cpu", torch.float32)
        lengths_cpu = selected_lengths.detach().to("cpu")
        return {
            self._owner_key(stream_slot, owner_ids): audio_cpu[index, :, : int(lengths_cpu[index])]
            for index, stream_slot in enumerate(slots)
        }

    def _materialize_pending(
        self,
        pending: _PendingOutputStep,
        *,
        wait: bool,
    ) -> dict[object, torch.Tensor]:
        if not wait and not self._event_query(pending.slot):
            return {}
        self._event_synchronize(pending.slot)
        slot = pending.slot
        if slot.audio is None or slot.lengths is None:
            raise RuntimeError("MOSS vocoder pending output slot is uninitialized")
        # Clone before returning the slot to the pool.  The returned tensors
        # are owned by the outgoing message and cannot alias a later replay.
        audio = slot.audio.clone()
        lengths = slot.lengths.clone().tolist()
        result = {
            owner: audio[index, :, : int(lengths[index])].contiguous()
            for index, owner in enumerate(pending.owners)
        }
        self._release_output_slot(slot)
        return result

    def flush_pending(self, *, wait: bool = True) -> dict[object, torch.Tensor]:
        """Return the previous step's output, if its D2H copy is ready."""
        pending = self._pending_output
        if pending is None:
            return {}
        try:
            result = self._materialize_pending(pending, wait=wait)
        except Exception:
            # Keep ownership on any failure; a later abort/shutdown can still
            # retire the slot instead of accidentally reusing a live buffer.
            raise
        if result:
            self._pending_output = None
        return result

    def step(
        self,
        slot_codes: dict[int, torch.Tensor],
        *,
        owner_ids: Mapping[int, object] | None = None,
        allow_overlap: bool = False,
    ) -> dict[object, torch.Tensor]:
        """Advance participating slots by one uniform-length step.

        ``owner_ids`` lets the scheduler keep a delayed output associated with
        the original request even if its codec slot is released and reused
        before the host copy is drained.  Callers that omit it retain the
        historical slot-keyed return contract.
        """
        if not slot_codes:
            return {}
        for slot, codes in slot_codes.items():
            if not isinstance(slot, int) or isinstance(slot, bool):
                raise TypeError(f"streaming slot id must be an int, got {slot!r}")
            if slot < 0 or slot >= self._batch_size:
                raise ValueError(
                    f"streaming slot {slot} is outside [0, {self._batch_size})"
                )
            if int(codes.ndim) != 2:
                raise ValueError(
                    f"streaming slot {slot} codes must have shape [NQ, T], "
                    f"got {tuple(codes.shape)}"
                )
            if int(codes.shape[0]) <= 0 or int(codes.shape[1]) <= 0:
                raise ValueError(
                    f"streaming slot {slot} codes must have positive NQ and T, "
                    f"got {tuple(codes.shape)}"
                )
        step_lengths = {int(codes.shape[1]) for codes in slot_codes.values()}
        if len(step_lengths) != 1:
            raise ValueError(
                f"streaming step requires a uniform length, got {sorted(step_lengths)}"
            )
        (step_t,) = step_lengths
        n_vq = int(next(iter(slot_codes.values())).shape[0])
        if any(int(codes.shape[0]) != n_vq for codes in slot_codes.values()):
            raise ValueError("all streaming slots must use the same quantizer count")
        slots = list(slot_codes)
        if self._compact_streaming:
            codes_step = torch.stack(
                [
                    codes.to(device=self._device, dtype=torch.long)
                    for codes in slot_codes.values()
                ],
                dim=1,
            )
            codes_lengths = torch.full(
                (len(slots),),
                step_t,
                dtype=torch.long,
                device=self._device,
            )
            state_slot_ids = torch.as_tensor(
                slots,
                dtype=torch.long,
                device=self._device,
            )
            exec_mask = torch.ones(
                len(slots), dtype=torch.bool, device=self._device
            )
            self._compact_batch_sizes[len(slots)] += 1
            self._compact_shape_sizes[(len(slots), step_t)] += 1
        else:
            codes_step = torch.zeros(
                n_vq,
                self._batch_size,
                step_t,
                dtype=torch.long,
                device=self._device,
            )
            codes_lengths = torch.zeros(
                self._batch_size, dtype=torch.long, device=self._device
            )
            exec_mask = torch.zeros(
                self._batch_size, dtype=torch.bool, device=self._device
            )
            for slot, codes in slot_codes.items():
                codes_step[:, slot, :] = codes.to(
                    device=self._device, dtype=torch.long
                )
                codes_lengths[slot] = step_t
                exec_mask[slot] = True
            state_slot_ids = self._all_slot_ids
        graphed = None
        graph_failed = False
        try:
            with torch.no_grad():
                if self._cg_runner is not None:
                    try:
                        if self._native_indexed and self._compact_streaming:
                            graphed = self._cg_runner.decode_step(
                                codes_step,
                                state_slot_ids,
                            )
                        else:
                            graphed = self._cg_runner.decode_step(
                                codes_step,
                                exec_mask,
                            )
                    except Exception:
                        graph_failed = True
                        raise
                if graphed is not None:
                    audio, audio_lengths = graphed
                else:
                    result = self._state_pool.decode_streaming_batch(
                        codes_step,
                        codes_lengths,
                        state_slot_ids,
                        exec_mask,
                    )
                    audio, audio_lengths = result.audio, result.audio_lengths
        except Exception:
            # Graphed step failed (in decode_step or async on the D2H): disable the runner so future
            # steps go eager; participants abort. An eager-path error does not disable it.
            if self._cg_runner is not None and (graph_failed or graphed is not None):
                logger.exception(
                    "MOSS vocoder CUDA-graph replay failed (in decode_step or on output "
                    "materialization); disabling runner, serving eager from here"
                )
                self._cg_runner = None
            raise
        if self._cg_runner is not None:
            if graphed is not None:
                self._cg_graph_t[step_t] += 1
            else:
                self._cg_eager_t[step_t] += 1
            self._cg_total_steps += 1
            if self._cg_total_steps % 2000 == 0:
                self._log_cg_stats()
        # The first window remains synchronous so time-to-first-audio does not
        # acquire a new event/pinned-buffer dependency.  Later windows launch
        # their D2H copy, run the next codec step, and drain the previous copy
        # after the launch.  If any optional operation fails, the candidate is
        # disabled and this step falls back to the original synchronous path.
        previous = self._pending_output
        overlap_candidate = bool(allow_overlap and self._output_overlap_active)
        # Preserve TTFC for the first decoded window. Once that window has
        # been returned, every later window can be staged one step ahead even
        # when the previous staged copy was already polled and drained.
        first_sync = overlap_candidate and not self._output_overlap_started
        pipeline = overlap_candidate and not first_sync
        current_slot: _PinnedOutputSlot | None = None
        if pipeline:
            current_slot = self._acquire_output_slot()
            if current_slot is None:
                # Pool exhaustion is bounded and rare; preserve order by
                # draining the old copy before doing a synchronous step.
                ready = self.flush_pending(wait=True)
                previous = None
                pipeline = False
            else:
                ready = {}
        else:
            ready = {}
            if previous is not None:
                # Final/tail paths explicitly disable overlap and must not
                # leave a prior window ahead of their terminal audio.
                ready.update(self.flush_pending(wait=True))
                previous = None

        if pipeline and current_slot is not None:
            try:
                current = self._launch_async_output(
                    current_slot,
                    audio,
                    audio_lengths,
                    slots,
                    owner_ids,
                )
            except Exception:
                logger.exception(
                    "MOSS vocoder asynchronous output materialization failed; "
                    "falling back to synchronous D2H"
                )
                self._output_overlap_active = False
                self._retire_output_slot(current_slot)
                # Drain the previous copy before exposing this step, keeping
                # per-request ordering identical to the synchronous path.
                if previous is not None:
                    ready.update(self.flush_pending(wait=True))
                ready.update(
                    self._materialize_sync_output(
                        audio, audio_lengths, slots, owner_ids
                    )
                )
                self._output_overlap_started = True
                return ready

            # Publish the new pending step before waiting on the old one.  A
            # flush failure therefore leaves both ownership records intact for
            # shutdown/abort handling.
            self._pending_output = current
            self._output_overlap_started = True
            if previous is not None:
                ready.update(self._materialize_pending(previous, wait=True))
            return ready

        ready.update(
            self._materialize_sync_output(audio, audio_lengths, slots, owner_ids)
        )
        if overlap_candidate:
            self._output_overlap_started = True
        return ready

    def decode_offline(
        self,
        codes_list: list[torch.Tensor],
        *,
        max_step_frames: int,
        max_batch_size: int,
    ) -> list[torch.Tensor]:
        """Decode complete utterances ``[n_vq, T]`` through offline slots in the
        persistent codec session."""
        if not codes_list:
            return []
        wavs: list[torch.Tensor] = []
        reserved = list(range(self._stream_slots, self._batch_size))
        requested_wave_size = min(max(int(max_batch_size), 1), len(codes_list))
        borrow_count = min(
            max(requested_wave_size - len(reserved), 0),
            len(self._free_stream_slots),
        )
        borrowed = self._free_stream_slots[-borrow_count:] if borrow_count else []
        if borrowed:
            del self._free_stream_slots[-borrow_count:]
        available = reserved + borrowed
        # Note(Chenchen Hong): Under stream saturation, offline batches degrade
        # to serial waves and block the scheduler pump until decoding completes.
        wave_size = min(requested_wave_size, len(available))
        decode_succeeded = False
        try:
            for wave_start in range(0, len(codes_list), wave_size):
                wave = codes_list[wave_start : wave_start + wave_size]
                slots = available[: len(wave)]
                self._reset_slots(slots)
                cursors = [0] * len(wave)
                chunks: list[list[torch.Tensor]] = [[] for _ in wave]
                while True:
                    remaining = [
                        int(codes.shape[1]) - cur for codes, cur in zip(wave, cursors)
                    ]
                    positive = [r for r in remaining if r > 0]
                    if not positive:
                        break
                    if any(r >= max_step_frames for r in positive):
                        step_t = max_step_frames
                    else:
                        step_t = min(positive)
                    plan = {
                        slots[i]: wave[i][:, cursors[i] : cursors[i] + step_t]
                        for i, rem in enumerate(remaining)
                        if rem >= step_t
                    }
                    decoded = self.step(plan)
                    for i in range(len(wave)):
                        if slots[i] in plan:
                            chunks[i].append(decoded[slots[i]])
                            cursors[i] += step_t
                for item_chunks in chunks:
                    wavs.append(torch.cat(item_chunks, dim=-1))
            decode_succeeded = True
        finally:
            if borrowed:
                try:
                    self._reset_slots(borrowed)
                except Exception:
                    logger.exception(
                        "MOSS vocoder failed to reset borrowed stream slots %s; "
                        "quarantining them",
                        borrowed,
                    )
                    if decode_succeeded:
                        raise
                else:
                    self._free_stream_slots.extend(borrowed)
        return wavs


@dataclass
class _LocalStreamState:
    slot: int | None = None
    pending: list[torch.Tensor] = field(default_factory=list)
    n_vq: int | None = None
    initial_chunk_frames: int = 0
    threshold: int = 0


@dataclass
class _CoalescedStepPlan:
    step_t: int
    slot_codes: dict[int, torch.Tensor]


class MossTTSLocalStreamingVocoderScheduler(
    StreamingVocoderBase[_LocalStreamState, _CoalescedStepPlan]
):
    """Decode MOSS-TTS Local codec rows incrementally on the v2 codec."""

    _can_batch_stream_chunks = True
    _can_batch_streaming_requests = True
    _stream_chunk_batch_distinct_requests = True

    def __init__(
        self,
        codec: Any,
        *,
        n_vq: int,
        sample_rate: int,
        stream_slots: int = 15,
        stream_chunk_frames: int = 25,
        initial_chunk_frames: int = 5,
        coalesce_floor_frames: int = 5,
        max_step_frames: int = 100,
        max_batch_size: int = 8,
        max_batch_wait_ms: int = 2,
        stream_chunk_batch_wait_ms: float = 0.0,
        cuda_graph: bool = True,
        cuda_graph_frames: list[int] | None = None,
        cuda_graph_min_free_gb: float = 3.0,
        compact_streaming: bool = False,
        compile_streaming_decode: bool = False,
        compile_streaming_decode_shapes: list[tuple[int, int]] | None = None,
        stream_output_overlap: bool = False,
    ) -> None:
        if stream_slots < 1:
            raise ValueError(f"stream_slots must be >= 1, got {stream_slots}")
        if not 0 < stream_chunk_frames <= max_step_frames:
            raise ValueError(
                "stream_chunk_frames must be in (0, max_step_frames], got "
                f"{stream_chunk_frames} (max_step_frames={max_step_frames})"
            )
        native_state_api = all(
            callable(getattr(codec, name, None))
            for name in (
                "initialize_decoder_state_pool",
                "reset_decoder_state_slots",
                "close_decoder_state_pool",
                "decode_streaming_batch",
                "decode_streaming_tensors",
            )
        )
        self._native_state_api = native_state_api
        legacy_missing = [
            name
            for name in ("streaming", "_set_streaming_exec_mask", "_decode_frame")
            if not callable(getattr(codec, name, None))
        ]
        if not native_state_api and legacy_missing:
            raise RuntimeError(
                f"MOSS-TTS Local streaming vocoder: codec is missing {legacy_missing}; "
                "the installed MOSS-Audio-Tokenizer-v2 version is incompatible"
            )
        if not callable(getattr(codec, "decode", None)):
            raise RuntimeError(
                "MOSS-TTS Local streaming vocoder: codec is missing callable "
                "decode for the non-streaming path"
            )
        if isinstance(codec.decoder, MossAudioTokenizerVocoderDecoder):
            nonstream_decoder = codec.decoder
        else:
            nonstream_decoder = MossAudioTokenizerVocoderDecoder(codec.decoder)
        logger.info(
            "MOSS-TTS Local non-streaming vocoder uses packed SGLang attention "
            "stages=%d",
            len(nonstream_decoder),
        )
        self._codec = codec
        self._nonstream_decoder = nonstream_decoder
        logger.info(
            "MOSS-TTS Local streaming vocoder execution: compact_streaming=%s "
            "codec_api=%s output_overlap=%s",
            bool(compact_streaming),
            "native-indexed" if native_state_api else "legacy-fixed-width",
            bool(stream_output_overlap),
        )
        self._stream_slots = int(stream_slots)
        # Coalesce up to one full set of streaming lanes per pump, not the offline batch width.
        self._stream_chunk_batch_max = self._stream_slots
        self._stream_chunk_frames = int(stream_chunk_frames)
        self._default_initial_chunk_frames = max(
            0, min(int(initial_chunk_frames), int(stream_chunk_frames))
        )
        self._coalesce_floor_frames = max(
            0, min(int(coalesce_floor_frames), int(stream_chunk_frames))
        )
        self._max_step_frames = int(max_step_frames)
        # Pure non-streaming traffic closes the idle session and uses the packed
        # batch path. Keep one overflow lane for progress while streams are live
        # instead of reserving half of every streaming CUDA graph for that case.
        self._offline_slots = _SESSION_RESERVED_OFFLINE_SLOTS
        self._n_vq = int(n_vq)
        self._session: _CodecStreamSession | None = None
        self._session_used_by_streaming = False
        self._cuda_graph = bool(cuda_graph)
        self._compact_streaming = bool(compact_streaming)
        self._compile_streaming_decode = bool(compile_streaming_decode)
        self._compile_streaming_decode_shapes = (
            None
            if compile_streaming_decode_shapes is None
            else sorted(
                {
                    (int(batch_size), int(frame_size))
                    for batch_size, frame_size in compile_streaming_decode_shapes
                }
            )
        )
        self._stream_output_overlap = bool(stream_output_overlap)
        self._cuda_graph_frames = (
            [int(t) for t in cuda_graph_frames] if cuda_graph_frames else None
        )
        self._cuda_graph_min_free_gb = float(cuda_graph_min_free_gb)
        if self._cuda_graph_frames is not None:
            too_large = [
                t for t in self._cuda_graph_frames if t > self._max_step_frames
            ]
            if too_large:
                raise ValueError(
                    f"cuda_graph_frames exceed max_step_frames={self._max_step_frames}: "
                    f"{too_large}"
                )
        super().__init__(
            self._vocode,
            batch_compute_fn=self._vocode_batch,
            sample_rate=sample_rate,
            stream_source_hint=_SOURCE_HINT,
            max_batch_size=max_batch_size,
            max_batch_wait_ms=max_batch_wait_ms,
            stream_chunk_batch_wait_ms=stream_chunk_batch_wait_ms,
        )

    def on_serving_stop(self) -> None:
        if self._session is not None:
            self._session.close()
            self._session = None
        self._session_used_by_streaming = False

    def _resolve_output_owner(self, key: object) -> str | None:
        if isinstance(key, str):
            return key
        for request_id, state in self._stream_state_items():
            if state.slot == key:
                return request_id
        return None

    def _emit_decoded_outputs(
        self,
        decoded: Mapping[object, torch.Tensor],
        *,
        current_ids: set[str],
    ) -> dict[str, torch.Tensor]:
        """Route immediate and one-step-delayed outputs without slot aliasing."""
        current: dict[str, torch.Tensor] = {}
        for key, waveform in decoded.items():
            request_id = self._resolve_output_owner(key)
            if request_id is None or self._is_aborted(request_id):
                continue
            if request_id in current_ids:
                current[request_id] = waveform
                continue
            self._mark_stream_emitted(request_id)
            self.outbox.put(self._stream_chunk_message(request_id, waveform))
        return current

    def _emit_pending_outputs(self, *, wait: bool) -> None:
        """Drain session output copies and enqueue any requests not in a step."""
        with self._state_lock:
            if self._session is None:
                return
            decoded = self._session.flush_pending(wait=wait)
            self._emit_decoded_outputs(decoded, current_ids=set())

    def _next_message(self):
        # Event.query() is non-blocking; polling here lets a lone stream emit
        # its final delayed window without waiting for another codec chunk.
        self._emit_pending_outputs(wait=False)
        return super()._next_message()

    def on_stream_done(self, request_id: str):
        # A pending threshold window must precede the terminal tail/result.
        self._emit_pending_outputs(wait=True)
        return super().on_stream_done(request_id)

    def on_stream_done_batch(self, request_ids: list[str]):
        # Drain the one-step output pipeline once before advancing any terminal
        # rows. The final batch itself always uses synchronous materialization.
        self._emit_pending_outputs(wait=True)
        return super().on_stream_done_batch(request_ids)

    def create_stream_state(self, request_id: str) -> _LocalStreamState:
        del request_id
        return _LocalStreamState()

    def latch_stream_contract(
        self,
        request_id: str,
        state: _LocalStreamState,
        source: StagePayload | Mapping[str, Any],
        *,
        origin: str,
    ) -> None:
        if origin == "payload":
            params = (
                source.request.params
                if isinstance(source.request.params, dict)
                else None
            )
            self._latch_thresholds(request_id, state, params)
            return
        metadata: Mapping[str, Any] = source
        n_vq = metadata.get("n_vq")
        if n_vq is not None:
            n_vq = int(n_vq)
            if state.n_vq is not None and state.n_vq != n_vq:
                raise ValueError(
                    f"MOSS-TTS Local stream n_vq changed for {request_id!r}: "
                    f"{state.n_vq} -> {n_vq}"
                )
            state.n_vq = n_vq
        if state.threshold == 0:
            self._latch_thresholds(request_id, state, metadata)

    def validate_chunk(
        self, request_id: str, state: _LocalStreamState, codes: torch.Tensor
    ) -> torch.Tensor:
        del request_id
        codes = codes.to(dtype=torch.long)
        n_vq = state.n_vq if state.n_vq is not None else self._n_vq
        if codes.ndim == 1 and int(codes.shape[0]) >= n_vq + 1:
            return codes[1 : 1 + n_vq]
        if codes.ndim == 2 and int(codes.shape[1]) >= n_vq + 1:
            return codes[:, 1 : 1 + n_vq]
        if codes.ndim not in (1, 2):
            shape_contract = "[channels] or [frames, channels]"
        else:
            shape_contract = f"at least {n_vq + 1} channels"
        raise ValueError(
            f"MOSS-TTS Local stream chunk must be {shape_contract}, "
            f"got {tuple(codes.shape)}"
        )

    def ingest(
        self, request_id: str, state: _LocalStreamState, codes: torch.Tensor
    ) -> None:
        del request_id
        if codes.ndim == 1:
            state.pending.append(codes)
        elif codes.ndim == 2:
            state.pending.extend(codes.unbind(0))
        else:
            raise ValueError(
                f"MOSS-TTS Local validated stream codes must be 1-D or 2-D, "
                f"got {tuple(codes.shape)}"
            )
        self._ensure_slot(state)

    def decode_delta(
        self, request_id: str, state: _LocalStreamState, *, is_final: bool
    ) -> torch.Tensor | None:
        """Stream-done drain: pending frames go through the request's session
        slot (released afterwards) or the offline lane when slot-starved;
        steady-state chunks decode through the coalesced step hooks instead."""
        del is_final
        audio_parts: list[torch.Tensor] = []
        if state.slot is None and state.pending:
            # Slot-starved: every frame is still buffered, decode offline.
            codes = torch.stack(state.pending, dim=1)
            state.pending = []
            audio_parts.extend(
                self._ensure_session_graphed().decode_offline(
                    [codes],
                    max_step_frames=self._max_step_frames,
                    max_batch_size=self._max_batch_size,
                )
            )
        elif state.slot is not None:
            session = self._ensure_session_graphed()
            # A threshold step may have an output copy in flight.  Final
            # tails must observe it before producing terminal audio.
            self._emit_pending_outputs(wait=True)
            while state.pending:
                step_t = min(len(state.pending), self._max_step_frames)
                codes = torch.stack(state.pending[:step_t], dim=1)
                del state.pending[:step_t]
                decoded = session.step(
                    {state.slot: codes},
                    owner_ids={state.slot: request_id},
                    allow_overlap=False,
                )
                self._emit_decoded_outputs(decoded, current_ids={request_id})
                waveform = decoded.get(request_id)
                if waveform is None:
                    raise RuntimeError(
                        f"MOSS vocoder final step produced no audio for {request_id!r}"
                    )
                audio_parts.append(waveform)
            session.release(state.slot)
            state.slot = None
        if not audio_parts:
            return None
        return torch.cat(audio_parts, dim=-1)

    def decode_final_batch(
        self, items: list[tuple[str, _LocalStreamState]]
    ) -> dict[str, torch.Tensor | None]:
        """Drain terminal rows together; only terminal rows may frame-pad."""
        outputs: dict[str, list[torch.Tensor]] = {
            request_id: [] for request_id, _ in items
        }
        session = self._ensure_session_graphed()

        unslotted = [
            (request_id, state)
            for request_id, state in items
            if state.slot is None and state.pending
        ]
        if unslotted:
            codes_list = [
                torch.stack(state.pending, dim=1) for _, state in unslotted
            ]
            wavs = session.decode_offline(
                codes_list,
                max_step_frames=self._max_step_frames,
                max_batch_size=self._max_batch_size,
            )
            for (request_id, state), waveform in zip(
                unslotted, wavs, strict=True
            ):
                state.pending.clear()
                outputs[request_id].append(waveform)

        slotted = [
            (request_id, state)
            for request_id, state in items
            if state.slot is not None
        ]
        for _, state in slotted:
            if not state.pending:
                assert state.slot is not None
                session.release(state.slot)
                state.slot = None

        active = [entry for entry in slotted if entry[1].pending]
        max_terminal_step = min(self._stream_chunk_frames, self._max_step_frames)
        while active:
            step_t = min(
                max(len(state.pending) for _, state in active),
                max_terminal_step,
            )
            slot_codes: dict[int, torch.Tensor] = {}
            owner_ids: dict[int, str] = {}
            real_frames: dict[str, int] = {}
            for request_id, state in active:
                assert state.slot is not None
                frame_count = min(len(state.pending), step_t)
                codes = torch.stack(state.pending[:frame_count], dim=1)
                if frame_count < step_t:
                    codes = torch.cat(
                        (
                            codes,
                            codes.new_zeros(
                                int(codes.shape[0]), step_t - frame_count
                            ),
                        ),
                        dim=1,
                    )
                slot_codes[state.slot] = codes
                owner_ids[state.slot] = request_id
                real_frames[request_id] = frame_count

            decoded = session.step(
                slot_codes,
                owner_ids=owner_ids,
                allow_overlap=False,
            )
            next_active: list[tuple[str, _LocalStreamState]] = []
            for request_id, state in active:
                frame_count = real_frames[request_id]
                waveform = decoded.get(request_id)
                if waveform is None:
                    raise RuntimeError(
                        f"MOSS vocoder final batch produced no audio for {request_id!r}"
                    )
                samples_per_frame = int(waveform.shape[-1]) // step_t
                outputs[request_id].append(
                    waveform[
                        ..., : frame_count * samples_per_frame
                    ].contiguous()
                )
                del state.pending[:frame_count]
                if state.pending:
                    next_active.append((request_id, state))
                else:
                    assert state.slot is not None
                    session.release(state.slot)
                    state.slot = None
            active = next_active

        return {
            request_id: (
                torch.cat(parts, dim=-1) if parts else None
            )
            for request_id, parts in outputs.items()
        }

    def stream_payload(self, request_id: str, waveform: torch.Tensor) -> dict[str, Any]:
        del request_id
        return audio_waveform_payload(
            waveform.detach().to("cpu", torch.float32),
            sample_rate=self._sample_rate,
            modality="audio",
            source_hint=f"{_SOURCE_HINT} streaming",
            keep_channels=True,
        )

    def fallback_full_decode(
        self, request_id: str, payload: StagePayload, state: _LocalStreamState
    ) -> torch.Tensor | None:
        del request_id, state
        return self._decode_payload_codes(payload)

    def final_result_data(
        self, request_id: str, payload: StagePayload, state: _LocalStreamState
    ) -> dict[str, Any]:
        del request_id, state
        final_data: dict[str, Any] = {
            "modality": "audio",
            "sample_rate": self._sample_rate,
        }
        usage = build_usage(MossTTSLocalState.from_dict(payload.data))
        if usage is not None:
            final_data["usage"] = usage
        return final_data

    def release_stream_resources(
        self, request_id: str, state: _LocalStreamState
    ) -> None:
        del request_id
        if state.slot is not None and self._session is not None:
            self._session.release(state.slot)

    def select_step_participants(self) -> list[tuple[str, _LocalStreamState]]:
        """Every stream whose buffer crossed its threshold is due; due streams
        coalesce with peers above the join floor into one forward."""
        join_floor = max(
            1, min(self._coalesce_floor_frames or 5, self._stream_chunk_frames)
        )
        slotted = [
            (request_id, state)
            for request_id, state in self._stream_state_items()
            if state.slot is not None and state.threshold > 0
        ]
        due = [
            entry for entry in slotted if len(entry[1].pending) >= entry[1].threshold
        ]
        if not due:
            return []
        floor = min(
            min(len(state.pending) for _, state in due),
            join_floor,
        )
        return [
            entry
            for entry in slotted
            if self._can_join_coalesced_step(entry[0], entry[1], floor)
        ]

    def build_step_plan(
        self, participants: list[tuple[str, _LocalStreamState]]
    ) -> _CoalescedStepPlan:
        """Uniform step capped at the steady chunk size and any un-emitted
        participant's first-chunk threshold; the base pump re-pumps remainder."""
        step_t = min(
            min(len(state.pending) for _, state in participants),
            self._stream_chunk_frames,
        )
        for request_id, state in participants:
            if not self._stream_has_emitted(request_id):
                step_t = min(step_t, state.threshold)
        return _CoalescedStepPlan(
            step_t=step_t,
            slot_codes={
                state.slot: torch.stack(state.pending[:step_t], dim=1)
                for _, state in participants
            },
        )

    def run_step(
        self,
        participants: list[tuple[str, _LocalStreamState]],
        plan: _CoalescedStepPlan,
    ) -> dict[str, torch.Tensor]:
        owner_ids = {
            state.slot: request_id
            for request_id, state in participants
            if state.slot is not None
        }
        decoded = self._ensure_session().step(
            plan.slot_codes,
            owner_ids=owner_ids,
            allow_overlap=self._stream_output_overlap,
        )
        current_ids = {request_id for request_id, _ in participants}
        out = self._emit_decoded_outputs(decoded, current_ids=current_ids)
        for request_id, state in participants:
            del state.pending[: plan.step_t]
            state.threshold = self._stream_chunk_frames
        return out

    def _can_join_coalesced_step(
        self, request_id: str, state: _LocalStreamState, floor: int
    ) -> bool:
        if len(state.pending) >= state.threshold:
            return True
        if not self._stream_has_emitted(request_id):
            return False
        return len(state.pending) >= floor

    def _ensure_session(self) -> _CodecStreamSession:
        if self._session is None:
            self._session = _CodecStreamSession(
                self._codec,
                stream_slots=self._stream_slots,
                offline_slots=self._offline_slots,
                n_vq=self._n_vq,
                compact_streaming=self._compact_streaming,
                compile_streaming_decode=self._compile_streaming_decode,
                compile_streaming_decode_shapes=(
                    self._compile_streaming_decode_shapes
                ),
                output_overlap=self._stream_output_overlap,
            )
        return self._session

    def _close_idle_startup_session_locked(self) -> None:
        if (
            self._session is not None
            and not self._session_used_by_streaming
            and not self._stream_states
            and not self._stream_payloads
        ):
            self._session.close()
            self._session = None

    def _cuda_graph_capture_frames(self) -> list[int]:
        """Step lengths T to capture. Config ``cuda_graph_frames`` overrides the default."""
        if self._cuda_graph_frames:
            # Validated at config (>= 1) and __init__ (<= max_step_frames); use as configured.
            return sorted(set(self._cuda_graph_frames))
        if self._compact_streaming and self._native_state_api:
            # Indexed graph replay requires an exact T because frame padding
            # advances causal decoder state.  The scheduler emits every
            # remainder length in [1, stream_chunk_frames], so capture the
            # complete bounded range by default; an explicit config list can
            # still select a smaller set when startup latency is prioritized.
            max_frame = min(self._stream_chunk_frames, self._max_step_frames)
            return list(range(1, max_frame + 1))
        max_frame = min(self._stream_chunk_frames, self._max_step_frames)
        return list(range(1, max_frame + 1))

    def _codec_on_cuda(self) -> bool:
        try:
            return next(self._codec.parameters()).device.type == "cuda"
        except StopIteration:
            return False

    def _ensure_session_graphed(self) -> _CodecStreamSession:
        """Live session with CUDA graphs captured (at most once). Streaming paths call this instead
        of _ensure_session so a session created after non-streaming traffic closed the graphed
        startup session is re-captured; a low-VRAM skip is remembered (no per-step re-probe).

        That first post-non-streaming streaming request pays a one-time warmup latency (the recapture
        runs synchronously here, fail-safe to eager on low VRAM); streaming-only traffic uses the
        factory session and never hits this path.
        """
        with self._state_lock:
            session = self._ensure_session()
            if (
                self._cuda_graph
                and (
                    not self._compact_streaming
                    or self._native_state_api
                )
                and not session.warmup_attempted
                and self._codec_on_cuda()
            ):
                try:
                    session.warmup_cuda_graph(
                        self._cuda_graph_capture_frames(),
                        min_free_gb=self._cuda_graph_min_free_gb,
                    )
                except Exception:
                    logger.exception(
                        "MOSS vocoder CUDA-graph capture failed; serving eager from this session"
                    )
            return session

    def warmup_now(self) -> None:
        """Capture the codec-decode graphs at factory-build time: codec loaded, GPU quiescent, and
        before the stage process is marked ready, so the serving loop never races a half-captured
        graph. No-op without a CUDA codec; best-effort, degrades to eager."""
        if (
            not self._cuda_graph
            or (self._compact_streaming and not self._native_state_api)
            or not self._codec_on_cuda()
        ):
            return
        session = self._ensure_session_graphed()
        if session.has_cuda_graph_runner():
            logger.info(
                "MOSS vocoder CUDA graphs captured at startup: T=%s",
                session.captured_frames(),
            )
        else:
            logger.warning(
                "MOSS vocoder CUDA graphs did not seal at startup (low VRAM); eager vocoder"
            )

    def _ensure_slot(self, state: _LocalStreamState) -> None:
        if state.slot is None:
            self._session_used_by_streaming = True
            state.slot = self._ensure_session_graphed().acquire()

    def _latch_thresholds(
        self,
        request_id: str,
        state: _LocalStreamState,
        params: Mapping[str, Any] | None,
    ) -> None:
        state.initial_chunk_frames = resolve_initial_codec_chunk_frames(
            params,
            steady_chunk_frames=self._stream_chunk_frames,
            default_frames=self._default_initial_chunk_frames,
        )
        if state.initial_chunk_frames > 0 and not self._stream_has_emitted(request_id):
            state.threshold = state.initial_chunk_frames
        else:
            state.threshold = self._stream_chunk_frames

    def _decode_payload_codes(self, payload: StagePayload) -> torch.Tensor | None:
        state = MossTTSLocalState.from_dict(payload.data)
        if state.audio_codes is None:
            return None
        rows = torch.as_tensor(state.audio_codes, dtype=torch.long)
        if rows.numel() == 0:
            return None
        codes = rows[:, : self._n_vq].transpose(0, 1).contiguous()
        self._session_used_by_streaming = True
        return self._ensure_session_graphed().decode_offline(
            [codes],
            max_step_frames=self._max_step_frames,
            max_batch_size=self._max_batch_size,
        )[0]

    def _prepare_codes(
        self, payload: StagePayload
    ) -> tuple[MossTTSLocalState, torch.Tensor | None]:
        state = MossTTSLocalState.from_dict(payload.data)
        if state.audio_codes is None:
            raise RuntimeError("MOSS-TTS Local vocoder requires audio_codes")
        codes = torch.as_tensor(state.audio_codes, dtype=torch.long)
        if codes.numel() == 0:
            # Emit no audio: only this request fails downstream, not the batch.
            return state, None
        return state, codes

    def _store_vocoder_result(
        self,
        payload: StagePayload,
        state: MossTTSLocalState,
        wav: torch.Tensor,
    ) -> StagePayload:
        # The v2 codec is natively stereo: keep [channels, samples] end to end.
        audio_payload = audio_waveform_payload(
            wav, source_hint=_SOURCE_HINT, keep_channels=True
        )
        state.audio_codes = None
        state.sample_rate = self._sample_rate
        payload.data = state.to_dict()
        payload.data.update(audio_payload)
        payload.data["sample_rate"] = state.sample_rate
        payload.data["modality"] = "audio"
        usage = build_usage(state)
        if usage is not None:
            payload.data["usage"] = usage
        return payload

    def _decode_codes_rows_nonstream(
        self, codes_list: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        n_vq = self._n_vq
        device = next(self._codec.parameters()).device
        codes_channels_first = [
            codes[:, :n_vq]
            .transpose(0, 1)
            .contiguous()
            .to(device=device, dtype=torch.long)
            for codes in codes_list
        ]
        max_len = max(int(codes.shape[1]) for codes in codes_channels_first)
        audio_codes = torch.zeros(
            n_vq,
            len(codes_channels_first),
            max_len,
            device=device,
            dtype=torch.long,
        )
        padding_mask = torch.zeros(
            len(codes_channels_first), max_len, device=device, dtype=torch.bool
        )
        for index, codes in enumerate(codes_channels_first):
            length = int(codes.shape[1])
            audio_codes[:, index, :length] = codes
            padding_mask[index, :length] = True

        decoded = self._codec.decode(
            audio_codes,
            padding_mask=padding_mask,
            num_quantizers=n_vq,
            return_dict=True,
            chunk_duration=None,
        )
        audio = decoded.audio
        audio_lengths = decoded.audio_lengths
        if audio is None or audio_lengths is None:
            raise RuntimeError(
                "audio_tokenizer.decode did not return audio/audio_lengths."
            )
        audio_cpu = audio.detach().to("cpu", torch.float32)
        lengths_cpu = audio_lengths.detach().to("cpu")
        return [
            audio_cpu[index, :, : int(lengths_cpu[index])].contiguous()
            for index in range(int(audio_cpu.shape[0]))
        ]

    def _decode_codes_rows(self, codes_list: list[torch.Tensor]) -> list[torch.Tensor]:
        """Decode ``[T, >=n_vq]`` row tensors to fp32 CPU waveforms."""
        with self._state_lock:
            self._close_idle_startup_session_locked()
        if self._session is None:
            if self._codec.decoder is not self._nonstream_decoder:
                original_decoder = self._codec.decoder
                self._codec.decoder = self._nonstream_decoder
                try:
                    return self._decode_codes_rows_nonstream(codes_list)
                finally:
                    self._codec.decoder = original_decoder
            return self._decode_codes_rows_nonstream(codes_list)
        channels_first = [
            codes[:, : self._n_vq].transpose(0, 1).contiguous() for codes in codes_list
        ]
        # abort() resets slots under _state_lock from other threads; serialize
        # every session access on the same lock.
        with self._state_lock:
            self._emit_pending_outputs(wait=True)
            wavs = self._session.decode_offline(
                channels_first,
                max_step_frames=self._max_step_frames,
                max_batch_size=self._max_batch_size,
            )
        return [wav.detach().to("cpu", torch.float32).contiguous() for wav in wavs]

    def _vocode_batch(self, payloads: list[StagePayload]) -> list[StagePayload]:
        prepared = [self._prepare_codes(payload) for payload in payloads]
        codes_list = [codes for _, codes in prepared if codes is not None]
        decoded = iter(self._decode_codes_rows(codes_list)) if codes_list else iter(())
        results = []
        for payload, (state, codes) in zip(payloads, prepared):
            if codes is None:
                state.audio_codes = None
                payload.data = state.to_dict()
                results.append(payload)
                continue
            results.append(self._store_vocoder_result(payload, state, next(decoded)))
        return results

    def _vocode(self, payload: StagePayload) -> StagePayload:
        return self._vocode_batch([payload])[0]


__all__ = ["MossTTSLocalStreamingVocoderScheduler"]
