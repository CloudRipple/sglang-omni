# SPDX-License-Identifier: Apache-2.0
"""Bucketed CUDA graphs for MOSS-TTS Delay packed vocoder decoding."""

from __future__ import annotations

import gc
import logging
import os
import threading
import traceback
from collections import Counter
from dataclasses import dataclass
from typing import Any

import torch

from sglang_omni.models.moss_tts_local.vocoder_decoder import (
    MossAudioTokenizerVocoderDecoder,
    MossVocoderStaticPackedPlan,
)

logger = logging.getLogger(__name__)

_DEFAULT_MIN_FRAME_BUCKET = 32
_DEFAULT_FRAME_BUCKET_STEP = 16
_DEFAULT_CAPTURE_WARMUPS = 3


@dataclass(frozen=True, order=True, slots=True)
class VocoderGraphKey:
    batch_size: int
    frames: int


@dataclass(slots=True)
class _CapturedVocoderGraph:
    key: VocoderGraphKey
    plan: MossVocoderStaticPackedPlan
    graph: torch.cuda.CUDAGraph
    static_hidden: torch.Tensor
    static_audio: torch.Tensor
    lock: threading.Lock


def make_vocoder_cuda_graph_keys(
    *,
    max_batch_size: int,
    max_frames: int,
) -> tuple[VocoderGraphKey, ...]:
    if max_batch_size < 1:
        raise ValueError("vocoder CUDA graph max_batch_size must be positive")
    if max_frames < 1:
        raise ValueError("vocoder CUDA graph max_frames must be positive")

    batch_buckets = []
    batch_size = 1
    while batch_size < max_batch_size:
        batch_buckets.append(batch_size)
        batch_size *= 2
    batch_buckets.append(max_batch_size)
    batch_buckets = sorted(set(batch_buckets))

    first_frame = min(_DEFAULT_MIN_FRAME_BUCKET, max_frames)
    frame_buckets = list(
        range(
            first_frame,
            max_frames + 1,
            _DEFAULT_FRAME_BUCKET_STEP,
        )
    )
    if frame_buckets[-1] != max_frames:
        frame_buckets.append(max_frames)

    return tuple(
        VocoderGraphKey(batch_size=batch, frames=frames)
        for batch in batch_buckets
        for frames in frame_buckets
    )


class MossTTSDelayVocoderCudaGraphRunner:
    """Capture and replay padded ``[B, hidden, T]`` decoder graphs.

    Quantizer/codebook work remains eager FP32. Graphs cover only the packed
    decoder under the configured FP16/BF16 autocast. Replay pads batch and frame
    dimensions to the smallest captured key and trims the borrowed graph output
    before returning a clone.
    """

    def __init__(
        self,
        decoder: MossAudioTokenizerVocoderDecoder,
        *,
        device: str | torch.device,
        autocast_dtype: torch.dtype,
        graph_keys: tuple[VocoderGraphKey, ...],
        min_free_gb: float,
        warmup_iters: int,
    ) -> None:
        self._decoder = decoder
        self._device = torch.device(device)
        if self._device.type != "cuda" or self._device.index is None:
            raise ValueError("MOSS vocoder CUDA graphs require a concrete CUDA device")
        if autocast_dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("MOSS vocoder CUDA graphs require FP16 or BF16 autocast")
        if not graph_keys:
            raise ValueError("MOSS vocoder CUDA graphs require at least one key")

        self._autocast_dtype = autocast_dtype
        self._graph_keys = graph_keys
        self._input_dimension = decoder.input_dimension()
        if min_free_gb < 0:
            raise ValueError("MOSS vocoder CUDA graph min_free_gb must be non-negative")
        self._min_free_bytes = int(float(min_free_gb) * (1024**3))
        self._warmup_iters = max(int(warmup_iters), 1)
        self._owner_pid = os.getpid()
        self._graphs: dict[VocoderGraphKey, _CapturedVocoderGraph] = {}
        self._pool: Any | None = None
        self._enabled = False
        self._disable_reason: str | None = None
        self._fallback_counts: Counter[str] = Counter()
        self._graph_hits: Counter[VocoderGraphKey] = Counter()
        self._graph_replays = 0

    @classmethod
    def build(
        cls,
        decoder: MossAudioTokenizerVocoderDecoder,
        *,
        device: str | torch.device,
        autocast_dtype: torch.dtype,
        max_batch_size: int,
        max_frames: int,
        min_free_gb: float = 8.0,
        warmup_iters: int = _DEFAULT_CAPTURE_WARMUPS,
    ) -> MossTTSDelayVocoderCudaGraphRunner:
        runner = cls(
            decoder,
            device=device,
            autocast_dtype=autocast_dtype,
            graph_keys=make_vocoder_cuda_graph_keys(
                max_batch_size=max_batch_size,
                max_frames=max_frames,
            ),
            min_free_gb=min_free_gb,
            warmup_iters=warmup_iters,
        )
        runner._capture()
        return runner

    def _capture(self) -> None:
        with torch.cuda.device(self._device):
            # Largest frame count first keeps the decoder's RoPE caches stable;
            # largest batch first lets the shared graph pool reach peak size
            # before smaller captures reuse it.
            ordered_keys = sorted(
                self._graph_keys,
                key=lambda key: (key.frames, key.batch_size),
                reverse=True,
            )
            for key in ordered_keys:
                free_bytes, _ = torch.cuda.mem_get_info(self._device)
                if free_bytes < self._min_free_bytes:
                    if not self._graphs:
                        self._disable_reason = "insufficient_free_memory"
                    logger.warning(
                        "MOSS-TTS Delay vocoder CUDA graph capture stopped: "
                        "free VRAM %.2f GiB < %.2f GiB before key=%s",
                        free_bytes / 1024**3,
                        self._min_free_bytes / 1024**3,
                        key,
                    )
                    break

                failure_traceback = None
                try:
                    self._graphs[key] = self._capture_key(key)
                except Exception:
                    failure_traceback = traceback.format_exc()
                if failure_traceback is not None:
                    self._graphs.pop(key, None)
                    logger.warning(
                        "MOSS-TTS Delay vocoder CUDA graph capture failed for "
                        "%s; this shape will use eager decode:\n%s",
                        key,
                        failure_traceback.rstrip(),
                    )
                    continue

                free_after, _ = torch.cuda.mem_get_info(self._device)
                if free_after < self._min_free_bytes:
                    self._disable_reason = "post_capture_memory_headroom"
                    logger.warning(
                        "MOSS-TTS Delay vocoder CUDA graphs would leave only "
                        "%.2f GiB free (< %.2f GiB); releasing captured graphs",
                        free_after / 1024**3,
                        self._min_free_bytes / 1024**3,
                    )
                    self._graphs.clear()
                    self._pool = None
                    gc.collect()
                    torch.cuda.empty_cache()
                    break

            self._enabled = bool(self._graphs)
            if not self._enabled and self._disable_reason is None:
                self._disable_reason = "no_graphs_captured"
            logger.info(
                "MOSS-TTS Delay vocoder CUDA graphs: captured=%d configured=%d "
                "dtype=%s keys=%s",
                len(self._graphs),
                len(self._graph_keys),
                self._autocast_dtype,
                sorted(self._graphs),
            )

    def _capture_key(self, key: VocoderGraphKey) -> _CapturedVocoderGraph:
        plan = self._decoder.build_static_packed_plan(
            batch_size=key.batch_size,
            input_frames=key.frames,
            device=self._device,
        )
        static_hidden = torch.zeros(
            key.batch_size,
            self._input_dimension,
            key.frames,
            device=self._device,
            dtype=torch.float32,
        )

        capture_stream = torch.cuda.Stream(device=self._device)
        current_stream = torch.cuda.current_stream(self._device)
        capture_stream.wait_stream(current_stream)
        with (
            torch.cuda.stream(capture_stream),
            torch.inference_mode(),
            torch.autocast(
                device_type="cuda",
                dtype=self._autocast_dtype,
            ),
        ):
            for _ in range(self._warmup_iters):
                self._decoder.forward_static_packed(static_hidden, plan)
        capture_stream.synchronize()
        current_stream.wait_stream(capture_stream)

        if self._pool is None:
            self._pool = torch.cuda.graph_pool_handle()
        graph = torch.cuda.CUDAGraph()
        try:
            with (
                torch.inference_mode(),
                torch.autocast(
                    device_type="cuda",
                    dtype=self._autocast_dtype,
                ),
            ):
                with torch.cuda.graph(
                    graph,
                    pool=self._pool,
                    stream=capture_stream,
                    capture_error_mode="thread_local",
                ):
                    static_audio = self._decoder.forward_static_packed(
                        static_hidden,
                        plan,
                    )
        finally:
            torch.cuda.set_stream(current_stream)
        current_stream.wait_stream(capture_stream)

        with (
            torch.inference_mode(),
            torch.autocast(
                device_type="cuda",
                dtype=self._autocast_dtype,
            ),
        ):
            eager_audio = self._decoder.forward_static_packed(
                static_hidden,
                plan,
            ).clone()
        graph.replay()
        torch.cuda.synchronize(self._device)
        if not torch.equal(eager_audio, static_audio):
            raise RuntimeError(f"CUDA graph output mismatch for {key}")

        return _CapturedVocoderGraph(
            key=key,
            plan=plan,
            graph=graph,
            static_hidden=static_hidden,
            static_audio=static_audio,
            lock=threading.Lock(),
        )

    def _select_key(self, batch_size: int, frames: int) -> VocoderGraphKey | None:
        return min(
            (
                key
                for key in self._graphs
                if key.batch_size >= batch_size and key.frames >= frames
            ),
            key=lambda key: (key.batch_size * key.frames, key.frames, key.batch_size),
            default=None,
        )

    @torch.no_grad()
    def run(
        self,
        hidden_states: torch.Tensor,
        input_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if os.getpid() != self._owner_pid:
            raise RuntimeError(
                "MOSS vocoder CUDA graph runner crossed a process boundary"
            )
        if not self._enabled:
            self._fallback_counts["disabled"] += 1
            return None
        if (
            hidden_states.ndim != 3
            or hidden_states.device != self._device
            or hidden_states.dtype != torch.float32
            or int(hidden_states.shape[1]) != self._input_dimension
        ):
            self._fallback_counts["invalid_input"] += 1
            return None
        if (
            input_lengths.ndim != 1
            or input_lengths.device != self._device
            or int(input_lengths.shape[0]) != int(hidden_states.shape[0])
        ):
            self._fallback_counts["invalid_lengths"] += 1
            return None

        batch_size = int(hidden_states.shape[0])
        frames = int(hidden_states.shape[2])
        key = self._select_key(batch_size, frames)
        if key is None:
            self._fallback_counts["key_miss"] += 1
            return None
        entry = self._graphs[key]
        output_factor = entry.plan.output_frames // entry.plan.input_frames

        failure_traceback = None
        audio = None
        try:
            with entry.lock:
                entry.static_hidden.zero_()
                entry.static_hidden[:batch_size, :, :frames].copy_(hidden_states)
                entry.graph.replay()
                audio = entry.static_audio[
                    :batch_size,
                    :,
                    : frames * output_factor,
                ].clone()
        except Exception:
            failure_traceback = traceback.format_exc()
        if failure_traceback is not None:
            self._disable_runtime("replay_failed")
            logger.error(
                "MOSS-TTS Delay vocoder CUDA graph replay failed; disabling "
                "graphs and retrying eagerly:\n%s",
                failure_traceback.rstrip(),
            )
            return None

        assert audio is not None
        self._graph_replays += 1
        self._graph_hits[key] += 1
        return audio, input_lengths * output_factor

    def _disable_runtime(self, reason: str) -> None:
        self._enabled = False
        self._disable_reason = reason
        self._graphs.clear()
        self._pool = None
        gc.collect()
        with torch.cuda.device(self._device):
            torch.cuda.empty_cache()

    def stats(self) -> dict[str, Any]:
        return {
            "enabled": self._enabled,
            "disable_reason": self._disable_reason,
            "configured_graphs": len(self._graph_keys),
            "captured_graphs": len(self._graphs),
            "captured_keys": [
                {"batch_size": key.batch_size, "frames": key.frames}
                for key in sorted(self._graphs)
            ],
            "graph_replays": self._graph_replays,
            "graph_hits": [
                {
                    "batch_size": key.batch_size,
                    "frames": key.frames,
                    "count": count,
                }
                for key, count in sorted(self._graph_hits.items())
            ],
            "fallback_counts": dict(sorted(self._fallback_counts.items())),
        }


__all__ = [
    "MossTTSDelayVocoderCudaGraphRunner",
    "VocoderGraphKey",
    "make_vocoder_cuda_graph_keys",
]
