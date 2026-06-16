# SPDX-License-Identifier: Apache-2.0
"""CUDA graph replay for the MOSS-TTS Local audio-tokenizer decoder."""

from __future__ import annotations

import logging
import math
import threading
from contextlib import nullcontext
from types import MethodType, SimpleNamespace
from typing import Any

import torch

logger = logging.getLogger(__name__)

codec_cuda_graph_capture_lock = threading.RLock()

_ROPE_CACHE_POSITIONS = 65536
_ROPE_CACHE_KEY_ATTR = "_sglang_omni_cuda_graph_rope_cache_key"
_ROPE_CACHE_COS_ATTR = "_sglang_omni_cuda_graph_rope_cos"
_ROPE_CACHE_SIN_ATTR = "_sglang_omni_cuda_graph_rope_sin"
_ROPE_CACHE_POS_ATTR = "_sglang_omni_cuda_graph_rope_positions"
_ROPE_ORIGINAL_FORWARD_ATTR = "_sglang_omni_original_rope_forward"
_ATTN_ORIGINAL_UPDATE_CACHE_ATTR = "_sglang_omni_original_update_streaming_cache"

_STATE_TENSOR_ATTRS = (
    "exec_mask",
    "offsets",
    "offset",
    "cached_keys",
    "cached_values",
    "cached_positions",
    "_flash_cached_keys",
    "_flash_cached_values",
)
_STATE_VALUE_ATTRS = ("offset_cpu",)


def _decoder_attention_modules(codec: Any) -> list[Any]:
    modules_by_id: dict[int, Any] = {}
    decoder = getattr(codec, "decoder", ())
    for decoder_module in decoder:
        modules = decoder_module.modules() if hasattr(decoder_module, "modules") else ()
        for module in modules:
            if hasattr(module, "attention_implementation"):
                modules_by_id.setdefault(id(module), module)
    return list(modules_by_id.values())


def _decoder_rope_modules(codec: Any) -> list[Any]:
    modules_by_id: dict[int, Any] = {}
    decoder = getattr(codec, "decoder", ())
    for decoder_module in decoder:
        modules = decoder_module.modules() if hasattr(decoder_module, "modules") else ()
        for module in modules:
            rope = getattr(module, "rope", None)
            if rope is not None and hasattr(rope, "max_period"):
                modules_by_id.setdefault(id(rope), rope)
    return list(modules_by_id.values())


def _decoder_rope_head_dims(codec: Any) -> dict[int, int]:
    head_dims: dict[int, int] = {}
    decoder = getattr(codec, "decoder", ())
    for decoder_module in decoder:
        modules = decoder_module.modules() if hasattr(decoder_module, "modules") else ()
        for module in modules:
            rope = getattr(module, "rope", None)
            embed_dim = getattr(module, "embed_dim", None)
            num_heads = getattr(module, "num_heads", None)
            if rope is None or embed_dim is None or num_heads is None:
                continue
            head_dims[id(rope)] = int(embed_dim) // int(num_heads)
    return head_dims


def _ensure_rope_cache(
    rope: Any,
    *,
    device: torch.device,
    head_dim: int,
    positions: int = _ROPE_CACHE_POSITIONS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cache_key = (str(device), int(head_dim), int(positions), float(rope.max_period))
    if getattr(rope, _ROPE_CACHE_KEY_ATTR, None) != cache_key:
        half_dim = int(head_dim) // 2
        ds = torch.arange(half_dim, device=device, dtype=torch.float32)
        freqs = torch.exp(ds * (-math.log(float(rope.max_period)) * 2 / int(head_dim)))
        position_ids_f = torch.arange(positions, device=device, dtype=torch.float32)
        phase = position_ids_f.view(-1, 1) * freqs.view(1, -1)
        setattr(rope, _ROPE_CACHE_KEY_ATTR, cache_key)
        setattr(rope, _ROPE_CACHE_COS_ATTR, torch.cos(phase))
        setattr(rope, _ROPE_CACHE_SIN_ATTR, torch.sin(phase))
        setattr(
            rope,
            _ROPE_CACHE_POS_ATTR,
            torch.arange(positions, device=device, dtype=torch.long),
        )
    return (
        getattr(rope, _ROPE_CACHE_COS_ATTR),
        getattr(rope, _ROPE_CACHE_SIN_ATTR),
        getattr(rope, _ROPE_CACHE_POS_ATTR),
    )


def _cached_rope_forward(
    self: Any,
    q: torch.Tensor,
    k: torch.Tensor,
    offset: torch.Tensor,
    time_before_heads: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if time_before_heads:
        batch_size, time_steps, _, head_dim = q.shape
    else:
        batch_size, _, time_steps, head_dim = q.shape
    if k.shape != q.shape:
        raise ValueError(
            f"Expected k.shape == q.shape, got k={tuple(k.shape)} q={tuple(q.shape)}"
        )
    if head_dim <= 0 or (head_dim % 2) != 0:
        raise ValueError(f"RoPE requires an even last dimension, got D={head_dim}")

    cos_cache, sin_cache, position_offsets = _ensure_rope_cache(
        self,
        device=q.device,
        head_dim=head_dim,
    )
    positions = offset.to(device=q.device, dtype=torch.long).view(
        -1, 1
    ) + position_offsets[:time_steps].view(1, -1)
    flat_positions = positions.reshape(-1)
    rotr = cos_cache.index_select(0, flat_positions).view(
        batch_size, time_steps, head_dim // 2
    )
    roti = sin_cache.index_select(0, flat_positions).view(
        batch_size, time_steps, head_dim // 2
    )
    if time_before_heads:
        rotr = rotr.view(batch_size, time_steps, 1, head_dim // 2)
        roti = roti.view(batch_size, time_steps, 1, head_dim // 2)
    else:
        rotr = rotr.view(batch_size, 1, time_steps, head_dim // 2)
        roti = roti.view(batch_size, 1, time_steps, head_dim // 2)

    dims = q.shape[:-1]
    q = q.view(*dims, head_dim // 2, 2)
    k = k.view(*dims, head_dim // 2, 2)

    qr, qi = q[..., 0].float(), q[..., 1].float()
    kr, ki = k[..., 0].float(), k[..., 1].float()
    qor = qr * rotr - qi * roti
    qoi = qr * roti + qi * rotr
    kor = kr * rotr - ki * roti
    koi = kr * roti + ki * rotr

    dtype = q.dtype
    qo = torch.stack([qor.to(dtype), qoi.to(dtype)], dim=-1)
    ko = torch.stack([kor.to(dtype), koi.to(dtype)], dim=-1)
    return qo.view(*dims, head_dim), ko.view(*dims, head_dim)


def patch_codec_rope_for_cuda_graph(codec: Any) -> None:
    """Make decoder RoPE graph-capturable by avoiding trig ops in capture."""

    try:
        device = next(codec.parameters()).device
    except Exception:
        return
    head_dims = _decoder_rope_head_dims(codec)
    for rope in _decoder_rope_modules(codec):
        if not hasattr(rope, _ROPE_ORIGINAL_FORWARD_ATTR):
            setattr(rope, _ROPE_ORIGINAL_FORWARD_ATTR, rope.forward)
            rope.forward = MethodType(_cached_rope_forward, rope)
        head_dim = head_dims.get(id(rope))
        if head_dim is not None:
            _ensure_rope_cache(rope, device=device, head_dim=head_dim)


def _cuda_graph_update_streaming_cache(
    self: Any,
    state: Any,
    cached_k: torch.Tensor,
    cached_v: torch.Tensor,
    cached_pos: torch.Tensor,
    k_all: torch.Tensor,
    v_all: torch.Tensor,
    pos_k: torch.Tensor,
) -> None:
    context = getattr(self, "context", None)
    original = getattr(self, _ATTN_ORIGINAL_UPDATE_CACHE_ATTR, None)
    if context is None:
        if callable(original):
            return original(state, cached_k, cached_v, cached_pos, k_all, v_all, pos_k)
        raise RuntimeError("CUDA graph codec attention requires finite context")

    state_cached_keys = getattr(state, "cached_keys", None)
    state_cached_values = getattr(state, "cached_values", None)
    state_cached_positions = getattr(state, "cached_positions", None)
    if (
        state_cached_keys is None
        or state_cached_values is None
        or state_cached_positions is None
    ):
        if callable(original):
            return original(state, cached_k, cached_v, cached_pos, k_all, v_all, pos_k)
        raise RuntimeError("CUDA graph codec attention cache is not initialized")

    exec_mask = state.exec_mask.view(-1, 1, 1, 1)
    exec_mask_pos = state.exec_mask.view(-1, 1)
    new_cached_k = k_all[:, :, -int(context) :, :].contiguous()
    new_cached_v = v_all[:, :, -int(context) :, :].contiguous()
    new_cached_pos = pos_k[:, -int(context) :].contiguous()
    state_cached_keys.copy_(torch.where(exec_mask, new_cached_k, cached_k))
    state_cached_values.copy_(torch.where(exec_mask, new_cached_v, cached_v))
    state_cached_positions.copy_(torch.where(exec_mask_pos, new_cached_pos, cached_pos))


def patch_codec_attention_cache_for_cuda_graph(codec: Any) -> None:
    """Keep streaming attention cache storage stable across graph replays."""

    for module in _decoder_attention_modules(codec):
        update_cache = getattr(module, "_update_streaming_cache", None)
        if not callable(update_cache):
            continue
        if hasattr(module, _ATTN_ORIGINAL_UPDATE_CACHE_ATTR):
            continue
        setattr(module, _ATTN_ORIGINAL_UPDATE_CACHE_ATTR, update_cache)
        module._update_streaming_cache = MethodType(
            _cuda_graph_update_streaming_cache, module
        )


def _set_attention_implementation(module: Any, attention_implementation: str) -> None:
    setter = getattr(module, "set_attention_implementation", None)
    if callable(setter):
        setter(attention_implementation)
    else:
        setattr(module, "attention_implementation", attention_implementation)


def set_codec_attention_backend_for_cuda_graph(codec: Any) -> None:
    """CUDA graph mode always uses the SDPA decoder attention path."""

    setter = getattr(codec, "set_attention_implementation", None)
    if callable(setter):
        setter("sdpa")
        return
    for module in _decoder_attention_modules(codec):
        _set_attention_implementation(module, "sdpa")


def ensure_codec_decoder_cuda_graph_surface(codec: Any) -> None:
    """Install the reference branch's decoder split if the HF codec lacks it."""

    if hasattr(codec, "_decode_hidden_states"):
        return
    if not hasattr(codec, "decoder") or not hasattr(
        codec, "_restore_channels_from_codec"
    ):
        return

    def _decode_hidden_states(
        self,
        decoder_hidden_states: torch.Tensor,
        codes_lengths: torch.Tensor,
    ) -> Any:
        autocast = (
            self._codec_inference_autocast()
            if hasattr(self, "_codec_inference_autocast")
            else nullcontext()
        )
        with autocast:
            audio, audio_lengths = decoder_hidden_states, codes_lengths
            for decoder_module in self.decoder:
                audio, audio_lengths = decoder_module(audio, audio_lengths)
        audio, audio_lengths = self._restore_channels_from_codec(audio, audio_lengths)
        return SimpleNamespace(audio=audio, audio_lengths=audio_lengths)

    codec._decode_hidden_states = MethodType(_decode_hidden_states, codec)


def codec_cuda_graph_supported(codec: Any) -> bool:
    if not torch.cuda.is_available():
        return False
    if not hasattr(codec, "_decode_hidden_states"):
        return False
    quantizer = getattr(codec, "quantizer", None)
    if quantizer is None or not hasattr(quantizer, "decode_codes"):
        return False
    try:
        param = next(codec.parameters())
    except Exception:
        return False
    return isinstance(param, torch.Tensor) and param.device.type == "cuda"


class AudioTokenizerDecoderCudaGraph:
    """Capture and replay the codec decoder while preserving streaming state.

    The quantizer lookup stays eager because it is cheap and can handle changing
    code values. The expensive decoder stack receives static hidden-state and
    length buffers under a CUDA graph keyed by static shape.
    """

    def __init__(self, codec: Any) -> None:
        if not codec_cuda_graph_supported(codec):
            raise RuntimeError(
                "MOSS-TTS Local codec does not expose a CUDA graph-compatible "
                "decoder surface"
            )
        self._codec = codec
        set_codec_attention_backend_for_cuda_graph(codec)
        patch_codec_rope_for_cuda_graph(codec)
        patch_codec_attention_cache_for_cuda_graph(codec)
        self._cache: dict[
            tuple[Any, ...],
            tuple[
                torch.cuda.CUDAGraph,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
            ],
        ] = {}
        self._disabled_reason: str | None = None
        self._streaming_states_cache: list[Any] | None = None

    @property
    def disabled_reason(self) -> str | None:
        return self._disabled_reason

    def decode_frame(
        self,
        codes: torch.Tensor,
        code_lengths: torch.Tensor,
        *,
        chunk_size: int,
        advance_frames: int | None = None,
        active_slots: tuple[int, ...] | None = None,
    ) -> Any:
        if self._disabled_reason is not None or codes.device.type != "cuda":
            return self._codec._decode_frame(codes, code_lengths)

        try:
            return self._decode_frame_graphed(
                codes,
                code_lengths,
                chunk_size=chunk_size,
                advance_frames=advance_frames,
                active_slots=active_slots,
            )
        except Exception as exc:
            self._disabled_reason = str(exc)
            logger.warning(
                "MOSS-TTS Local audio-tokenizer CUDA graph capture disabled; "
                "falling back to eager decoder: %s "
                "(chunk_size=%s, active_slots=%s, cache_entries=%s)",
                exc,
                chunk_size,
                active_slots,
                len(self._cache),
                exc_info=True,
            )
            return self._codec._decode_frame(codes, code_lengths)

    def close(self) -> None:
        self._cache.clear()

    def _decode_frame_graphed(
        self,
        codes: torch.Tensor,
        code_lengths: torch.Tensor,
        *,
        chunk_size: int,
        advance_frames: int | None,
        active_slots: tuple[int, ...] | None,
    ) -> Any:
        codec = self._codec
        with torch.cuda.device(codes.device):
            decoder_hidden_states = codec.quantizer.decode_codes(codes).float()
            graph_key = (
                str(codes.device),
                tuple(decoder_hidden_states.shape),
                str(decoder_hidden_states.dtype),
                tuple(code_lengths.shape),
                str(code_lengths.dtype),
                getattr(codec, "compute_dtype_name", None),
                int(chunk_size),
            )

            cached = self._cache.get(graph_key)
            if cached is None:
                cached = self._capture_graph(
                    graph_key, decoder_hidden_states, code_lengths
                )

            graph, static_hidden, static_lengths, audio, audio_lengths = cached
            static_hidden.copy_(decoder_hidden_states)
            static_lengths.copy_(code_lengths)
            graph.replay()
            self._correct_streaming_offsets(
                static_lengths,
                chunk_size=int(chunk_size),
                fallback_advance_frames=int(
                    advance_frames if advance_frames is not None else chunk_size
                ),
            )
            return SimpleNamespace(audio=audio, audio_lengths=audio_lengths)

    def _capture_graph(
        self,
        graph_key: tuple[Any, ...],
        decoder_hidden_states: torch.Tensor,
        code_lengths: torch.Tensor,
    ) -> tuple[
        torch.cuda.CUDAGraph,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        static_hidden = decoder_hidden_states.new_zeros(decoder_hidden_states.shape)
        static_lengths = code_lengths.new_zeros(code_lengths.shape)
        static_hidden.copy_(decoder_hidden_states)
        static_lengths.copy_(code_lengths)

        snapshots = self._snapshot_streaming_states()
        try:
            with codec_cuda_graph_capture_lock:
                device = decoder_hidden_states.device
                current_stream = torch.cuda.current_stream(device=device)
                capture_stream = torch.cuda.Stream(device=device)
                capture_stream.wait_stream(current_stream)
                with torch.cuda.stream(capture_stream):
                    _ = self._codec._decode_hidden_states(static_hidden, static_lengths)
                capture_stream.synchronize()
                self._restore_streaming_states(snapshots)
                # Restore enqueues tensor copies on the caller's stream; graph
                # capture must observe those copies before recording decoder ops.
                capture_stream.wait_stream(current_stream)

                with torch.cuda.device(device):
                    cuda_graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(
                        cuda_graph,
                        stream=capture_stream,
                        capture_error_mode="relaxed",
                    ):
                        decoder_output = self._codec._decode_hidden_states(
                            static_hidden, static_lengths
                        )
                current_stream.wait_stream(capture_stream)
            if decoder_output.audio is None or decoder_output.audio_lengths is None:
                raise RuntimeError(
                    "audio-tokenizer decoder graph capture returned empty audio"
                )

            captured = (
                cuda_graph,
                static_hidden,
                static_lengths,
                decoder_output.audio,
                decoder_output.audio_lengths,
            )
            self._cache[graph_key] = captured

            # Capture itself advances the codec streaming state. Restore here;
            # the caller performs the first replay through the common path.
            self._restore_streaming_states(snapshots)
            return captured
        except Exception:
            self._restore_streaming_states(snapshots)
            raise

    def _collect_streaming_states(self) -> list[Any]:
        states: list[Any] = []
        decoder = getattr(self._codec, "decoder", ())
        for decoder_module in decoder:
            modules = (
                decoder_module.modules() if hasattr(decoder_module, "modules") else ()
            )
            for module in modules:
                state = getattr(module, "_streaming_state", None)
                if state is not None:
                    states.append(state)
        return states

    def _streaming_states(self) -> list[Any]:
        if self._streaming_states_cache is None:
            self._streaming_states_cache = self._collect_streaming_states()
        return self._streaming_states_cache

    def _snapshot_streaming_states(self) -> list[tuple[Any, dict[str, Any]]]:
        snapshots: list[tuple[Any, dict[str, Any]]] = []
        for state in self._streaming_states():
            state_snapshot: dict[str, Any] = {}
            for attr in _STATE_TENSOR_ATTRS:
                if not hasattr(state, attr):
                    continue
                value = getattr(state, attr)
                state_snapshot[attr] = (
                    value.clone() if isinstance(value, torch.Tensor) else value
                )
            for attr in _STATE_VALUE_ATTRS:
                if hasattr(state, attr):
                    state_snapshot[attr] = getattr(state, attr)
            snapshots.append((state, state_snapshot))
        return snapshots

    def _restore_streaming_states(
        self, snapshots: list[tuple[Any, dict[str, Any]]]
    ) -> None:
        for state, state_snapshot in snapshots:
            for attr, snapshot_value in state_snapshot.items():
                current_value = getattr(state, attr, None)
                if isinstance(snapshot_value, torch.Tensor):
                    if (
                        isinstance(current_value, torch.Tensor)
                        and current_value.shape == snapshot_value.shape
                        and current_value.dtype == snapshot_value.dtype
                        and current_value.device == snapshot_value.device
                    ):
                        current_value.copy_(snapshot_value)
                    else:
                        setattr(state, attr, snapshot_value.clone())
                elif snapshot_value is None and isinstance(current_value, torch.Tensor):
                    if attr == "cached_positions":
                        current_value.fill_(-1)
                    else:
                        current_value.zero_()
                else:
                    setattr(state, attr, snapshot_value)

    def _correct_streaming_offsets(
        self,
        code_lengths: torch.Tensor,
        *,
        chunk_size: int,
        fallback_advance_frames: int,
    ) -> None:
        for state in self._streaming_states():
            corrected_tensor_offset = False
            for attr in ("offset", "offsets"):
                value = getattr(state, attr, None)
                if (
                    not isinstance(value, torch.Tensor)
                    or value.ndim == 0
                    or value.shape[0] != code_lengths.shape[0]
                ):
                    continue

                lengths = code_lengths.to(
                    device=value.device, dtype=value.dtype, non_blocking=True
                )
                exec_mask = getattr(state, "exec_mask", None)
                if (
                    isinstance(exec_mask, torch.Tensor)
                    and exec_mask.ndim > 0
                    and exec_mask.shape[0] == value.shape[0]
                ):
                    mask = exec_mask.to(
                        device=value.device, dtype=torch.bool, non_blocking=True
                    )
                else:
                    mask = lengths > 0
                correction = lengths - int(chunk_size)
                value.add_(torch.where(mask, correction, torch.zeros_like(correction)))
                corrected_tensor_offset = True

            if hasattr(state, "offset_cpu"):
                # Older codec variants keep a scalar Python offset. They cannot
                # represent per-slot lengths, so preserve the caller's uniform
                # fallback behavior for those implementations.
                if not corrected_tensor_offset:
                    state.offset_cpu += int(fallback_advance_frames)
