# SPDX-License-Identifier: Apache-2.0
"""MOSS-Audio-Tokenizer runtime."""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F
import torchaudio
from torch import nn
from transformers import AutoModel

from sglang_omni.models.moss_tts.attention import (
    AUTO_ATTENTION_BACKEND,
    PACKED_FLASH_ATTENTION_BACKEND,
    _SDPA_ATTENTION_BACKEND,
    AttentionBackendResolution,
    MossAudioTokenizerAttention,
    MossPackedRopeCache,
    PositionIdsCache,
    _StreamingModule,
    _StreamingState,
    merge_attention_backend_resolutions,
    pack_padded_sequence,
    pack_padded_sequence_from_host_lengths,
    pack_unpadded_sequence,
    unpack_packed_sequence,
    unpack_packed_sequence_from_indices,
    unpack_unpadded_sequence,
    validate_attention_backend,
)
from sglang_omni.models.moss_tts.streaming_codec import StreamingExecutionContext
from sglang_omni.models.moss_tts.hf_loading import moss_transformers_processor_compat
from sglang_omni.models.moss_tts.vocoder_quantizer import (
    MossAudioTokenizerQuantizerDecoder,
)
from sglang_omni.models.weight_loader import (
    load_module,
    load_weights_by_prefix,
    resolve_model_path,
)

logger = logging.getLogger(__name__)

DEFAULT_MOSS_TTS_AUDIO_TOKENIZER = "OpenMOSS-Team/MOSS-Audio-Tokenizer"
DEFAULT_MOSS_TTS_LOCAL_AUDIO_TOKENIZER = "OpenMOSS-Team/MOSS-Audio-Tokenizer-v2"
SDPA_ATTENTION_BACKEND = _SDPA_ATTENTION_BACKEND
_LOUDNESS_TARGET_DBFS = -20.0
_LOUDNESS_GAIN_MIN_DB = -3.0
_LOUDNESS_GAIN_MAX_DB = 3.0

# Compatibility names retained by the repository-owned codec loader replayed
# from the development branch. The optimized attention implementation lives in
# ``moss_tts.attention`` on the perf branch.
_AUTO_ATTENTION_BACKEND = AUTO_ATTENTION_BACKEND
_PACKED_FLASH_ATTENTION_BACKEND = PACKED_FLASH_ATTENTION_BACKEND
_AttentionBackendResolution = AttentionBackendResolution
_merge_attention_backend_resolutions = merge_attention_backend_resolutions
_validate_attention_backend = validate_attention_backend


def resolve_moss_audio_attention_backend(
    attention_backend: str,
    attention_implementation: str | None,
) -> str:
    """Resolve the legacy Transformers attention hint to a codec backend."""
    validate_attention_backend(attention_backend)
    if attention_implementation not in (
        None,
        "flash_attention_2",
        SDPA_ATTENTION_BACKEND,
    ):
        raise ValueError(
            "attention_implementation must be None, 'flash_attention_2', "
            f"or 'sdpa'; got {attention_implementation!r}"
        )
    if attention_backend != AUTO_ATTENTION_BACKEND:
        return attention_backend
    if attention_implementation == SDPA_ATTENTION_BACKEND:
        return SDPA_ATTENTION_BACKEND
    return AUTO_ATTENTION_BACKEND


def resolve_moss_audio_sample_rate(model: Any, config: Any) -> int:
    """Read the sampling rate from either runtime model or checkpoint config."""
    for value in (
        getattr(model, "sampling_rate", None),
        getattr(config, "sampling_rate", None),
        getattr(config, "sample_rate", None),
    ):
        if value is not None:
            return int(value)
    raise ValueError(
        "MOSS-Audio-Tokenizer model/config lacks sampling_rate or sample_rate"
    )


def _log_attention_backend_resolution(
    component: str,
    *,
    requested_backend: str,
    resolution: AttentionBackendResolution,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    if resolution.fallback_reason is not None:
        logger.warning(
            "MOSS audio-tokenizer %s attention_backend=%r falls back to SDPA "
            "on device=%s, dtype=%s: %s",
            component,
            requested_backend,
            device,
            dtype,
            resolution.fallback_reason,
        )
        return
    logger.info(
        "MOSS audio-tokenizer %s selected attention backend %s on device=%s, "
        "dtype=%s",
        component,
        resolution.backend,
        device,
        dtype,
    )


class _MossAudioTokenizerV1FeedForward(nn.Module):
    """Expose MOSS-Audio-Tokenizer v1 Linear-GELU-Linear weights."""

    def __init__(
        self,
        linear1: nn.Module,
        linear2: nn.Module,
        activation: Any,
    ) -> None:
        super().__init__()
        self.linear1 = linear1
        self.linear2 = linear2
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.activation(self.linear1(x)))


def _feed_forward(module: nn.Module) -> nn.Module:
    ffn = getattr(module, "ffn", None)
    if ffn is not None:
        return ffn
    if (
        getattr(module, "linear1", None) is None
        or getattr(module, "linear2", None) is None
    ):
        raise ValueError("MOSS-Audio-Tokenizer transformer layer has no supported FFN")
    return _MossAudioTokenizerV1FeedForward(
        module.linear1,
        module.linear2,
        module.activation,
    )


class MossAudioTokenizerTransformerLayer(nn.Module):
    """One shared MOSS-Audio-Tokenizer transformer layer."""

    def __init__(
        self,
        *,
        norm1: nn.Module,
        self_attn: MossAudioTokenizerAttention,
        layer_scale_1: nn.Module,
        norm2: nn.Module,
        ffn: nn.Module,
        layer_scale_2: nn.Module,
    ) -> None:
        super().__init__()
        self.norm1 = norm1
        self.self_attn = self_attn
        self.layer_scale_1 = layer_scale_1
        self.norm2 = norm2
        self.ffn = ffn
        self.layer_scale_2 = layer_scale_2
        assert callable(self.ffn), "MOSS-Audio-Tokenizer layer requires an FFN"

    @classmethod
    def from_module(
        cls,
        module: nn.Module,
        *,
        attention_backend: str = AUTO_ATTENTION_BACKEND,
        packed_rope_cache: MossPackedRopeCache | None = None,
    ) -> MossAudioTokenizerTransformerLayer:
        return cls(
            norm1=module.norm1,
            self_attn=MossAudioTokenizerAttention.from_module(
                module.self_attn,
                attention_backend=attention_backend,
                packed_rope_cache=packed_rope_cache,
            ),
            layer_scale_1=module.layer_scale_1,
            norm2=module.norm2,
            ffn=_feed_forward(module),
            layer_scale_2=module.layer_scale_2,
        )

    @classmethod
    def from_config(
        cls,
        *,
        d_model: int,
        num_heads: int,
        dim_feedforward: int,
        causal: bool,
        context: int | None,
        rope: nn.Module | None,
        norm: str,
        layer_scale: float | None,
        gating: str,
        moss_audio_tokenizer_v1_weights: bool,
        device: str | torch.device | None,
        dtype: torch.dtype | None,
        attention_backend: str = AUTO_ATTENTION_BACKEND,
        packed_rope_cache: MossPackedRopeCache | None = None,
    ) -> MossAudioTokenizerTransformerLayer:
        if gating != "none":
            raise ValueError(
                "repository-local MOSS encoder currently supports gating='none'; "
                f"got {gating!r}"
            )
        if moss_audio_tokenizer_v1_weights:
            ffn: nn.Module = _MossAudioTokenizerV1FeedForward(
                nn.Linear(
                    d_model,
                    dim_feedforward,
                    bias=False,
                    device=device,
                    dtype=dtype,
                ),
                nn.Linear(
                    dim_feedforward,
                    d_model,
                    bias=False,
                    device=device,
                    dtype=dtype,
                ),
                nn.GELU(),
            )
        else:
            ffn = nn.Sequential(
                nn.Linear(
                    d_model,
                    dim_feedforward,
                    bias=False,
                    device=device,
                    dtype=dtype,
                ),
                nn.GELU(),
                nn.Linear(
                    dim_feedforward,
                    d_model,
                    bias=False,
                    device=device,
                    dtype=dtype,
                ),
            )
        if layer_scale is None:
            layer_scale_1: nn.Module = nn.Identity()
            layer_scale_2: nn.Module = nn.Identity()
        else:
            layer_scale_1 = _LayerScale(
                d_model,
                init=float(layer_scale),
                device=device,
                dtype=dtype,
            )
            layer_scale_2 = _LayerScale(
                d_model,
                init=float(layer_scale),
                device=device,
                dtype=dtype,
            )
        self_attn = MossAudioTokenizerAttention(
            in_proj=nn.Linear(
                d_model,
                3 * d_model,
                bias=False,
                device=device,
                dtype=dtype,
            ),
            out_proj=nn.Linear(
                d_model,
                d_model,
                bias=False,
                device=device,
                dtype=dtype,
            ),
            embed_dim=d_model,
            num_heads=num_heads,
            causal=causal,
            context=context,
            rope=rope,
            attention_implementation=(
                None if moss_audio_tokenizer_v1_weights else "flash_attention_2"
            ),
            attention_backend=attention_backend,
            packed_rope_cache=packed_rope_cache,
        )
        return cls(
            norm1=_create_norm(norm, d_model, device=device, dtype=dtype),
            self_attn=self_attn,
            layer_scale_1=layer_scale_1,
            norm2=_create_norm(norm, d_model, device=device, dtype=dtype),
            ffn=ffn,
            layer_scale_2=layer_scale_2,
        )

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x = residual.to(x) + self.layer_scale_1(self.self_attn(x, **kwargs))
        residual = x
        x = self.norm2(x)
        x = residual.to(x) + self.layer_scale_2(self.ffn(x))
        return x


@dataclass
class _TransformerStreamingState(_StreamingState):
    offsets: torch.Tensor

    def reset(self, reset_mask: torch.Tensor) -> None:
        super().reset(reset_mask)
        reset_mask = reset_mask.to(device=self.device, dtype=torch.bool)
        self.offsets.copy_(
            torch.where(reset_mask, torch.zeros_like(self.offsets), self.offsets)
        )

    def reset_slots(self, state_slot_ids: torch.Tensor) -> None:
        if state_slot_ids.numel() == 0:
            return
        slots = state_slot_ids.to(device=self.device, dtype=torch.long)
        self.offsets.index_fill_(0, slots, 0)
        self.exec_mask.index_fill_(0, slots, True)


class MossAudioTokenizerTransformer(_StreamingModule):
    """Shared MOSS-Audio-Tokenizer transformer body."""

    def __init__(
        self,
        *,
        layers: Sequence[MossAudioTokenizerTransformerLayer],
        positional_embedding: str,
        positional_scale: float,
        max_period: float,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.positional_embedding = positional_embedding
        self.positional_scale = float(positional_scale)
        self.max_period = float(max_period)

    def _init_streaming_state(self, batch_size: int):
        device = self._streaming_device()
        return _TransformerStreamingState(
            int(batch_size),
            device,
            offsets=torch.zeros(batch_size, device=device, dtype=torch.long),
        )

    @classmethod
    def from_module(
        cls,
        module: nn.Module,
        *,
        attention_backend: str = AUTO_ATTENTION_BACKEND,
    ) -> MossAudioTokenizerTransformer:
        max_period = float(module.max_period)
        packed_rope_cache = MossPackedRopeCache(max_period=max_period)
        return cls(
            layers=[
                MossAudioTokenizerTransformerLayer.from_module(
                    layer,
                    attention_backend=attention_backend,
                    packed_rope_cache=packed_rope_cache,
                )
                for layer in module.layers
            ],
            positional_embedding=module.positional_embedding,
            positional_scale=float(module.positional_scale),
            max_period=max_period,
        )

    @classmethod
    def from_config(
        cls,
        *,
        d_model: int,
        num_heads: int,
        num_layers: int,
        dim_feedforward: int,
        causal: bool,
        context: int | None,
        positional_embedding: str,
        max_period: float,
        positional_scale: float,
        norm: str,
        layer_scale: float | None,
        gating: str,
        moss_audio_tokenizer_v1_weights: bool,
        device: str | torch.device | None,
        dtype: torch.dtype | None,
        attention_backend: str = AUTO_ATTENTION_BACKEND,
    ) -> MossAudioTokenizerTransformer:
        rope = (
            _RotaryEmbedding(max_period)
            if positional_embedding in {"rope", "sin_rope"}
            else None
        )
        packed_rope_cache = MossPackedRopeCache(max_period=max_period)
        return cls(
            layers=[
                MossAudioTokenizerTransformerLayer.from_config(
                    d_model=d_model,
                    num_heads=num_heads,
                    dim_feedforward=dim_feedforward,
                    causal=causal,
                    context=context,
                    rope=rope,
                    norm=norm,
                    layer_scale=layer_scale,
                    gating=gating,
                    moss_audio_tokenizer_v1_weights=moss_audio_tokenizer_v1_weights,
                    device=device,
                    dtype=dtype,
                    attention_backend=attention_backend,
                    packed_rope_cache=packed_rope_cache,
                )
                for _ in range(num_layers)
            ],
            positional_embedding=positional_embedding,
            positional_scale=positional_scale,
            max_period=max_period,
        )

    def resolve_attention_backend(
        self,
        device: torch.device,
        dtype: torch.dtype | None,
    ) -> AttentionBackendResolution:
        return merge_attention_backend_resolutions(
            [
                layer.self_attn.resolve_attention_backend(device, dtype)
                for layer in self.layers
            ]
        )

    def supports_packed_attention(
        self,
        device: torch.device,
        dtype: torch.dtype | None,
    ) -> bool:
        return bool(self.layers) and all(
            layer.self_attn.supports_packed_attention(device, dtype)
            for layer in self.layers
        )

    def supports_packed_flash(
        self,
        device: torch.device,
        dtype: torch.dtype | None,
    ) -> bool:
        return self.supports_packed_attention(device, dtype)

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        execution_context = kwargs.pop("execution_context", None)
        if execution_context is not None and not isinstance(
            execution_context, StreamingExecutionContext
        ):
            raise TypeError(
                "execution_context must be a StreamingExecutionContext or None"
            )
        state = self._streaming_state
        if state is not None and not isinstance(state, _TransformerStreamingState):
            raise RuntimeError("invalid MOSS transformer streaming state")
        if state is None and execution_context is not None:
            raise RuntimeError(
                "streaming execution context requires an active transformer state"
            )
        if self.positional_embedding in {"sin", "sin_rope"}:
            if x.dim() == 3:
                if state is None:
                    offsets = torch.zeros(1, device=x.device, dtype=torch.long)
                elif execution_context is None:
                    offsets = state.offsets
                else:
                    execution_context.validate(
                        batch_size=x.shape[0],
                        state_capacity=state.batch_size,
                        device=x.device,
                    )
                    offsets = state.offsets.index_select(
                        0, execution_context.state_slot_ids
                    )
                positions = torch.arange(x.shape[1], device=x.device).view(1, -1)
                positions = positions + offsets.view(-1, 1)
            else:
                if state is not None:
                    raise ValueError("streaming transformer requires dense inputs")
                positions = kwargs.get("position_ids")
                if positions is None:
                    raise ValueError(
                        "packed transformer inputs require position_ids for "
                        "sinusoidal embeddings"
                    )
            pos_emb = create_sin_embedding(
                positions,
                x.shape[-1],
                max_period=self.max_period,
                dtype=x.dtype,
            )
            x = x + self.positional_scale * pos_emb
        for layer in self.layers:
            x = layer(x, execution_context=execution_context, **kwargs)
        if state is not None:
            if x.dim() != 3:
                raise ValueError("streaming transformer requires dense inputs")
            if execution_context is None:
                state.offsets.copy_(
                    torch.where(
                        state.exec_mask,
                        state.offsets + x.shape[1],
                        state.offsets,
                    )
                )
            else:
                offsets = state.offsets.index_select(
                    0, execution_context.state_slot_ids
                )
                next_offsets = torch.where(
                    execution_context.valid_rows,
                    offsets + x.shape[1],
                    offsets,
                )
                state.offsets.index_copy_(
                    0,
                    execution_context.state_slot_ids,
                    next_offsets,
                )
        return x


class MossAudioTokenizerProjectedTransformer(nn.Module):
    """Shared projected Transformer stage with the MOSS-Audio-Tokenizer tensor layout."""

    def __init__(
        self,
        *,
        input_proj: nn.Module,
        transformer: MossAudioTokenizerTransformer,
        output_proj: nn.Module,
    ) -> None:
        super().__init__()
        self.module_type = "Transformer"
        self.downsample_ratio = 1
        self.input_proj = input_proj
        self.transformer = transformer
        self.output_proj = output_proj
        self._position_ids_cache = PositionIdsCache()

    @property
    def is_streaming(self) -> bool:
        return self.transformer.is_streaming

    @classmethod
    def from_module(
        cls,
        module: nn.Module,
        *,
        attention_backend: str = AUTO_ATTENTION_BACKEND,
    ) -> MossAudioTokenizerProjectedTransformer:
        return cls(
            input_proj=module.input_proj,
            transformer=MossAudioTokenizerTransformer.from_module(
                module.transformer,
                attention_backend=attention_backend,
            ),
            output_proj=module.output_proj,
        )

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        *,
        context: int | None,
        moss_audio_tokenizer_v1_weights: bool,
        device: str | torch.device | None,
        dtype: torch.dtype | None,
        attention_backend: str = AUTO_ATTENTION_BACKEND,
    ) -> MossAudioTokenizerProjectedTransformer:
        input_dimension = int(config["input_dimension"])
        output_dimension = int(config["output_dimension"])
        d_model = int(config["d_model"])
        input_proj: nn.Module = (
            nn.Linear(
                input_dimension,
                d_model,
                bias=False,
                device=device,
                dtype=dtype,
            )
            if not moss_audio_tokenizer_v1_weights or input_dimension != d_model
            else nn.Identity()
        )
        output_proj: nn.Module = (
            nn.Linear(
                d_model,
                output_dimension,
                bias=False,
                device=device,
                dtype=dtype,
            )
            if not moss_audio_tokenizer_v1_weights or d_model != output_dimension
            else nn.Identity()
        )
        return cls(
            input_proj=input_proj,
            transformer=MossAudioTokenizerTransformer.from_config(
                d_model=d_model,
                num_heads=int(config["num_heads"]),
                num_layers=int(config["num_layers"]),
                dim_feedforward=int(config.get("dim_feedforward", 2048)),
                causal=bool(config.get("causal", False)),
                context=context,
                positional_embedding=str(config.get("positional_embedding", "sin")),
                max_period=float(config.get("max_period", 10_000)),
                positional_scale=float(config.get("positional_scale", 1.0)),
                norm=str(config.get("norm", "layer_norm")),
                layer_scale=config.get("layer_scale"),
                gating=str(config.get("gating", "none")),
                moss_audio_tokenizer_v1_weights=moss_audio_tokenizer_v1_weights,
                device=device,
                dtype=dtype,
                attention_backend=attention_backend,
            ),
            output_proj=output_proj,
        )

    def forward(
        self,
        x: torch.Tensor,
        input_lengths: torch.Tensor,
        *,
        input_lengths_cpu: Sequence[int] | None = None,
        execution_context: StreamingExecutionContext | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if execution_context is not None and not self.is_streaming:
            raise RuntimeError(
                "streaming execution context requires an active projected transformer"
            )
        x = self.input_proj(x.transpose(1, 2))
        backend = None
        if not self.is_streaming:
            backend = self.transformer.resolve_attention_backend(
                x.device,
                MossAudioTokenizerAttention.get_backend_dtype(x),
            ).backend
        if backend == PACKED_FLASH_ATTENTION_BACKEND:
            batch_size, max_seqlen, _ = x.shape
            if input_lengths_cpu is not None:
                if len(input_lengths_cpu) != batch_size:
                    raise ValueError(
                        "input_lengths_cpu must match the decoder batch size"
                    )
                max_valid_seqlen = max(map(int, input_lengths_cpu), default=0)
            else:
                max_valid_seqlen = int(input_lengths.max().item()) if max_seqlen else 0
            if max_valid_seqlen == 0:
                x = x.new_zeros(x.shape)
            else:
                is_unpadded_single = batch_size == 1 and max_valid_seqlen == max_seqlen
                if is_unpadded_single:
                    packed_x, cu_seqlens, position_ids = pack_unpadded_sequence(
                        x,
                        self._position_ids_cache,
                    )
                    valid_mask = None
                    flat_indices = None
                elif input_lengths_cpu is not None:
                    packed_x, flat_indices, cu_seqlens, position_ids = (
                        pack_padded_sequence_from_host_lengths(
                            x,
                            input_lengths,
                            input_lengths_cpu,
                        )
                    )
                    valid_mask = None
                else:
                    packed_x, valid_mask, cu_seqlens, position_ids = (
                        pack_padded_sequence(x, input_lengths)
                    )
                    flat_indices = None
                first_attention = self.transformer.layers[0].self_attn
                local_flash_plan = first_attention.build_local_causal_flash_plan(
                    cu_seqlens,
                    max_seqlen=max_valid_seqlen,
                    sequence_lengths=input_lengths_cpu,
                )
                packed_x = self.transformer(
                    packed_x,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_valid_seqlen,
                    position_ids=position_ids,
                    input_lengths=input_lengths,
                    local_flash_plan=local_flash_plan,
                    **kwargs,
                )
                if is_unpadded_single:
                    x = unpack_unpadded_sequence(packed_x)
                elif flat_indices is not None:
                    x = unpack_packed_sequence_from_indices(
                        packed_x,
                        flat_indices,
                        batch_size,
                        max_seqlen,
                    )
                else:
                    assert valid_mask is not None
                    x = unpack_packed_sequence(
                        packed_x,
                        valid_mask,
                        batch_size,
                        max_seqlen,
                    )
        else:
            x = self.transformer(
                x,
                input_lengths=input_lengths,
                execution_context=execution_context,
                **kwargs,
            )
        return self.output_proj(x).transpose(1, 2), input_lengths

    def supports_packed_attention(
        self,
        device: torch.device,
        dtype: torch.dtype | None,
    ) -> bool:
        return self.transformer.supports_packed_attention(device, dtype)

    def supports_packed_flash(
        self,
        device: torch.device,
        dtype: torch.dtype | None,
    ) -> bool:
        return self.supports_packed_attention(device, dtype)

    def resolve_attention_backend(
        self,
        device: str | torch.device,
        dtype: torch.dtype | None,
    ) -> AttentionBackendResolution:
        return self.transformer.resolve_attention_backend(torch.device(device), dtype)


# note (Zhang Yiyang): Non-streaming decoder wrapper.


class MossAudioTokenizerVocoderDecoder(nn.ModuleList):
    """Iterable MOSS-Audio-Tokenizer vocoder decoder with patched projected transformers."""

    def __init__(
        self,
        decoder: nn.Module,
        *,
        attention_backend: str = AUTO_ATTENTION_BACKEND,
    ) -> None:
        super().__init__()
        stages = list(decoder)
        assert (
            stages
        ), "MOSS-Audio-Tokenizer vocoder decoder must be a non-empty stage list"
        self.attention_backend = validate_attention_backend(attention_backend)
        # Register stages directly on the ModuleList so checkpoint keys stay
        # compatible with the historical ``0.<param>`` layout.
        self.extend(self._wrap_stage(stage) for stage in stages)

    @classmethod
    def from_module(
        cls,
        decoder: nn.Module,
        *,
        attention_backend: str = AUTO_ATTENTION_BACKEND,
    ) -> "MossAudioTokenizerVocoderDecoder":
        if isinstance(decoder, cls):
            return decoder
        return cls(decoder, attention_backend=attention_backend)

    def _wrap_stage(self, stage: nn.Module) -> nn.Module:
        module_type = stage.module_type
        if module_type == "Transformer":
            return MossAudioTokenizerProjectedTransformer.from_module(
                stage,
                attention_backend=self.attention_backend,
            )
        if module_type == "PatchedPretransform":
            return stage
        raise ValueError(
            f"unsupported MOSS-Audio-Tokenizer vocoder decoder stage {stage.__class__.__name__} "
            f"with module_type={module_type!r}"
        )

    def supports_packed_attention(
        self,
        device: str | torch.device,
        dtype: torch.dtype | None,
    ) -> bool:
        device = torch.device(device)
        transformer_stages = [
            stage
            for stage in self
            if isinstance(stage, MossAudioTokenizerProjectedTransformer)
        ]
        return bool(transformer_stages) and all(
            stage.supports_packed_attention(device, dtype)
            for stage in transformer_stages
        )

    def supports_packed_flash(
        self,
        device: str | torch.device,
        dtype: torch.dtype | None,
    ) -> bool:
        return self.supports_packed_attention(device, dtype)

    def resolve_attention_backend(
        self,
        device: str | torch.device,
        dtype: torch.dtype | None,
    ) -> AttentionBackendResolution:
        device = torch.device(device)
        return merge_attention_backend_resolutions(
            [
                stage.resolve_attention_backend(device, dtype)
                for stage in self
                if isinstance(stage, MossAudioTokenizerProjectedTransformer)
            ]
        )

    @staticmethod
    def _update_cpu_lengths(
        stage: nn.Module,
        input_lengths: Sequence[int],
    ) -> list[int]:
        if isinstance(stage, MossAudioTokenizerProjectedTransformer):
            return list(map(int, input_lengths))
        patch_size = int(getattr(stage, "patch_size", 0))
        if patch_size <= 0:
            raise ValueError(
                "MOSS-Audio-Tokenizer patched pretransform requires patch_size > 0"
            )
        if bool(getattr(stage, "is_downsample", False)):
            return [int(length) // patch_size for length in input_lengths]
        return [int(length) * patch_size for length in input_lengths]

    def output_lengths(self, input_lengths: Sequence[int]) -> list[int]:
        output_lengths = list(map(int, input_lengths))
        for stage in self:
            output_lengths = self._update_cpu_lengths(stage, output_lengths)
        return output_lengths

    def forward(
        self,
        x: torch.Tensor,
        input_lengths: torch.Tensor,
        *,
        input_lengths_cpu: Sequence[int] | None = None,
        execution_context: StreamingExecutionContext | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cpu_lengths = (
            None if input_lengths_cpu is None else list(map(int, input_lengths_cpu))
        )
        for stage in self:
            if isinstance(stage, MossAudioTokenizerProjectedTransformer):
                x, input_lengths = stage(
                    x,
                    input_lengths,
                    input_lengths_cpu=cpu_lengths,
                    execution_context=execution_context,
                )
            else:
                x, input_lengths = stage(x, input_lengths)
            if cpu_lengths is not None:
                cpu_lengths = self._update_cpu_lengths(stage, cpu_lengths)
        return x, input_lengths


def create_sin_embedding(
    positions: torch.Tensor,
    dim: int,
    *,
    max_period: float = 10_000,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create the sinusoidal embedding expected by the shared stage wrapper."""

    if dim % 2:
        raise ValueError(f"sinusoidal embedding requires an even dim, got {dim}")
    half_dim = dim // 2
    if half_dim <= 1:
        raise ValueError(f"sinusoidal embedding requires dim >= 4, got {dim}")
    positions = positions.to(dtype).unsqueeze(-1)
    dimensions = torch.arange(half_dim, device=positions.device, dtype=dtype)
    period = torch.full(
        (),
        float(max_period),
        device=positions.device,
        dtype=dtype,
    )
    phase = positions / (period ** (dimensions / (half_dim - 1)))
    return torch.cat((torch.cos(phase), torch.sin(phase)), dim=-1)


def _torch_dtype(dtype: str | torch.dtype) -> torch.dtype:
    return getattr(torch, dtype) if isinstance(dtype, str) else dtype


def _model_floating_dtype(model: Any) -> torch.dtype:
    parameters = getattr(model, "parameters", None)
    if not callable(parameters):
        return torch.float32
    return next(
        (
            parameter.dtype
            for parameter in parameters()
            if parameter.is_floating_point()
        ),
        torch.float32,
    )


class MossTTSAudioTokenizer:
    """Processor-compatible wrapper around a separately loaded codec model."""

    def __init__(self, model: Any, *, device: str) -> None:
        self.model = model
        self.device = str(device)
        self.dtype = _model_floating_dtype(model)
        self.sample_rate = int(model.config.sampling_rate)

    def _autocast(self) -> Any:
        device_type = torch.device(self.device).type
        if device_type == "cuda" and self.dtype in {torch.float16, torch.bfloat16}:
            return torch.autocast(device_type=device_type, dtype=self.dtype)
        return nullcontext()

    def encode_waveforms(
        self,
        waveforms: list[tuple[torch.Tensor, int]],
        *,
        num_quantizers: int | None = None,
    ) -> list[torch.Tensor]:
        if not waveforms:
            raise ValueError("waveforms must contain at least one waveform")
        prepared = [
            self._prepare_waveform(wav, sample_rate) for wav, sample_rate in waveforms
        ]

        with torch.inference_mode(), self._autocast():
            if hasattr(self.model, "batch_encode"):
                encoded = self.model.batch_encode(
                    prepared,
                    num_quantizers=num_quantizers,
                )
            else:
                max_length = max(int(wav.shape[-1]) for wav in prepared)
                input_values = torch.zeros(
                    len(prepared),
                    1,
                    max_length,
                    device=self.device,
                    dtype=torch.float32,
                )
                padding_mask = torch.zeros(
                    len(prepared),
                    max_length,
                    device=self.device,
                    dtype=torch.bool,
                )
                for index, wav in enumerate(prepared):
                    length = int(wav.shape[-1])
                    input_values[index, 0, :length] = wav
                    padding_mask[index, :length] = True
                encoded = self.model.encode(
                    input_values,
                    padding_mask=padding_mask,
                    num_quantizers=num_quantizers,
                    return_dict=True,
                )

        audio_codes = encoded.audio_codes
        audio_codes_lengths = encoded.audio_codes_lengths
        if audio_codes is None or audio_codes_lengths is None:
            raise RuntimeError(
                "MOSS-Audio-Tokenizer encode returned empty "
                "audio_codes/audio_codes_lengths"
            )
        codes_cpu = audio_codes.detach().to(device="cpu", dtype=torch.long)
        lengths_cpu = audio_codes_lengths.detach().to("cpu")
        return [
            codes_cpu[:, index, : int(lengths_cpu[index])].transpose(0, 1).contiguous()
            for index in range(int(codes_cpu.shape[1]))
        ]

    def _prepare_waveform(self, wav: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        if wav.ndim != 2:
            raise ValueError(
                f"expected waveform with shape [channels, samples], got {tuple(wav.shape)}"
            )
        if int(wav.shape[0]) > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)
        if int(sample_rate) != self.sample_rate:
            wav = torchaudio.functional.resample(
                waveform=wav,
                orig_freq=int(sample_rate),
                new_freq=self.sample_rate,
            )
        wav = self._loudness_normalize(wav.squeeze(0))
        return wav.to(device=self.device, dtype=torch.float32)

    def decode_codes(
        self,
        codes: torch.Tensor | list[torch.Tensor],
    ) -> list[torch.Tensor]:
        if isinstance(codes, torch.Tensor):
            codes = [codes]
        if not codes:
            return []

        codes_nq_t = [
            item.transpose(0, 1).contiguous().to(device=self.device, dtype=torch.long)
            for item in codes
        ]
        num_quantizers = int(codes_nq_t[0].shape[0])
        if any(int(item.shape[0]) != num_quantizers for item in codes_nq_t):
            raise ValueError("all audio-code rows must use the same quantizer count")
        max_length = max(int(item.shape[1]) for item in codes_nq_t)
        audio_codes = torch.zeros(
            num_quantizers,
            len(codes_nq_t),
            max_length,
            device=self.device,
            dtype=torch.long,
        )
        padding_mask = torch.zeros(
            len(codes_nq_t),
            max_length,
            device=self.device,
            dtype=torch.bool,
        )
        for index, item in enumerate(codes_nq_t):
            length = int(item.shape[1])
            audio_codes[:, index, :length] = item
            padding_mask[index, :length] = True

        with torch.inference_mode(), self._autocast():
            decoded = self.model.decode(
                audio_codes,
                padding_mask=padding_mask,
                return_dict=True,
                chunk_duration=8,
            )
        audio = decoded.audio
        audio_lengths = decoded.audio_lengths
        if audio is None or audio_lengths is None:
            raise RuntimeError(
                "MOSS-Audio-Tokenizer decode returned empty audio/audio_lengths"
            )
        audio_cpu = audio.detach().to(device="cpu", dtype=torch.float32)
        lengths_cpu = audio_lengths.detach().to("cpu")
        return [
            audio_cpu[index, 0, : int(lengths_cpu[index])].contiguous()
            for index in range(int(audio_cpu.shape[0]))
        ]

    @staticmethod
    def _loudness_normalize(wav: torch.Tensor) -> torch.Tensor:
        wav = wav.to(torch.float32)
        if wav.numel() == 0:
            return wav
        current_dbfs = 10.0 * torch.log10(torch.mean(wav**2) + 1e-9)
        gain = float(_LOUDNESS_TARGET_DBFS - current_dbfs)
        gain = max(_LOUDNESS_GAIN_MIN_DB, min(gain, _LOUDNESS_GAIN_MAX_DB))
        return wav * (10.0 ** (gain / 20.0))


def load_moss_tts_audio_tokenizer(
    model_path: str = DEFAULT_MOSS_TTS_AUDIO_TOKENIZER,
    *,
    device: str = "cpu",
    dtype: str | torch.dtype = "float32",
) -> MossTTSAudioTokenizer:
    logger.info(f"Loading MOSS-Audio-Tokenizer from {model_path} on {device}")
    try:
        with moss_transformers_processor_compat():
            model = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True,
            )
    except Exception as exc:
        raise RuntimeError(
            "MOSS-TTS support requires OpenMOSS-Team/MOSS-Audio-Tokenizer"
        ) from exc
    model.eval()
    move_kwargs: dict[str, Any] = {"device": device}
    if device != "cpu":
        move_kwargs["dtype"] = _torch_dtype(dtype)
    model.to(**move_kwargs)
    return MossTTSAudioTokenizer(model, device=device)

_AUDIO_DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


def resolve_moss_audio_dtype(
    dtype: str | torch.dtype | None,
    *,
    name: str,
    allow_none: bool,
) -> torch.dtype | None:
    if dtype is None:
        if allow_none:
            return None
    elif isinstance(dtype, str):
        resolved = _AUDIO_DTYPES.get(dtype.lower())
        if resolved is not None:
            return resolved
    elif isinstance(dtype, torch.dtype) and dtype in _AUDIO_DTYPES.values():
        return dtype
    allowed = "float32, bfloat16"
    if allow_none:
        allowed += ", or null"
    raise ValueError(f"{name} must be {allowed}; got {dtype!r}")


def _validate_audio_dtypes(
    *,
    component_dtype: torch.dtype,
    component_name: str,
    compute_dtype: torch.dtype | None,
) -> None:
    if component_dtype not in (torch.float32, torch.bfloat16):
        raise ValueError(
            f"{component_name} must be torch.float32 or torch.bfloat16; "
            f"got {component_dtype!r}"
        )
    if compute_dtype not in (None, torch.float32, torch.bfloat16):
        raise ValueError(
            "compute_dtype must be torch.float32, torch.bfloat16, or None; "
            f"got {compute_dtype!r}"
        )



@dataclass
class MossAudioTokenizerEncoderOutput:
    """Output contract shared with the upstream audio-tokenizer encoder."""

    audio_codes: torch.Tensor
    audio_codes_lengths: torch.Tensor
    encoder_hidden_states: torch.Tensor


@dataclass
class MossAudioTokenizerDecoderOutput:
    """Output contract shared by full and incremental decoder execution."""

    audio: torch.Tensor
    audio_lengths: torch.Tensor


class _LayerScale(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        init: float,
        device: str | torch.device | None,
        dtype: torch.dtype | None,
    ) -> None:
        super().__init__()
        self.scale = nn.Parameter(
            torch.full((channels,), init, device=device, dtype=dtype)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * x


class _RMSNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        eps: float,
        device: str | torch.device | None,
        dtype: torch.dtype | None,
        compute_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.eps = float(eps)
        self.compute_dtype = compute_dtype
        self.alpha = nn.Parameter(torch.ones((1, 1, dim), device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output_dtype = x.dtype
        if self.compute_dtype is not None:
            x = x.to(self.compute_dtype)
        variance = self.eps + torch.mean(x**2, dim=-1, keepdim=True)
        alpha = self.alpha.to(variance)
        if x.dim() == 2:
            alpha = alpha.view(1, -1)
        return (x * (alpha * torch.rsqrt(variance))).to(output_dtype)


def _create_norm(
    norm: str,
    dim: int,
    *,
    device: str | torch.device | None,
    dtype: torch.dtype | None,
) -> nn.Module:
    if norm == "layer_norm":
        return nn.LayerNorm(dim, eps=1e-5, device=device, dtype=dtype)
    if norm == "rms_norm":
        return _RMSNorm(dim, eps=1e-5, device=device, dtype=dtype)
    if norm == "rms_norm_f32":
        return _RMSNorm(
            dim,
            eps=1e-8,
            device=device,
            # note (Zhang Yiyang): This norm explicitly computes in FP32. Keep
            # its scale in FP32 as well so the forward path does not recast the
            # parameter every call.
            dtype=torch.float32,
            compute_dtype=torch.float32,
        )
    raise ValueError(f"unsupported MOSS audio-tokenizer norm: {norm!r}")


def _restore_fp32_compute_parameters(module: nn.Module) -> None:
    """Keep parameters of explicitly FP32 compute modules in FP32."""
    for submodule in module.modules():
        if not isinstance(submodule, _RMSNorm):
            continue
        if submodule.compute_dtype is not torch.float32:
            continue
        with torch.no_grad():
            submodule.alpha.data = submodule.alpha.data.to(dtype=torch.float32)


class _RotaryEmbedding(nn.Module):
    def __init__(self, max_period: float) -> None:
        super().__init__()
        self.max_period = float(max_period)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        offset: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, _, sequence_length, head_dim = q.shape
        frequencies = torch.exp(
            torch.arange(
                head_dim // 2,
                device=q.device,
                dtype=torch.float32,
            )
            * (-math.log(self.max_period) * 2 / head_dim)
        )
        positions = offset.float().view(batch_size, 1, 1, 1) + torch.arange(
            sequence_length,
            device=q.device,
            dtype=torch.float32,
        ).view(1, 1, sequence_length, 1)
        phase = positions * frequencies.view(1, 1, 1, -1)
        cos = torch.cos(phase)
        sin = torch.sin(phase)

        def rotate(x: torch.Tensor) -> torch.Tensor:
            shape = x.shape
            pairs = x.float().view(*shape[:-1], head_dim // 2, 2)
            real, imag = pairs[..., 0], pairs[..., 1]
            return (
                torch.stack(
                    (real * cos - imag * sin, real * sin + imag * cos),
                    dim=-1,
                )
                .to(x.dtype)
                .view(shape)
            )

        return rotate(q), rotate(k)


class _PatchedPretransform(nn.Module):
    def __init__(self, patch_size: int, *, is_downsample: bool) -> None:
        super().__init__()
        self.module_type = "PatchedPretransform"
        self.patch_size = int(patch_size)
        self.downsample_ratio = self.patch_size
        self.is_downsample = bool(is_downsample)

    def forward(
        self,
        x: torch.Tensor,
        input_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, channels, length = x.shape
        if self.is_downsample:
            x = (
                x.reshape(batch_size, channels, -1, self.patch_size)
                .permute(0, 1, 3, 2)
                .reshape(batch_size, channels * self.patch_size, -1)
            )
            return x, input_lengths // self.patch_size
        if channels % self.patch_size:
            raise ValueError(
                "MOSS vocoder patch stage requires channels divisible by "
                f"patch_size, got channels={channels}, patch_size={self.patch_size}"
            )
        output_channels = channels // self.patch_size
        x = (
            x.reshape(batch_size, output_channels, self.patch_size, length)
            .permute(0, 1, 3, 2)
            .reshape(batch_size, output_channels, length * self.patch_size)
        )
        return x, input_lengths * self.patch_size


def _weight_normalized_conv1d(*args: Any, **kwargs: Any) -> nn.Module:
    return nn.utils.parametrizations.weight_norm(nn.Conv1d(*args, **kwargs))


class _LFQ(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        codebook_size: int,
        codebook_dim: int,
        device: str | torch.device | None,
    ) -> None:
        super().__init__()
        self.in_proj = (
            _weight_normalized_conv1d(
                input_dim,
                codebook_dim,
                kernel_size=1,
                device=device,
                dtype=torch.float32,
            )
            if input_dim != codebook_dim
            else nn.Identity()
        )
        self.out_proj = (
            _weight_normalized_conv1d(
                codebook_dim,
                input_dim,
                kernel_size=1,
                device=device,
                dtype=torch.float32,
            )
            if input_dim != codebook_dim
            else nn.Identity()
        )
        self.codebook = nn.Embedding(
            codebook_size,
            codebook_dim,
            device=device,
            dtype=torch.float32,
        )

    def forward(
        self,
        z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.in_proj(z.float()).float()
        flat = F.normalize(encoded.transpose(1, 2).reshape(-1, encoded.shape[1]))
        codebook = F.normalize(self.codebook.weight.float())
        distance = (
            flat.pow(2).sum(1, keepdim=True)
            - 2 * flat @ codebook.t()
            + codebook.pow(2).sum(1, keepdim=True).t()
        )
        indices = (-distance).max(1)[1].reshape(z.shape[0], -1)
        quantized = F.embedding(indices, self.codebook.weight).transpose(1, 2)
        quantized = encoded + (quantized - encoded).detach()
        return self.out_proj(quantized.float()).float(), indices

    def decode_code(self, indices: torch.Tensor) -> torch.Tensor:
        quantized = F.embedding(indices, self.codebook.weight).transpose(1, 2)
        return self.out_proj(quantized.float()).float()


class _ResidualLFQ(nn.Module):
    def __init__(
        self,
        config: dict[str, Any],
        *,
        device: str | torch.device | None,
    ) -> None:
        super().__init__()
        input_dim = int(config.get("input_dim", 1024))
        rvq_dim = int(config.get("rvq_dim") or input_dim)
        output_dim = int(config.get("output_dim") or input_dim)
        self.rvq_dim = rvq_dim
        self.num_quantizers = int(config.get("num_quantizers", 32))
        codebook_size = int(config.get("codebook_size", 1024))
        codebook_dim = int(config.get("codebook_dim", 8))
        self.input_proj = (
            _weight_normalized_conv1d(
                input_dim,
                rvq_dim,
                kernel_size=1,
                device=device,
                dtype=torch.float32,
            )
            if input_dim != rvq_dim
            else nn.Identity()
        )
        self.output_proj = (
            _weight_normalized_conv1d(
                rvq_dim,
                output_dim,
                kernel_size=1,
                device=device,
                dtype=torch.float32,
            )
            if rvq_dim != output_dim
            else nn.Identity()
        )
        self.quantizers = nn.ModuleList(
            [
                _LFQ(
                    input_dim=rvq_dim,
                    codebook_size=codebook_size,
                    codebook_dim=codebook_dim,
                    device=device,
                )
                for _ in range(self.num_quantizers)
            ]
        )
        # Built after checkpoint loading.  Keeping the cache optional preserves
        # the reference path for construction/tests and for unsupported module
        # variants while removing repeated embedding/projection setup in the
        # inference codec.
        self._decode_cache: MossAudioTokenizerQuantizerDecoder | None = None

    @torch.no_grad()
    def build_decode_cache(self) -> None:
        """Cache one batched codebook gather and fixed projection weights."""
        self._decode_cache = MossAudioTokenizerQuantizerDecoder(self)

    def clear_decode_cache(self) -> None:
        """Drop the inference-only cache and restore the reference decode path."""
        self._decode_cache = None

    @torch.no_grad()
    def forward(
        self,
        z: torch.Tensor,
        input_lengths: torch.Tensor,
        num_quantizers: int | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.autocast(device_type="cuda", enabled=False):
            z = self.input_proj(z.float()).float()
            batch_size, _, max_time = z.shape
            mask = torch.arange(max_time, device=z.device).expand(
                batch_size, max_time
            ) < input_lengths.unsqueeze(1)
            residual = z.clone()
            quantized = torch.zeros_like(z)
            indices = []
            count = (
                self.num_quantizers if num_quantizers is None else int(num_quantizers)
            )
            if not 0 < count <= self.num_quantizers:
                raise ValueError(
                    f"num_quantizers must be in [1, {self.num_quantizers}], got {count}"
                )
            update_mask = mask.unsqueeze(1)
            for quantizer in self.quantizers[:count]:
                current, current_indices = quantizer(residual * update_mask)
                quantized += current * update_mask
                residual -= current * update_mask
                indices.append(current_indices)
            return (
                self.output_proj(quantized.float()).float(),
                torch.stack(indices),
                input_lengths,
            )

    @torch.no_grad()
    def decode_codes(self, codes: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=codes.device.type, enabled=False):
            if codes.ndim != 3:
                raise ValueError(
                    "MOSS quantizer codes must be [N, B, T], got "
                    f"{tuple(codes.shape)}"
                )
            count, batch_size, frames = map(int, codes.shape)
            if not 0 < count <= self.num_quantizers:
                raise ValueError(
                    "MOSS quantizer codebook count must be within "
                    f"[1, {self.num_quantizers}], got {count}"
                )
            if (
                self._decode_cache is not None
                and codes.device == self._decode_cache.device
            ):
                return self._decode_cache.decode_codes(codes)
            decoded = torch.zeros(
                batch_size,
                self.rvq_dim,
                frames,
                device=codes.device,
                dtype=torch.float32,
            )
            for index, quantizer in enumerate(self.quantizers[:count]):
                decoded += quantizer.decode_code(codes[index]).float()
            return self.output_proj(decoded.float()).float()


class MossAudioTokenizerEncoder(nn.Module):
    """Inference-only MOSS codec encoder with repository-owned execution code."""

    def __init__(
        self,
        config: dict[str, Any],
        *,
        parameter_device: str | torch.device | None = None,
        encoder_dtype: torch.dtype = torch.bfloat16,
        compute_dtype: torch.dtype | None = None,
        attention_backend: str = _AUTO_ATTENTION_BACKEND,
    ) -> None:
        super().__init__()
        _validate_audio_dtypes(
            component_dtype=encoder_dtype,
            component_name="encoder_dtype",
            compute_dtype=compute_dtype,
        )
        self.config = SimpleNamespace(**config)
        sampling_rate = config.get("sampling_rate") or config.get("sample_rate")
        if sampling_rate is None:
            raise ValueError("MOSS audio-tokenizer config lacks sampling_rate")
        self.sampling_rate = int(sampling_rate)
        self.downsample_rate = int(config["downsample_rate"])
        self.number_channels = int(config.get("number_channels", 1))
        self.enable_channel_interleave = bool(
            config.get("enable_channel_interleave", self.number_channels > 1)
        )
        configured_compute_dtype = str(config.get("compute_dtype", "bf16"))
        resolved_compute_dtype = {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp32": None,
            "float32": None,
        }.get(configured_compute_dtype)
        if configured_compute_dtype not in {
            "bf16",
            "bfloat16",
            "fp32",
            "float32",
        }:
            raise ValueError(
                f"unsupported codec compute_dtype: {configured_compute_dtype!r}"
            )
        requested_compute_dtype = (
            resolved_compute_dtype if compute_dtype is None else compute_dtype
        )
        self.compute_dtype = (
            None
            if requested_compute_dtype is torch.float32
            else requested_compute_dtype
        )
        # note (Zhang Yiyang): The compute policy is also the materialized dtype
        # for the encoder weights. This avoids relying on autocast to recast
        # FP32 parameters on every request. The quantizer is intentionally kept
        # FP32 below.
        self.encoder_dtype = (
            torch.float32
            if requested_compute_dtype is None
            or requested_compute_dtype is torch.float32
            else requested_compute_dtype
        )
        self.attention_backend = resolve_moss_audio_attention_backend(
            attention_backend,
            config.get("attention_implementation"),
        )
        self._uses_moss_audio_tokenizer_v1_weights = "number_channels" not in config

        default_context_duration = float(
            config.get("causal_transformer_context_duration", 10.0)
        )
        channel_factor = (
            self.number_channels
            if self.enable_channel_interleave and self.number_channels > 1
            else 1
        )
        frame_rate = float(self.sampling_rate * channel_factor)
        stages: list[nn.Module] = []
        for stage_config_raw in config["encoder_kwargs"]:
            stage_config = dict(stage_config_raw)
            module_type = stage_config["module_type"]
            if module_type == "PatchedPretransform":
                stage = _PatchedPretransform(
                    int(stage_config["patch_size"]),
                    is_downsample=True,
                )
            elif module_type == "Transformer":
                context_duration = float(
                    stage_config.pop("context_duration", default_context_duration)
                )
                stage = MossAudioTokenizerProjectedTransformer.from_config(
                    stage_config,
                    context=int(round(frame_rate * context_duration)),
                    moss_audio_tokenizer_v1_weights=(
                        self._uses_moss_audio_tokenizer_v1_weights
                    ),
                    device=parameter_device,
                    dtype=self.encoder_dtype,
                    attention_backend=self.attention_backend,
                )
            else:
                raise ValueError(f"unsupported MOSS encoder stage: {module_type!r}")
            stages.append(stage)
            frame_rate /= int(getattr(stage, "downsample_ratio", 1))
        self.encoder = nn.ModuleList(stages)

        quantizer_config = dict(config["quantizer_kwargs"])
        quantizer_type = quantizer_config.get(
            "quantizer_type", config.get("quantizer_type", "rlfq")
        )
        if quantizer_type not in {"rlfq", "random_prefix_rlfq"}:
            raise ValueError(
                "repository-local MOSS encoder supports residual LFQ checkpoints; "
                f"got quantizer_type={quantizer_type!r}"
            )
        self.quantizer = _ResidualLFQ(
            quantizer_config,
            device=parameter_device,
        )

    def supports_packed_attention(self) -> bool:
        device = next(self.parameters()).device
        return all(
            not isinstance(stage, MossAudioTokenizerProjectedTransformer)
            or stage.supports_packed_flash(device, self.encoder_dtype)
            for stage in self.encoder
        )

    def supports_packed_flash(self) -> bool:
        return self.supports_packed_attention()

    def resolve_attention_backend(
        self,
        device: str | torch.device,
        dtype: torch.dtype | None = None,
    ) -> _AttentionBackendResolution:
        device = torch.device(device)
        return _merge_attention_backend_resolutions(
            [
                stage.resolve_attention_backend(
                    device,
                    self.encoder_dtype if dtype is None else dtype,
                )
                for stage in self.encoder
                if isinstance(stage, MossAudioTokenizerProjectedTransformer)
            ]
        )

    def _prepare_waveform_batch(
        self,
        waveforms: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        if not waveforms:
            raise ValueError("waveforms must contain at least one item")
        device = waveforms[0].device
        dtype = waveforms[0].dtype
        normalized = []
        lengths_cpu = []
        for index, waveform in enumerate(waveforms):
            if self.number_channels == 1 and waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            if waveform.dim() != 2 or waveform.shape[0] != self.number_channels:
                raise ValueError(
                    f"waveforms[{index}] must have shape "
                    f"({self.number_channels}, T), got {tuple(waveform.shape)}"
                )
            normalized.append(waveform)
            lengths_cpu.append(int(waveform.shape[-1]))
        max_length = max(lengths_cpu)
        batch = torch.zeros(
            len(normalized),
            self.number_channels,
            max_length,
            device=device,
            dtype=dtype,
        )
        for index, waveform in enumerate(normalized):
            batch[index, :, : waveform.shape[-1]] = waveform
        lengths = torch.tensor(lengths_cpu, device=device, dtype=torch.long)
        return batch, lengths, lengths_cpu

    def _flatten_channels(
        self,
        input_values: torch.Tensor,
        input_lengths: torch.Tensor,
        input_lengths_cpu: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        remainder = input_values.shape[-1] % self.downsample_rate
        if remainder:
            input_values = F.pad(
                input_values,
                (0, self.downsample_rate - remainder),
            )
        if self.number_channels > 1 and self.enable_channel_interleave:
            input_values = (
                input_values.transpose(1, 2)
                .contiguous()
                .view(input_values.shape[0], 1, -1)
            )
            input_lengths = input_lengths * self.number_channels
            input_lengths_cpu = [
                length * self.number_channels for length in input_lengths_cpu
            ]
        return input_values, input_lengths, input_lengths_cpu

    @torch.no_grad()
    def batch_encode(
        self,
        waveforms: list[torch.Tensor],
        num_quantizers: int | None = None,
        chunk_duration: float | None = None,
    ) -> MossAudioTokenizerEncoderOutput:
        if chunk_duration is not None:
            raise ValueError(
                "repository-local MOSS encoder only supports full non-streaming "
                "batch_encode (chunk_duration=None)"
            )
        hidden, lengths, lengths_cpu = self._prepare_waveform_batch(waveforms)
        hidden, lengths, lengths_cpu = self._flatten_channels(
            hidden,
            lengths,
            lengths_cpu,
        )
        hidden = hidden.to(dtype=self.encoder_dtype)
        for stage in self.encoder:
            if isinstance(stage, MossAudioTokenizerProjectedTransformer):
                hidden, lengths = stage(
                    hidden,
                    lengths,
                    input_lengths_cpu=lengths_cpu,
                )
            else:
                hidden, lengths = stage(hidden, lengths)
                lengths_cpu = [
                    length // int(stage.downsample_ratio) for length in lengths_cpu
                ]
        _, codes, code_lengths = self.quantizer(
            hidden.float(),
            lengths,
            num_quantizers,
        )
        max_valid_length = max(lengths_cpu, default=0)
        return MossAudioTokenizerEncoderOutput(
            audio_codes=codes[:, :, :max_valid_length],
            audio_codes_lengths=code_lengths,
            encoder_hidden_states=hidden[:, :, :max_valid_length].float(),
        )


class MossAudioEncoder:
    """Prepare reference audio and encode it with a shared MOSS encoder."""

    def __init__(self, model: MossAudioTokenizerEncoder, *, device: str) -> None:
        self.model = model
        self.device = str(device)
        config = model.config
        self.sample_rate = int(
            getattr(model, "sampling_rate", getattr(config, "sampling_rate"))
        )
        self.number_channels = int(
            getattr(model, "number_channels", getattr(config, "number_channels", 1))
        )

    def encode_paths(
        self,
        paths: list[str | PathLike[str]],
        *,
        num_quantizers: int,
    ) -> list[torch.Tensor]:
        if not paths:
            raise ValueError("paths must contain at least one audio path")
        return self.encode_waveforms(
            self.load_paths(paths),
            num_quantizers=num_quantizers,
        )

    def load_paths(
        self,
        paths: list[str | PathLike[str]],
    ) -> list[tuple[torch.Tensor, int]]:
        import torchaudio

        waveforms = []
        for path in paths:
            waveform, sample_rate = torchaudio.load(path)
            if int(sample_rate) != self.sample_rate:
                waveform = torchaudio.functional.resample(
                    waveform=waveform,
                    orig_freq=int(sample_rate),
                    new_freq=self.sample_rate,
                )
            waveforms.append((waveform, self.sample_rate))
        return waveforms

    def encode_wavs(
        self,
        waveforms: list[torch.Tensor],
        sample_rate: int,
        *,
        num_quantizers: int,
    ) -> list[torch.Tensor]:
        return self.encode_waveforms(
            [(waveform, int(sample_rate)) for waveform in waveforms],
            num_quantizers=num_quantizers,
        )

    def encode_waveforms(
        self,
        waveforms: list[tuple[torch.Tensor, int]],
        *,
        num_quantizers: int,
    ) -> list[torch.Tensor]:
        if not waveforms:
            raise ValueError("waveforms must contain at least one waveform")
        prepared = [
            self._prepare_waveform(waveform, sample_rate)
            for waveform, sample_rate in waveforms
        ]
        with torch.inference_mode():
            encoded = self.model.batch_encode(
                prepared,
                num_quantizers=int(num_quantizers),
            )
        codes = encoded.audio_codes
        lengths = encoded.audio_codes_lengths
        if codes is None or lengths is None:
            raise RuntimeError(
                "MOSS audio encoder returned empty audio_codes/audio_codes_lengths"
            )
        codes = codes.detach().to(device="cpu", dtype=torch.long)
        lengths = lengths.detach().to("cpu")
        return [
            codes[:, index, : int(lengths[index])].transpose(0, 1).contiguous()
            for index in range(int(codes.shape[1]))
        ]

    def _prepare_waveform(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
    ) -> torch.Tensor:
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.ndim != 2:
            raise ValueError(
                "expected waveform with shape [channels, samples], got "
                f"{tuple(waveform.shape)}"
            )
        if self.number_channels == 1:
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
        else:
            if waveform.shape[0] == 1:
                waveform = waveform.repeat(self.number_channels, 1)
            elif waveform.shape[0] > self.number_channels:
                waveform = waveform[: self.number_channels]
            if waveform.shape[0] != self.number_channels:
                raise ValueError(
                    f"expected {self.number_channels} audio channels, "
                    f"got {waveform.shape[0]}"
                )
        if int(sample_rate) != self.sample_rate:
            import torchaudio

            waveform = torchaudio.functional.resample(
                waveform=waveform,
                orig_freq=int(sample_rate),
                new_freq=self.sample_rate,
            )
        waveform = self._loudness_normalize(waveform)
        if self.number_channels == 1:
            waveform = waveform.squeeze(0)
        return waveform.to(device=self.device, dtype=torch.float32)

    @staticmethod
    def _loudness_normalize(waveform: torch.Tensor) -> torch.Tensor:
        waveform = waveform.to(torch.float32)
        if waveform.numel() == 0:
            return waveform
        current_dbfs = 10.0 * torch.log10(torch.mean(waveform**2) + 1e-9)
        gain = float(_LOUDNESS_TARGET_DBFS - current_dbfs)
        gain = max(_LOUDNESS_GAIN_MIN_DB, min(gain, _LOUDNESS_GAIN_MAX_DB))
        return waveform * (10.0 ** (gain / 20.0))


def _normalize_moss_audio_tokenizer_v1_transformer_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Map MOSS v1 weight names onto the shared Transformer modules."""

    replacements = (
        (".self_attn.in_projs.0.", ".self_attn.in_proj."),
        (".self_attn.out_projs.0.", ".self_attn.out_proj."),
        (".linear1.", ".ffn.linear1."),
        (".linear2.", ".ffn.linear2."),
    )
    normalized = {}
    for name, tensor in state_dict.items():
        normalized_name = name
        for old, new in replacements:
            normalized_name = normalized_name.replace(old, new)
        normalized[normalized_name] = tensor
    return normalized


def load_moss_audio_encoder(
    model_path: str,
    *,
    device: str | torch.device,
    encoder_dtype: torch.dtype = torch.bfloat16,
    compute_dtype: torch.dtype | None = None,
    attention_backend: str = _AUTO_ATTENTION_BACKEND,
) -> MossAudioEncoder:
    """Load only encoder/quantizer weights without executing checkpoint code."""

    resolved_path = resolve_model_path(str(model_path))
    config_path = Path(resolved_path) / "config.json"
    with config_path.open(encoding="utf-8") as config_file:
        config = json.load(config_file)
    if config.get("model_type") != "moss-audio-tokenizer":
        raise ValueError(
            f"expected model_type='moss-audio-tokenizer' in {config_path}, "
            f"got {config.get('model_type')!r}"
        )

    model = MossAudioTokenizerEncoder(
        config,
        parameter_device="meta",
        encoder_dtype=encoder_dtype,
        compute_dtype=compute_dtype,
        attention_backend=attention_backend,
    )
    target_device = torch.device(device)
    backend_resolution = model.resolve_attention_backend(target_device)
    _log_attention_backend_resolution(
        "encoder",
        requested_backend=model.attention_backend,
        resolution=backend_resolution,
        device=target_device,
        dtype=model.encoder_dtype,
    )
    if model._uses_moss_audio_tokenizer_v1_weights:
        state_dict = load_weights_by_prefix(
            str(resolved_path),
            prefix="encoder.",
        )
        state_dict = _normalize_moss_audio_tokenizer_v1_transformer_state_dict(
            state_dict
        )
        try:
            model.encoder.load_state_dict(state_dict, strict=True, assign=True)
        except TypeError:
            model.encoder.load_state_dict(state_dict, strict=True)
        model.encoder = model.encoder.to(device=device, dtype=model.encoder_dtype)
        _restore_fp32_compute_parameters(model.encoder)
        model.encoder.eval()
    else:
        model.encoder = load_module(
            model.encoder,
            str(resolved_path),
            prefix="encoder.",
            dtype=model.encoder_dtype,
            device=device,
            strict=True,
        )
        _restore_fp32_compute_parameters(model.encoder)
    model.quantizer = load_module(
        model.quantizer,
        str(resolved_path),
        prefix="quantizer.",
        dtype=torch.float32,
        device=device,
        strict=True,
    )
    model.eval()
    logger.info(
        "Loaded repository-local MOSS audio encoder from %s on %s "
        "(channels=%d, encoder_dtype=%s, compute_dtype=%s)",
        resolved_path,
        device,
        model.number_channels,
        model.encoder_dtype,
        model.compute_dtype,
    )
    return MossAudioEncoder(model, device=str(device))


class MossAudioTokenizerVocoder(_StreamingModule):
    """Inference-only MOSS quantizer and waveform decoder."""

    def __init__(
        self,
        config: dict[str, Any],
        *,
        parameter_device: str | torch.device | None = None,
        decoder_dtype: torch.dtype = torch.bfloat16,
        compute_dtype: torch.dtype | None = None,
        attention_backend: str = _AUTO_ATTENTION_BACKEND,
    ) -> None:
        super().__init__()
        _validate_audio_dtypes(
            component_dtype=decoder_dtype,
            component_name="decoder_dtype",
            compute_dtype=compute_dtype,
        )
        self.config = SimpleNamespace(**config)
        sampling_rate = config.get("sampling_rate") or config.get("sample_rate")
        if sampling_rate is None:
            raise ValueError("MOSS audio-tokenizer config lacks sampling_rate")
        self.sampling_rate = int(sampling_rate)
        self.downsample_rate = int(config["downsample_rate"])
        self.number_channels = int(config.get("number_channels", 1))
        self.enable_channel_interleave = bool(
            config.get("enable_channel_interleave", True)
        )
        if not hasattr(self.config, "sampling_rate"):
            self.config.sampling_rate = self.sampling_rate
        if not hasattr(self.config, "number_channels"):
            self.config.number_channels = self.number_channels
        if not hasattr(self.config, "enable_channel_interleave"):
            self.config.enable_channel_interleave = self.enable_channel_interleave
        self.decoder_dtype = decoder_dtype if compute_dtype is None else compute_dtype
        self.compute_dtype = None if compute_dtype is torch.float32 else compute_dtype
        self.attention_backend = resolve_moss_audio_attention_backend(
            attention_backend,
            config.get("attention_implementation"),
        )
        self._uses_moss_audio_tokenizer_v1_weights = "number_channels" not in config

        quantizer_config = dict(config["quantizer_kwargs"])
        quantizer_type = quantizer_config.get(
            "quantizer_type", config.get("quantizer_type", "rlfq")
        )
        if quantizer_type not in {"rlfq", "random_prefix_rlfq"}:
            raise ValueError(
                "repository-local MOSS vocoder supports residual LFQ checkpoints; "
                f"got quantizer_type={quantizer_type!r}"
            )
        self.quantizer = _ResidualLFQ(
            quantizer_config,
            device=parameter_device,
        )

        default_context_duration = float(
            config.get("causal_transformer_context_duration", 10.0)
        )
        frame_rate = float(self.sampling_rate) / self.downsample_rate
        stages: list[nn.Module] = []
        for stage_config_raw in config["decoder_kwargs"]:
            stage_config = dict(stage_config_raw)
            module_type = stage_config["module_type"]
            if module_type == "PatchedPretransform":
                stage = _PatchedPretransform(
                    int(stage_config["patch_size"]),
                    is_downsample=False,
                )
            elif module_type == "Transformer":
                context_duration = float(
                    stage_config.pop("context_duration", default_context_duration)
                )
                stage = MossAudioTokenizerProjectedTransformer.from_config(
                    stage_config,
                    context=int(round(frame_rate * context_duration)),
                    moss_audio_tokenizer_v1_weights=(
                        self._uses_moss_audio_tokenizer_v1_weights
                    ),
                    device=parameter_device,
                    dtype=self.decoder_dtype,
                    attention_backend=self.attention_backend,
                )
            else:
                raise ValueError(f"unsupported MOSS decoder stage: {module_type!r}")
            stages.append(stage)
            if isinstance(stage, _PatchedPretransform):
                frame_rate *= stage.patch_size
        self.decoder: nn.Module = nn.ModuleList(stages)
        # Native Local streaming owns decoder state independently from the
        # execution batch.  The fields are populated after checkpoint loading,
        # when ``self.decoder`` has been wrapped and moved to its final device.
        self._decoder_streaming_modules: list[_StreamingModule] = []
        self._decoder_state_capacity = 0
        self._decoder_real_state_capacity = 0
        self._decoder_scratch_capacity = 0

    def _decoder_device(self) -> torch.device:
        parameter = next(self.decoder.parameters(), None)
        if parameter is not None:
            return parameter.device
        return self._streaming_device()

    def _start_decoder_state_pool(self, total_capacity: int) -> None:
        if self._decoder_streaming_modules:
            raise RuntimeError("MOSS decoder state pool is already initialized")
        modules = [
            module
            for module in self.decoder.modules()
            if isinstance(module, _StreamingModule)
        ]
        if not modules:
            raise RuntimeError("MOSS decoder has no streaming-capable stages")
        if any(module._streaming_state is not None for module in modules):
            raise RuntimeError("MOSS decoder is already in a streaming session")
        states = [module._init_streaming_state(total_capacity) for module in modules]
        for module, state in zip(modules, states, strict=True):
            module._streaming_state = state
        self._decoder_streaming_modules = modules

    def initialize_decoder_state_pool(
        self,
        state_capacity: int,
        scratch_capacity: int = 0,
    ) -> None:
        """Allocate persistent decoder state independently of execution width."""
        if not isinstance(state_capacity, int) or isinstance(state_capacity, bool):
            raise TypeError("state_capacity must be an int")
        if not isinstance(scratch_capacity, int) or isinstance(scratch_capacity, bool):
            raise TypeError("scratch_capacity must be an int")
        if state_capacity <= 0 or scratch_capacity < 0:
            raise ValueError(
                "state_capacity must be > 0 and scratch_capacity must be >= 0; "
                f"got state_capacity={state_capacity}, scratch_capacity={scratch_capacity}"
            )
        total_capacity = state_capacity + scratch_capacity
        self._start_decoder_state_pool(total_capacity)
        self._decoder_state_capacity = total_capacity
        self._decoder_real_state_capacity = state_capacity
        self._decoder_scratch_capacity = scratch_capacity

    def close_decoder_state_pool(self) -> None:
        """Release all decoder streaming state and its persistent cache rows."""
        if not self._decoder_streaming_modules:
            return
        for module in reversed(self._decoder_streaming_modules):
            module._streaming_state = None
        self._decoder_streaming_modules = []
        self._decoder_state_capacity = 0
        self._decoder_real_state_capacity = 0
        self._decoder_scratch_capacity = 0

    def reset_decoder_state_slots(self, state_slot_ids: torch.Tensor) -> None:
        """Reset only the requested decoder state slots."""
        if not self._decoder_streaming_modules:
            raise RuntimeError("MOSS decoder state pool is not initialized")
        if state_slot_ids.ndim != 1:
            raise ValueError(
                f"state_slot_ids must be rank 1, got {tuple(state_slot_ids.shape)}"
            )
        if state_slot_ids.dtype != torch.long:
            raise TypeError("state_slot_ids must have dtype torch.long")
        device = self._decoder_device()
        if state_slot_ids.device != device:
            raise ValueError(
                f"state_slot_ids must be on {device}, got {state_slot_ids.device}"
            )
        if state_slot_ids.numel() == 0:
            return
        if device.type != "cuda":
            if bool(torch.any(state_slot_ids < 0)) or bool(
                torch.any(state_slot_ids >= self._decoder_state_capacity)
            ):
                raise ValueError(
                    "state_slot_ids must be in "
                    f"[0, {self._decoder_state_capacity}), got "
                    f"{state_slot_ids.detach().to('cpu').tolist()}"
                )
            if torch.unique(state_slot_ids).numel() != state_slot_ids.numel():
                raise ValueError("state_slot_ids must be unique")
        for module in self._decoder_streaming_modules:
            state = module._streaming_state
            if state is None:
                raise RuntimeError("MOSS decoder streaming state was unexpectedly closed")
            reset_slots = getattr(state, "reset_slots", None)
            if not callable(reset_slots):
                raise RuntimeError(
                    f"{module.__class__.__name__} has no indexed state reset"
                )
            reset_slots(state_slot_ids)

    def _validate_decoder_streaming_inputs(
        self,
        codes: torch.Tensor,
        codes_lengths: torch.Tensor,
        state_slot_ids: torch.Tensor,
        valid_rows: torch.Tensor,
        *,
        scratch_rows_are_disposable: bool = False,
    ) -> StreamingExecutionContext:
        if not self._decoder_streaming_modules:
            raise RuntimeError("MOSS decoder state pool is not initialized")
        if codes.ndim != 3:
            raise ValueError(
                "codes must have shape [num_quantizers, batch, time], "
                f"got {tuple(codes.shape)}"
            )
        device = self._decoder_device()
        if codes.device != device:
            raise ValueError(f"codes must be on {device}, got {codes.device}")
        if codes.dtype != torch.long:
            raise TypeError("codes must have dtype torch.long")
        _, batch_size, frame_count = codes.shape
        if batch_size <= 0 or frame_count <= 0:
            raise ValueError(
                f"codes must have positive batch/time dimensions, got B={batch_size}, T={frame_count}"
            )
        if codes_lengths.shape != (batch_size,):
            raise ValueError(
                f"codes_lengths must have shape ({batch_size},), got "
                f"{tuple(codes_lengths.shape)}"
            )
        if codes_lengths.dtype != torch.long or codes_lengths.device != device:
            raise TypeError("codes_lengths must be a torch.long tensor on the codec device")
        if device.type != "cuda" and (
            bool(torch.any(codes_lengths < 0))
            or bool(torch.any(codes_lengths > frame_count))
        ):
            raise ValueError(
                f"codes_lengths must be in [0, {frame_count}], got "
                f"{codes_lengths.detach().to('cpu').tolist()}"
            )
        context = StreamingExecutionContext(
            state_slot_ids,
            valid_rows,
            scratch_rows_are_disposable=bool(scratch_rows_are_disposable),
        )
        context.validate(
            batch_size=batch_size,
            state_capacity=self._decoder_state_capacity,
            real_state_capacity=self._decoder_real_state_capacity,
            device=device,
            check_values=device.type != "cuda",
        )
        if (
            device.type != "cuda"
            and torch.unique(state_slot_ids).numel() != state_slot_ids.numel()
        ):
            raise ValueError("state_slot_ids must be unique")
        return context

    @torch.no_grad()
    def decode_streaming_batch(
        self,
        codes: torch.Tensor,
        codes_lengths: torch.Tensor,
        state_slot_ids: torch.Tensor,
        valid_rows: torch.Tensor,
    ) -> MossAudioTokenizerDecoderOutput:
        audio, audio_lengths = self.decode_streaming_tensors(
            codes,
            codes_lengths,
            state_slot_ids,
            valid_rows,
        )
        return MossAudioTokenizerDecoderOutput(audio, audio_lengths)

    @torch.no_grad()
    def decode_streaming_tensors(
        self,
        codes: torch.Tensor,
        codes_lengths: torch.Tensor,
        state_slot_ids: torch.Tensor,
        valid_rows: torch.Tensor,
        *,
        scratch_rows_are_disposable: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Tensor-only indexed decode boundary for scheduler/graph callers."""
        execution_context = self._validate_decoder_streaming_inputs(
            codes,
            codes_lengths,
            state_slot_ids,
            valid_rows,
            scratch_rows_are_disposable=scratch_rows_are_disposable,
        )
        return self._decode_frame_tensors(
            codes,
            codes_lengths,
            execution_context=execution_context,
        )

    def _reset_streaming_slots(self, reset_mask: torch.Tensor) -> None:
        """Compatibility reset hook used by the legacy state-pool adapter."""
        if not self._decoder_streaming_modules:
            raise RuntimeError("MOSS decoder state pool is not initialized")
        if reset_mask.shape != (self._decoder_state_capacity,):
            raise ValueError(
                "reset_mask must have shape "
                f"({self._decoder_state_capacity},), got {tuple(reset_mask.shape)}"
            )
        slots = torch.nonzero(reset_mask, as_tuple=False).flatten()
        self.reset_decoder_state_slots(slots.to(device=self._decoder_device()))

    def resolve_attention_backend(
        self,
        device: str | torch.device,
        dtype: torch.dtype | None = None,
    ) -> _AttentionBackendResolution:
        device = torch.device(device)
        decoder_dtype = self.decoder_dtype if dtype is None else dtype
        if isinstance(self.decoder, MossAudioTokenizerVocoderDecoder):
            return self.decoder.resolve_attention_backend(device, decoder_dtype)
        return _merge_attention_backend_resolutions(
            [
                stage.resolve_attention_backend(device, decoder_dtype)
                for stage in self.decoder
                if isinstance(stage, MossAudioTokenizerProjectedTransformer)
            ]
        )

    def _restore_channels_from_codec(
        self,
        audio: torch.Tensor,
        audio_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.number_channels == 1 or not self.enable_channel_interleave:
            return audio.float(), audio_lengths
        if audio.shape[1] != 1:
            raise ValueError(
                "interleaved MOSS decoder output must have one codec channel, "
                f"got {audio.shape[1]}"
            )
        audio = (
            audio.squeeze(1)
            .contiguous()
            .view(audio.shape[0], -1, self.number_channels)
            .transpose(1, 2)
            .contiguous()
            .float()
        )
        return (
            audio,
            torch.div(
                audio_lengths,
                self.number_channels,
                rounding_mode="floor",
            ),
        )

    @torch.no_grad()
    def _decode_frame_tensors(
        self,
        codes: torch.Tensor,
        codes_lengths: torch.Tensor | None = None,
        *,
        codes_lengths_cpu: Sequence[int] | None = None,
        execution_context: StreamingExecutionContext | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if codes.dim() != 3:
            raise ValueError(
                "MOSS audio codes must have shape [num_quantizers, batch, time], "
                f"got {tuple(codes.shape)}"
            )
        _, batch_size, time = codes.shape
        if codes_lengths is None:
            codes_lengths = torch.full(
                (batch_size,),
                time,
                device=codes.device,
                dtype=torch.long,
            )
        elif codes_lengths.shape != (batch_size,):
            raise ValueError(
                "codes_lengths must have shape "
                f"({batch_size},), got {tuple(codes_lengths.shape)}"
            )
        codes_lengths = codes_lengths.to(device=codes.device, dtype=torch.long)
        check_length_values = execution_context is None or codes.device.type != "cuda"
        if check_length_values and (
            bool(torch.any(codes_lengths < 0))
            or bool(torch.any(codes_lengths > time))
        ):
            raise ValueError(
                f"codes_lengths must be in [0, {time}], got "
                f"{codes_lengths.detach().to('cpu').tolist()}"
            )
        if codes_lengths_cpu is not None and len(codes_lengths_cpu) != batch_size:
            raise ValueError(
                "codes_lengths_cpu must match the decoder batch size, got "
                f"{len(codes_lengths_cpu)} and {batch_size}"
            )
        hidden = self.quantizer.decode_codes(codes).to(dtype=self.decoder_dtype)
        decoder = self.decoder
        if isinstance(decoder, MossAudioTokenizerVocoderDecoder):
            audio, audio_lengths = decoder(
                hidden,
                codes_lengths,
                input_lengths_cpu=codes_lengths_cpu,
                execution_context=execution_context,
            )
        else:
            audio, audio_lengths = hidden, codes_lengths
            for stage in decoder:
                if isinstance(stage, MossAudioTokenizerProjectedTransformer):
                    audio, audio_lengths = stage(
                        audio,
                        audio_lengths,
                        execution_context=execution_context,
                    )
                else:
                    audio, audio_lengths = stage(audio, audio_lengths)
        audio, audio_lengths = self._restore_channels_from_codec(
            audio,
            audio_lengths,
        )
        if execution_context is not None:
            invalid = ~execution_context.valid_rows
            audio = audio.masked_fill(invalid.view(-1, 1, 1), 0)
            audio_lengths = audio_lengths.masked_fill(invalid, 0)
        return audio, audio_lengths

    @torch.no_grad()
    def _decode_frame(
        self,
        codes: torch.Tensor,
        codes_lengths: torch.Tensor | None = None,
        *,
        codes_lengths_cpu: Sequence[int] | None = None,
        execution_context: StreamingExecutionContext | None = None,
    ) -> MossAudioTokenizerDecoderOutput:
        audio, audio_lengths = self._decode_frame_tensors(
            codes,
            codes_lengths,
            codes_lengths_cpu=codes_lengths_cpu,
            execution_context=execution_context,
        )
        return MossAudioTokenizerDecoderOutput(audio, audio_lengths)

    @staticmethod
    def _plan_batch_stream_step(
        remaining: torch.Tensor,
        max_step_length: int,
    ) -> tuple[int, torch.Tensor]:
        positive_mask = remaining > 0
        if not bool(positive_mask.any().item()):
            raise RuntimeError("cannot plan a streaming decode with no remaining codes")
        if max_step_length > 0:
            full_step_mask = remaining >= max_step_length
            if bool(full_step_mask.any().item()):
                return max_step_length, full_step_mask
        step_length = int(remaining[positive_mask].min().item())
        if max_step_length > 0:
            step_length = min(step_length, max_step_length)
        return step_length, remaining >= step_length

    @torch.no_grad()
    def batch_decode(
        self,
        codes_list: list[torch.Tensor],
        *,
        num_quantizers: int | None = None,
        chunk_duration: float | None = None,
    ) -> MossAudioTokenizerDecoderOutput:
        if not codes_list:
            raise ValueError("codes_list must contain at least one code tensor")
        quantizer_counts = [int(codes.shape[0]) for codes in codes_list]
        if num_quantizers is None:
            num_quantizers = quantizer_counts[0]
            if any(count != num_quantizers for count in quantizer_counts):
                raise ValueError("all code tensors must use the same quantizer count")
        elif min(quantizer_counts) < num_quantizers:
            raise ValueError(
                "num_quantizers exceeds at least one code tensor's quantizer count"
            )
        device = codes_list[0].device
        lengths_cpu = [int(codes.shape[-1]) for codes in codes_list]
        lengths = torch.tensor(lengths_cpu, device=device, dtype=torch.long)
        max_length = max(lengths_cpu)
        audio_codes = torch.zeros(
            num_quantizers,
            len(codes_list),
            max_length,
            device=device,
            dtype=torch.long,
        )
        for index, codes in enumerate(codes_list):
            codes = codes[:num_quantizers].to(device=device, dtype=torch.long)
            audio_codes[:, index, : codes.shape[-1]] = codes
        if chunk_duration is None:
            return self._decode_frame(
                audio_codes,
                lengths,
                codes_lengths_cpu=lengths_cpu,
            )
        if chunk_duration <= 0:
            raise ValueError("chunk_duration must be positive")
        chunk_samples = int(round(chunk_duration * self.sampling_rate))
        if chunk_samples <= 0 or chunk_samples % self.downsample_rate:
            raise ValueError(
                "chunk_duration * sampling_rate must be a positive multiple of "
                f"downsample_rate={self.downsample_rate}"
            )
        chunk_frames = chunk_samples // self.downsample_rate
        cursors = torch.zeros_like(lengths)
        audio_chunks: list[list[torch.Tensor]] = [[] for _ in codes_list]
        with self.streaming(batch_size=len(codes_list)):
            while bool((cursors < lengths).any().item()):
                step_frames, active_mask = self._plan_batch_stream_step(
                    lengths - cursors,
                    chunk_frames,
                )
                step_codes = torch.zeros(
                    num_quantizers,
                    len(codes_list),
                    step_frames,
                    device=device,
                    dtype=torch.long,
                )
                step_lengths = torch.zeros_like(lengths)
                active_indices = torch.nonzero(active_mask, as_tuple=False).flatten()
                for index in active_indices.tolist():
                    start = int(cursors[index].item())
                    end = start + step_frames
                    step_codes[:, index] = audio_codes[:, index, start:end]
                    step_lengths[index] = step_frames
                self._set_streaming_exec_mask(active_mask)
                result = self._decode_frame(step_codes, step_lengths)
                for index in active_indices.tolist():
                    audio_length = int(result.audio_lengths[index].item())
                    if audio_length:
                        audio_chunks[index].append(
                            result.audio[index, :, :audio_length].clone()
                        )
                    cursors[index] += step_frames
        output_lengths = torch.tensor(
            [sum(chunk.shape[-1] for chunk in chunks) for chunks in audio_chunks],
            device=device,
            dtype=torch.long,
        )
        max_output_length = int(output_lengths.max().item())
        output_channels = next(
            (int(chunks[0].shape[0]) for chunks in audio_chunks if chunks),
            self.number_channels,
        )
        audio = torch.zeros(
            len(codes_list),
            output_channels,
            max_output_length,
            device=device,
            dtype=torch.float32,
        )
        for index, chunks in enumerate(audio_chunks):
            if chunks:
                waveform = torch.cat(chunks, dim=-1)
                audio[index, :, : waveform.shape[-1]] = waveform
        return MossAudioTokenizerDecoderOutput(audio, output_lengths)

    @torch.no_grad()
    def decode(
        self,
        audio_codes: torch.Tensor,
        *,
        padding_mask: torch.Tensor | None = None,
        num_quantizers: int | None = None,
        return_dict: bool = True,
        chunk_duration: float | None = None,
    ) -> MossAudioTokenizerDecoderOutput | tuple[torch.Tensor, torch.Tensor]:
        if audio_codes.dim() != 3:
            raise ValueError(
                "audio_codes must have shape [num_quantizers, batch, time], "
                f"got {tuple(audio_codes.shape)}"
            )
        if num_quantizers is not None:
            if not 0 < num_quantizers <= audio_codes.shape[0]:
                raise ValueError(
                    "num_quantizers must be in "
                    f"[1, {audio_codes.shape[0]}], got {num_quantizers}"
                )
            audio_codes = audio_codes[:num_quantizers]
        batch_size, max_length = audio_codes.shape[1:]
        if padding_mask is None:
            lengths = torch.full(
                (batch_size,),
                max_length,
                device=audio_codes.device,
                dtype=torch.long,
            )
            lengths_cpu = [max_length] * batch_size
        else:
            if padding_mask.shape != (batch_size, max_length):
                raise ValueError(
                    "padding_mask must have shape "
                    f"({batch_size}, {max_length}), got {tuple(padding_mask.shape)}"
                )
            lengths = padding_mask.to(device=audio_codes.device, dtype=torch.bool).sum(
                dim=-1, dtype=torch.long
            )
            lengths_cpu = lengths.detach().to("cpu").tolist()
        if chunk_duration is None:
            result = self._decode_frame(
                audio_codes,
                lengths,
                codes_lengths_cpu=lengths_cpu,
            )
        else:
            result = self.batch_decode(
                [
                    audio_codes[:, index, : int(lengths[index].item())]
                    for index in range(batch_size)
                ],
                num_quantizers=int(audio_codes.shape[0]),
                chunk_duration=chunk_duration,
            )
        if return_dict:
            return result
        return result.audio, result.audio_lengths


class MossAudioVocoder:
    """Narrow wrapper used by the MOSS-TTS Delay vocoder stage."""

    def __init__(self, model: MossAudioTokenizerVocoder, *, device: str) -> None:
        self.model = model
        self.device = str(device)
        self.sample_rate = int(model.sampling_rate)

    @torch.no_grad()
    def decode_codes(
        self,
        codes: torch.Tensor | list[torch.Tensor],
    ) -> list[torch.Tensor]:
        if isinstance(codes, torch.Tensor):
            codes = [codes]
        if not codes:
            return []
        codes_nq_t = [
            item.transpose(0, 1).contiguous().to(device=self.device, dtype=torch.long)
            for item in codes
        ]
        count = int(codes_nq_t[0].shape[0])
        if any(int(item.shape[0]) != count for item in codes_nq_t):
            raise ValueError("all audio-code rows must use the same quantizer count")
        lengths_cpu = [int(item.shape[1]) for item in codes_nq_t]
        max_length = max(lengths_cpu)
        audio_codes = torch.zeros(
            count,
            len(codes_nq_t),
            max_length,
            device=self.device,
            dtype=torch.long,
        )
        for index, item in enumerate(codes_nq_t):
            audio_codes[:, index, : item.shape[1]] = item
        lengths = torch.tensor(lengths_cpu, device=self.device, dtype=torch.int32)
        hidden = self.model.quantizer.decode_codes(audio_codes)
        decoder = self.model.decoder
        if not isinstance(decoder, MossAudioTokenizerVocoderDecoder):
            # Directly constructed test/tools models may still hold the raw
            # ModuleList.  Wrap it lazily so the public decode path remains
            # compatible with checkpoints loaded through the factory.
            decoder = MossAudioTokenizerVocoderDecoder.from_module(
                decoder,
                attention_backend=self.model.attention_backend,
            )
        hidden = hidden.to(dtype=self.model.decoder_dtype)
        audio, _ = decoder(
            hidden,
            lengths,
            input_lengths_cpu=lengths_cpu,
        )
        output_lengths = decoder.output_lengths(lengths_cpu)
        return [
            audio[index, 0, :length].detach().to(device="cpu", dtype=torch.float32)
            for index, length in enumerate(output_lengths)
        ]


def load_moss_audio_vocoder(
    model_path: str,
    *,
    device: str | torch.device,
    decoder_dtype: torch.dtype = torch.bfloat16,
    compute_dtype: torch.dtype | None = None,
    attention_backend: str = _AUTO_ATTENTION_BACKEND,
) -> MossAudioVocoder:
    """Load only quantizer/decoder weights without executing checkpoint code."""

    resolved_path = resolve_model_path(str(model_path))
    config_path = Path(resolved_path) / "config.json"
    with config_path.open(encoding="utf-8") as config_file:
        config = json.load(config_file)
    if config.get("model_type") != "moss-audio-tokenizer":
        raise ValueError(
            f"expected model_type='moss-audio-tokenizer' in {config_path}, "
            f"got {config.get('model_type')!r}"
        )

    model = MossAudioTokenizerVocoder(
        config,
        parameter_device="meta",
        decoder_dtype=decoder_dtype,
        compute_dtype=compute_dtype,
        attention_backend=attention_backend,
    )
    target_device = torch.device(device)
    backend_resolution = model.resolve_attention_backend(target_device)
    _log_attention_backend_resolution(
        "vocoder",
        requested_backend=model.attention_backend,
        resolution=backend_resolution,
        device=target_device,
        dtype=model.decoder_dtype,
    )
    model.quantizer = load_module(
        model.quantizer,
        str(resolved_path),
        prefix="quantizer.",
        dtype=torch.float32,
        device=device,
        strict=True,
    )
    model.quantizer.build_decode_cache()
    if model._uses_moss_audio_tokenizer_v1_weights:
        state_dict = load_weights_by_prefix(
            str(resolved_path),
            prefix="decoder.",
        )
        state_dict = _normalize_moss_audio_tokenizer_v1_transformer_state_dict(
            state_dict
        )
        try:
            model.decoder.load_state_dict(state_dict, strict=True, assign=True)
        except TypeError:
            model.decoder.load_state_dict(state_dict, strict=True)
        model.decoder = model.decoder.to(device=device, dtype=model.decoder_dtype)
        _restore_fp32_compute_parameters(model.decoder)
        model.decoder.eval()
    else:
        model.decoder = load_module(
            model.decoder,
            str(resolved_path),
            prefix="decoder.",
            dtype=model.decoder_dtype,
            device=device,
            strict=True,
        )
        _restore_fp32_compute_parameters(model.decoder)
    model.decoder = MossAudioTokenizerVocoderDecoder(
        model.decoder,
        attention_backend=model.attention_backend,
    )
    model.eval()
    logger.info(
        "Loaded repository-local MOSS audio vocoder from %s on %s "
        "(decoder_dtype=%s, compute_dtype=%s)",
        resolved_path,
        device,
        model.decoder_dtype,
        model.compute_dtype,
    )
    return MossAudioVocoder(model, device=str(device))
