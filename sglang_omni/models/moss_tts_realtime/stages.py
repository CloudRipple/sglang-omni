# SPDX-License-Identifier: Apache-2.0
"""Stage factories for MOSS-TTS-Realtime."""

from __future__ import annotations

import logging
import threading
from collections.abc import Mapping
from numbers import Integral
from typing import Any

import numpy as np
import torch

from sglang_omni.models.moss_tts.hf_loading import (
    moss_transformers_processor_compat,
    resolve_moss_checkpoint,
)
from sglang_omni.models.moss_tts_realtime.config import (
    DEFAULT_MOSS_TTS_REALTIME_CODEC_MODEL,
    MossTTSRealtimeResourceLimits,
)
from sglang_omni.models.moss_tts_realtime.request_builders import (
    cleanup_prepared_moss_tts_realtime_request,
    preprocess_moss_tts_realtime_payload,
    set_moss_tts_realtime_preprocessing_context,
)
from sglang_omni.models.moss_tts_realtime.streaming_vocoder import (
    MossTTSRealtimeStreamingVocoderScheduler,
)
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.utils.audio import load_audio

logger = logging.getLogger(__name__)

_INSTALL_HINT = (
    "MOSS-TTS-Realtime requires the trusted custom code published with the "
    "model and OpenMOSS-Team/MOSS-Audio-Tokenizer."
)
_MAX_REFERENCE_SECONDS = 100.0
_DEFAULT_LIMITS = MossTTSRealtimeResourceLimits()


def _resolve_codec_device(device: str | None, gpu_id: int | None) -> str:
    if device:
        return device
    if gpu_id is not None:
        return f"cuda:{int(gpu_id)}"
    return "cuda:0"


def load_moss_tts_realtime_codec(
    model_path: str = DEFAULT_MOSS_TTS_REALTIME_CODEC_MODEL,
    *,
    device: str = "cuda:0",
) -> Any:
    checkpoint_dir = resolve_moss_checkpoint(model_path)
    logger.info(
        "Loading MOSS-TTS-Realtime codec from %s on %s",
        checkpoint_dir,
        device,
    )
    try:
        from transformers import AutoModel

        with moss_transformers_processor_compat():
            codec = AutoModel.from_pretrained(
                checkpoint_dir,
                trust_remote_code=True,
            )
    except Exception as exc:
        raise RuntimeError(_INSTALL_HINT) from exc
    codec.eval()
    codec.to(device)
    return codec


def _strict_processor_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def bind_moss_tts_realtime_processor_config(config: Any, processor: Any) -> Any:
    """Validate processor-owned metadata and attach it to the HF config."""
    expected = {
        "channels": int(config.rvq),
        "audio_channel_pad": int(config.audio_pad_token),
        "audio_pad_token_id": int(config.reference_audio_pad),
        "text_pad_token_id": int(config.text_pad),
    }
    for name, expected_value in expected.items():
        actual = _strict_processor_int(
            getattr(processor, name, None),
            f"processor.{name}",
        )
        if actual != expected_value:
            raise ValueError(
                f"processor.{name} must match model config "
                f"{expected_value}, got {actual}"
            )

    audio_bos_token = _strict_processor_int(
        getattr(processor, "audio_bos_token", None),
        "processor.audio_bos_token",
    )
    audio_eos_token = _strict_processor_int(
        getattr(processor, "audio_eos_token", None),
        "processor.audio_eos_token",
    )
    delay_tokens_len = _strict_processor_int(
        getattr(processor, "delay_tokens_len", None),
        "processor.delay_tokens_len",
        minimum=1,
    )
    special_audio_tokens = {
        int(config.audio_pad_token),
        audio_bos_token,
        audio_eos_token,
    }
    if len(special_audio_tokens) != 3:
        raise ValueError("processor audio pad, BOS, and EOS tokens must be distinct")
    if max(special_audio_tokens) >= int(config.audio_vocab_size):
        raise ValueError("processor audio special tokens exceed audio_vocab_size")

    config.audio_bos_token = audio_bos_token
    config.audio_eos_token = audio_eos_token
    config.delay_tokens_len = delay_tokens_len
    processor.model_config = config
    return config


def load_moss_tts_realtime_processor(model_path: str) -> Any:
    """Load the model config and processor through the checkpoint auto map."""

    checkpoint_dir = resolve_moss_checkpoint(model_path)
    logger.info("Loading MOSS-TTS-Realtime processor from %s", checkpoint_dir)
    try:
        from transformers import AutoConfig, AutoProcessor

        with moss_transformers_processor_compat():
            model_config = AutoConfig.from_pretrained(
                checkpoint_dir,
                trust_remote_code=True,
            )
            processor = AutoProcessor.from_pretrained(
                checkpoint_dir,
                trust_remote_code=True,
            )
            bind_moss_tts_realtime_processor_config(model_config, processor)
    except Exception as exc:
        raise RuntimeError(_INSTALL_HINT) from exc
    return processor


def _normalize_audio_source(value: Any) -> Any:
    if not isinstance(value, Mapping):
        return value
    for key in ("audio", "audio_path", "path", "bytes"):
        candidate = value.get(key)
        if candidate is not None:
            return candidate
    encoded = value.get("base64") or value.get("data")
    if encoded is not None:
        if isinstance(encoded, str) and encoded.startswith("data:"):
            return encoded
        media_type = value.get("media_type") or "audio/wav"
        return f"data:{media_type};base64,{encoded}"
    raise ValueError("audio reference mapping contains no supported audio source")


class MossTTSRealtimeAudioEncoder:
    """Serialize reference encoding through the checkpoint's legacy codec."""

    def __init__(self, codec: Any, *, device: str) -> None:
        self.codec = codec
        self.device = torch.device(device)
        self._lock = threading.Lock()
        config = getattr(codec, "config", None)
        if config is None:
            raise ValueError("MOSS-TTS-Realtime codec must expose config")
        self.sample_rate = int(
            getattr(config, "sampling_rate", 0) or getattr(config, "sample_rate", 0)
        )
        if self.sample_rate < 1:
            raise ValueError("MOSS-TTS-Realtime codec sample rate must be positive")

    def _waveform(self, value: Any) -> torch.Tensor:
        source = _normalize_audio_source(value)
        if isinstance(source, torch.Tensor):
            waveform = source.detach().to(dtype=torch.float32, device="cpu")
        elif isinstance(source, np.ndarray):
            waveform = torch.from_numpy(np.ascontiguousarray(source, dtype=np.float32))
        else:
            waveform = torch.from_numpy(
                load_audio(
                    source,
                    source_name="MOSS-TTS-Realtime reference",
                    target_sample_rate=self.sample_rate,
                    mono=True,
                )
            )
        if waveform.ndim == 2:
            waveform = (
                waveform.mean(dim=0) if waveform.shape[0] > 1 else waveform.squeeze(0)
            )
        if waveform.ndim != 1:
            raise ValueError("reference waveform must normalize to mono rank 1")
        waveform = waveform.contiguous().to(dtype=torch.float32)
        if waveform.numel() == 0:
            raise ValueError("reference waveform must not be empty")
        duration_s = waveform.numel() / self.sample_rate
        if duration_s > _MAX_REFERENCE_SECONDS:
            raise ValueError(
                f"reference audio is {duration_s:.1f}s long, limit is "
                f"{_MAX_REFERENCE_SECONDS:.0f}s"
            )
        return waveform

    def encode(self, value: Any) -> Any:
        waveform = self._waveform(value).to(self.device)
        with self._lock, torch.inference_mode():
            return self.codec.encode(
                waveform.unsqueeze(0),
                return_dict=True,
            )


def create_preprocessing_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    codec_model_path: str | None = None,
    max_concurrency: int = 8,
) -> SimpleScheduler:
    resolved_device = _resolve_codec_device(device, gpu_id)
    processor = load_moss_tts_realtime_processor(model_path)
    codec = load_moss_tts_realtime_codec(
        codec_model_path or DEFAULT_MOSS_TTS_REALTIME_CODEC_MODEL,
        device=resolved_device,
    )
    audio_encoder = MossTTSRealtimeAudioEncoder(codec, device=resolved_device)
    set_moss_tts_realtime_preprocessing_context(
        processor=processor,
        audio_encoder=audio_encoder,
    )
    return SimpleScheduler(
        preprocess_moss_tts_realtime_payload,
        abort_callback=cleanup_prepared_moss_tts_realtime_request,
        max_concurrency=max_concurrency,
    )


def create_sglang_tts_engine_executor(
    model_path: str,
    *,
    device: str = "cuda:0",
    gpu_id: int | None = None,
    dtype: str = "bfloat16",
    server_args_overrides: dict[str, Any] | None = None,
    max_seq_len: int | None = None,
    total_gpu_memory_fraction: float | None = None,
    codec_mem_reserve: float = 0.0,
    max_sessions: int = _DEFAULT_LIMITS.max_sessions,
    max_held_sessions: int = _DEFAULT_LIMITS.max_held_sessions,
    max_active_turns: int = _DEFAULT_LIMITS.max_active_turns,
    max_pending_text_tokens: int = _DEFAULT_LIMITS.max_pending_text_tokens,
    max_pending_text_bytes: int = _DEFAULT_LIMITS.max_pending_text_bytes,
    max_input_updates: int = _DEFAULT_LIMITS.max_input_updates,
    max_turn_frames: int = _DEFAULT_LIMITS.max_turn_frames,
    terminal_tombstone_limit: int = _DEFAULT_LIMITS.terminal_tombstone_limit,
    input_idle_timeout_s: float = _DEFAULT_LIMITS.input_idle_timeout_s,
    turn_timeout_s: float = _DEFAULT_LIMITS.turn_timeout_s,
    session_idle_ttl_s: float = _DEFAULT_LIMITS.session_idle_ttl_s,
) -> Any:
    from sglang_omni.models.moss_tts_realtime.engine_builder import (
        MossTTSRealtimeEngineBuilder,
    )

    return MossTTSRealtimeEngineBuilder(
        max_seq_len=max_seq_len,
        total_gpu_memory_fraction=total_gpu_memory_fraction,
        codec_mem_reserve=codec_mem_reserve,
        max_sessions=max_sessions,
        max_held_sessions=max_held_sessions,
        max_active_turns=max_active_turns,
        max_pending_text_tokens=max_pending_text_tokens,
        max_pending_text_bytes=max_pending_text_bytes,
        max_input_updates=max_input_updates,
        max_turn_frames=max_turn_frames,
        terminal_tombstone_limit=terminal_tombstone_limit,
        input_idle_timeout_s=input_idle_timeout_s,
        turn_timeout_s=turn_timeout_s,
        session_idle_ttl_s=session_idle_ttl_s,
    ).build(
        model_path,
        device=device,
        gpu_id=gpu_id,
        dtype=dtype,
        server_args_overrides=server_args_overrides,
    )


create_tts_engine_executor = create_sglang_tts_engine_executor


def create_vocoder_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    codec_model_path: str | None = None,
    stream_slots: int = 16,
    max_batch_size: int = 8,
    max_batch_wait_ms: int = 2,
) -> MossTTSRealtimeStreamingVocoderScheduler:
    resolved_device = _resolve_codec_device(device, gpu_id)
    processor = load_moss_tts_realtime_processor(model_path)
    codec = load_moss_tts_realtime_codec(
        codec_model_path or DEFAULT_MOSS_TTS_REALTIME_CODEC_MODEL,
        device=resolved_device,
    )
    return MossTTSRealtimeStreamingVocoderScheduler(
        codec,
        n_vq=int(processor.model_config.rvq),
        stream_slots=stream_slots,
        max_batch_size=max_batch_size,
        max_batch_wait_ms=max_batch_wait_ms,
    )
