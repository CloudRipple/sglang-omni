# SPDX-License-Identifier: Apache-2.0
"""Standalone MOSS-Audio-Tokenizer loader for MOSS-TTS Delay."""

from __future__ import annotations

import logging
from typing import Any, Literal

import torch
import torch.nn as nn
import torchaudio
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModel

from sglang_omni.models.moss_tts.hf_loading import moss_transformers_processor_compat
from sglang_omni.models.weight_loader import load_module

logger = logging.getLogger(__name__)

DEFAULT_MOSS_TTS_AUDIO_TOKENIZER = "OpenMOSS-Team/MOSS-Audio-Tokenizer"
_LOUDNESS_TARGET_DBFS = -20.0
_LOUDNESS_GAIN_MIN_DB = -3.0
_LOUDNESS_GAIN_MAX_DB = 3.0
_CODEC_COMPONENTS = ("encoder", "decoder")

CodecComponent = Literal["encoder", "decoder"]


def _torch_dtype(dtype: str | torch.dtype) -> torch.dtype:
    return getattr(torch, dtype) if isinstance(dtype, str) else dtype


def _validate_component_model(model: nn.Module) -> None:
    for name in (*_CODEC_COMPONENTS, "quantizer"):
        if not isinstance(getattr(model, name, None), nn.Module):
            raise RuntimeError(
                f"MOSS-TTS audio tokenizer does not expose a {name!r} module"
            )


def _raise_for_meta_state(model: nn.Module) -> None:
    meta_state = [
        name
        for name, tensor in (
            *model.named_parameters(),
            *model.named_buffers(),
        )
        if tensor.is_meta
    ]
    if meta_state:
        preview = ", ".join(meta_state[:8])
        if len(meta_state) > 8:
            preview += f", ... ({len(meta_state)} total)"
        raise RuntimeError(
            "MOSS-TTS component codec has unmaterialized state outside the "
            f"supported component prefixes: {preview}"
        )


def _state_nbytes(model: nn.Module) -> int:
    return sum(
        tensor.numel() * tensor.element_size()
        for tensor in (*model.parameters(), *model.buffers())
    )


def _load_component_model(
    model_path: str,
    *,
    component: CodecComponent,
    device: str,
    dtype: str | torch.dtype,
) -> nn.Module:
    config = AutoConfig.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    with init_empty_weights(include_buffers=True):
        model = AutoModel.from_config(
            config,
            trust_remote_code=True,
        )
    _validate_component_model(model)

    unused_component = "decoder" if component == "encoder" else "encoder"
    setattr(model, unused_component, nn.ModuleList())

    load_dtype = None
    if torch.device(device).type != "cpu":
        load_dtype = _torch_dtype(dtype)
    setattr(
        model,
        component,
        load_module(
            getattr(model, component),
            model_path,
            prefix=f"{component}.",
            dtype=load_dtype,
            device=device,
            strict=True,
        ),
    )
    model.quantizer = load_module(
        model.quantizer,
        model_path,
        prefix="quantizer.",
        dtype=load_dtype,
        device=device,
        strict=True,
    )
    model.eval()
    _raise_for_meta_state(model)
    logger.info(
        "Loaded MOSS-TTS audio tokenizer component=%s (%.3f GiB)",
        component,
        _state_nbytes(model) / (1024**3),
    )
    return model


class MossTTSAudioTokenizer:
    """Processor-compatible wrapper around a separately loaded codec model."""

    def __init__(self, model: Any, *, device: str) -> None:
        self.model = model
        self.device = str(device)
        self.sample_rate = int(model.config.sampling_rate)

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

        with torch.inference_mode():
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
                "MOSS-TTS audio tokenizer encode returned empty "
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

        with torch.inference_mode():
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
                "MOSS-TTS audio tokenizer decode returned empty audio/audio_lengths"
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
    component: CodecComponent | None = None,
) -> MossTTSAudioTokenizer:
    if component is not None and component not in _CODEC_COMPONENTS:
        raise ValueError(
            f"component must be one of {_CODEC_COMPONENTS!r} or None; got {component!r}"
        )
    logger.info(
        "Loading MOSS-TTS audio tokenizer from %s on %s (component=%s)",
        model_path,
        device,
        component or "full",
    )
    try:
        with moss_transformers_processor_compat():
            if component is None:
                model = AutoModel.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                )
            else:
                model = _load_component_model(
                    model_path,
                    component=component,
                    device=device,
                    dtype=dtype,
                )
    except Exception as exc:
        raise RuntimeError(
            "MOSS-TTS support requires OpenMOSS-Team/MOSS-Audio-Tokenizer; "
            f"failed to load component {component or 'full'!r}: {exc}"
        ) from exc
    if component is None:
        model.eval()
        move_kwargs: dict[str, Any] = {"device": device}
        if torch.device(device).type != "cpu":
            move_kwargs["dtype"] = _torch_dtype(dtype)
        model.to(**move_kwargs)
    return MossTTSAudioTokenizer(model, device=device)
