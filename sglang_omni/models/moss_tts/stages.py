# SPDX-License-Identifier: Apache-2.0
"""Stage factories for the MOSS-TTS Delay pipeline."""

from __future__ import annotations

import concurrent.futures
import logging
import os
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, TypeAlias
from urllib.parse import unquote, urlparse

import torch
from transformers import AutoConfig, AutoTokenizer

from sglang_omni.models.moss_tts.audio_tokenizer import (
    DEFAULT_MOSS_TTS_AUDIO_TOKENIZER,
    MossTTSAudioTokenizer,
    load_moss_tts_audio_tokenizer,
)
from sglang_omni.models.moss_tts.delay_pattern import split_moss_audio_segments
from sglang_omni.models.moss_tts.engine_builder import MossTtsEngineBuilder
from sglang_omni.models.moss_tts.hf_loading import (
    load_moss_processor_class,
    moss_transformers_processor_compat,
)
from sglang_omni.models.moss_tts.payload_types import (
    MossTTSState,
    moss_tts_special_token_defaults,
    resolve_moss_audio_pad_code,
)
from sglang_omni.models.moss_tts.request_builders import (
    cleanup_prepared_moss_tts_request,
    preprocess_moss_tts_payload,
    set_moss_tts_preprocessing_context,
)
from sglang_omni.models.moss_tts.streaming_vocoder import MossStreamingVocoderScheduler
from sglang_omni.preprocessing.cache_key import hash_bytes, reference_path_cache_key
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.pipeline_state import build_usage
from sglang_omni.scheduling.pipeline_state import load_state as _load_pipeline_state
from sglang_omni.scheduling.pipeline_state import store_state as _store_pipeline_state
from sglang_omni.scheduling.reference_encoder import (
    ReferenceEncodeKey,
    ReferenceEncodeService,
    TensorReferenceEncodeHook,
)
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.scheduling.vocoder_base import BatchVocoderBase
from sglang_omni.utils.audio import decode_audio_data_uri, load_audio
from sglang_omni.utils.audio_payload import audio_waveform_payload

logger = logging.getLogger(__name__)

_MOSS_TTS_INSTALL_HINT = (
    "MOSS-TTS support requires the upstream custom Transformers code. "
    "Launch with trust_remote_code=True and make sure the checkpoint can load "
    "OpenMOSS-Team/MOSS-Audio-Tokenizer."
)
_MAX_REFERENCE_SECONDS = 100.0


@dataclass(frozen=True)
class _MossTTSReferenceInput:
    source_kind: str
    source: str
    raw: bytes | None = None


@dataclass(frozen=True, eq=False)
class _PreparedReferenceEncodeJob:
    item: _MossTTSReferenceInput
    waveform: torch.Tensor
    sample_rate: int


_ReferenceEncodeJob: TypeAlias = _PreparedReferenceEncodeJob


def load_state(payload: StagePayload) -> MossTTSState:
    return _load_pipeline_state(payload, MossTTSState)


def store_state(payload: StagePayload, state: MossTTSState) -> StagePayload:
    return _store_pipeline_state(payload, state)


def _normalize_moss_processor_config(processor: Any) -> None:
    model_config = getattr(processor, "model_config", None)
    if model_config is None:
        return
    audio_vocab_size = int(getattr(model_config, "audio_vocab_size", 1024) or 1024)
    for attr, default in moss_tts_special_token_defaults(audio_vocab_size):
        if getattr(model_config, attr, None) is None:
            setattr(model_config, attr, default)


def _audio_tokenizer_model_path_from_processor_dict(
    processor_dict: dict[str, Any],
) -> str | None:
    model_path = processor_dict.get("audio_tokenizer_name_or_path")
    audio_tokenizer_dict = processor_dict.get("audio_tokenizer")
    if isinstance(audio_tokenizer_dict, dict):
        model_path = (
            audio_tokenizer_dict.get("audio_tokenizer_name_or_path") or model_path
        )
    return str(model_path) if model_path else None


def _load_moss_processor(
    model_path: str,
) -> Any:
    logger.info(f"Loading MOSS-TTS processor from {model_path} without codec")
    try:
        with moss_transformers_processor_compat():
            processor_cls = load_moss_processor_class(model_path)
            processor_dict, _ = processor_cls.get_processor_dict(model_path)
            audio_tokenizer_model_path = (
                _audio_tokenizer_model_path_from_processor_dict(processor_dict)
            )
            model_config = AutoConfig.from_pretrained(
                model_path,
                trust_remote_code=True,
            )
            if audio_tokenizer_model_path:
                # processor_cls.from_pretrained normally resolves this metadata
                # before loading the codec. Preserve the same selection while
                # constructing the processor without a codec instance.
                model_config.audio_tokenizer_name_or_path = audio_tokenizer_model_path
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
            )
            processor = processor_cls(
                tokenizer=tokenizer,
                audio_tokenizer=None,
                model_config=model_config,
            )
    except Exception as exc:
        raise RuntimeError(_MOSS_TTS_INSTALL_HINT) from exc

    _normalize_moss_processor_config(processor)
    return processor


def _resolve_audio_tokenizer_model_path(
    processor: Any,
    codec_model_path: str | None,
) -> str:
    return str(
        codec_model_path
        or getattr(processor.model_config, "audio_tokenizer_name_or_path", None)
        or DEFAULT_MOSS_TTS_AUDIO_TOKENIZER
    )


def _resolve_codec_device(device: str | None, gpu_id: int | None) -> str:
    if device:
        return device
    if gpu_id is not None:
        return f"cuda:{int(gpu_id)}"
    return "cuda:0"


class _BatchedReferenceEncoder:
    """Coalesce concurrent Delay reference encodes into codec batches."""

    MAX_REFERENCE_SECONDS = _MAX_REFERENCE_SECONDS
    ENCODE_TIMEOUT_S = 120.0

    def __init__(
        self,
        audio_tokenizer: MossTTSAudioTokenizer,
        *,
        n_vq: int,
        max_batch_size: int = 8,
        max_batch_wait_ms: int = 4,
    ) -> None:
        self._audio_tokenizer = audio_tokenizer
        self._n_vq = int(n_vq)
        self._max_batch_size = max(int(max_batch_size), 1)
        self._max_wait_s = max(float(max_batch_wait_ms), 0.0) / 1000.0
        self._queue: queue.Queue[
            tuple[_ReferenceEncodeJob, concurrent.futures.Future[torch.Tensor]]
        ] = queue.Queue()
        self._load_lock = threading.Lock()
        self._load_inflight: dict[
            _MossTTSReferenceInput,
            concurrent.futures.Future[tuple[torch.Tensor, int]],
        ] = {}
        self._thread = threading.Thread(
            target=self._worker,
            name="moss-tts-ref-encode",
            daemon=True,
        )
        self._thread.start()

    @staticmethod
    def normalize_source(source: str | os.PathLike[str]) -> _MossTTSReferenceInput:
        source = os.fsdecode(source)
        if source.startswith("data:"):
            raw = decode_audio_data_uri(source)
            if raw is None:
                raise ValueError("MOSS-TTS reference is not a valid audio data URI")
            return _MossTTSReferenceInput("data_uri", source, raw)
        if source.startswith(("http://", "https://")):
            return _MossTTSReferenceInput("url", source)
        if source.startswith("file://"):
            return _MossTTSReferenceInput(
                "path",
                unquote(urlparse(source).path),
            )
        return _MossTTSReferenceInput("path", source)

    def encode(self, source: str | os.PathLike[str]) -> torch.Tensor:
        return self.encode_input(self.normalize_source(source))

    def encode_input(self, item: _MossTTSReferenceInput) -> torch.Tensor:
        waveform, sample_rate = self._load_input(item)
        future: concurrent.futures.Future[torch.Tensor] = concurrent.futures.Future()
        self._queue.put(
            (
                _PreparedReferenceEncodeJob(item, waveform, sample_rate),
                future,
            )
        )
        return future.result(timeout=self.ENCODE_TIMEOUT_S)

    def _load_input(self, item: _MossTTSReferenceInput) -> tuple[torch.Tensor, int]:
        # URLs are intentionally uncacheable because their content may change.
        # Load them independently in the caller's preprocessing worker.
        if item.source_kind == "url":
            return _load_reference_waveform(self._audio_tokenizer, item)

        leader: concurrent.futures.Future[tuple[torch.Tensor, int]] | None = None
        with self._load_lock:
            follower = self._load_inflight.get(item)
            if follower is None:
                leader = concurrent.futures.Future()
                self._load_inflight[item] = leader

        if follower is not None:
            return follower.result(timeout=self.ENCODE_TIMEOUT_S)

        assert leader is not None
        try:
            loaded = _load_reference_waveform(self._audio_tokenizer, item)
        except BaseException as exc:
            leader.set_exception(exc)
            with self._load_lock:
                self._load_inflight.pop(item, None)
            raise
        leader.set_result(loaded)
        return loaded

    def _release_loaded_inputs(
        self,
        batch: list[
            tuple[_ReferenceEncodeJob, concurrent.futures.Future[torch.Tensor]]
        ],
    ) -> None:
        with self._load_lock:
            for job, _ in batch:
                if job.item.source_kind != "url":
                    self._load_inflight.pop(job.item, None)

    def _drain_batch(
        self,
    ) -> list[tuple[_ReferenceEncodeJob, concurrent.futures.Future[torch.Tensor]]]:
        batch = [self._queue.get()]
        deadline = time.monotonic() + self._max_wait_s
        while len(batch) < self._max_batch_size:
            try:
                if self._max_wait_s > 0:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    batch.append(self._queue.get(timeout=remaining))
                else:
                    batch.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return batch

    def _worker(self) -> None:
        while True:
            batch = self._drain_batch()
            try:
                results = self._encode_batch(batch)
            except BaseException as exc:
                logger.exception("MOSS-TTS reference encode worker failed")
                results = {index: exc for index in range(len(batch))}
            for index, (_, future) in enumerate(batch):
                outcome = results.get(index)
                if isinstance(outcome, BaseException):
                    future.set_exception(
                        RuntimeError(f"reference encode failed: {outcome}")
                    )
                elif outcome is None:
                    future.set_exception(
                        RuntimeError("reference encode produced no codes")
                    )
                else:
                    future.set_result(outcome)
            self._release_loaded_inputs(batch)

    def _encode_batch(
        self,
        batch: list[
            tuple[_ReferenceEncodeJob, concurrent.futures.Future[torch.Tensor]]
        ],
    ) -> dict[int, torch.Tensor | BaseException]:
        results: dict[int, torch.Tensor | BaseException] = {}
        group_indices: list[list[int]] = []
        waveforms: list[tuple[torch.Tensor, int]] = []
        item_to_group: dict[_MossTTSReferenceInput, int] = {}
        for index, (job, _) in enumerate(batch):
            group = (
                item_to_group.get(job.item) if job.item.source_kind != "url" else None
            )
            if group is None:
                group = len(waveforms)
                waveforms.append((job.waveform, job.sample_rate))
                group_indices.append([])
                if job.item.source_kind != "url":
                    item_to_group[job.item] = group
            group_indices[group].append(index)

        try:
            encoded = self._audio_tokenizer.encode_waveforms(
                waveforms,
                num_quantizers=self._n_vq,
            )
            if len(encoded) != len(waveforms):
                raise RuntimeError(
                    "MOSS-TTS audio tokenizer returned an unexpected batch size: "
                    f"{len(encoded)} != {len(waveforms)}"
                )
            for indices, codes in zip(group_indices, encoded):
                for index in indices:
                    results[index] = codes
            return results
        except Exception:
            logger.exception(
                "MOSS-TTS batched reference encode failed; retrying per item"
            )

        for indices, waveform in zip(group_indices, waveforms):
            try:
                codes = self._audio_tokenizer.encode_waveforms(
                    [waveform],
                    num_quantizers=self._n_vq,
                )[0]
            except Exception as exc:
                codes = exc
            for index in indices:
                results[index] = codes
        return results


class _DirectReferenceEncoder:
    """Keep CPU codec calls parallel while sharing normalization and caching."""

    ENCODE_TIMEOUT_S = _BatchedReferenceEncoder.ENCODE_TIMEOUT_S

    def __init__(
        self,
        audio_tokenizer: MossTTSAudioTokenizer,
        *,
        n_vq: int,
    ) -> None:
        self._audio_tokenizer = audio_tokenizer
        self._n_vq = int(n_vq)

    @staticmethod
    def normalize_source(source: str | os.PathLike[str]) -> _MossTTSReferenceInput:
        return _BatchedReferenceEncoder.normalize_source(source)

    def encode(self, source: str | os.PathLike[str]) -> torch.Tensor:
        return self.encode_input(self.normalize_source(source))

    def encode_input(self, item: _MossTTSReferenceInput) -> torch.Tensor:
        waveform = _load_reference_waveform(self._audio_tokenizer, item)
        return self._audio_tokenizer.encode_waveforms(
            [waveform],
            num_quantizers=self._n_vq,
        )[0]


def _load_reference_waveform(
    audio_tokenizer: MossTTSAudioTokenizer,
    item: _MossTTSReferenceInput,
) -> tuple[torch.Tensor, int]:
    source: str | bytes = item.raw if item.raw is not None else item.source
    waveform = load_audio(
        source,
        source_name="MOSS-TTS reference",
        target_sample_rate=audio_tokenizer.sample_rate,
        mono=True,
    )
    duration = len(waveform) / max(audio_tokenizer.sample_rate, 1)
    if duration > _MAX_REFERENCE_SECONDS:
        raise ValueError(
            f"reference audio is {duration:.1f}s long, limit is "
            f"{_MAX_REFERENCE_SECONDS:.0f}s"
        )
    return torch.from_numpy(waveform), audio_tokenizer.sample_rate


class _MossTTSReferenceEncodeHook(TensorReferenceEncodeHook[_MossTTSReferenceInput]):
    model_id = "moss_tts_delay"
    encoder_id = "moss_audio_tokenizer"
    artifact_kind = "moss_tts_reference_codes"
    storage_dtype = torch.int32
    output_dtype = torch.long

    def __init__(
        self,
        encoder: _BatchedReferenceEncoder | _DirectReferenceEncoder,
        *,
        codec_model_path: str,
        n_vq: int,
    ) -> None:
        self._encoder = encoder
        self.model_revision = str(codec_model_path)
        model = getattr(encoder._audio_tokenizer, "model", None)
        try:
            parameter_dtype = str(next(model.parameters()).dtype)
        except (AttributeError, StopIteration):
            parameter_dtype = "unknown"
        config = (
            f"n_vq:{int(n_vq)}|sample_rate:"
            f"{getattr(encoder._audio_tokenizer, 'sample_rate', 'unknown')}|device:"
            f"{getattr(encoder._audio_tokenizer, 'device', 'unknown')}|dtype:"
            f"{parameter_dtype}"
        )
        self.encoder_config_hash = hash_bytes(config.encode("utf-8"))

    def normalize_input(self, raw_input: Any) -> _MossTTSReferenceInput:
        if isinstance(raw_input, _MossTTSReferenceInput):
            return raw_input
        return self._encoder.normalize_source(raw_input)

    def input_key(self, item: _MossTTSReferenceInput) -> str | None:
        if item.source_kind == "path":
            return reference_path_cache_key(item.source)
        if item.source_kind == "data_uri":
            if item.raw is None:
                raise ValueError("MOSS-TTS data URI reference has no decoded bytes")
            return f"bytes:{hash_bytes(item.raw)}"
        return None

    def encode_one(self, item: _MossTTSReferenceInput) -> torch.Tensor:
        return self._encoder.encode_input(item)

    def revalidate(
        self,
        item: _MossTTSReferenceInput,
        key: ReferenceEncodeKey,
    ) -> bool:
        return (
            item.source_kind != "path"
            or reference_path_cache_key(item.source) == key.input_key
        )


class _MossTTSReferenceEncoder:
    """Cache and in-flight merge boundary around the batched codec encoder."""

    def __init__(
        self,
        encoder: _BatchedReferenceEncoder | _DirectReferenceEncoder,
        *,
        codec_model_path: str,
        n_vq: int,
        max_items: int | None = 8192,
        max_bytes: int | None = 64 * 1024 * 1024,
    ) -> None:
        self._service = ReferenceEncodeService(
            _MossTTSReferenceEncodeHook(
                encoder,
                codec_model_path=codec_model_path,
                n_vq=n_vq,
            ),
            max_items=max_items,
            max_bytes=max_bytes,
            timeout_s=_BatchedReferenceEncoder.ENCODE_TIMEOUT_S + 10,
            log_prefix="MOSS-TTS ref cache",
        )

    def encode(self, source: str | os.PathLike[str]) -> torch.Tensor:
        normalized = _BatchedReferenceEncoder.normalize_source(source)
        return self._service.get_or_encode(
            normalized,
            desc=(
                "data-URI"
                if normalized.source_kind == "data_uri"
                else repr(normalized.source)
            ),
        )

    def stats(self) -> dict[str, int]:
        return self._service.stats()


def create_preprocessing_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    dtype: str = "float32",
    codec_model_path: str | None = None,
    max_concurrency: int = 16,
    encode_batch_size: int = 8,
    encode_batch_wait_ms: int = 4,
    ref_audio_cache: bool = True,
    ref_audio_cache_max_items: int = 8192,
    ref_audio_cache_max_bytes: int = 64 * 1024 * 1024,
) -> SimpleScheduler:
    env_toggle = os.environ.get("MOSS_REF_AUDIO_CACHE")
    if env_toggle is not None:
        ref_audio_cache = env_toggle.strip().lower() not in (
            "0",
            "false",
            "no",
            "off",
            "",
        )
    device = _resolve_codec_device(device, gpu_id)
    processor = _load_moss_processor(model_path)
    resolved_codec_model_path = _resolve_audio_tokenizer_model_path(
        processor,
        codec_model_path,
    )
    audio_tokenizer = load_moss_tts_audio_tokenizer(
        resolved_codec_model_path,
        device=device,
        dtype=dtype,
    )
    if str(device).startswith("cuda"):
        reference_encoder: Any = _BatchedReferenceEncoder(
            audio_tokenizer,
            n_vq=int(processor.model_config.n_vq),
            max_batch_size=encode_batch_size,
            max_batch_wait_ms=encode_batch_wait_ms,
        )
    else:
        reference_encoder = _DirectReferenceEncoder(
            audio_tokenizer,
            n_vq=int(processor.model_config.n_vq),
        )
    if ref_audio_cache:
        reference_encoder = _MossTTSReferenceEncoder(
            reference_encoder,
            codec_model_path=resolved_codec_model_path,
            n_vq=int(processor.model_config.n_vq),
            max_items=ref_audio_cache_max_items,
            max_bytes=ref_audio_cache_max_bytes,
        )
    set_moss_tts_preprocessing_context(
        processor=processor,
        reference_encoder=reference_encoder,
    )
    # GPU codec work is coalesced through one batch queue; explicit CPU configs
    # keep the prior parallel single-item calls, because CPU batching serializes
    # the workers and regresses real SeedTTS throughput.
    return SimpleScheduler(
        preprocess_moss_tts_payload,
        abort_callback=cleanup_prepared_moss_tts_request,
        max_concurrency=max_concurrency,
    )


def create_sglang_tts_engine_executor(
    model_path: str,
    *,
    device: str = "cuda:0",
    gpu_id: int | None = None,
    dtype: str = "bfloat16",
    server_args_overrides: dict[str, Any] | None = None,
) -> Any:
    return MossTtsEngineBuilder().build(
        model_path,
        device=device,
        gpu_id=gpu_id,
        dtype=dtype,
        server_args_overrides=server_args_overrides,
    )


create_tts_engine_executor = create_sglang_tts_engine_executor


class _MossTTSVocoder(BatchVocoderBase):
    def __init__(
        self,
        processor: Any,
        audio_tokenizer: MossTTSAudioTokenizer,
        device: str,
    ) -> None:
        self._processor = processor
        self._audio_tokenizer = audio_tokenizer
        self._device = device

    def prepare_item(self, payload: StagePayload) -> tuple[MossTTSState, torch.Tensor]:
        state = load_state(payload)
        if state.delayed_audio_codes is None:
            raise RuntimeError("MOSS-TTS vocoder requires delayed_audio_codes")
        delayed_codes = torch.as_tensor(state.delayed_audio_codes, dtype=torch.long)
        if delayed_codes.numel() == 0:
            raise RuntimeError("MOSS-TTS generated no delayed audio codes")
        return state, delayed_codes

    def _decode_audio(
        self,
        state: MossTTSState,
        delayed_codes: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        delayed_codes = delayed_codes.to(device=self._device, dtype=torch.long)
        audio_pad_code = resolve_moss_audio_pad_code(
            getattr(self._processor, "model_config", None)
        )
        segments = split_moss_audio_segments(
            delayed_codes,
            audio_pad_code=audio_pad_code,
            assistant_start_length=int(state.assistant_start_length),
        )
        decoded = []
        for segment in segments:
            decoded.extend(self._audio_tokenizer.decode_codes([segment]))
        if not decoded:
            raise RuntimeError("MOSS-TTS vocoder decoded no audio segments")
        waveforms = [
            torch.as_tensor(wav).detach().reshape(-1).to("cpu") for wav in decoded
        ]
        waveform = torch.cat(waveforms, dim=0)
        sample_rate = int(
            getattr(self._audio_tokenizer, "sample_rate", 0)
            or getattr(
                getattr(self._processor, "model_config", None), "sampling_rate", 0
            )
            or state.sample_rate
            or 24000
        )
        return waveform, sample_rate

    async def decode_batch(
        self, items: list[tuple[MossTTSState, torch.Tensor]]
    ) -> list[tuple[torch.Tensor, int]]:
        return [self._decode_audio(state, codes) for state, codes in items]

    def store_result(
        self,
        payload: StagePayload,
        state: MossTTSState,
        wav: torch.Tensor,
        sample_rate: int,
    ) -> StagePayload:
        audio_payload = audio_waveform_payload(wav, source_hint="MOSS-TTS")
        state.delayed_audio_codes = None
        state.sample_rate = int(sample_rate)
        payload = store_state(payload, state)
        payload.data.update(audio_payload)
        payload.data["sample_rate"] = state.sample_rate
        payload.data["modality"] = "audio"
        usage = build_usage(state)
        if usage is not None:
            payload.data["usage"] = usage
        return payload


def create_vocoder_executor(
    model_path: str,
    *,
    device: str = "cuda:0",
    gpu_id: int | None = None,
    dtype: str = "float32",
    codec_model_path: str | None = None,
    max_batch_size: int = 8,
    max_batch_wait_ms: int = 2,
    stream_stride: int = 8,
    stream_followup_stride: int = 8,
    stream_overlap_tokens: int = 8,
    stream_holdback_tokens: int = 1,
    initial_chunk_frames: int = 0,
) -> MossStreamingVocoderScheduler:
    if gpu_id is not None:
        device = f"cuda:{gpu_id}"
    processor = _load_moss_processor(model_path)
    audio_tokenizer = load_moss_tts_audio_tokenizer(
        _resolve_audio_tokenizer_model_path(processor, codec_model_path),
        device=device,
        dtype=dtype,
    )

    vocoder = _MossTTSVocoder(processor, audio_tokenizer, device)
    return MossStreamingVocoderScheduler(
        vocoder,
        stream_stride=stream_stride,
        stream_followup_stride=stream_followup_stride,
        stream_overlap_tokens=stream_overlap_tokens,
        stream_holdback_tokens=stream_holdback_tokens,
        initial_chunk_frames=initial_chunk_frames,
        max_batch_size=max_batch_size,
        max_batch_wait_ms=max_batch_wait_ms,
    )
