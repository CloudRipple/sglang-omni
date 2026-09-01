# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for MOSS-TTS Local (v1.5)."""

from __future__ import annotations

import logging
import math
from typing import Any, ClassVar

from pydantic import Field, field_validator

from sglang_omni.config import (
    EngineArgs,
    EngineStageConfig,
    FactoryArgs,
    PipelineConfig,
    StageConfig,
)
from sglang_omni.utils.cpu import bounded_intraop_threads

_PKG = "sglang_omni.models.moss_tts_local"
# Keep reference encoding with AR so process-scoped SGLang accounting includes
# its codec allocation. The vocoder is isolated: its Python-heavy packed decode
# otherwise stalls the AR scheduler thread under ordinary serving concurrency.
_COLOCATED_PREPROCESSING_GPU_MEMORY_FRACTION = 0.15
_COLOCATED_AR_GPU_MEMORY_FRACTION = 0.67
_COLOCATED_VOCODER_GPU_MEMORY_FRACTION = 0.18
_AR_MEM_FRACTION_STATIC = 0.85
_REF_AUDIO_CACHE_MAX_ITEMS = 8192
_REF_AUDIO_CACHE_MAX_BYTES = 64 * 1024 * 1024
_PREPROCESSING_MAX_CONCURRENCY = 16
_MAX_PIPELINE_INTRAOP_THREADS = 8
logger = logging.getLogger(__name__)


def _validate_prefill_coalesce_requests(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"prefill_coalesce_requests must be a number, got {value!r}")
    if isinstance(value, float) and not (math.isfinite(value) and value.is_integer()):
        raise ValueError(
            "prefill_coalesce_requests must be a finite integer, "
            f"got {value!r}"
        )
    requests = int(value)
    if requests < 0:
        raise ValueError("prefill_coalesce_requests must be >= 0")
    if requests == 1:
        logger.warning(
            "prefill_coalesce_requests=1 disables coalescing: the admission "
            "gate only engages at >= 2"
        )
    return requests


def _validate_prefill_coalesce_wait_ms(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"prefill_coalesce_wait_ms must be a number, got {value!r}")
    try:
        wait_ms = float(value)
    except (TypeError, OverflowError) as exc:
        raise ValueError("prefill_coalesce_wait_ms must be a finite value > 0") from exc
    if not (math.isfinite(wait_ms) and wait_ms > 0):
        raise ValueError("prefill_coalesce_wait_ms must be a finite value > 0")
    return wait_ms


def _stages(
    *,
    codec_device: str,
    colocated: bool,
    vocoder_gpu: int = 0,
) -> list[StageConfig]:
    """Build the local pipeline stages.

    The Python-heavy vocoder always gets its own OS process. ``colocated``
    controls only GPU memory accounting: the default puts all three stages on
    one GPU, while the split variant puts codec work and the vocoder on GPU 1
    and leaves the AR engine on GPU 0.
    """
    return [
        StageConfig(
            name="preprocessing",
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_preprocessing_executor",
            factory=FactoryArgs(
                device=codec_device,
                dtype="bfloat16",
                compute_dtype="bfloat16",
                attention_backend="auto",
                max_concurrency=_PREPROCESSING_MAX_CONCURRENCY,
            ),
            gpu_memory_fraction=(
                _COLOCATED_PREPROCESSING_GPU_MEMORY_FRACTION if colocated else None
            ),
            gpu=0,
            next="tts_engine",
        ),
        EngineStageConfig(
            name="tts_engine",
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_sglang_tts_engine_executor",
            factory=FactoryArgs(dtype="bfloat16"),
            engine=EngineArgs(
                mem_fraction_static=None if colocated else _AR_MEM_FRACTION_STATIC
            ),
            gpu_memory_fraction=(
                _COLOCATED_AR_GPU_MEMORY_FRACTION if colocated else None
            ),
            gpu=0,
            next="vocoder",
            stream_to=["vocoder"],
        ),
        StageConfig(
            name="vocoder",
            process="vocoder",
            factory_path=f"{_PKG}.stages.create_vocoder_executor",
            # The repository-owned vocoder loader fixes the decoder and compute
            # dtypes; only placement/device is a factory-level input.
            factory=FactoryArgs(device=codec_device),
            gpu_memory_fraction=(
                _COLOCATED_VOCODER_GPU_MEMORY_FRACTION if colocated else None
            ),
            gpu=vocoder_gpu,
            terminal=True,
            can_accept_stream_before_payload=True,
        ),
    ]


class MossTTSLocalPipelineConfig(PipelineConfig):
    """Single-GPU MOSS-TTS Local pipeline."""

    architecture: ClassVar[str] = "MossTTSLocalModel"
    requires_model_capabilities: ClassVar[bool] = True
    architecture_aliases: ClassVar[tuple[str, ...]] = (
        "MossTTSLocal",
        "MossTTSLocalForConditionalGeneration",
    )
    additional_speech_languages: ClassVar[frozenset[str]] = frozenset(
        {
            "Cantonese",
            "Arabic",
            "Czech",
            "Danish",
            "Dutch",
            "Finnish",
            "Greek",
            "Hebrew",
            "Hindi",
            "Hungarian",
            "Macedonian",
            "Malay",
            "Persian (Farsi)",
            "Polish",
            "Romanian",
            "Swahili",
            "Swedish",
            "Tagalog",
            "Thai",
            "Turkish",
            "Vietnamese",
        }
    )

    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        "tts_engine": EngineStageConfig,
    }

    @classmethod
    def process_local_edges(cls) -> frozenset[tuple[str, str]]:
        # Prepared requests are held in a process-local queue between these stages.
        return frozenset({("preprocessing", "tts_engine")})

    stages: list[StageConfig] = Field(
        default_factory=lambda: _stages(codec_device="cuda:0", colocated=True)
    )

    # Streaming-vocoder knobs. They are injected into the canonical stage
    # factory group by ``stage_factory_kwargs`` below.
    cuda_graph: bool = True
    cuda_graph_frames: list[int] | None = None
    cuda_graph_min_free_gb: float = 3.0
    compact_streaming: bool = False
    compile_streaming_decode: bool = False
    compile_streaming_decode_shapes: list[tuple[int, int]] | None = None
    stream_output_overlap: bool = False
    stream_transport_batch_frames: int | None = None
    compile_ar_backbone: bool = False
    compact_topk_sampling: bool = False
    compile_local_frame: bool = False
    compile_local_frame_batch_size: int = 16
    compile_local_frame_batch_sizes: list[int] | None = None
    prefill_coalesce_requests: int = 2
    prefill_coalesce_wait_ms: float = 6.0
    ref_audio_cache: bool = True
    ref_audio_cache_max_items: int = _REF_AUDIO_CACHE_MAX_ITEMS
    ref_audio_cache_max_bytes: int = _REF_AUDIO_CACHE_MAX_BYTES

    @field_validator("prefill_coalesce_requests", mode="before")
    @classmethod
    def _validate_prefill_requests_field(cls, value: object) -> object:
        return _validate_prefill_coalesce_requests(value)

    @field_validator("prefill_coalesce_wait_ms", mode="before")
    @classmethod
    def _validate_prefill_wait_field(cls, value: object) -> object:
        return _validate_prefill_coalesce_wait_ms(value)

    def stage_factory_kwargs(self, stage_name: str) -> dict[str, Any]:
        """Pass Local streaming knobs to the canonical stage API."""
        if stage_name == "preprocessing":
            return {
                "ref_audio_cache": self.ref_audio_cache,
                "ref_audio_cache_max_items": self.ref_audio_cache_max_items,
                "ref_audio_cache_max_bytes": self.ref_audio_cache_max_bytes,
            }
        if stage_name == "tts_engine":
            kwargs: dict[str, Any] = {
                "compile_ar_backbone": self.compile_ar_backbone,
                "compile_local_frame": self.compile_local_frame,
                "compile_local_frame_batch_size": self.compile_local_frame_batch_size,
                "compile_local_frame_batch_sizes": self.compile_local_frame_batch_sizes,
                "prefill_coalesce_requests": self.prefill_coalesce_requests,
                "prefill_coalesce_wait_ms": self.prefill_coalesce_wait_ms,
            }
            if self.stream_transport_batch_frames is not None:
                kwargs["stream_transport_batch_frames"] = (
                    self.stream_transport_batch_frames
                )
            engine_stage = self.stage_named("tts_engine")
            if engine_stage.gpu_memory_fraction is not None:
                # Colocated layouts budget codec memory with stage fractions.
                kwargs["codec_mem_reserve"] = 0.0
            return kwargs
        if stage_name == "vocoder":
            return {
                "cuda_graph": self.cuda_graph,
                "cuda_graph_frames": self.cuda_graph_frames,
                "cuda_graph_min_free_gb": self.cuda_graph_min_free_gb,
                "compact_streaming": self.compact_streaming,
                "compile_streaming_decode": self.compile_streaming_decode,
                "compile_streaming_decode_shapes": self.compile_streaming_decode_shapes,
                "stream_output_overlap": self.stream_output_overlap,
            }
        return {}

    def resolved_env_defaults(self) -> dict[str, str]:
        preprocessing = next(
            (stage for stage in self.stages if stage.name == "preprocessing"),
            None,
        )
        if preprocessing is None:
            return dict(self.env_defaults)
        configured_workers = (
            preprocessing.factory.max_concurrency
        )
        preprocessing_workers = max(
            int(
                configured_workers
                if configured_workers is not None
                else _PREPROCESSING_MAX_CONCURRENCY
            ),
            1,
        )
        derived = {
            "OMP_NUM_THREADS": str(
                bounded_intraop_threads(
                    worker_count=preprocessing_workers,
                    max_threads=_MAX_PIPELINE_INTRAOP_THREADS,
                )
            )
        }
        if self.compact_topk_sampling:
            derived["MOSS_TTS_LOCAL_COMPACT_TOPK"] = "1"
        return {**derived, **self.env_defaults}

    def model_post_init(self, __context: Any = None) -> None:
        super().model_post_init(__context)
        stage_by_name = {stage.name: stage for stage in self.stages}
        vocoder = stage_by_name.get("vocoder")
        tts_engine = stage_by_name.get("tts_engine")
        if (
            vocoder is not None
            and tts_engine is not None
            and vocoder.process == tts_engine.process
        ):
            raise ValueError(
                "MOSS-TTS Local vocoder must run in an independent OS process; "
                "placing it with tts_engine is unsupported"
            )
        if self.ref_audio_cache_max_items < 1:
            raise ValueError(
                "ref_audio_cache_max_items must be >= 1; got "
                f"{self.ref_audio_cache_max_items}"
            )
        if self.ref_audio_cache_max_bytes < 1:
            raise ValueError(
                "ref_audio_cache_max_bytes must be >= 1; got "
                f"{self.ref_audio_cache_max_bytes}"
            )
        if self.cuda_graph_min_free_gb < 0:
            raise ValueError(
                "cuda_graph_min_free_gb must be >= 0 (0 disables the VRAM headroom "
                f"guard); got {self.cuda_graph_min_free_gb}"
            )
        if self.cuda_graph_frames is not None:
            if not self.cuda_graph_frames:
                raise ValueError(
                    "cuda_graph_frames must be non-empty; set `cuda_graph: false` to "
                    "disable graphs, or leave it null to use the default capture set"
                )
            invalid = [t for t in self.cuda_graph_frames if t < 1]
            if invalid:
                raise ValueError(
                    "cuda_graph_frames entries must be positive ints (>= 1); "
                    f"got {invalid}"
                )
        if self.compile_streaming_decode_shapes is not None:
            normalized_compile_shapes = sorted(
                {
                    (int(batch_size), int(frame_size))
                    for batch_size, frame_size in self.compile_streaming_decode_shapes
                }
            )
            if (
                not normalized_compile_shapes
                or min(min(shape) for shape in normalized_compile_shapes) < 1
            ):
                raise ValueError(
                    "compile_streaming_decode_shapes must contain positive "
                    f"(B,T) pairs; got {self.compile_streaming_decode_shapes}"
                )
            self.compile_streaming_decode_shapes = normalized_compile_shapes
        if self.stream_transport_batch_frames is not None:
            self.stream_transport_batch_frames = int(self.stream_transport_batch_frames)
            if self.stream_transport_batch_frames < 1:
                raise ValueError(
                    "stream_transport_batch_frames must be >= 1; got "
                    f"{self.stream_transport_batch_frames}"
                )
        if self.compile_local_frame_batch_size < 1:
            raise ValueError(
                "compile_local_frame_batch_size must be >= 1; got "
                f"{self.compile_local_frame_batch_size}"
            )
        if self.compile_local_frame_batch_sizes is not None:
            normalized_compile_sizes = sorted(
                {int(size) for size in self.compile_local_frame_batch_sizes}
            )
            if not normalized_compile_sizes or normalized_compile_sizes[0] < 1:
                raise ValueError(
                    "compile_local_frame_batch_sizes must contain positive ints; "
                    f"got {self.compile_local_frame_batch_sizes}"
                )
            self.compile_local_frame_batch_sizes = normalized_compile_sizes
        if self.compact_topk_sampling:
            self.env_defaults.setdefault("MOSS_TTS_LOCAL_COMPACT_TOPK", "1")

    def supports_uploaded_voice_references(self) -> bool:
        return True


class MossTTSLocalColocatedPipelineConfig(MossTTSLocalPipelineConfig):
    """Backward-compatible alias for the default single-GPU pipeline."""

    stages: list[StageConfig] = Field(
        default_factory=lambda: _stages(codec_device="cuda:0", colocated=True)
    )


class MossTTSLocalSplitPipelineConfig(MossTTSLocalPipelineConfig):
    """Two-GPU variant with codec work and vocoder isolation on GPU 1."""

    stages: list[StageConfig] = Field(
        default_factory=lambda: _stages(
            codec_device="cuda:1",
            colocated=False,
            vocoder_gpu=1,
        )
    )


EntryClass = MossTTSLocalPipelineConfig

Variants = {
    "default": MossTTSLocalPipelineConfig,
    "colocated": MossTTSLocalColocatedPipelineConfig,
    "split": MossTTSLocalSplitPipelineConfig,
}
