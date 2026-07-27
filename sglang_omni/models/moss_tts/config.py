# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for MOSS-TTS Delay."""

from __future__ import annotations

from typing import Any, ClassVar

from sglang_omni.config import PipelineConfig, StageConfig

_PKG = "sglang_omni.models.moss_tts"
_REF_AUDIO_CACHE_MAX_ITEMS = 8192
_REF_AUDIO_CACHE_MAX_BYTES = 64 * 1024 * 1024


class MossTTSPipelineConfig(PipelineConfig):
    """MOSS-TTS Delay pipeline: preprocessing -> AR engine -> vocoder."""

    architecture: ClassVar[str] = "MossTTSDelayModel"
    requires_model_capabilities: ClassVar[bool] = True
    architecture_aliases: ClassVar[tuple[str, ...]] = (
        "MossTTSDelay",
        "MossTTSDelayForConditionalGeneration",
        "MossTTSDelayWithCodec",
        "MossTTSDelayWithCodecModel",
    )

    @classmethod
    def mem_fraction_role_to_stage(cls) -> dict[str, str]:
        return {"talker": "tts_engine"}

    @classmethod
    def talker_sglang_role_to_stage(cls) -> dict[str, str]:
        return {"talker": "tts_engine"}

    @classmethod
    def generation_sglang_role_to_stage(cls) -> dict[str, str]:
        return {"generation": "tts_engine"}

    @classmethod
    def process_safe_edges(cls) -> frozenset[tuple[str, str]]:
        # Note (Akazaakane): preprocessing -> tts_engine is excluded because
        # preprocessing publishes into the module-level PreparedRequestQueue that the
        # AR stage pops in-process. The vocoder loads its own processor and reads
        # delayed codes from MossTTSState.
        return frozenset({("tts_engine", "vocoder")})

    @classmethod
    def process_edge_resources(
        cls,
    ) -> dict[tuple[str, str], dict[str, float]]:
        return {
            ("tts_engine", "vocoder"): {
                "preprocessing": 0.05,
                "tts_engine": 0.85,
                "vocoder": 0.10,
            }
        }

    model_path: str
    ref_audio_cache: bool = True
    ref_audio_cache_max_items: int = _REF_AUDIO_CACHE_MAX_ITEMS
    ref_audio_cache_max_bytes: int = _REF_AUDIO_CACHE_MAX_BYTES
    stages: list[StageConfig] = [
        StageConfig(
            name="preprocessing",
            process="pipeline",
            factory=f"{_PKG}.stages.create_preprocessing_executor",
            factory_args={"dtype": "float32"},
            gpu=0,
            next="tts_engine",
        ),
        StageConfig(
            name="tts_engine",
            process="pipeline",
            factory=f"{_PKG}.stages.create_sglang_tts_engine_executor",
            factory_args={"dtype": "bfloat16"},
            gpu=0,
            next="vocoder",
            stream_to=["vocoder"],
        ),
        StageConfig(
            name="vocoder",
            process="pipeline",
            factory=f"{_PKG}.stages.create_vocoder_executor",
            factory_args={"dtype": "float32"},
            gpu=0,
            terminal=True,
            can_accept_stream_before_payload=True,
        ),
    ]

    def model_post_init(self, __context: Any = None) -> None:
        super().model_post_init(__context)
        merged_keys = (
            frozenset(__context.get("config_manager_merged_keys", ()))
            if isinstance(__context, dict)
            else frozenset()
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
        for stage_index, stage in enumerate(self.stages):
            if stage.factory.endswith("create_preprocessing_executor"):
                cache_args = {
                    "ref_audio_cache": self.ref_audio_cache,
                    "ref_audio_cache_max_items": self.ref_audio_cache_max_items,
                    "ref_audio_cache_max_bytes": self.ref_audio_cache_max_bytes,
                }
                for name, value in cache_args.items():
                    stage_override_merged = bool(
                        {
                            f"stages.{stage.name}.factory_args.{name}",
                            f"stages.{stage_index}.factory_args.{name}",
                        }
                        & merged_keys
                    )
                    if name in merged_keys and not stage_override_merged:
                        stage.factory_args[name] = value
                    else:
                        stage.factory_args.setdefault(name, value)

    def supports_uploaded_voice_references(self) -> bool:
        return True


EntryClass = MossTTSPipelineConfig
