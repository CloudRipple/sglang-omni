# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect

from sglang_omni.models.moss_transcribe_diarize.config import (
    MossTranscribeDiarizePipelineConfig,
)
from sglang_omni.models.moss_transcribe_diarize.stages import (
    create_sglang_moss_transcribe_diarize_executor,
)
from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY


def test_moss_transcribe_diarize_config_uses_single_batched_stage() -> None:
    config = MossTranscribeDiarizePipelineConfig(
        model_path="OpenMOSS-Team/MOSS-Transcribe-Diarize"
    )

    assert config.entry_stage == "asr"
    assert [stage.name for stage in config.stages] == ["asr"]
    assert config.terminal_stages == ["asr"]
    assert config.gpu_placement == {"asr": 0}
    assert config.stages[0].factory.endswith(
        "create_sglang_moss_transcribe_diarize_executor"
    )
    assert config.stages[0].factory_args["device"] == "cuda:0"
    assert config.stages[0].factory_args["max_running_requests"] == 16
    assert config.stages[0].factory_args["request_build_max_workers"] == 2
    assert config.stages[0].factory_args["request_build_max_pending"] == 16
    assert (
        PIPELINE_CONFIG_REGISTRY.get_config(
            "MossTranscribeDiarizeForConditionalGeneration"
        )
        is MossTranscribeDiarizePipelineConfig
    )
    assert MossTranscribeDiarizePipelineConfig.mem_fraction_role_to_stage() == {
        "asr": "asr"
    }
    assert MossTranscribeDiarizePipelineConfig.generation_sglang_role_to_stage() == {
        "generation": "asr"
    }


def test_moss_transcribe_diarize_stage_reserves_encoder_headroom() -> None:
    signature = inspect.signature(create_sglang_moss_transcribe_diarize_executor)

    assert signature.parameters["max_running_requests"].default == 16
    assert signature.parameters["mem_fraction_static"].default == 0.80
    assert signature.parameters["request_build_max_workers"].default == 2
    assert signature.parameters["request_build_max_pending"].default == 16
    assert signature.parameters["mm_embedding_cache_size_bytes"].default == 0
