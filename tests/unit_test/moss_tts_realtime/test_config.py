# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from pydantic import ValidationError

from sglang_omni.config import PipelineConfig, StageConfig, resolve_stage_factory_args
from sglang_omni.models.moss_tts_realtime import text_delta
from sglang_omni.models.moss_tts_realtime.config import (
    DEFAULT_MOSS_TTS_REALTIME_CODEC_MODEL,
    MossTTSRealtimePipelineConfig,
    MossTTSRealtimeResourceLimits,
    MossTTSRealtimeSplitPipelineConfig,
)
from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY
from sglang_omni.pipeline.realtime_coordinator import RealtimeCoordinator
from sglang_omni.utils.imports import import_string


def test_default_pipeline_declares_realtime_streaming_topology() -> None:
    config = MossTTSRealtimePipelineConfig(model_path="fake-model")
    stages = {stage.name: stage for stage in config.stages}

    assert list(stages) == ["preprocessing", "tts_engine", "vocoder"]
    assert stages["preprocessing"].next == "tts_engine"
    assert stages["tts_engine"].next == "vocoder"
    assert stages["tts_engine"].stream_to == ["vocoder"]
    assert stages["tts_engine"].can_accept_stream_before_payload is True
    assert config.realtime_input_stage == "tts_engine"
    assert stages["vocoder"].terminal is True
    assert stages["vocoder"].can_accept_stream_before_payload is True

    assert stages["preprocessing"].factory_args["codec_model_path"] == (
        DEFAULT_MOSS_TTS_REALTIME_CODEC_MODEL
    )
    assert "enable_streaming_session" not in stages["tts_engine"].factory_args
    assert not any(
        key.startswith("local_cuda_graph") for key in stages["tts_engine"].factory_args
    )
    assert stages["tts_engine"].factory_args["max_active_turns"] == 16
    assert stages["tts_engine"].factory_args["session_idle_ttl_s"] == 300.0
    assert stages["vocoder"].factory_args["stream_slots"] == 16


def test_realtime_coordinator_factory_is_model_scoped() -> None:
    assert PipelineConfig.coordinator_factory is None
    assert "coordinator_factory" not in PipelineConfig.model_fields
    assert (
        import_string(MossTTSRealtimePipelineConfig.coordinator_factory)
        is RealtimeCoordinator
    )


def test_tts_engine_runtime_resolves_context_and_colocated_budget() -> None:
    config = MossTTSRealtimePipelineConfig(model_path="fake-model")
    stage = next(stage for stage in config.stages if stage.name == "tts_engine")

    args = resolve_stage_factory_args(stage, config)

    assert args["model_path"] == "fake-model"
    assert args["gpu_id"] == 0
    assert "max_seq_len" not in args
    assert args["total_gpu_memory_fraction"] == pytest.approx(0.90)
    assert args["codec_mem_reserve"] == pytest.approx(0.15)


def test_split_pipeline_targets_second_visible_gpu_for_codec() -> None:
    config = MossTTSRealtimeSplitPipelineConfig(model_path="fake-model")
    stages = {stage.name: stage for stage in config.stages}

    assert stages["preprocessing"].factory_args["device"] == "cuda:1"
    assert stages["vocoder"].factory_args["device"] == "cuda:1"
    assert stages["tts_engine"].runtime.sglang_server_args.mem_fraction_static == 0.85


def test_pipeline_config_round_trip_and_defaults_are_independent() -> None:
    first = MossTTSRealtimePipelineConfig(model_path="fake-model")
    second = MossTTSRealtimePipelineConfig(model_path="fake-model")
    first.stages[0].factory_args["test_only"] = True

    assert "test_only" not in second.stages[0].factory_args

    restored = MossTTSRealtimePipelineConfig.model_validate(first.model_dump())
    assert restored.model_dump() == first.model_dump()


def test_pipeline_builds_model_owned_speech_websocket_handler(monkeypatch) -> None:
    loaded_paths: list[str] = []

    class _Tokenizer:
        def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
            del text, add_special_tokens
            return []

    tokenizer = _Tokenizer()

    def _load(model_path: str):
        loaded_paths.append(model_path)
        return tokenizer

    monkeypatch.setattr(text_delta, "load_moss_tts_realtime_text_tokenizer", _load)
    config = MossTTSRealtimePipelineConfig(model_path="fake-model")

    handler = config.build_speech_realtime_handler()

    assert loaded_paths == ["fake-model"]
    assert callable(handler)


def test_pipeline_resource_limits_authoritatively_override_stage_defaults() -> None:
    stages = MossTTSRealtimePipelineConfig(model_path="fake-model").stages
    stage_by_name = {stage.name: stage for stage in stages}
    stage_by_name["tts_engine"].factory_args.update(
        max_sessions=999,
        max_session_rows=999,
        max_held_kv_tokens=999,
        codec_slots=999,
        turn_timeout_s=999.0,
    )
    stage_by_name["vocoder"].factory_args["stream_slots"] = 999
    limits = MossTTSRealtimeResourceLimits(
        max_sessions=7,
        max_held_sessions=5,
        max_active_turns=3,
        max_pending_text_tokens=64,
        max_pending_text_bytes=2048,
        max_input_updates=32,
        max_turn_frames=40,
        terminal_tombstone_limit=77,
        input_idle_timeout_s=1.5,
        turn_timeout_s=2.5,
        session_idle_ttl_s=3.5,
    )

    config = MossTTSRealtimePipelineConfig(
        model_path="fake-model",
        limits=limits,
        stages=stages,
    )

    stage_by_name = {stage.name: stage for stage in config.stages}
    engine_args = stage_by_name["tts_engine"].factory_args
    assert "enable_streaming_session" not in engine_args
    for key, value in limits.model_dump().items():
        assert engine_args[key] == value
    assert "max_session_rows" not in engine_args
    assert "max_held_kv_tokens" not in engine_args
    assert "codec_slots" not in engine_args
    assert stage_by_name["vocoder"].factory_args["stream_slots"] == (
        limits.max_active_turns
    )


@pytest.mark.parametrize(
    "field_name",
    ["max_session_rows", "max_held_kv_tokens", "codec_slots"],
)
def test_resource_limits_reject_derived_fields(field_name: str) -> None:
    assert field_name not in MossTTSRealtimeResourceLimits.model_fields
    with pytest.raises(ValidationError, match=field_name):
        MossTTSRealtimeResourceLimits(**{field_name: 1})


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"max_sessions": 0}, "max_sessions"),
        ({"max_sessions": 2, "max_held_sessions": 3}, "max_held_sessions"),
        ({"max_sessions": 2, "max_active_turns": 3}, "max_active_turns"),
        ({"input_idle_timeout_s": 0}, "input_idle_timeout_s"),
    ],
)
def test_resource_limits_reject_invalid_values(
    overrides: dict[str, int | float], match: str
) -> None:
    with pytest.raises(ValidationError, match=match):
        MossTTSRealtimeResourceLimits(**overrides)


def test_pipeline_rejects_empty_codec_path() -> None:
    with pytest.raises(ValidationError, match="codec_model_path"):
        MossTTSRealtimePipelineConfig(
            model_path="fake-model",
            codec_model_path=" ",
        )


def test_pipeline_realtime_input_stage_must_exist_and_accept_early_updates() -> None:
    with pytest.raises(ValidationError, match="realtime_input_stage.*not defined"):
        MossTTSRealtimePipelineConfig(
            model_path="fake-model",
            realtime_input_stage="missing",
        )

    with pytest.raises(ValidationError, match="must accept updates"):
        MossTTSRealtimePipelineConfig(
            model_path="fake-model",
            stages=[
                StageConfig(
                    name="tts_engine",
                    process="pipeline",
                    factory="fake.factory",
                    terminal=True,
                )
            ],
        )


def test_pipeline_registry_exposes_hf_architecture_and_aliases() -> None:
    for architecture in (
        "MossTTSRealtime",
        "MossTTSRealtimeModel",
        "MossTTSRealtimeForConditionalGeneration",
    ):
        assert (
            PIPELINE_CONFIG_REGISTRY.get_config(architecture)
            is MossTTSRealtimePipelineConfig
        )
