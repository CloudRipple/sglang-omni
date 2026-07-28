# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import pytest

from sglang_omni.models.moss_tts_realtime.engine_builder import (
    MossTTSRealtimeEngineBuilder,
)


def _builder(**overrides: Any) -> MossTTSRealtimeEngineBuilder:
    values: dict[str, Any] = {
        "max_seq_len": 40960,
        "total_gpu_memory_fraction": 0.90,
        "codec_mem_reserve": 0.15,
        "max_sessions": 7,
        "max_held_sessions": 5,
        "max_active_turns": 3,
        "max_pending_text_tokens": 64,
        "max_pending_text_bytes": 2048,
        "max_input_updates": 32,
        "max_turn_frames": 40,
        "terminal_tombstone_limit": 77,
        "input_idle_timeout_s": 1.5,
        "turn_timeout_s": 2.5,
        "session_idle_ttl_s": 3.5,
    }
    values.update(overrides)
    return MossTTSRealtimeEngineBuilder(**values)


def test_falls_back_when_process_accounting_is_unavailable(monkeypatch) -> None:
    from sglang_omni.utils import gpu_memory

    builder = _builder()
    builder.gpu_id = 1
    overrides = builder.generation_defaults(dtype="bfloat16")
    monkeypatch.setattr(gpu_memory, "get_process_gpu_memory_bytes", lambda _: None)

    builder.adjust_overrides(overrides)

    assert overrides["mem_fraction_static"] == pytest.approx(0.75)
    assert builder.profile_total_gpu_memory_fraction is None


@pytest.mark.parametrize(
    ("max_sessions", "max_held_sessions", "max_active_turns", "expected"),
    [
        (10, 2, 3, 5),
        (7, 5, 3, 7),
    ],
)
def test_reserves_request_slots_for_held_sessions(
    max_sessions: int,
    max_held_sessions: int,
    max_active_turns: int,
    expected: int,
) -> None:
    builder = _builder(
        max_sessions=max_sessions,
        max_held_sessions=max_held_sessions,
        max_active_turns=max_active_turns,
    )

    assert builder.request_pool_capacity == expected
    assert (
        builder.generation_defaults(dtype="bfloat16")["max_running_requests"]
        == expected
    )
