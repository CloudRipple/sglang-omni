# SPDX-License-Identifier: Apache-2.0
"""CPU product tests for the MOSS-TTS-Realtime streaming vocoder."""

from __future__ import annotations

import queue
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

from sglang_omni.models.moss_tts_realtime.payload_types import MossTTSRealtimeState
from sglang_omni.models.moss_tts_realtime.streaming_vocoder import (
    MossTTSRealtimeStreamingVocoderScheduler,
    _CodecStreamSession,
    _LegacyCodecStreamingStateAdapter,
)
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto import OmniRequest, StagePayload
from tests.unit_test.moss_tts_realtime.runtime_config import MODEL_CONFIG

N_VQ = int(MODEL_CONFIG.rvq)
SAMPLE_RATE = 24000
SAMPLES_PER_FRAME = 1920


class _FakeStreamingState:
    def __init__(self, batch_size: int) -> None:
        self.device = torch.device("cpu")
        self.offsets = torch.zeros(batch_size, dtype=torch.long)
        self.exec_mask = torch.ones(batch_size, dtype=torch.bool)

    def set_exec_mask(self, exec_mask: torch.Tensor) -> None:
        self.exec_mask.copy_(exec_mask.to(dtype=torch.bool))

    def reset(self, reset_mask: torch.Tensor) -> None:
        self.offsets[reset_mask] = 0
        self.exec_mask[reset_mask] = True


class _FakeStateModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._streaming_state: _FakeStreamingState | None = None


class FakeLegacyCodec(nn.Module):
    """Legacy surface: module states exist, but no top-level exec-mask setter."""

    def __init__(self) -> None:
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))
        self.state_modules = nn.ModuleList([_FakeStateModule(), _FakeStateModule()])
        self.config = SimpleNamespace(
            sampling_rate=SAMPLE_RATE,
            downsample_rate=SAMPLES_PER_FRAME,
            quantizer_kwargs={"num_quantizers": 32, "codebook_size": 1024},
        )
        self.frame_calls: list[tuple[tuple[int, ...], int]] = []
        self.batch_decode_calls = 0
        self.streaming_batch_sizes: list[int] = []
        self.fail_next_decode = False

    @contextmanager
    def streaming(self, batch_size: int):
        if any(module._streaming_state is not None for module in self.state_modules):
            raise RuntimeError("already streaming")
        self.streaming_batch_sizes.append(batch_size)
        for module in self.state_modules:
            module._streaming_state = _FakeStreamingState(batch_size)
        try:
            yield
        finally:
            for module in self.state_modules:
                module._streaming_state = None

    def _decode_frame(self, codes: torch.Tensor, codes_lengths: torch.Tensor):
        if self.fail_next_decode:
            self.fail_next_decode = False
            raise RuntimeError("injected codec failure")
        _, batch_size, step_frames = codes.shape
        states = [module._streaming_state for module in self.state_modules]
        active = tuple(
            index
            for index in range(batch_size)
            if states[0] is None or bool(states[0].exec_mask[index])
        )
        self.frame_calls.append((active, step_frames))
        audio = torch.zeros(batch_size, 1, step_frames * SAMPLES_PER_FRAME)
        audio_lengths = torch.zeros(batch_size, dtype=torch.long)
        state_count = len(states)
        for batch_index in range(batch_size):
            length = int(codes_lengths[batch_index])
            if length == 0:
                continue
            if states[0] is not None and not bool(states[0].exec_mask[batch_index]):
                continue
            if states[0] is None:
                offset_units = 0
            else:
                offset_units = sum(
                    int(state.offsets[batch_index]) for state in states if state
                )
            for frame_index in range(length):
                value = float(codes[:, batch_index, frame_index].sum())
                value += 1000.0 * (offset_units + state_count * frame_index)
                start = frame_index * SAMPLES_PER_FRAME
                audio[
                    batch_index,
                    0,
                    start : start + SAMPLES_PER_FRAME,
                ] = value
            audio_lengths[batch_index] = length * SAMPLES_PER_FRAME
            for state in states:
                if state is not None and bool(state.exec_mask[batch_index]):
                    state.offsets[batch_index] += length
        return SimpleNamespace(audio=audio, audio_lengths=audio_lengths)

    def batch_decode(
        self,
        codes_list: list[torch.Tensor],
        *,
        num_quantizers: int | None = None,
    ):
        self.batch_decode_calls += 1
        if num_quantizers is None:
            num_quantizers = int(codes_list[0].shape[0])
        max_frames = max(int(codes.shape[1]) for codes in codes_list)
        batch = torch.zeros(
            num_quantizers,
            len(codes_list),
            max_frames,
            dtype=torch.long,
        )
        lengths = torch.zeros(len(codes_list), dtype=torch.long)
        for index, codes in enumerate(codes_list):
            frames = int(codes.shape[1])
            batch[:, index, :frames] = codes[:num_quantizers]
            lengths[index] = frames
        return self._decode_frame(batch, lengths)


class _FakeCudaGraphRunner:
    def __init__(
        self,
        frames: list[int],
        *,
        fail: bool = False,
        length_delta: int = 0,
    ) -> None:
        self._frames = sorted(set(frames))
        self._fail = fail
        self._length_delta = int(length_delta)
        self.decode_calls: list[int] = []

    def captured_frames(self) -> list[int]:
        return list(self._frames)

    def decode_step(
        self,
        codes: torch.Tensor,
        exec_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        frame_count = int(codes.shape[2])
        self.decode_calls.append(frame_count)
        if self._fail:
            raise RuntimeError("injected graph replay failure")
        if frame_count not in self._frames:
            return None
        batch_size = int(codes.shape[1])
        audio = torch.zeros(
            batch_size,
            1,
            frame_count * SAMPLES_PER_FRAME,
        )
        audio_lengths = exec_mask.to(dtype=torch.long) * (
            frame_count * SAMPLES_PER_FRAME + self._length_delta
        )
        return audio, audio_lengths


def _rows(frames: int, *, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, 100, (frames, N_VQ), generator=generator)


def _reference(rows: torch.Tensor) -> np.ndarray:
    waveform = np.empty(rows.shape[0] * SAMPLES_PER_FRAME, dtype=np.float32)
    state_count = 2
    for frame_index, row in enumerate(rows):
        value = float(row.sum()) + 1000.0 * state_count * frame_index
        start = frame_index * SAMPLES_PER_FRAME
        waveform[start : start + SAMPLES_PER_FRAME] = value
    return waveform


def _metadata(**extra: Any) -> dict[str, Any]:
    return {
        "stream": True,
        "modality": "audio_codes",
        "n_vq": N_VQ,
        **extra,
    }


def _stream_item(row: torch.Tensor, chunk_id: int, **metadata: Any) -> StreamItem:
    return StreamItem(
        chunk_id=chunk_id,
        data=row.clone(),
        from_stage="tts_engine",
        metadata=_metadata(**metadata),
    )


def _payload(
    rows: torch.Tensor,
    *,
    request_id: str,
    stream: bool,
) -> StagePayload:
    state = MossTTSRealtimeState(
        session_id=f"session-{request_id}",
        turn_id=request_id,
        audio_codes=rows.clone(),
    )
    state.prompt_tokens = 3
    state.completion_tokens = int(rows.shape[0])
    state.engine_time_s = 0.25
    return StagePayload(
        request_id=request_id,
        request=OmniRequest(inputs="", params={"stream": stream}),
        data=state.to_dict(),
    )


def _scheduler(
    *,
    stream_slots: int = 2,
    max_batch_size: int = 4,
) -> tuple[MossTTSRealtimeStreamingVocoderScheduler, FakeLegacyCodec]:
    codec = FakeLegacyCodec()
    scheduler = MossTTSRealtimeStreamingVocoderScheduler(
        codec,
        n_vq=N_VQ,
        stream_slots=stream_slots,
        max_batch_size=max_batch_size,
        max_batch_wait_ms=0,
    )
    return scheduler, codec


def _active_fake_states(codec: FakeLegacyCodec) -> list[_FakeStreamingState]:
    states = [module._streaming_state for module in codec.state_modules]
    assert all(state is not None for state in states)
    return [state for state in states if state is not None]


def _drain(scheduler: MossTTSRealtimeStreamingVocoderScheduler) -> list[Any]:
    messages = []
    while True:
        try:
            messages.append(scheduler.outbox.get_nowait())
        except queue.Empty:
            return messages


def _decode_audio(data: dict[str, Any]) -> np.ndarray:
    assert data["audio_waveform_dtype"] == "float32"
    waveform = np.frombuffer(data["audio_waveform"], dtype=np.float32)
    return waveform.reshape(data["audio_waveform_shape"])


def _stream_audio(messages: list[Any], request_id: str) -> np.ndarray:
    chunks = [
        _decode_audio(message.data)
        for message in messages
        if message.request_id == request_id and message.type == "stream"
    ]
    assert chunks
    assert all(chunk.ndim == 1 for chunk in chunks)
    return np.concatenate(chunks)


def _finish(
    scheduler: MossTTSRealtimeStreamingVocoderScheduler,
    rows: torch.Tensor,
    *,
    request_id: str,
) -> None:
    scheduler._on_done(request_id)
    scheduler._on_streaming_new_request(
        request_id,
        _payload(rows, request_id=request_id, stream=True),
    )


def test_ramp_final_flush_and_terminal_order() -> None:
    scheduler, codec = _scheduler(stream_slots=1)
    rows = _rows(8, seed=1)

    for index, row in enumerate(rows):
        scheduler._on_chunk("req", _stream_item(row, index))
    _finish(scheduler, rows, request_id="req")

    messages = _drain(scheduler)
    assert [message.type for message in messages] == [
        "stream",
        "stream",
        "stream",
        "stream",
        "result",
    ]
    assert [
        _decode_audio(message.data).shape[0]
        for message in messages
        if message.type == "stream"
    ] == [
        SAMPLES_PER_FRAME,
        2 * SAMPLES_PER_FRAME,
        3 * SAMPLES_PER_FRAME,
        2 * SAMPLES_PER_FRAME,
    ]
    assert [step for _, step in codec.frame_calls] == [1, 2, 3, 2]
    np.testing.assert_array_equal(_stream_audio(messages, "req"), _reference(rows))
    assert messages[-1].data.data["sample_rate"] == SAMPLE_RATE
    assert scheduler._session is not None
    assert scheduler._session.active_leases == 0


def test_equal_ramp_steps_coalesce_without_cross_slot_drift() -> None:
    scheduler, codec = _scheduler(stream_slots=2)
    rows_a = _rows(6, seed=2)
    rows_b = _rows(6, seed=3)
    items = []
    for index in range(6):
        items.extend(
            [
                ("a", _stream_item(rows_a[index], index)),
                ("b", _stream_item(rows_b[index], index)),
            ]
        )

    scheduler.on_stream_chunk_batch(items)
    _finish(scheduler, rows_a, request_id="a")
    _finish(scheduler, rows_b, request_id="b")

    messages = _drain(scheduler)
    assert codec.frame_calls == [((0, 1), 1), ((0, 1), 2), ((0, 1), 3)]
    np.testing.assert_array_equal(_stream_audio(messages, "a"), _reference(rows_a))
    np.testing.assert_array_equal(_stream_audio(messages, "b"), _reference(rows_b))


def test_staggered_requests_keep_their_exact_next_ramp_size() -> None:
    scheduler, codec = _scheduler(stream_slots=2)
    rows_a = _rows(3, seed=31)
    rows_b = _rows(3, seed=32)
    scheduler._on_chunk("a", _stream_item(rows_a[0], 0))
    scheduler.on_stream_chunk_batch(
        [
            ("a", _stream_item(rows_a[1], 1)),
            ("a", _stream_item(rows_a[2], 2)),
            ("b", _stream_item(rows_b[0], 0)),
        ]
    )
    scheduler.on_stream_chunk_batch(
        [
            ("b", _stream_item(rows_b[1], 1)),
            ("b", _stream_item(rows_b[2], 2)),
        ]
    )
    _finish(scheduler, rows_a, request_id="a")
    _finish(scheduler, rows_b, request_id="b")

    assert [step for _, step in codec.frame_calls] == [1, 1, 2, 2]
    messages = _drain(scheduler)
    np.testing.assert_array_equal(_stream_audio(messages, "a"), _reference(rows_a))
    np.testing.assert_array_equal(_stream_audio(messages, "b"), _reference(rows_b))


def test_codec_session_shares_one_exec_mask_and_resets_every_state() -> None:
    codec = FakeLegacyCodec()
    session = _CodecStreamSession(
        codec,
        stream_slots=2,
        n_vq=N_VQ,
        samples_per_frame=SAMPLES_PER_FRAME,
    )
    states = _active_fake_states(codec)
    shared_exec_mask = states[0].exec_mask

    assert all(state.exec_mask is shared_exec_mask for state in states)
    session._state_adapter.set_exec_mask(torch.tensor([False, False]))
    for index, state in enumerate(states, start=1):
        state.offsets.fill_(index)

    session._state_adapter.reset_slots([1], batch_size=2)

    assert torch.equal(shared_exec_mask, torch.tensor([False, True]))
    for index, state in enumerate(states, start=1):
        assert torch.equal(state.offsets, torch.tensor([index, 0]))
        assert state.exec_mask is shared_exec_mask
    session.close()


@pytest.mark.parametrize("mismatch", ["shape", "dtype", "device"])
def test_codec_state_adapter_rejects_incompatible_exec_masks(mismatch: str) -> None:
    codec = FakeLegacyCodec()
    with codec.streaming(2):
        states = _active_fake_states(codec)
        original_exec_masks = [state.exec_mask for state in states]
        if mismatch == "shape":
            states[1].exec_mask = torch.ones(3, dtype=torch.bool)
        elif mismatch == "dtype":
            states[1].exec_mask = torch.ones(2, dtype=torch.long)
        else:
            states[1].device = torch.device("meta")
            states[1].exec_mask = torch.ones(2, dtype=torch.bool, device="meta")

        with pytest.raises(RuntimeError, match="cannot share exec_mask"):
            _LegacyCodecStreamingStateAdapter(codec, device=torch.device("cpu"))

        assert states[0].exec_mask is original_exec_masks[0]


@pytest.mark.parametrize("stream_slots", [1, 16])
def test_shared_exec_mask_only_advances_active_eager_slot(stream_slots: int) -> None:
    codec = FakeLegacyCodec()
    session = _CodecStreamSession(
        codec,
        stream_slots=stream_slots,
        n_vq=N_VQ,
        samples_per_frame=SAMPLES_PER_FRAME,
    )
    slot = session.acquire()

    decoded = session.step({slot: _rows(1, seed=33).transpose(0, 1)})[slot]

    assert decoded.shape == (1, SAMPLES_PER_FRAME)
    assert codec.frame_calls == [((slot,), 1)]
    for state in _active_fake_states(codec):
        expected_offsets = torch.zeros(stream_slots, dtype=torch.long)
        expected_offsets[slot] = 1
        assert torch.equal(state.offsets, expected_offsets)
    session.release(slot)
    session.close()


def test_masked_release_preserves_peer_and_reused_slot_starts_fresh() -> None:
    scheduler, _ = _scheduler(stream_slots=2)
    rows_a = _rows(3, seed=4)
    rows_b = _rows(6, seed=5)
    scheduler.on_stream_chunk_batch(
        [
            *(("a", _stream_item(row, index)) for index, row in enumerate(rows_a)),
            *(("b", _stream_item(row, index)) for index, row in enumerate(rows_b[:3])),
        ]
    )
    released_slot = scheduler._stream_states["a"].slot

    _finish(scheduler, rows_a, request_id="a")
    for index, row in enumerate(rows_b[3:], start=3):
        scheduler._on_chunk("b", _stream_item(row, index))

    rows_c = _rows(1, seed=6)
    scheduler._on_chunk("c", _stream_item(rows_c[0], 0))
    assert scheduler._stream_states["c"].slot == released_slot
    _finish(scheduler, rows_c, request_id="c")
    _finish(scheduler, rows_b, request_id="b")

    messages = _drain(scheduler)
    np.testing.assert_array_equal(_stream_audio(messages, "b"), _reference(rows_b))
    np.testing.assert_array_equal(_stream_audio(messages, "c"), _reference(rows_c))


def test_slot_exhaustion_errors_without_displacing_live_request() -> None:
    scheduler, _ = _scheduler(stream_slots=1)
    row = _rows(1, seed=7)[0]
    scheduler._on_chunk("live", _stream_item(row, 0))
    scheduler._on_chunk("overflow", _stream_item(row, 0))

    messages = _drain(scheduler)
    error = next(message for message in messages if message.request_id == "overflow")
    assert error.type == "error"
    assert "slots are exhausted" in str(error.data)
    assert "live" in scheduler._stream_states
    assert "overflow" not in scheduler._stream_states
    assert scheduler._session is not None
    assert scheduler._session.active_leases == 1


def test_codec_model_info_tracks_acquire_release_reuse_and_exhaustion() -> None:
    scheduler, _ = _scheduler(stream_slots=1)
    row = _rows(1, seed=71)[0]
    initial = scheduler.admin("model_info")["data"]
    assert initial["codec_slot_capacity"] == 1
    assert initial["codec_active_slots"] == 0
    assert initial["codec_free_slots"] == 1
    assert initial["codec_decoder_dtype"] == "float32"

    scheduler._on_chunk("live", _stream_item(row, 0))
    leased_slot = scheduler._stream_states["live"].slot
    scheduler._on_chunk("overflow", _stream_item(row, 0))
    active = scheduler.admin("model_info")["data"]
    assert active["codec_active_slots"] == 1
    assert active["codec_free_slots"] == 0
    assert active["codec_live_stream_states"] == 1
    assert active["codec_active_slots_high_water"] == 1
    assert active["codec_pending_frames_high_water"] == 1
    assert active["codec_slot_acquire_total"] == 1
    assert active["codec_slot_exhaustion_total"] == 1

    scheduler.abort("live")
    scheduler._on_chunk("reused", _stream_item(row, 0))
    assert scheduler._stream_states["reused"].slot == leased_slot
    scheduler.abort("reused")
    released = scheduler.admin("model_info")["data"]
    assert released["codec_active_slots"] == 0
    assert released["codec_free_slots"] == 1
    assert released["codec_live_stream_states"] == 0
    assert released["codec_slot_acquire_total"] == 2
    assert released["codec_slot_release_total"] == 2
    assert released["codec_slot_exhaustion_total"] == 1
    assert released["codec_slot_reset_error_total"] == 0


def test_codec_reset_failure_quarantines_slot_and_reports_error() -> None:
    scheduler, codec = _scheduler(stream_slots=1)
    row = _rows(1, seed=72)[0]
    scheduler._on_chunk("live", _stream_item(row, 0))
    streaming_state = codec.state_modules[0]._streaming_state
    assert streaming_state is not None

    def fail_reset(reset_mask: torch.Tensor) -> None:
        del reset_mask
        raise RuntimeError("injected codec reset failure")

    streaming_state.reset = fail_reset
    with pytest.raises(RuntimeError, match="injected codec reset failure"):
        scheduler.clear_stream_state("live")

    snapshot = scheduler.admin("model_info")["data"]
    assert snapshot["codec_active_slots"] == 0
    assert snapshot["codec_free_slots"] == 0
    assert snapshot["codec_quarantined_slots"] == 1
    assert snapshot["codec_live_stream_states"] == 0
    assert snapshot["codec_slot_acquire_total"] == 1
    assert snapshot["codec_slot_release_total"] == 0
    assert snapshot["codec_slot_reset_error_total"] == 1


def test_shared_codec_step_failure_aborts_every_participant() -> None:
    scheduler, codec = _scheduler(stream_slots=2)
    rows_a = _rows(1, seed=33)
    rows_b = _rows(1, seed=34)
    codec.fail_next_decode = True

    scheduler.on_stream_chunk_batch(
        [
            ("a", _stream_item(rows_a[0], 0)),
            ("b", _stream_item(rows_b[0], 0)),
        ]
    )

    messages = _drain(scheduler)
    assert [(message.request_id, message.type) for message in messages] == [
        ("a", "error"),
        ("b", "error"),
    ]
    assert all("injected codec failure" in str(message.data) for message in messages)
    assert not scheduler._stream_states
    assert scheduler._session is not None
    assert scheduler._session.active_leases == 0


@pytest.mark.parametrize(
    ("codes", "metadata", "match"),
    [
        (torch.zeros(15, dtype=torch.long), {}, "shape"),
        (torch.zeros(16, dtype=torch.float32), {}, "integer dtype"),
        (torch.full((16,), 1024, dtype=torch.long), {}, "must be in"),
        (torch.zeros(16, dtype=torch.long), {"n_vq": 8}, "n_vq"),
        (
            torch.zeros(16, dtype=torch.long),
            {"sample_rate": 48000},
            "sample_rate",
        ),
    ],
)
def test_invalid_stream_chunk_aborts_and_releases_slot(
    codes: torch.Tensor,
    metadata: dict[str, Any],
    match: str,
) -> None:
    scheduler, _ = _scheduler(stream_slots=1)
    item = StreamItem(
        chunk_id=0,
        data=codes,
        from_stage="tts_engine",
        metadata=_metadata(**metadata),
    )
    scheduler._on_chunk("bad", item)

    messages = _drain(scheduler)
    assert len(messages) == 1
    assert messages[0].type == "error"
    assert match in str(messages[0].data)
    assert "bad" not in scheduler._stream_states
    assert scheduler._session is not None
    assert scheduler._session.active_leases == 0


def test_abort_is_idempotent_and_late_chunks_do_not_reacquire() -> None:
    scheduler, _ = _scheduler(stream_slots=1)
    row = _rows(1, seed=8)[0]
    scheduler._on_chunk("req", _stream_item(row, 0))
    scheduler.abort("req")
    scheduler.abort("req")
    scheduler._on_chunk("req", _stream_item(row, 1))

    assert scheduler._session is not None
    assert scheduler._session.active_leases == 0
    assert "req" not in scheduler._stream_states


def test_stop_releases_all_live_slots_and_closes_context() -> None:
    scheduler, codec = _scheduler(stream_slots=2)
    rows = _rows(1, seed=9)
    scheduler._on_chunk("a", _stream_item(rows[0], 0))
    scheduler._on_chunk("b", _stream_item(rows[0], 0))

    scheduler.stop()

    assert scheduler._session is None
    assert not scheduler._stream_states
    assert all(module._streaming_state is None for module in codec.state_modules)


def test_serving_start_opens_fixed_slot_codec_session() -> None:
    scheduler, codec = _scheduler(stream_slots=3)

    scheduler.on_serving_start()

    assert scheduler._session is not None
    assert scheduler._session.free_slots == 3
    assert codec.streaming_batch_sizes == [3]
    scheduler.on_serving_stop()


def test_codec_session_routes_captured_shapes_and_falls_back_to_eager() -> None:
    codec = FakeLegacyCodec()
    session = _CodecStreamSession(
        codec,
        stream_slots=1,
        n_vq=N_VQ,
        samples_per_frame=SAMPLES_PER_FRAME,
    )
    slot = session.acquire()
    graph = _FakeCudaGraphRunner([1])
    session._cg_runner = graph

    graphed = session.step({slot: _rows(1, seed=91).transpose(0, 1)})[slot]
    eager = session.step({slot: _rows(2, seed=92).transpose(0, 1)})[slot]

    assert graphed.shape == (1, SAMPLES_PER_FRAME)
    assert eager.shape == (1, 2 * SAMPLES_PER_FRAME)
    assert graph.decode_calls == [1, 2]
    assert codec.frame_calls == [((slot,), 2)]
    assert session._cg_graph_frames == {1: 1}
    assert session._cg_eager_frames == {2: 1}
    session.release(slot)
    session.close()


def test_codec_session_disables_graph_after_replay_failure() -> None:
    codec = FakeLegacyCodec()
    session = _CodecStreamSession(
        codec,
        stream_slots=1,
        n_vq=N_VQ,
        samples_per_frame=SAMPLES_PER_FRAME,
    )
    slot = session.acquire()
    session._cg_runner = _FakeCudaGraphRunner([1], fail=True)
    codes = _rows(1, seed=93).transpose(0, 1)

    with pytest.raises(RuntimeError, match="graph replay failure"):
        session.step({slot: codes})

    assert session._cg_runner is None
    decoded = session.step({slot: codes})[slot]
    assert decoded.shape == (1, SAMPLES_PER_FRAME)
    assert codec.frame_calls == [((slot,), 1)]
    session.release(slot)
    session.close()


def test_codec_session_disables_graph_after_invalid_replay_output() -> None:
    codec = FakeLegacyCodec()
    session = _CodecStreamSession(
        codec,
        stream_slots=1,
        n_vq=N_VQ,
        samples_per_frame=SAMPLES_PER_FRAME,
    )
    slot = session.acquire()
    session._cg_runner = _FakeCudaGraphRunner([1], length_delta=1)

    with pytest.raises(RuntimeError, match="unexpected active length"):
        session.step({slot: _rows(1, seed=95).transpose(0, 1)})

    assert session._cg_runner is None
    session.release(slot)
    session.close()


def test_codec_session_attempts_cuda_graph_warmup_once(monkeypatch) -> None:
    from sglang_omni.models.moss_tts_realtime import vocoder_cuda_graph

    calls: list[tuple[list[int], float]] = []

    class FakeCaptureRunner:
        def __init__(self, *args: Any, min_free_gb: float, **kwargs: Any) -> None:
            del args, kwargs
            self._min_free_gb = min_free_gb
            self._frames: list[int] = []

        def warmup(self, frames: list[int]) -> None:
            calls.append((list(frames), self._min_free_gb))
            self._frames = list(frames)

        def captured_frames(self) -> list[int]:
            return list(self._frames)

    monkeypatch.setattr(
        vocoder_cuda_graph,
        "MossTTSRealtimeVocoderCudaGraphRunner",
        FakeCaptureRunner,
    )
    codec = FakeLegacyCodec()
    session = _CodecStreamSession(
        codec,
        stream_slots=2,
        n_vq=N_VQ,
        samples_per_frame=SAMPLES_PER_FRAME,
    )
    slot = session.acquire()
    session.step({slot: _rows(1, seed=94).transpose(0, 1)})

    assert session.warmup_cuda_graph([1, 2], min_free_gb=4.5) == [1, 2]
    assert session.warmup_cuda_graph([3], min_free_gb=8.0) == [1, 2]
    assert calls == [([1, 2], 4.5)]
    for module in codec.state_modules:
        state = module._streaming_state
        assert state is not None
        assert torch.equal(state.offsets, torch.zeros(2, dtype=torch.long))
    session.release(slot)
    session.close()


def test_scheduler_default_cuda_graph_frames_cover_dense_catchup_range(
    monkeypatch,
) -> None:
    calls: list[tuple[list[int], float]] = []
    monkeypatch.setattr(
        MossTTSRealtimeStreamingVocoderScheduler,
        "_codec_on_cuda",
        lambda self: True,
    )

    def fake_warmup(
        self: _CodecStreamSession,
        frames: list[int],
        *,
        min_free_gb: float = 3.0,
    ) -> list[int]:
        self.warmup_attempted = True
        calls.append((list(frames), min_free_gb))
        return []

    monkeypatch.setattr(_CodecStreamSession, "warmup_cuda_graph", fake_warmup)
    scheduler, _ = _scheduler(stream_slots=1)

    scheduler.warmup_now()
    scheduler.warmup_now()

    assert calls == [(list(range(1, 13)), 3.0)]
    assert scheduler._session is not None
    assert scheduler._session.warmup_attempted
    scheduler.on_serving_stop()


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"cuda_graph_frames": []}, "must not be empty"),
        ({"cuda_graph_frames": [0]}, "step range"),
        ({"cuda_graph_frames": [26]}, "step range"),
        ({"cuda_graph_min_free_gb": -1.0}, "non-negative"),
    ],
)
def test_scheduler_rejects_invalid_cuda_graph_settings(
    kwargs: dict[str, Any],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        MossTTSRealtimeStreamingVocoderScheduler(
            FakeLegacyCodec(),
            n_vq=N_VQ,
            **kwargs,
        )


def test_scheduler_accepts_explicit_capture_through_25_frames() -> None:
    scheduler = MossTTSRealtimeStreamingVocoderScheduler(
        FakeLegacyCodec(),
        n_vq=N_VQ,
        cuda_graph_frames=[25, 1, 25],
    )

    assert scheduler._cuda_graph_capture_frames() == [1, 25]


def test_dense_cuda_graph_range_drains_existing_backlog_without_waiting() -> None:
    scheduler, _ = _scheduler(stream_slots=1)
    session = scheduler._ensure_session()
    graph = _FakeCudaGraphRunner(list(range(1, 13)))
    session._cg_runner = graph
    rows = _rows(20, seed=96)

    scheduler.on_stream_chunk_batch(
        [("req", _stream_item(row, index)) for index, row in enumerate(rows)]
    )

    assert graph.decode_calls == [1, 2, 12, 5]
    messages = _drain(scheduler)
    assert [
        _decode_audio(message.data).shape[0]
        for message in messages
        if message.type == "stream"
    ] == [
        1 * SAMPLES_PER_FRAME,
        2 * SAMPLES_PER_FRAME,
        12 * SAMPLES_PER_FRAME,
        5 * SAMPLES_PER_FRAME,
    ]
    snapshot = scheduler.admin("model_info")["data"]
    assert snapshot["codec_resource_totals"]["codec_catchup_step_total"] == 2
    assert snapshot["codec_resource_totals"]["codec_catchup_frame_total"] == 17
    scheduler.abort("req")


def test_dense_cuda_graph_range_never_waits_to_fill_a_larger_shape() -> None:
    scheduler, _ = _scheduler(stream_slots=1)
    session = scheduler._ensure_session()
    graph = _FakeCudaGraphRunner(list(range(1, 13)))
    session._cg_runner = graph
    rows = _rows(6, seed=97)

    for index, row in enumerate(rows):
        scheduler._on_chunk("req", _stream_item(row, index))

    assert graph.decode_calls == [1, 2, 3]
    snapshot = scheduler.admin("model_info")["data"]
    assert snapshot["codec_cuda_graph_default_max_frames"] == 12
    assert snapshot["codec_resource_totals"].get("codec_catchup_step_total", 0) == 0
    scheduler.abort("req")


def test_idle_offline_batch_uses_full_batch_decode() -> None:
    scheduler, codec = _scheduler(stream_slots=2)
    rows_a = _rows(4, seed=10)
    rows_b = _rows(2, seed=11)
    payloads = [
        _payload(rows_a, request_id="a", stream=False),
        _payload(rows_b, request_id="b", stream=False),
    ]

    results = scheduler._vocode_batch(payloads)

    assert codec.batch_decode_calls == 1
    np.testing.assert_array_equal(
        _decode_audio(results[0].data),
        _reference(rows_a),
    )
    np.testing.assert_array_equal(
        _decode_audio(results[1].data),
        _reference(rows_b),
    )


def test_done_before_payload_falls_back_to_terminal_audio_codes_once() -> None:
    scheduler, codec = _scheduler(stream_slots=1)
    rows = _rows(4, seed=35)

    scheduler._on_done("req")
    scheduler._on_streaming_new_request(
        "req",
        _payload(rows, request_id="req", stream=True),
    )

    messages = _drain(scheduler)
    assert [message.type for message in messages] == ["stream", "result"]
    assert codec.batch_decode_calls == 1
    np.testing.assert_array_equal(_stream_audio(messages, "req"), _reference(rows))


def test_audio_eos_without_real_frames_emits_only_terminal_result() -> None:
    scheduler, codec = _scheduler(stream_slots=1)
    rows = torch.empty((0, N_VQ), dtype=torch.long)

    scheduler._on_done("req")
    scheduler._on_streaming_new_request(
        "req",
        _payload(rows, request_id="req", stream=True),
    )

    messages = _drain(scheduler)
    assert [message.type for message in messages] == ["result"]
    assert codec.batch_decode_calls == 0
    assert scheduler._session is not None
    assert scheduler._session.active_leases == 0


def test_offline_decode_borrows_free_slot_without_advancing_live_stream() -> None:
    scheduler, codec = _scheduler(stream_slots=2)
    live_rows = _rows(6, seed=12)
    for index, row in enumerate(live_rows[:3]):
        scheduler._on_chunk("live", _stream_item(row, index))
    offline_rows = _rows(4, seed=13)

    result = scheduler._vocode(
        _payload(offline_rows, request_id="offline", stream=False)
    )
    for index, row in enumerate(live_rows[3:], start=3):
        scheduler._on_chunk("live", _stream_item(row, index))
    _finish(scheduler, live_rows, request_id="live")

    messages = _drain(scheduler)
    assert codec.batch_decode_calls == 0
    np.testing.assert_array_equal(_decode_audio(result.data), _reference(offline_rows))
    np.testing.assert_array_equal(
        _stream_audio(messages, "live"),
        _reference(live_rows),
    )


def test_offline_decode_fails_when_all_fixed_slots_are_leased() -> None:
    scheduler, _ = _scheduler(stream_slots=1)
    row = _rows(1, seed=14)[0]
    scheduler._on_chunk("live", _stream_item(row, 0))

    with pytest.raises(RuntimeError, match="no free slot"):
        scheduler._vocode(
            _payload(_rows(2, seed=15), request_id="offline", stream=False)
        )
