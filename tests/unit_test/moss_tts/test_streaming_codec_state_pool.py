# SPDX-License-Identifier: Apache-2.0
"""Contract tests for the indexed MOSS codec streaming state pool."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sglang_omni.models.moss_tts.audio_tokenizer import MossAudioTokenizerVocoder
from sglang_omni.models.moss_tts.streaming_codec import (
    MossAudioTokenizerStreamingStatePool,
)


class _LegacyState:
    def __init__(self, batch_size: int) -> None:
        self.offsets = torch.zeros(batch_size, dtype=torch.long)
        self.exec_mask = torch.ones(batch_size, dtype=torch.bool)

    def reset(self, reset_mask: torch.Tensor) -> None:
        self.offsets[reset_mask] = 0
        self.exec_mask[reset_mask] = True


class _LegacyCodec(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))
        self._streaming_state: _LegacyState | None = None
        self.decode_calls = 0

    @contextmanager
    def streaming(self, batch_size: int):
        if self._streaming_state is not None:
            raise RuntimeError("nested streaming context")
        self._streaming_state = _LegacyState(batch_size)
        try:
            yield
        finally:
            self._streaming_state = None

    def _set_streaming_exec_mask(self, exec_mask: torch.Tensor) -> None:
        assert self._streaming_state is not None
        self._streaming_state.exec_mask = exec_mask.clone()

    def _decode_frame(
        self,
        codes: torch.Tensor,
        codes_lengths: torch.Tensor,
    ) -> SimpleNamespace:
        assert self._streaming_state is not None
        self.decode_calls += 1
        _, batch_size, frame_count = codes.shape
        audio = torch.zeros(batch_size, 1, frame_count * 2)
        lengths = torch.zeros(batch_size, dtype=torch.long)
        for slot in range(batch_size):
            if not bool(self._streaming_state.exec_mask[slot]):
                continue
            length = int(codes_lengths[slot])
            offset = int(self._streaming_state.offsets[slot])
            for frame in range(length):
                audio[slot, 0, frame * 2 : frame * 2 + 2] = offset + frame + 1
            lengths[slot] = length * 2
            self._streaming_state.offsets[slot] += length
        return SimpleNamespace(audio=audio, audio_lengths=lengths)


class _NativeCodec(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))
        self.calls: list[tuple[str, tuple[int, ...]]] = []

    def initialize_decoder_state_pool(
        self, state_capacity: int, scratch_capacity: int = 0
    ) -> None:
        self.calls.append(("initialize", (state_capacity, scratch_capacity)))

    def reset_decoder_state_slots(self, slot_ids: torch.Tensor) -> None:
        self.calls.append(("reset", tuple(slot_ids.tolist())))

    def close_decoder_state_pool(self) -> None:
        self.calls.append(("close", ()))

    def decode_streaming_batch(
        self,
        codes: torch.Tensor,
        lengths: torch.Tensor,
        slot_ids: torch.Tensor,
        valid_rows: torch.Tensor,
    ) -> SimpleNamespace:
        del lengths, slot_ids
        self.calls.append(("batch", tuple(codes.shape)))
        frame_values = codes.sum(dim=0).sum(dim=0).float()
        audio = frame_values.view(1, 1, -1).expand(codes.shape[1], -1, -1).contiguous()
        return SimpleNamespace(audio=audio, audio_lengths=valid_rows.long())

    def decode_streaming_tensors(
        self,
        codes: torch.Tensor,
        lengths: torch.Tensor,
        slot_ids: torch.Tensor,
        valid_rows: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        result = self.decode_streaming_batch(codes, lengths, slot_ids, valid_rows)
        return result.audio, result.audio_lengths


def _codes(values: list[list[int]]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.long).transpose(0, 1).contiguous()


def test_legacy_pool_supports_sparse_slots_and_reset_reuse() -> None:
    codec = _LegacyCodec()
    pool = MossAudioTokenizerStreamingStatePool(codec, n_vq=2)
    pool.initialize_decoder_state_pool(state_capacity=3, scratch_capacity=1)

    codes = torch.stack([_codes([[1, 2], [3, 4]]), _codes([[5, 6], [7, 8]])], dim=1)
    lengths = torch.tensor([2, 1], dtype=torch.long)
    slots = torch.tensor([2, 0], dtype=torch.long)
    valid = torch.ones(2, dtype=torch.bool)
    result = pool.decode_streaming_batch(codes, lengths, slots, valid)
    assert result.audio.shape == (2, 1, 4)
    assert result.audio_lengths.tolist() == [4, 2]
    assert result.audio[0, 0, :4].tolist() == [1.0, 1.0, 2.0, 2.0]

    # A scratch row must not advance state or leak stale output.
    scratch = pool.decode_streaming_batch(
        torch.zeros(2, 1, 1, dtype=torch.long),
        torch.zeros(1, dtype=torch.long),
        torch.tensor([3], dtype=torch.long),
        torch.tensor([False]),
    )
    assert scratch.audio_lengths.tolist() == [0]

    pool.reset_decoder_state_slots(torch.tensor([0], dtype=torch.long))
    reused = pool.decode_streaming_batch(
        torch.ones(2, 1, 1, dtype=torch.long),
        torch.ones(1, dtype=torch.long),
        torch.tensor([0], dtype=torch.long),
        torch.tensor([True]),
    )
    assert reused.audio[0, 0, :2].tolist() == [1.0, 1.0]
    assert reused.audio_lengths.tolist() == [2]
    pool.close_decoder_state_pool()
    assert codec._streaming_state is None


def test_pool_rejects_duplicate_valid_slots_and_scratch_leases() -> None:
    codec = _LegacyCodec()
    pool = MossAudioTokenizerStreamingStatePool(codec, n_vq=1)
    pool.initialize_decoder_state_pool(state_capacity=2, scratch_capacity=1)
    codes = torch.zeros(1, 2, 1, dtype=torch.long)
    lengths = torch.ones(2, dtype=torch.long)
    with pytest.raises(ValueError, match="unique state slots"):
        pool.decode_streaming_batch(
            codes,
            lengths,
            torch.tensor([0, 0], dtype=torch.long),
            torch.ones(2, dtype=torch.bool),
        )
    with pytest.raises(ValueError, match="real decoder state slots"):
        pool.decode_streaming_batch(
            codes[:, :1],
            lengths[:1],
            torch.tensor([2], dtype=torch.long),
            torch.tensor([True]),
        )
    pool.close_decoder_state_pool()


def test_native_pool_delegates_indexed_contract() -> None:
    codec = _NativeCodec()
    pool = MossAudioTokenizerStreamingStatePool(codec, n_vq=1)
    pool.initialize_decoder_state_pool(2, scratch_capacity=1)
    assert codec.calls == [("initialize", (2, 1))]
    output = pool.decode_streaming_batch(
        torch.ones(1, 1, 3, dtype=torch.long),
        torch.full((1,), 3, dtype=torch.long),
        torch.tensor([1], dtype=torch.long),
        torch.tensor([True]),
    )
    assert output.audio.shape == (1, 1, 3)
    assert ("batch", (1, 1, 3)) in codec.calls
    pool.reset_decoder_state_slots(torch.tensor([1], dtype=torch.long))
    pool.close_decoder_state_pool()
    assert codec.calls[-2:] == [("reset", (1,)), ("close", ())]


def _tiny_repository_vocoder() -> MossAudioTokenizerVocoder:
    config = {
        "sampling_rate": 8,
        "downsample_rate": 1,
        "number_channels": 1,
        "enable_channel_interleave": False,
        "quantizer_kwargs": {
            "input_dim": 4,
            "rvq_dim": 4,
            "output_dim": 4,
            "num_quantizers": 1,
            "codebook_size": 8,
            "codebook_dim": 4,
            "quantizer_type": "rlfq",
        },
        "decoder_kwargs": [
            {
                "module_type": "Transformer",
                "input_dimension": 4,
                "output_dimension": 4,
                "d_model": 4,
                "num_heads": 2,
                "num_layers": 1,
                "dim_feedforward": 8,
                "causal": True,
                "positional_embedding": "sin",
                "max_period": 10_000,
                "positional_scale": 1.0,
                "norm": "layer_norm",
                "gating": "none",
                "context_duration": 4.0,
            }
        ],
    }
    torch.manual_seed(7)
    model = MossAudioTokenizerVocoder(
        config,
        parameter_device="cpu",
        decoder_dtype=torch.float32,
        compute_dtype=torch.float32,
        attention_backend="sdpa",
    )
    model.eval()
    return model


def test_repository_codec_native_pool_keeps_compact_slots_isolated() -> None:
    model = _tiny_repository_vocoder()
    model.initialize_decoder_state_pool(state_capacity=3, scratch_capacity=1)
    try:
        first_codes = torch.tensor(
            [[[1, 2], [3, 4]]],
            dtype=torch.long,
        )
        second_codes = torch.tensor(
            [[[5, 6], [7, 0]]],
            dtype=torch.long,
        )
        slots = torch.tensor([2, 0], dtype=torch.long)
        valid = torch.ones(2, dtype=torch.bool)

        first = model.decode_streaming_batch(
            first_codes,
            torch.tensor([2, 2], dtype=torch.long),
            slots,
            valid,
        )
        second = model.decode_streaming_batch(
            second_codes,
            torch.tensor([2, 2], dtype=torch.long),
            slots,
            valid,
        )
        assert first.audio.shape[0] == 2
        assert second.audio.shape[0] == 2
        assert first.audio_lengths.tolist() == [2, 2]
        assert second.audio_lengths.tolist() == [2, 2]

        # A graph-padding row is returned as zero and must not advance slot 1.
        scratch = model.decode_streaming_batch(
            torch.zeros(1, 1, 1, dtype=torch.long),
            torch.zeros(1, dtype=torch.long),
            torch.tensor([1], dtype=torch.long),
            torch.tensor([False]),
        )
        assert scratch.audio_lengths.tolist() == [0]

        resumed = model.decode_streaming_batch(
            torch.ones(1, 1, 1, dtype=torch.long),
            torch.ones(1, dtype=torch.long),
            torch.tensor([1], dtype=torch.long),
            torch.tensor([True]),
        )
        model.reset_decoder_state_slots(torch.tensor([1], dtype=torch.long))
        reset = model.decode_streaming_batch(
            torch.ones(1, 1, 1, dtype=torch.long),
            torch.ones(1, dtype=torch.long),
            torch.tensor([1], dtype=torch.long),
            torch.tensor([True]),
        )
        assert resumed.audio_lengths.tolist() == [1]
        assert reset.audio_lengths.tolist() == [1]
        assert torch.allclose(resumed.audio, reset.audio)
    finally:
        model.close_decoder_state_pool()


def test_repository_codec_disposable_graph_scratch_matches_safe_active_rows() -> None:
    safe_model = _tiny_repository_vocoder()
    fast_model = _tiny_repository_vocoder()
    for model in (safe_model, fast_model):
        model.initialize_decoder_state_pool(state_capacity=1, scratch_capacity=1)
    try:
        slots = torch.tensor([0, 1], dtype=torch.long)
        valid = torch.tensor([True, False])
        lengths = torch.tensor([2, 2], dtype=torch.long)
        first_codes = torch.tensor([[[1, 2], [6, 7]]], dtype=torch.long)

        safe_audio, safe_lengths = safe_model.decode_streaming_tensors(
            first_codes,
            lengths,
            slots,
            valid,
        )
        fast_audio, fast_lengths = fast_model.decode_streaming_tensors(
            first_codes,
            lengths,
            slots,
            valid,
            scratch_rows_are_disposable=True,
        )

        assert safe_lengths.tolist() == fast_lengths.tolist() == [2, 0]
        assert torch.equal(safe_audio[0], fast_audio[0])
        assert torch.count_nonzero(fast_audio[1]) == 0

        # Mutating a non-leaseable scratch row must not affect the real row's
        # subsequent streaming state.
        next_codes = torch.tensor([[[3], [5]]], dtype=torch.long)
        next_lengths = torch.ones(2, dtype=torch.long)
        safe_next, _ = safe_model.decode_streaming_tensors(
            next_codes,
            next_lengths,
            slots,
            valid,
        )
        fast_next, _ = fast_model.decode_streaming_tensors(
            next_codes,
            next_lengths,
            slots,
            valid,
            scratch_rows_are_disposable=True,
        )
        assert torch.equal(safe_next[0], fast_next[0])
    finally:
        safe_model.close_decoder_state_pool()
        fast_model.close_decoder_state_pool()
