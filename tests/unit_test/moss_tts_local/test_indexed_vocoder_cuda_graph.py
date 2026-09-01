# SPDX-License-Identifier: Apache-2.0
"""Contract tests for the native indexed vocoder CUDA-graph runner."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sglang_omni.models.moss_tts_local.indexed_vocoder_cuda_graph import (
    MossIndexedVocoderCudaGraphRunner,
)


class _CpuCodec(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))


class _GraphCodec(_CpuCodec):
    def __init__(self) -> None:
        super().__init__()
        self.disposable_flags: list[bool] = []

    def decode_streaming_tensors(
        self,
        codes,
        codes_lengths,
        state_slot_ids,
        valid_rows,
        *,
        scratch_rows_are_disposable=False,
    ):
        del state_slot_ids
        self.disposable_flags.append(bool(scratch_rows_are_disposable))
        audio = codes.sum(dim=0, keepdim=False).to(torch.float32).unsqueeze(1)
        return audio, codes_lengths.masked_fill(~valid_rows, 0)


def test_runner_rejects_graph_buckets_without_scratch_rows() -> None:
    with pytest.raises(ValueError, match="scratch_capacity"):
        MossIndexedVocoderCudaGraphRunner(
            _CpuCodec(),
            real_state_capacity=4,
            scratch_capacity=1,
            batch_sizes=[1, 2],
            frame_sizes=[5],
            num_quantizers=32,
        )


def test_runner_is_eager_only_on_cpu() -> None:
    runner = MossIndexedVocoderCudaGraphRunner(
        _CpuCodec(),
        real_state_capacity=4,
        scratch_capacity=2,
        batch_sizes=[1, 2],
        frame_sizes=[5, 25],
        num_quantizers=32,
    )
    assert runner.warmup() == []
    assert runner.capture_sizes == []
    codes = torch.zeros(32, 1, 5, dtype=torch.long)
    slots = torch.zeros(1, dtype=torch.long)
    assert runner.decode_step(codes, slots) is None


def test_compile_adapter_is_opt_in_and_dynamic(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_compile(module, **kwargs):
        calls.append(kwargs)
        return module

    monkeypatch.setattr(torch, "compile", fake_compile)
    runner = MossIndexedVocoderCudaGraphRunner(
        _CpuCodec(),
        real_state_capacity=2,
        scratch_capacity=1,
        batch_sizes=[1],
        frame_sizes=[5],
        num_quantizers=4,
        compile_decode=True,
        compile_mode="default",
    )

    compiled = runner._ensure_compiled_decode()

    assert compiled is not None
    assert runner.compile_requested is True
    assert calls == [{"dynamic": True, "mode": "default"}]
    assert runner.compiled_capture_sizes == []


def test_compile_adapter_can_target_fixed_capture_shapes(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_compile(module, **kwargs):
        calls.append(kwargs)
        return module

    monkeypatch.setattr(torch, "compile", fake_compile)
    runner = MossIndexedVocoderCudaGraphRunner(
        _CpuCodec(),
        real_state_capacity=16,
        scratch_capacity=16,
        batch_sizes=[8, 12, 16],
        frame_sizes=[5, 25],
        num_quantizers=4,
        compile_decode=True,
        compile_decode_shapes=[(12, 5)],
        compile_mode="default",
    )

    compiled = runner._ensure_compiled_decode()

    assert compiled is not None
    assert runner.compile_shapes == [(12, 5)]
    assert runner._compile_shape_enabled(12, 5) is True
    assert runner._compile_shape_enabled(8, 5) is False
    assert runner._compile_shape_enabled(12, 25) is False
    assert calls == [{"dynamic": False, "mode": "default"}]


@pytest.mark.parametrize("shapes", [[(0, 5)], [(12, 0)], [(4, 5)]])
def test_compile_shapes_must_be_positive_and_in_capture_grid(shapes) -> None:
    with pytest.raises(ValueError, match="compile_decode_shapes"):
        MossIndexedVocoderCudaGraphRunner(
            _CpuCodec(),
            real_state_capacity=16,
            scratch_capacity=16,
            batch_sizes=[8, 12, 16],
            frame_sizes=[5, 25],
            num_quantizers=4,
            compile_decode=True,
            compile_decode_shapes=shapes,
        )


def test_graph_adapter_marks_padding_scratch_state_disposable() -> None:
    codec = _GraphCodec()
    runner = MossIndexedVocoderCudaGraphRunner(
        codec,
        real_state_capacity=2,
        scratch_capacity=2,
        batch_sizes=[2],
        frame_sizes=[3],
        num_quantizers=4,
    )
    codes = torch.ones(4, 2, 3, dtype=torch.long)
    lengths = torch.full((2,), 3, dtype=torch.long)
    slots = torch.tensor([0, 2], dtype=torch.long)
    valid = torch.tensor([True, False])

    audio, audio_lengths = runner._graph_decode(codes, lengths, slots, valid)

    assert codec.disposable_flags == [True]
    assert audio.shape == (2, 1, 3)
    assert audio_lengths.tolist() == [3, 0]


def test_staging_shrunk_batch_restores_retired_rows_to_scratch_slots() -> None:
    entry = SimpleNamespace(
        static_state_slot_ids=torch.tensor([4, 5, 6, 7], dtype=torch.long),
        scratch_state_slot_ids=torch.tensor([4, 5, 6, 7], dtype=torch.long),
        static_valid_rows=torch.zeros(4, dtype=torch.bool),
        active_batch_size=0,
    )

    MossIndexedVocoderCudaGraphRunner._stage_active_rows(
        entry,
        torch.tensor([0, 1, 2, 3], dtype=torch.long),
    )
    assert entry.static_state_slot_ids.tolist() == [0, 1, 2, 3]
    assert entry.static_valid_rows.tolist() == [True, True, True, True]

    MossIndexedVocoderCudaGraphRunner._stage_active_rows(
        entry,
        torch.tensor([1, 0], dtype=torch.long),
    )
    assert entry.static_state_slot_ids.tolist() == [1, 0, 6, 7]
    assert entry.static_valid_rows.tolist() == [True, True, False, False]

    MossIndexedVocoderCudaGraphRunner._stage_active_rows(
        entry,
        torch.tensor([2, 3, 0], dtype=torch.long),
    )
    assert entry.static_state_slot_ids.tolist() == [2, 3, 0, 7]
    assert entry.static_valid_rows.tolist() == [True, True, True, False]
