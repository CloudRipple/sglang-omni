# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from sglang_omni.models.moss_tts.vocoder_cuda_graph import (
    MossTTSDelayVocoderCudaGraphRunner,
    VocoderGraphKey,
    make_vocoder_cuda_graph_keys,
)


def test_vocoder_cuda_graph_keys_bucket_batch_and_frames() -> None:
    keys = make_vocoder_cuda_graph_keys(max_batch_size=8, max_frames=80)

    assert keys == tuple(
        VocoderGraphKey(batch_size=batch_size, frames=frames)
        for batch_size in (1, 2, 4, 8)
        for frames in (32, 48, 64, 80)
    )


def test_vocoder_cuda_graph_keys_include_non_aligned_limits() -> None:
    assert make_vocoder_cuda_graph_keys(max_batch_size=3, max_frames=33) == (
        VocoderGraphKey(1, 32),
        VocoderGraphKey(1, 33),
        VocoderGraphKey(2, 32),
        VocoderGraphKey(2, 33),
        VocoderGraphKey(3, 32),
        VocoderGraphKey(3, 33),
    )


def test_vocoder_cuda_graph_selects_smallest_covering_bucket() -> None:
    runner = object.__new__(MossTTSDelayVocoderCudaGraphRunner)
    runner._graphs = {
        VocoderGraphKey(batch_size=batch_size, frames=frames): None
        for batch_size in (1, 2, 4, 8)
        for frames in (32, 48, 64)
    }

    assert runner._select_key(3, 49) == VocoderGraphKey(4, 64)
    assert runner._select_key(1, 33) == VocoderGraphKey(1, 48)
    assert runner._select_key(9, 32) is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_batch_size": 0, "max_frames": 32},
        {"max_batch_size": 1, "max_frames": 0},
    ],
)
def test_vocoder_cuda_graph_keys_reject_non_positive_limits(kwargs) -> None:
    with pytest.raises(ValueError):
        make_vocoder_cuda_graph_keys(**kwargs)
