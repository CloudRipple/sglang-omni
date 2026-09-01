# SPDX-License-Identifier: Apache-2.0
"""Correctness tests for MOSS-TTS sampling kernels."""

from __future__ import annotations

import pytest
import torch
from sglang.kernels.ops.sampling.murmur_hash import murmur_hash32
from sglang.srt.layers.sampler import multinomial_with_seed

from sglang_omni.models.moss_tts.sampling_kernels import (
    sample_seeded_branchless,
    sample_seeded_compact_topk,
    seeded_gumbel_argmax,
)

pytestmark = pytest.mark.accelerator

_UINT32_MAX_HASH_POSITION = 1_707_985_137


@pytest.mark.parametrize("equal_scores", [False, True], ids=["random", "tied"])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_seeded_gumbel_argmax_matches_production_shape(
    equal_scores: bool,
) -> None:
    device = torch.device("cuda")
    rows, vocab = 32, 1025
    scores = (
        torch.zeros(rows, vocab, device=device, dtype=torch.float32)
        if equal_scores
        else torch.randn(rows, vocab, device=device, dtype=torch.float32)
    )
    seeds = torch.tensor([20260720], device=device, dtype=torch.long).expand(rows)
    positions = torch.arange(rows, device=device, dtype=torch.long)
    output = torch.empty(rows, device=device, dtype=torch.long)

    expected = multinomial_with_seed(scores, seeds, positions).view(-1)
    actual = seeded_gumbel_argmax(scores, seeds, positions, output)
    torch.cuda.synchronize()

    assert seeds.stride(0) == 0
    assert torch.equal(expected, actual)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_seeded_gumbel_argmax_matches_uint32_max_hash() -> None:
    device = torch.device("cuda")
    scores = torch.tensor([[-100.0, 0.0]], device=device)
    seeds = torch.tensor([0], device=device, dtype=torch.long)
    # This position makes MurmurHash(seed=0, position, token_id=0) UINT32_MAX.
    positions = torch.tensor(
        [_UINT32_MAX_HASH_POSITION], device=device, dtype=torch.long
    )
    output = torch.empty(1, device=device, dtype=torch.long)

    hashes = murmur_hash32(
        seeds.to(torch.uint64),
        positions,
        torch.arange(scores.shape[1], device=device),
    )
    expected = multinomial_with_seed(scores, seeds, positions).view(-1)
    actual = seeded_gumbel_argmax(scores, seeds, positions, output)
    torch.cuda.synchronize()

    assert hashes[0, 0].item() == torch.iinfo(torch.uint32).max
    assert expected.item() == 0
    assert torch.equal(expected, actual)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_seeded_gumbel_argmax_rejects_strided_output() -> None:
    device = torch.device("cuda")
    rows, vocab = 2, 16
    scores = torch.randn(rows, vocab, device=device, dtype=torch.float32)
    seeds = torch.arange(rows, device=device, dtype=torch.long)
    positions = torch.arange(rows, device=device, dtype=torch.long)
    output = torch.empty(rows * 2, device=device, dtype=torch.long)[::2]

    assert output.stride(0) == 2
    with pytest.raises(ValueError, match="output must have stride 1"):
        seeded_gumbel_argmax(scores, seeds, positions, output)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_compact_topk_preserves_seeded_default_distribution() -> None:
    device = torch.device("cuda")
    torch.manual_seed(20260821)
    logits = torch.randn(8, 1024, device=device, dtype=torch.float32)
    temperature = torch.full((8,), 1.7, device=device)
    top_p = torch.full((8,), 0.8, device=device)
    top_k = torch.full((8,), 25, device=device, dtype=torch.long)
    seeds = torch.arange(8, device=device, dtype=torch.long) + 17
    positions = torch.arange(8, device=device, dtype=torch.long) * 13

    expected = sample_seeded_branchless(
        logits,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        seeds=seeds,
        positions=positions,
    )
    actual = sample_seeded_compact_topk(
        logits,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        seeds=seeds,
        positions=positions,
    )
    torch.cuda.synchronize()
    assert torch.equal(expected, actual)
