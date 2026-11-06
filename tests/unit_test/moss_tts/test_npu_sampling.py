# SPDX-License-Identifier: Apache-2.0
"""Ascend seeded sampling parity with scalar Murmur3 reference vectors."""

import pytest
import torch

from sglang_omni.models.moss_tts.sampling_kernels import (
    multinomial_with_seed,
    sample_seeded_branchless,
    sample_seeded_npu,
)
from sglang_omni.platforms import current_platform


@pytest.mark.accelerator
@pytest.mark.skipif(not current_platform.is_npu(), reason="requires Ascend NPU")
@pytest.mark.parametrize("vocab", [2, 1024])
@torch.no_grad()
def test_seeded_sampling_matches_reference_after_graph_updates(vocab: int) -> None:
    """The previous NPU hash changed columns through unsigned vector widening."""
    device = torch.device("npu:0")
    device_module = torch.get_device_module(device)
    backend = current_platform.get_device_graph_backend(device)
    assert backend is not None
    scores = (
        torch.zeros(4, vocab)
        if vocab == 2
        else torch.linspace(-3, 3, vocab).repeat(4, 1)
    )
    if vocab == 2:
        scores[0] = torch.tensor([-100.0, 0.0])
    else:
        pass
    scores = scores.T.contiguous().to(device).T
    seeds = torch.empty((4, 2), device=device, dtype=torch.int64)[:, 0]
    positions = torch.empty_like(seeds)
    cases = [
        (
            [0, 42, -1, 0x123456789ABCDEF],
            [1_707_985_137, 13, 2**32 + 9, 19],
            [1, 1, 0, 1] if vocab == 2 else [0, 1012, 905, 1020],
        ),
        (
            [42, 17, 2**63 - 1, -(2**63)],
            [26, 3, 39, 52],
            [1, 0, 1, 1] if vocab == 2 else [960, 968, 991, 951],
        ),
    ]
    seeds.copy_(torch.tensor(cases[0][0]))
    positions.copy_(torch.tensor(cases[0][1]))
    stream = device_module.Stream()
    stream.wait_stream(device_module.current_stream())
    with device_module.stream(stream):
        multinomial_with_seed(scores, seeds, positions)
    device_module.current_stream().wait_stream(stream)
    with backend.capture(stream=stream) as graph:
        captured = multinomial_with_seed(scores, seeds, positions)
    for seed_values, position_values, expected in cases:
        seeds.copy_(torch.tensor(seed_values))
        positions.copy_(torch.tensor(position_values))
        actual = multinomial_with_seed(scores, seeds, positions)
        graph.replay()
        expected = torch.tensor(expected, dtype=torch.int64).view(-1, 1)
        assert torch.equal(actual.cpu(), expected)
        assert torch.equal(captured.cpu(), expected)


@pytest.mark.accelerator
@pytest.mark.skipif(not current_platform.is_npu(), reason="requires Ascend NPU")
@pytest.mark.parametrize("batch_size,vocab_size", [(1, 2), (3, 17), (16, 1024)])
@torch.no_grad()
def test_fused_sampling_matches_native_after_graph_updates(
    batch_size: int, vocab_size: int
) -> None:
    """Fusion must preserve mixed sampling controls, ties and updated graph inputs."""
    device = torch.device("npu:0")
    device_module = torch.get_device_module(device)
    backend = current_platform.get_device_graph_backend(device)
    assert backend is not None
    torch.manual_seed(42)
    logits = torch.randn(vocab_size, batch_size, device=device).T
    seeds = torch.arange(batch_size, device=device, dtype=torch.int64)
    positions = torch.full(
        (batch_size,), 1_707_985_137, device=device, dtype=torch.int64
    )
    temperature = torch.full((batch_size,), 1.7, device=device)
    top_p = torch.full((batch_size,), 0.8, device=device)
    top_k = torch.full((batch_size,), 25, device=device, dtype=torch.int64)
    controls = dict(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        seeds=seeds,
        positions=positions,
    )
    for _ in range(3):
        sample_seeded_npu(logits, **controls)
    device_module.synchronize()
    with backend.capture() as graph:
        captured = sample_seeded_npu(logits, **controls)
    for iteration in range(3):
        logits.normal_()
        if iteration == 1:
            logits.zero_()
        else:
            pass
        temperature.fill_(1.7)
        temperature[1::3] = 0
        top_p.fill_((1.0, 0.8, 0.1)[iteration])
        top_k.fill_((0, 25, 1)[iteration])
        if vocab_size == 2 and iteration < 2:
            # note (Zhang Yiyang): These seeds hash token zero at position zero to UINT32_MAX and zero.
            seeds[0] = (3_666_503_354, 1_460_908_073)[iteration]
            positions[0] = 0
            logits[0] = logits.new_tensor(([-100.0, 0.0], [0.0, -100.0])[iteration])
            temperature[0] = 1.0
            top_p[0] = 1.0
            top_k[0] = 0
        else:
            pass
        expected = sample_seeded_branchless(logits, **controls)
        actual = sample_seeded_npu(logits, **controls)
        graph.replay()
        assert torch.equal(actual, expected)
        assert torch.equal(captured, expected)
        seeds.add_(2**32 + 5)
        positions.add_(13)
