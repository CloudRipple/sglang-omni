# SPDX-License-Identifier: Apache-2.0
"""Shared 32-bit seeded hashing for Ascend kernels."""

import triton
import triton.language as tl


@triton.jit
def mix32(value: tl.tensor, key: tl.tensor) -> tl.tensor:
    # note (cocoa): Masks prevent sign extension in Ascend right-shift lowering.
    key = (key * 0xCC9E2D51).to(tl.uint32)
    key = (key << 15) | ((key >> 17) & 0x7FFF)
    key = (key * 0x1B873593).to(tl.uint32)
    value = value ^ key
    value = (value << 13) | ((value >> 19) & 0x1FFF)
    return (value * 5 + 0xE6546B64).to(tl.uint32)


@triton.jit
def hash32_values(
    seeds: tl.tensor, positions: tl.tensor, row: tl.tensor, column: tl.tensor
) -> tl.tensor:
    low = tl.load(seeds + row * 2).to(tl.uint32, bitcast=True)
    high = tl.load(seeds + row * 2 + 1).to(tl.uint32, bitcast=True)
    position = tl.load(positions + row * 2).to(tl.uint32, bitcast=True)
    value = tl.full(column.shape, 0, tl.uint32)
    value = mix32(value, low)
    value = mix32(value, high)
    value = mix32(value, position)
    value = mix32(value, column.to(tl.uint32)) ^ 16
    value = value ^ ((value >> 16) & 0xFFFF)
    value = (value * 0x85EBCA6B).to(tl.uint32)
    value = value ^ ((value >> 13) & 0x7FFFF)
    value = (value * 0xC2B2AE35).to(tl.uint32)
    return value ^ ((value >> 16) & 0xFFFF)
