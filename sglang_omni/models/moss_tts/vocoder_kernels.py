# SPDX-License-Identifier: Apache-2.0
"""Shared inference kernels for the MOSS audio-tokenizer vocoder."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - depends on runtime image
    triton = None
    tl = None


_EXACT_ROPE_BLOCK_SIZE = 256
_EXACT_ROPE_NUM_WARPS = 4
_SPARSE_KV_MAX_BLOCK_T = 16
_SPARSE_KV_NUM_WARPS = 4
_SPARSE_KV_GATHER_BLOCK_SIZE = 1024
_SPARSE_KV_GATHER_NUM_WARPS = 8


if triton is not None:

    @triton.jit
    def _sparse_ring_kv_commit_kernel(
        cached_keys,
        cached_values,
        state_slot_ids,
        write_indices,
        current_k,
        current_v,
        stride_kb: tl.constexpr,
        stride_kh: tl.constexpr,
        stride_kt: tl.constexpr,
        stride_kd: tl.constexpr,
        stride_vb: tl.constexpr,
        stride_vh: tl.constexpr,
        stride_vt: tl.constexpr,
        stride_vd: tl.constexpr,
        num_heads: tl.constexpr,
        chunk_length: tl.constexpr,
        cache_capacity: tl.constexpr,
        head_dim: tl.constexpr,
        block_t: tl.constexpr,
    ):
        batch_head = tl.program_id(0)
        time_block = tl.program_id(1)
        batch = batch_head // num_heads
        head = batch_head % num_heads
        times = time_block * block_t + tl.arange(0, block_t)
        dims = tl.arange(0, head_dim)
        mask = times[:, None] < chunk_length

        slot = tl.load(state_slot_ids + batch)
        positions = tl.load(
            write_indices + batch * chunk_length + times,
            mask=times < chunk_length,
            other=0,
        )
        cache_offsets = (
            ((slot * num_heads + head) * cache_capacity + positions[:, None])
            * head_dim
            + dims[None, :]
        )
        key_offsets = (
            batch * stride_kb
            + head * stride_kh
            + times[:, None] * stride_kt
            + dims[None, :] * stride_kd
        )
        value_offsets = (
            batch * stride_vb
            + head * stride_vh
            + times[:, None] * stride_vt
            + dims[None, :] * stride_vd
        )
        keys = tl.load(current_k + key_offsets, mask=mask)
        values = tl.load(current_v + value_offsets, mask=mask)
        tl.store(cached_keys + cache_offsets, keys, mask=mask)
        tl.store(cached_values + cache_offsets, values, mask=mask)

    @triton.jit
    def _sparse_ring_kv_prewrite_gather_kernel(
        cached_keys,
        cached_values,
        state_slot_ids,
        write_indices,
        current_k,
        current_v,
        row_keys,
        row_values,
        stride_kb: tl.constexpr,
        stride_kh: tl.constexpr,
        stride_kt: tl.constexpr,
        stride_kd: tl.constexpr,
        stride_vb: tl.constexpr,
        stride_vh: tl.constexpr,
        stride_vt: tl.constexpr,
        stride_vd: tl.constexpr,
        num_heads: tl.constexpr,
        chunk_length: tl.constexpr,
        cache_capacity: tl.constexpr,
        head_dim: tl.constexpr,
        block_size: tl.constexpr,
    ):
        batch_head = tl.program_id(0)
        batch = batch_head // num_heads
        head = batch_head % num_heads
        elements = tl.program_id(1) * block_size + tl.arange(0, block_size)
        cache_elements: tl.constexpr = cache_capacity * head_dim
        valid = elements < cache_elements
        cache_time = elements // head_dim
        dim = elements % head_dim

        slot = tl.load(state_slot_ids + batch)
        write_start = tl.load(write_indices + batch * chunk_length)
        relative_time = (cache_time - write_start + cache_capacity) % cache_capacity
        is_current = valid & (relative_time < chunk_length)

        cache_offsets = (
            ((slot * num_heads + head) * cache_capacity + cache_time) * head_dim
            + dim
        )
        row_offsets = batch_head * cache_elements + elements
        key_offsets = (
            batch * stride_kb
            + head * stride_kh
            + relative_time * stride_kt
            + dim * stride_kd
        )
        value_offsets = (
            batch * stride_vb
            + head * stride_vh
            + relative_time * stride_vt
            + dim * stride_vd
        )

        old_keys = tl.load(cached_keys + cache_offsets, mask=valid & ~is_current)
        old_values = tl.load(cached_values + cache_offsets, mask=valid & ~is_current)
        new_keys = tl.load(current_k + key_offsets, mask=is_current)
        new_values = tl.load(current_v + value_offsets, mask=is_current)
        keys = tl.where(is_current, new_keys, old_keys)
        values = tl.where(is_current, new_values, old_values)

        tl.store(row_keys + row_offsets, keys, mask=valid)
        tl.store(row_values + row_offsets, values, mask=valid)
        tl.store(cached_keys + cache_offsets, new_keys, mask=is_current)
        tl.store(cached_values + cache_offsets, new_values, mask=is_current)

else:
    _sparse_ring_kv_commit_kernel = None
    _sparse_ring_kv_prewrite_gather_kernel = None


if triton is not None and hasattr(tl, "inline_asm_elementwise"):

    @triton.jit
    def _exact_interleaved_rope_kernel(
        q,
        k,
        cos_sin,
        positions,
        total_pairs,
        stride_qt,
        stride_qh,
        stride_kt,
        stride_kh,
        num_heads: tl.constexpr,
        head_dim: tl.constexpr,
        block_size: tl.constexpr,
    ):
        half_dim: tl.constexpr = head_dim // 2
        offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
        mask = offsets < total_pairs
        token = offsets // (num_heads * half_dim)
        pair_in_token = offsets % (num_heads * half_dim)
        head = pair_in_token // half_dim
        pair = pair_in_token % half_dim
        q_base = q + token * stride_qt + head * stride_qh + pair * 2
        k_base = k + token * stride_kt + head * stride_kh + pair * 2
        position = tl.load(positions + token, mask=mask, other=0)
        cos = tl.load(
            cos_sin + position * head_dim + pair,
            mask=mask,
        ).to(tl.float32)
        sin = tl.load(
            cos_sin + position * head_dim + half_dim + pair,
            mask=mask,
        ).to(tl.float32)
        qr = tl.load(q_base, mask=mask).to(tl.float32)
        qi = tl.load(q_base + 1, mask=mask).to(tl.float32)
        kr = tl.load(k_base, mask=mask).to(tl.float32)
        ki = tl.load(k_base + 1, mask=mask).to(tl.float32)

        # Note (Zhang Yiyang): The source implementation materializes each FP32
        # multiply before the add/subtract. Explicit rounding prevents contraction
        # into FMAs and keeps the fused kernel bitwise-equivalent after the
        # low-precision store.
        qor, qoi, kor, koi = tl.inline_asm_elementwise(
            asm="""
            {
                .reg .f32 a;
                .reg .f32 b;
                mul.rn.f32 a, $4, $8;
                mul.rn.f32 b, $5, $9;
                sub.rn.f32 $0, a, b;
                mul.rn.f32 a, $4, $9;
                mul.rn.f32 b, $5, $8;
                add.rn.f32 $1, a, b;
                mul.rn.f32 a, $6, $8;
                mul.rn.f32 b, $7, $9;
                sub.rn.f32 $2, a, b;
                mul.rn.f32 a, $6, $9;
                mul.rn.f32 b, $7, $8;
                add.rn.f32 $3, a, b;
            }
            """,
            constraints="=f,=f,=f,=f,f,f,f,f,f,f",
            args=[qr, qi, kr, ki, cos, sin],
            dtype=(tl.float32, tl.float32, tl.float32, tl.float32),
            is_pure=True,
            pack=1,
        )
        tl.store(q_base, qor, mask=mask)
        tl.store(q_base + 1, qoi, mask=mask)
        tl.store(k_base, kor, mask=mask)
        tl.store(k_base + 1, koi, mask=mask)

else:
    _exact_interleaved_rope_kernel = None


def apply_exact_interleaved_rope_inplace(
    q: torch.Tensor,
    k: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    position_ids: torch.Tensor,
) -> bool:
    """Apply source-equivalent interleaved RoPE in one CUDA kernel if supported."""

    if (
        _exact_interleaved_rope_kernel is None
        or torch.version.hip is not None
        or q.device.type != "cuda"
        or k.device != q.device
        or cos_sin_cache.device != q.device
        or position_ids.device != q.device
        or q.dtype not in (torch.bfloat16, torch.float16)
        or k.dtype != q.dtype
        or cos_sin_cache.dtype != torch.float32
        or position_ids.dtype not in (torch.int32, torch.int64)
        or q.ndim != 3
        or k.shape != q.shape
        or cos_sin_cache.ndim != 2
        or int(cos_sin_cache.shape[0]) == 0
        or int(cos_sin_cache.shape[1]) != int(q.shape[2])
        or position_ids.ndim != 1
        or int(position_ids.numel()) != int(q.shape[0])
        or q.stride(2) != 1
        or k.stride(2) != 1
        or position_ids.stride(0) != 1
        or not cos_sin_cache.is_contiguous()
    ):
        return False

    tokens, num_heads, head_dim = map(int, q.shape)
    if tokens == 0 or num_heads <= 0 or head_dim <= 0 or head_dim % 2 != 0:
        return False
    total_pairs = tokens * num_heads * (head_dim // 2)
    block_size = _EXACT_ROPE_BLOCK_SIZE
    _exact_interleaved_rope_kernel[(triton.cdiv(total_pairs, block_size),)](
        q,
        k,
        cos_sin_cache,
        position_ids,
        total_pairs,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        num_heads=num_heads,
        head_dim=head_dim,
        block_size=block_size,
        num_warps=_EXACT_ROPE_NUM_WARPS,
    )
    return True


def commit_sparse_ring_kv_inplace(
    cached_keys: torch.Tensor,
    cached_values: torch.Tensor,
    state_slot_ids: torch.Tensor,
    write_indices: torch.Tensor,
    current_k: torch.Tensor,
    current_v: torch.Tensor,
) -> bool:
    """Commit strided K/V ring cells in one CUDA kernel when supported."""

    if (
        _sparse_ring_kv_commit_kernel is None
        or torch.version.hip is not None
        or cached_keys.device.type != "cuda"
        or cached_values.device != cached_keys.device
        or state_slot_ids.device != cached_keys.device
        or write_indices.device != cached_keys.device
        or current_k.device != cached_keys.device
        or current_v.device != cached_keys.device
        or cached_keys.dtype not in (torch.bfloat16, torch.float16)
        or cached_values.dtype != cached_keys.dtype
        or current_k.dtype != cached_keys.dtype
        or current_v.dtype != cached_keys.dtype
        or cached_keys.ndim != 4
        or cached_values.shape != cached_keys.shape
        or current_k.ndim != 4
        or current_v.shape != current_k.shape
        or state_slot_ids.ndim != 1
        or write_indices.ndim != 2
        or state_slot_ids.dtype != torch.long
        or write_indices.dtype != torch.long
        or not cached_keys.is_contiguous()
        or not cached_values.is_contiguous()
        or not write_indices.is_contiguous()
        or state_slot_ids.stride(0) != 1
        or current_k.stride(3) != 1
        or current_v.stride(3) != 1
    ):
        return False

    batch_size, num_heads, chunk_length, head_dim = map(int, current_k.shape)
    if (
        batch_size <= 0
        or num_heads <= 0
        or chunk_length <= 0
        or head_dim <= 0
        or head_dim > 256
        or head_dim & (head_dim - 1)
        or cached_keys.shape[1] != num_heads
        or cached_keys.shape[3] != head_dim
        or cached_keys.shape[2] <= 0
        or state_slot_ids.shape != (batch_size,)
        or write_indices.shape != (batch_size, chunk_length)
    ):
        return False

    cache_capacity = int(cached_keys.shape[2])
    block_t = min(triton.next_power_of_2(chunk_length), _SPARSE_KV_MAX_BLOCK_T)
    grid = (batch_size * num_heads, triton.cdiv(chunk_length, block_t))
    _sparse_ring_kv_commit_kernel[grid](
        cached_keys,
        cached_values,
        state_slot_ids,
        write_indices,
        current_k,
        current_v,
        current_k.stride(0),
        current_k.stride(1),
        current_k.stride(2),
        current_k.stride(3),
        current_v.stride(0),
        current_v.stride(1),
        current_v.stride(2),
        current_v.stride(3),
        num_heads=num_heads,
        chunk_length=chunk_length,
        cache_capacity=cache_capacity,
        head_dim=head_dim,
        block_t=block_t,
        num_warps=_SPARSE_KV_NUM_WARPS,
    )
    return True


def prewrite_and_gather_sparse_ring_kv(
    cached_keys: torch.Tensor,
    cached_values: torch.Tensor,
    state_slot_ids: torch.Tensor,
    write_indices: torch.Tensor,
    current_k: torch.Tensor,
    current_v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Prewrite current K/V and gather both persistent rows in one kernel."""

    if (
        _sparse_ring_kv_prewrite_gather_kernel is None
        or torch.version.hip is not None
        or cached_keys.device.type != "cuda"
        or cached_values.device != cached_keys.device
        or state_slot_ids.device != cached_keys.device
        or write_indices.device != cached_keys.device
        or current_k.device != cached_keys.device
        or current_v.device != cached_keys.device
        or cached_keys.dtype not in (torch.bfloat16, torch.float16)
        or cached_values.dtype != cached_keys.dtype
        or current_k.dtype != cached_keys.dtype
        or current_v.dtype != cached_keys.dtype
        or cached_keys.ndim != 4
        or cached_values.shape != cached_keys.shape
        or current_k.ndim != 4
        or current_v.shape != current_k.shape
        or state_slot_ids.ndim != 1
        or write_indices.ndim != 2
        or state_slot_ids.dtype != torch.long
        or write_indices.dtype != torch.long
        or not cached_keys.is_contiguous()
        or not cached_values.is_contiguous()
        or not write_indices.is_contiguous()
        or state_slot_ids.stride(0) != 1
        or current_k.stride(3) != 1
        or current_v.stride(3) != 1
    ):
        return None

    batch_size, num_heads, chunk_length, head_dim = map(int, current_k.shape)
    if (
        batch_size <= 0
        or num_heads <= 0
        or chunk_length <= 0
        or head_dim <= 0
        or head_dim > 256
        or head_dim & (head_dim - 1)
        or cached_keys.shape[1] != num_heads
        or cached_keys.shape[3] != head_dim
        or cached_keys.shape[2] < chunk_length
        or state_slot_ids.shape != (batch_size,)
        or write_indices.shape != (batch_size, chunk_length)
    ):
        return None

    cache_capacity = int(cached_keys.shape[2])
    row_keys = torch.empty(
        (batch_size, num_heads, cache_capacity, head_dim),
        device=cached_keys.device,
        dtype=cached_keys.dtype,
    )
    row_values = torch.empty_like(row_keys)
    cache_elements = cache_capacity * head_dim
    block_size = min(
        triton.next_power_of_2(cache_elements),
        _SPARSE_KV_GATHER_BLOCK_SIZE,
    )
    num_warps = (
        4
        if block_size < _SPARSE_KV_GATHER_BLOCK_SIZE
        else _SPARSE_KV_GATHER_NUM_WARPS
    )
    grid = (batch_size * num_heads, triton.cdiv(cache_elements, block_size))
    _sparse_ring_kv_prewrite_gather_kernel[grid](
        cached_keys,
        cached_values,
        state_slot_ids,
        write_indices,
        current_k,
        current_v,
        row_keys,
        row_values,
        current_k.stride(0),
        current_k.stride(1),
        current_k.stride(2),
        current_k.stride(3),
        current_v.stride(0),
        current_v.stride(1),
        current_v.stride(2),
        current_v.stride(3),
        num_heads=num_heads,
        chunk_length=chunk_length,
        cache_capacity=cache_capacity,
        head_dim=head_dim,
        block_size=block_size,
        num_warps=num_warps,
    )
    return row_keys, row_values


__all__ = [
    "apply_exact_interleaved_rope_inplace",
    "commit_sparse_ring_kv_inplace",
    "prewrite_and_gather_sparse_ring_kv",
]
