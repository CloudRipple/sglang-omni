# SPDX-License-Identifier: Apache-2.0
"""Ascend Triton kernels for MOSS-TTS and MOSS-Audio-Tokenizer."""

try:
    import triton
    import triton.language as tl

except ImportError:
    triton = None
    tl = None

if triton is not None:

    @triton.jit
    def reset_streaming_state_npu_kernel(
        position_words: tl.tensor,
        offset_words: tl.tensor,
        exec_mask: tl.tensor,
        slot_words: tl.tensor,
        context: tl.constexpr,
        block_size: tl.constexpr,
    ) -> None:
        slot = tl.load(slot_words + tl.program_id(0).to(tl.int64) * 2)
        block = tl.program_id(1)
        index = block * block_size + tl.arange(0, block_size)
        row = position_words + slot.to(tl.int64) * context * 2
        tl.store(row + index, -1, index < context * 2)
        if block == 0:
            tl.store(offset_words + slot.to(tl.int64) * 2 + tl.arange(0, 2), 0)
            tl.store(exec_mask + slot, True)
        else:
            pass

    @triton.jit
    def exact_interleaved_rope_npu_kernel(
        q: tl.tensor,
        k: tl.tensor,
        cos_sin: tl.tensor,
        position_words: tl.tensor,
        q_token_stride: tl.constexpr,
        k_token_stride: tl.constexpr,
        position_word_stride: tl.constexpr,
        num_heads: tl.constexpr,
        head_dim: tl.constexpr,
        head_block_size: tl.constexpr,
    ) -> None:
        token = tl.program_id(0)
        heads = tl.arange(0, head_block_size)
        dimensions = tl.arange(0, head_dim)
        pairs = tl.arange(0, head_dim // 2)
        position = tl.load(position_words + token.to(tl.int64) * position_word_stride)
        cache_row = cos_sin + position.to(tl.int64) * head_dim
        cos = tl.load(cache_row + pairs)
        sin = tl.load(cache_row + head_dim // 2 + pairs)
        q_row = q + token.to(tl.int64) * q_token_stride
        k_row = k + token.to(tl.int64) * k_token_stride
        offsets = heads[:, None] * head_dim + dimensions[None, :]
        mask = heads[:, None] < num_heads
        q_values = tl.load(q_row + offsets, mask, other=0).to(tl.float32)
        k_values = tl.load(k_row + offsets, mask, other=0).to(tl.float32)
        qr, qi = tl.split(tl.reshape(q_values, (head_block_size, head_dim // 2, 2)))
        kr, ki = tl.split(tl.reshape(k_values, (head_block_size, head_dim // 2, 2)))
        q_output = tl.join(
            qr * cos[None, :] - qi * sin[None, :], qr * sin[None, :] + qi * cos[None, :]
        )
        k_output = tl.join(
            kr * cos[None, :] - ki * sin[None, :], kr * sin[None, :] + ki * cos[None, :]
        )
        tl.store(
            q_row + offsets, tl.reshape(q_output, (head_block_size, head_dim)), mask
        )
        tl.store(
            k_row + offsets, tl.reshape(k_output, (head_block_size, head_dim)), mask
        )

    @triton.jit(do_not_specialize=["chunk_length"])
    def streaming_kv_commit_npu_kernel(
        cached_k: tl.tensor,
        cached_v: tl.tensor,
        all_k: tl.tensor,
        all_v: tl.tensor,
        slot_words: tl.tensor,
        valid_rows: tl.tensor,
        num_heads: tl.constexpr,
        context: tl.constexpr,
        chunk_length: tl.tensor,
        head_dim: tl.constexpr,
        block_size: tl.constexpr,
    ) -> None:
        blocks_per_head: tl.constexpr = tl.cdiv(context * head_dim, block_size)
        batch_head = tl.program_id(0) // blocks_per_head
        row = batch_head // num_heads
        head = batch_head % num_heads
        index = (tl.program_id(0) % blocks_per_head) * block_size + tl.arange(
            0, block_size
        )
        slot = tl.load(slot_words + row.to(tl.int64) * 2)
        valid = tl.load(valid_rows + row) != 0
        mask = (index < context * head_dim) & valid
        # note (Zhang Yiyang): Scalar slot addressing and contiguous offsets keep Ascend transfers vectorized.
        source = (
            batch_head.to(tl.int64) * (context + chunk_length) + chunk_length
        ) * head_dim
        target = (slot.to(tl.int64) * num_heads + head) * context * head_dim
        k = tl.load((all_k + source) + index, mask, other=0)
        v = tl.load((all_v + source) + index, mask, other=0)
        tl.store((cached_k + target) + index, k, mask)
        tl.store((cached_v + target) + index, v, mask)

    @triton.jit(do_not_specialize=["chunk_length"])
    def streaming_positions_commit_npu_kernel(
        cached_position_words: tl.tensor,
        offset_words: tl.tensor,
        position_words: tl.tensor,
        slot_words: tl.tensor,
        valid_rows: tl.tensor,
        context: tl.constexpr,
        chunk_length: tl.tensor,
        block_size: tl.constexpr,
    ) -> None:
        row = tl.program_id(0)
        slot = tl.load(slot_words + row.to(tl.int64) * 2)
        valid = tl.load(valid_rows + row) != 0
        index = tl.arange(0, block_size)
        mask = (index < context * 2) & valid
        source = (row.to(tl.int64) * (context + chunk_length) + chunk_length) * 2
        target = slot.to(tl.int64) * context * 2
        words = tl.load((position_words + source) + index, mask, other=0)
        tl.store((cached_position_words + target) + index, words, mask)
        # note (Zhang Yiyang): Preserve int64 positions with native 32-bit word copies and carry arithmetic on Ascend.
        slot_offset_words = offset_words + slot.to(tl.int64) * 2
        low = tl.load(slot_offset_words, valid, other=0).to(tl.uint32)
        high = tl.load(slot_offset_words + 1, valid, other=0)
        next_low = (low + chunk_length).to(tl.uint32)
        next_high = high + (next_low < low).to(tl.int32)
        tl.store(slot_offset_words, next_low.to(tl.int32), valid)
        tl.store(slot_offset_words + 1, next_high, valid)

else:
    reset_streaming_state_npu_kernel = None
    exact_interleaved_rope_npu_kernel = None
    streaming_kv_commit_npu_kernel = None
    streaming_positions_commit_npu_kernel = None
