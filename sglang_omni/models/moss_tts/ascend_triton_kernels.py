# SPDX-License-Identifier: Apache-2.0
"""Ascend Triton kernels for MOSS-TTS and MOSS-Audio-Tokenizer."""

try:
    import triton
    import triton.language as tl

    from sglang_omni.sampling.ascend_kernels import hash32_values
except ImportError:
    triton = None
    tl = None

if triton is not None:

    @triton.jit
    def mask_sorted_top_k_npu_kernel(
        scores: tl.tensor,
        top_k_words: tl.tensor,
        masked_scores: tl.tensor,
        vocab_size: tl.constexpr,
        block_size: tl.constexpr,
    ) -> None:
        row = tl.program_id(0)
        ranks = tl.arange(0, block_size)
        valid = ranks < vocab_size
        k = tl.load(top_k_words + row * 2)
        high = tl.load(top_k_words + row * 2 + 1)
        active = (high == 0) & (k > 0) & (k < vocab_size)
        safe_k = tl.where(active, k, 1)
        row_offset = row.to(tl.int64) * vocab_size
        threshold = tl.load(scores + row_offset + safe_k - 1)
        threshold = tl.where(active, threshold, -float("inf"))
        value = tl.load(scores + row_offset + ranks, valid, other=-float("inf"))
        value = tl.where(value < threshold, -float("inf"), value)
        tl.store(masked_scores + row_offset + ranks, value, valid)

    @triton.jit
    def sample_nucleus_npu_kernel(
        sorted_scores: tl.tensor,
        sorted_token_id_words: tl.tensor,
        cumulative_probabilities: tl.tensor,
        logits: tl.tensor,
        top_p: tl.tensor,
        temperature: tl.tensor,
        seed_words: tl.tensor,
        position_words: tl.tensor,
        output_words: tl.tensor,
        vocab_size: tl.constexpr,
        logit_row_stride: tl.constexpr,
        logit_column_stride: tl.constexpr,
        block_size: tl.constexpr,
    ) -> None:
        row = tl.program_id(0)
        ranks = tl.arange(0, block_size)
        valid = ranks < vocab_size
        row_offset = row.to(tl.int64) * vocab_size
        token_ids = tl.load(
            sorted_token_id_words + row_offset * 2 + ranks * 2, valid, other=0
        )
        previous_probability = tl.load(
            cumulative_probabilities + row_offset + ranks - 1,
            valid & (ranks > 0),
            other=0,
        )
        p = tl.load(top_p + row)
        temp = tl.load(temperature + row)
        value = tl.load(sorted_scores + row_offset + ranks, valid, other=-float("inf"))
        remove = (ranks > 0) & (p > 0) & (p < 1) & (previous_probability > p)
        value = tl.where(remove, -float("inf"), value)
        invalid_probability = (
            tl.sum((valid & ((value != value) | (value == float("inf")))).to(tl.int32))
            > 0
        )
        no_score = tl.max(value, 0) == -float("inf")
        hashes = hash32_values(seed_words, position_words, row, token_ids)
        high = ((hashes >> 16) & 0xFFFF).to(tl.int32).to(tl.float32)
        low = (hashes & 0xFFFF).to(tl.int32).to(tl.float32)
        uniform = high * (1.0 / 65536.0) + low * (1.0 / 4294967296.0)
        # note (Zhang Yiyang): Preserve native NPU FP32 hash conversion, log roundings and endpoint clamps.
        log_uniform = tl.minimum(
            tl.maximum(tl.log(uniform), -3.4028234663852886e38),
            -2.3283064365386963e-10,
        )
        noise = -tl.log(-log_uniform)
        draw = tl.where(valid, value + noise, -float("inf"))
        maximum = tl.max(draw, 0)
        sampled = tl.min(tl.where(valid & (draw == maximum), token_ids, 2147483647), 0)
        original = tl.load(
            logits + row.to(tl.int64) * logit_row_stride + ranks * logit_column_stride,
            valid,
            other=-float("inf"),
        )
        nan_index = tl.min(
            tl.where(valid & (original != original), ranks, 2147483647), 0
        )
        greedy_max = tl.max(tl.where(original == original, original, -float("inf")), 0)
        greedy = tl.min(
            tl.where(valid & (original == greedy_max), ranks, 2147483647), 0
        )
        greedy = tl.where(nan_index != 2147483647, nan_index, greedy)
        output = tl.where(
            (temp <= 0) | (temp != temp) | invalid_probability | no_score,
            greedy,
            sampled,
        )
        tl.store(output_words + row * 2, output)
        tl.store(output_words + row * 2 + 1, 0)

    @triton.jit(do_not_specialize=["position"])
    def local_rotary_cache_npu_kernel(
        qkv: tl.tensor,
        cosine: tl.tensor,
        sine: tl.tensor,
        query: tl.tensor,
        key_cache: tl.tensor,
        value_cache: tl.tensor,
        position: tl.tensor,
        num_heads: tl.constexpr,
        head_dim: tl.constexpr,
        max_positions: tl.constexpr,
        block_size: tl.constexpr,
    ) -> None:
        batch_head = tl.program_id(0)
        batch = batch_head // num_heads
        head = batch_head % num_heads
        channel = tl.arange(0, block_size)
        valid = channel < head_dim
        row = (
            qkv
            + batch.to(tl.int64) * 3 * num_heads * head_dim
            + head.to(tl.int64) * head_dim
        )
        cos = tl.load(cosine + channel, valid, other=0).to(tl.float32)
        sin = tl.load(sine + channel, valid, other=0).to(tl.float32)
        cos, _ = tl.split(tl.reshape(cos, (block_size // 2, 2)))
        sin, _ = tl.split(tl.reshape(sin, (block_size // 2, 2)))
        q = tl.load(row + channel, valid, other=0).to(tl.float32)
        k = tl.load(row + num_heads * head_dim + channel, valid, other=0).to(tl.float32)
        v = tl.load(row + 2 * num_heads * head_dim + channel, valid, other=0)
        qr, qi = tl.split(tl.reshape(q, (block_size // 2, 2)))
        kr, ki = tl.split(tl.reshape(k, (block_size // 2, 2)))
        qr_out = (qr * cos).to(tl.bfloat16).to(tl.float32) + (-qi * sin).to(
            tl.bfloat16
        ).to(tl.float32)
        qi_out = (qi * cos).to(tl.bfloat16).to(tl.float32) + (qr * sin).to(
            tl.bfloat16
        ).to(tl.float32)
        kr_out = (kr * cos).to(tl.bfloat16).to(tl.float32) + (-ki * sin).to(
            tl.bfloat16
        ).to(tl.float32)
        ki_out = (ki * cos).to(tl.bfloat16).to(tl.float32) + (kr * sin).to(
            tl.bfloat16
        ).to(tl.float32)
        q = tl.reshape(tl.join(qr_out, qi_out), (block_size,))
        k = tl.reshape(tl.join(kr_out, ki_out), (block_size,))
        tl.store(query + batch_head.to(tl.int64) * head_dim + channel, q, valid)
        cache_offset = (
            batch_head.to(tl.int64) * max_positions * head_dim + position * head_dim
        )
        tl.store(key_cache + cache_offset + channel, k, valid)
        tl.store(value_cache + cache_offset + channel, v, valid)

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
    mask_sorted_top_k_npu_kernel = None
    sample_nucleus_npu_kernel = None
    local_rotary_cache_npu_kernel = None
    reset_streaming_state_npu_kernel = None
    exact_interleaved_rope_npu_kernel = None
    streaming_kv_commit_npu_kernel = None
    streaming_positions_commit_npu_kernel = None
