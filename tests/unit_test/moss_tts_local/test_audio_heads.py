# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the MOSS-TTS Local untied audio heads.

The 2.0 runtime layout trains ``audio_lm_heads`` separately from the audio
embedding tables, while v1.5 ships them tied (or converted checkpoints omit
them). These CPU-only tests pin the load/forward contract:

- ``load_weights`` routes ``audio_lm_heads.{i}.weight`` into the dedicated
  tables, ties missing heads onto the embedding rows, and rejects a partial
  head set;
- both frame-decode paths sample logits through the head tables while the
  next-local-position feedback stays on the embedding rows.
"""

from __future__ import annotations

import types
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from sglang_omni.models.moss_tts_local.sglang_model import MossTTSLocalSGLangModel

_HIDDEN = 16
_N_VQ = 2
_AUDIO_VOCAB = 7
_TEXT_VOCAB = 11
_CHANNELS = _N_VQ + 1


def _bind(stub: SimpleNamespace, *names: str) -> SimpleNamespace:
    for name in names:
        setattr(
            stub,
            name,
            types.MethodType(getattr(MossTTSLocalSGLangModel, name), stub),
        )
    return stub


def _embedding_weight(vocab: int, seed: int) -> torch.nn.Parameter:
    gen = torch.Generator().manual_seed(seed)
    return torch.nn.Parameter(torch.randn(vocab, _HIDDEN, generator=gen))


class _FakeEmbedding(torch.nn.Module):
    """Callable embedding table mimicking VocabParallelEmbedding's weight."""

    def __init__(self, vocab: int, seed: int) -> None:
        super().__init__()
        self.weight = _embedding_weight(vocab, seed)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(ids, self.weight)


def _linear(out: int, seed: int) -> torch.nn.Linear:
    layer = torch.nn.Linear(_HIDDEN, out, bias=False)
    with torch.no_grad():
        layer.weight.copy_(
            torch.randn(out, _HIDDEN, generator=torch.Generator().manual_seed(seed))
        )
    return layer


def _model_stub() -> SimpleNamespace:
    """Minimal stand-in exposing what load_weights/decode_frame touch."""
    embedding_list = [
        # Text channel table; the extra audio tables carry one pad row.
        _FakeEmbedding(_TEXT_VOCAB, seed=100),
        *(_FakeEmbedding(_AUDIO_VOCAB + 1, seed=200 + c) for c in range(_N_VQ)),
    ]
    stub = SimpleNamespace(
        config=SimpleNamespace(
            audio_vocab_size=_AUDIO_VOCAB,
            audio_assistant_slot_token_id=_TEXT_VOCAB - 1,
            channels=_CHANNELS,
            vocab_size_list=[_TEXT_VOCAB] + [_AUDIO_VOCAB + 1] * _N_VQ,
        ),
        n_vq=_N_VQ,
        dtype=torch.float32,
        embedding_list=embedding_list,
        audio_lm_heads=[_linear(_AUDIO_VOCAB, seed=300 + c) for c in range(_N_VQ)],
        local_text_lm_head=_linear(2, seed=400),
        local_transformer=SimpleNamespace(step=lambda hidden, position: hidden * 0.5),
        model=SimpleNamespace(start_layer=0, end_layer=36),
    )
    params_dict: dict[str, torch.nn.Parameter] = {}
    for idx, layer in enumerate(embedding_list):
        params_dict[f"embedding_list.{idx}.weight"] = layer.weight
    for idx, head in enumerate(stub.audio_lm_heads):
        params_dict[f"audio_lm_heads.{idx}.weight"] = head.weight
    params_dict["local_text_lm_head.weight"] = stub.local_text_lm_head.weight
    params_dict["model.embed_tokens.weight"] = torch.nn.Parameter(
        torch.zeros(_TEXT_VOCAB, _HIDDEN)
    )
    stub.named_parameters = lambda: iter(params_dict.items())
    # Staticmethods on the class: assign the plain functions, no MethodType.
    stub._load_param = MossTTSLocalSGLangModel._load_param
    stub._map_audio_embedding_name = MossTTSLocalSGLangModel._map_audio_embedding_name
    return _bind(
        stub,
        "_audio_embedding_weight",
        "_audio_head_weight",
        "_resolve_audio_head_ties",
        "_zero_audio_pad_rows",
        "decode_frame",
        "_decode_frame_graphable",
        "load_weights",
    )


def _ckpt_weights(
    *, include_audio_heads: bool, n_heads: int = _N_VQ
) -> list[tuple[str, torch.Tensor]]:
    gen = torch.Generator().manual_seed(0)
    weights: list[tuple[str, torch.Tensor]] = [
        (
            "transformer.embed_tokens.weight",
            torch.randn(_TEXT_VOCAB, _HIDDEN, generator=gen),
        ),
        ("text_lm_head.weight", torch.randn(_TEXT_VOCAB, _HIDDEN, generator=gen)),
    ]
    for channel in range(_N_VQ):
        weights.append(
            (
                f"audio_embeddings.{channel}.weight",
                torch.randn(_AUDIO_VOCAB, _HIDDEN, generator=gen),
            )
        )
    if include_audio_heads:
        for channel in range(n_heads):
            weights.append(
                (
                    f"audio_lm_heads.{channel}.weight",
                    torch.randn(_AUDIO_VOCAB, _HIDDEN, generator=gen),
                )
            )
    weights.append(
        ("local_text_lm_head.weight", torch.randn(2, _HIDDEN, generator=gen))
    )
    return weights


def test_loads_untied_audio_heads() -> None:
    stub = _model_stub()
    weights = _ckpt_weights(include_audio_heads=True)
    expected = {name: w for name, w in weights}
    stub.load_weights(iter(weights))

    for channel in range(_N_VQ):
        assert torch.equal(
            stub.audio_lm_heads[channel].weight,
            expected[f"audio_lm_heads.{channel}.weight"],
        )
        # Untied: head storage stays separate from the embedding rows.
        assert not torch.equal(
            stub.audio_lm_heads[channel].weight,
            expected[f"audio_embeddings.{channel}.weight"],
        )
        assert (
            stub.audio_lm_heads[channel].weight.data_ptr()
            != stub.embedding_list[channel + 1].weight.data_ptr()
        )
    # The embedding tables still load from audio_embeddings, with a zeroed pad row.
    for channel in range(_N_VQ):
        table = stub.embedding_list[channel + 1].weight
        assert torch.equal(
            table[:_AUDIO_VOCAB], expected[f"audio_embeddings.{channel}.weight"]
        )
        assert torch.all(table[_AUDIO_VOCAB:] == 0)


def test_missing_audio_heads_tie_to_embeddings() -> None:
    stub = _model_stub()
    weights = _ckpt_weights(include_audio_heads=False)
    expected = {name: w for name, w in weights}
    stub.load_weights(iter(weights))

    for channel in range(_N_VQ):
        head = stub.audio_lm_heads[channel].weight
        embedding_rows = stub.embedding_list[channel + 1].weight[:_AUDIO_VOCAB]
        assert torch.equal(head, expected[f"audio_embeddings.{channel}.weight"])
        # Aliased onto the same storage, not a copy.
        assert head.data_ptr() == embedding_rows.data_ptr()


def test_serialized_tie_aliases_back_onto_embeddings() -> None:
    """A checkpoint storing audio_lm_heads == audio_embeddings loads, then
    aliases the heads onto the embedding rows instead of keeping duplicates."""
    stub = _model_stub()
    weights = _ckpt_weights(include_audio_heads=False)
    expected = {name: w for name, w in weights}
    # Re-add the head tensors as exact copies of the embedding rows: the
    # serialized-tie layout (the v1.5 release shape).
    weights += [
        (f"audio_lm_heads.{c}.weight", expected[f"audio_embeddings.{c}.weight"])
        for c in range(_N_VQ)
    ]
    stub.load_weights(iter(weights))

    for channel in range(_N_VQ):
        head = stub.audio_lm_heads[channel].weight
        embedding_rows = stub.embedding_list[channel + 1].weight[:_AUDIO_VOCAB]
        assert torch.equal(head, embedding_rows)
        assert head.data_ptr() == embedding_rows.data_ptr()


def test_partial_audio_heads_rejected() -> None:
    stub = _model_stub()
    weights = _ckpt_weights(include_audio_heads=True, n_heads=1)
    with pytest.raises(ValueError, match="untied heads must be complete"):
        stub.load_weights(iter(weights))


def test_text_lm_head_stays_tied_via_embedding() -> None:
    stub = _model_stub()
    weights = _ckpt_weights(include_audio_heads=True)
    expected = {name: w for name, w in weights}
    stub.load_weights(iter(weights))
    # text_lm_head is skipped; embedding_list.0 follows embed_tokens.
    assert torch.equal(
        stub.embedding_list[0].weight, expected["transformer.embed_tokens.weight"]
    )


def _frame_stub() -> SimpleNamespace:
    stub = _model_stub()
    stub._sample_seeded_branchless = lambda logits, **kwargs: logits.argmax(-1)
    return stub


def test_decode_frame_uses_heads_for_logits_and_embeddings_for_feedback() -> None:
    stub = _frame_stub()
    seen: list[tuple[int, torch.Tensor]] = []

    def sample_audio(logits: torch.Tensor, channel: int) -> torch.Tensor:
        seen.append((channel, logits))
        return logits.argmax(-1)

    def step(hidden: torch.Tensor, position: int) -> torch.Tensor:
        return hidden * 0.5

    stub.local_transformer = SimpleNamespace(step=step)
    stub.decode_frame(
        torch.ones(2, _HIDDEN),
        sample_text=lambda logits: logits.argmin(-1),
        sample_audio=sample_audio,
    )

    assert [c for c, _ in seen] == list(range(_N_VQ))
    # Channel 0 logits come from the head table on local step 0's output.
    hidden0 = torch.ones(2, _HIDDEN) * 0.5
    assert torch.equal(
        seen[0][1], F.linear(hidden0, stub.audio_lm_heads[0].weight).float()
    )
    # Channel 1 is fed the *embedding* row of channel 0's code, stepped once.
    code0 = seen[0][1].argmax(-1)
    embed0 = F.embedding(code0, stub.embedding_list[1].weight[:_AUDIO_VOCAB])
    assert torch.equal(
        seen[1][1],
        F.linear(embed0 * 0.5, stub.audio_lm_heads[1].weight).float(),
    )


def test_graphable_frame_uses_same_head_embedding_split() -> None:
    stub = _frame_stub()
    batch = 2
    stop_choice, codes, feedback = stub._decode_frame_graphable(
        torch.ones(batch, _HIDDEN),
        text_temperature=torch.ones(batch),
        text_top_p=torch.ones(batch),
        text_top_k=torch.full((batch,), 50, dtype=torch.long),
        audio_temperature=torch.ones(batch),
        audio_top_p=torch.ones(batch),
        audio_top_k=torch.full((batch,), 25, dtype=torch.long),
        seeds=torch.zeros(batch, dtype=torch.long),
        base_positions=torch.zeros(batch, dtype=torch.long),
    )
    assert stop_choice.shape == (batch,)
    assert codes.shape == (batch, _N_VQ)
    # feedback = slot embedding + sum over channels of embedding rows.
    slot_ids = torch.full_like(codes[:, 0], stub.config.audio_assistant_slot_token_id)
    expected = stub.embedding_list[0].weight[slot_ids]
    for channel in range(_N_VQ):
        expected = expected + F.embedding(
            codes[:, channel], stub.embedding_list[channel + 1].weight[:_AUDIO_VOCAB]
        )
    assert torch.allclose(feedback, expected)
    # Argmax sampling through the (distinct) head tables decides the codes.
    hidden0 = torch.ones(batch, _HIDDEN) * 0.5  # step(ones, 0)
    expected_code = F.linear(hidden0, stub.audio_lm_heads[0].weight).argmax(-1)
    assert torch.equal(codes[:, 0], expected_code)
    hidden1 = (
        F.embedding(expected_code, stub.embedding_list[1].weight[:_AUDIO_VOCAB]) * 0.5
    )  # step(embed_row, 1)
    expected_code = F.linear(hidden1, stub.audio_lm_heads[1].weight).argmax(-1)
    assert torch.equal(codes[:, 1], expected_code)
