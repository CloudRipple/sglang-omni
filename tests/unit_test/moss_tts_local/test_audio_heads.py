# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the MOSS-TTS Local untied audio heads.

The 2.0 runtime layout trains ``audio_lm_heads`` separately from the audio
embedding tables, while v1.5 ships them tied (or converted checkpoints omit
them). These tests pin the load/forward contract and CUDA graph replay:

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
        hidden_size=_HIDDEN,
        _audio_heads_initialized=False,
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
        "_load_audio_weights",
        "_resolve_audio_head_ties",
        "_zero_audio_pad_rows",
        "decode_frame",
        "_decode_frame_graphable",
        "load_weights",
    )


def _ckpt_weights(
    *,
    include_audio_heads: bool,
    n_heads: int = _N_VQ,
    tied_audio_heads: bool = False,
    seed: int = 0,
) -> list[tuple[str, torch.Tensor]]:
    gen = torch.Generator().manual_seed(seed)
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
        embeddings = dict(weights)
        for channel in range(n_heads):
            weights.append(
                (
                    f"audio_lm_heads.{channel}.weight",
                    (
                        embeddings[f"audio_embeddings.{channel}.weight"].clone()
                        if tied_audio_heads
                        else torch.randn(_AUDIO_VOCAB, _HIDDEN, generator=gen)
                    ),
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
    weights = _ckpt_weights(include_audio_heads=True, tied_audio_heads=True)
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


@pytest.mark.parametrize("initial_tied", [False, True])
def test_incremental_audio_weight_updates(initial_tied: bool) -> None:
    stub = _model_stub()
    stub.load_weights(
        _ckpt_weights(include_audio_heads=True, tied_audio_heads=initial_tied)
    )
    heads = [head.weight.detach().clone() for head in stub.audio_lm_heads]
    embeddings = [
        stub._audio_embedding_weight(c).detach().clone() for c in range(_N_VQ)
    ]

    text_head = stub.local_text_lm_head.weight.detach().clone() + 1
    stub.load_weights([("local_text_lm_head.weight", text_head)])
    torch.testing.assert_close(stub.local_text_lm_head.weight, text_head)
    for channel in range(_N_VQ):
        torch.testing.assert_close(stub._audio_head_weight(channel), heads[channel])

    new_head = heads[0] + 1
    stub.load_weights([("audio_lm_heads.0.weight", new_head)])
    torch.testing.assert_close(stub._audio_head_weight(0), new_head)
    torch.testing.assert_close(stub._audio_head_weight(1), heads[1])
    for channel in range(_N_VQ):
        torch.testing.assert_close(
            stub._audio_embedding_weight(channel), embeddings[channel]
        )

    new_embedding = embeddings[1] + 1
    stub.load_weights([("audio_embeddings.1.weight", new_embedding)])
    torch.testing.assert_close(stub._audio_embedding_weight(1), new_embedding)
    torch.testing.assert_close(
        stub._audio_head_weight(1), new_embedding if initial_tied else heads[1]
    )
    torch.testing.assert_close(stub._audio_head_weight(0), new_head)


@pytest.mark.parametrize(
    "initial_tied,next_tied,heads_first",
    [
        (False, False, False),
        (False, True, False),
        (True, True, False),
        (True, False, False),
        (True, False, True),
    ],
)
def test_audio_weight_reload(
    initial_tied: bool, next_tied: bool, heads_first: bool
) -> None:
    stub = _model_stub()
    stub.load_weights(
        _ckpt_weights(include_audio_heads=True, tied_audio_heads=initial_tied)
    )
    head_ptrs = [head.weight.data_ptr() for head in stub.audio_lm_heads]
    embedding_ptrs = [stub._audio_embedding_weight(c).data_ptr() for c in range(_N_VQ)]
    weights = _ckpt_weights(
        include_audio_heads=True, tied_audio_heads=next_tied, seed=1
    )
    expected = dict(weights)
    stub.load_weights(reversed(weights) if heads_first else iter(weights))

    for channel in range(_N_VQ):
        head = stub._audio_head_weight(channel)
        embedding = stub._audio_embedding_weight(channel)
        torch.testing.assert_close(head, expected[f"audio_lm_heads.{channel}.weight"])
        torch.testing.assert_close(
            embedding, expected[f"audio_embeddings.{channel}.weight"]
        )
        assert embedding.data_ptr() == embedding_ptrs[channel]
        if initial_tied and not next_tied:
            assert head.data_ptr() != head_ptrs[channel]
        else:
            assert head.data_ptr() == head_ptrs[channel]


def _frame_stub() -> SimpleNamespace:
    stub = _model_stub()
    stub._sample_seeded_branchless = lambda logits, **kwargs: logits.argmax(-1)
    return stub


def _frame_inputs(batch: int, *, device="cpu", dtype=torch.float32) -> dict:
    return dict(
        hidden_states=torch.ones(batch, _HIDDEN, device=device, dtype=dtype),
        text_temperature=torch.ones(batch, device=device),
        text_top_p=torch.ones(batch, device=device),
        text_top_k=torch.full((batch,), 50, device=device, dtype=torch.long),
        audio_temperature=torch.ones(batch, device=device),
        audio_top_p=torch.ones(batch, device=device),
        audio_top_k=torch.full((batch,), 25, device=device, dtype=torch.long),
        seeds=torch.zeros(batch, device=device, dtype=torch.long),
        base_positions=torch.zeros(batch, device=device, dtype=torch.long),
    )


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
    stop_choice, codes, feedback = stub._decode_frame_graphable(**_frame_inputs(batch))
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


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("initial_tied", [False, True, None])
def test_cuda_graph_replay_after_audio_weight_updates(
    initial_tied: bool | None,
) -> None:
    stub = _frame_stub()
    stub.device = torch.device("cuda", torch.cuda.current_device())
    stub.dtype = torch.bfloat16
    for _, param in stub.named_parameters():
        param.data = param.data.to(device=stub.device, dtype=stub.dtype)
    stub._decode_input_embedding = SimpleNamespace(weight=torch.empty(2, _HIDDEN))
    stub.local_transformer._ensure_kv_cache = lambda *args: None
    stub.local_transformer.freeze_kv_cache = lambda: None
    stub._ensure_frame_sampler_compile = lambda: None
    _bind(stub, "init_frame_decode_graphs", "decode_frame_graphed")
    # note (Zhang Yiyang): Dummy loading captures before the first load_weights.
    if initial_tied is not None:
        stub.load_weights(
            _ckpt_weights(include_audio_heads=True, tied_audio_heads=initial_tied)
        )
    stub.init_frame_decode_graphs([1, 2])

    for seed, next_tied in enumerate((not initial_tied, bool(initial_tied)), start=1):
        previous_graphs = {bs: entry[0] for bs, entry in stub._frame_graphs.items()}
        previous_heads = [head.weight.data_ptr() for head in stub.audio_lm_heads]
        stub.load_weights(
            _ckpt_weights(
                include_audio_heads=True, tied_audio_heads=next_tied, seed=seed
            )
        )
        storage_changed = initial_tied is None or (initial_tied and seed == 1)
        for channel in range(_N_VQ):
            assert (
                stub._audio_head_weight(channel).data_ptr() != previous_heads[channel]
            ) == storage_changed
        for batch in (1, 2):
            assert (
                stub._frame_graphs[batch][0] is not previous_graphs[batch]
            ) == storage_changed
            inputs = _frame_inputs(batch, device=stub.device, dtype=stub.dtype)
            expected = stub._decode_frame_graphable(**inputs)
            replayed = stub.decode_frame_graphed(**inputs)
            for actual, reference in zip(replayed, expected):
                torch.testing.assert_close(actual, reference, rtol=0, atol=0)
