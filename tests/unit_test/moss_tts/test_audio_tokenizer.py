# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file


class _TinyCodec(nn.Module):
    def __init__(self, config, *, with_shared_state: bool = False) -> None:
        super().__init__()
        self.config = config
        self.encoder = nn.Linear(2, 2, bias=False)
        self.decoder = nn.Linear(2, 2, bias=False)
        self.quantizer = nn.Linear(2, 2, bias=False)
        if with_shared_state:
            self.shared = nn.Parameter(torch.ones(1))


def _patch_tiny_codec(
    monkeypatch: pytest.MonkeyPatch,
    *,
    with_shared_state: bool = False,
) -> None:
    from sglang_omni.models.moss_tts import audio_tokenizer

    config = SimpleNamespace(sampling_rate=24000)
    monkeypatch.setattr(
        audio_tokenizer.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: config,
    )
    monkeypatch.setattr(
        audio_tokenizer.AutoModel,
        "from_config",
        lambda *args, **kwargs: _TinyCodec(
            config,
            with_shared_state=with_shared_state,
        ),
    )


def _write_component_checkpoint(tmp_path, component: str) -> torch.Tensor:
    selected = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    quantizer = torch.full((2, 2), 7.0)
    save_file({f"{component}.weight": selected}, tmp_path / "selected.safetensors")
    save_file({"quantizer.weight": quantizer}, tmp_path / "quantizer.safetensors")

    unused = "decoder" if component == "encoder" else "encoder"
    index = {
        "weight_map": {
            f"{component}.weight": "selected.safetensors",
            "quantizer.weight": "quantizer.safetensors",
            f"{unused}.weight": "missing-unused.safetensors",
        }
    }
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(index),
        encoding="utf-8",
    )
    return selected


@pytest.mark.parametrize(
    ("component", "unused"),
    [("encoder", "decoder"), ("decoder", "encoder")],
)
def test_component_loader_reads_only_selected_prefix_and_quantizer(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    component: str,
    unused: str,
) -> None:
    from sglang_omni.models.moss_tts.audio_tokenizer import (
        load_moss_tts_audio_tokenizer,
    )

    _patch_tiny_codec(monkeypatch)
    expected = _write_component_checkpoint(tmp_path, component)

    tokenizer = load_moss_tts_audio_tokenizer(
        str(tmp_path),
        device="cpu",
        dtype="bfloat16",
        component=component,
    )

    loaded_component = getattr(tokenizer.model, component)
    assert torch.equal(loaded_component.weight, expected)
    assert loaded_component.weight.dtype == torch.float32
    assert torch.equal(
        tokenizer.model.quantizer.weight,
        torch.full((2, 2), 7.0),
    )
    assert isinstance(getattr(tokenizer.model, unused), nn.ModuleList)
    assert len(getattr(tokenizer.model, unused)) == 0
    assert all(not parameter.is_meta for parameter in tokenizer.model.parameters())


def test_component_loader_forwards_cuda_dtype(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang_omni.models.moss_tts import audio_tokenizer

    _patch_tiny_codec(monkeypatch)
    calls: list[tuple[str, torch.dtype | None, str]] = []

    def load_module(module, model_path, *, prefix, dtype, device, strict):
        del module, model_path
        assert strict is True
        calls.append((prefix, dtype, device))
        loaded = nn.Linear(2, 2, bias=False)
        return loaded.to(dtype=dtype)

    monkeypatch.setattr(audio_tokenizer, "load_module", load_module)

    tokenizer = audio_tokenizer.load_moss_tts_audio_tokenizer(
        str(tmp_path),
        device="cuda:7",
        dtype="bfloat16",
        component="decoder",
    )

    assert tokenizer.device == "cuda:7"
    assert calls == [
        ("decoder.", torch.bfloat16, "cuda:7"),
        ("quantizer.", torch.bfloat16, "cuda:7"),
    ]
    assert {parameter.dtype for parameter in tokenizer.model.parameters()} == {
        torch.bfloat16
    }


def test_component_loader_rejects_unmaterialized_shared_state(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang_omni.models.moss_tts import audio_tokenizer

    _patch_tiny_codec(monkeypatch, with_shared_state=True)

    def load_module(module, *args, **kwargs):
        del module, args, kwargs
        return nn.Linear(2, 2, bias=False)

    monkeypatch.setattr(audio_tokenizer, "load_module", load_module)

    with pytest.raises(RuntimeError, match="unmaterialized state.*shared"):
        audio_tokenizer.load_moss_tts_audio_tokenizer(
            str(tmp_path),
            device="cpu",
            component="encoder",
        )
