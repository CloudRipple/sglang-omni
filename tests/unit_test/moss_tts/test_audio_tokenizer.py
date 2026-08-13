# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from accelerate import init_empty_weights
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


def test_component_loader_forwards_cuda_dtypes(
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
        return nn.Linear(2, 2, bias=False).to(dtype=dtype)

    monkeypatch.setattr(audio_tokenizer, "load_module", load_module)

    tokenizer = audio_tokenizer.load_moss_tts_audio_tokenizer(
        str(tmp_path),
        device="cuda:7",
        dtype="bfloat16",
        component="encoder",
    )

    assert tokenizer.device == "cuda:7"
    assert calls == [
        ("encoder.", torch.bfloat16, "cuda:7"),
        ("quantizer.", torch.float32, "cuda:7"),
    ]


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


@pytest.mark.parametrize("component", ["encoder", "decoder"])
def test_mixed_component_load_converts_only_autocast_weights(
    tmp_path,
    component: str,
) -> None:
    from sglang_omni.models.moss_tts.audio_tokenizer import _load_mixed_dtype_component

    class TinyComponent(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(2, 2)
            self.norm = nn.LayerNorm(2)

    source = TinyComponent()
    save_file(
        {f"{component}.{name}": tensor for name, tensor in source.state_dict().items()},
        tmp_path / "model.safetensors",
    )
    with init_empty_weights(include_buffers=True):
        module = TinyComponent()

    module = _load_mixed_dtype_component(
        module,
        str(tmp_path),
        component=component,
        storage_dtype=torch.float32,
        compute_dtype=torch.bfloat16,
        device="cpu",
    )

    assert module.linear.weight.dtype == torch.bfloat16
    assert module.linear.bias.dtype == torch.float32
    assert module.norm.weight.dtype == torch.float32
    assert module.norm.bias.dtype == torch.float32
    assert all(not parameter.is_meta for parameter in module.parameters())


def test_component_decoder_uses_configured_compute_autocast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang_omni.models.moss_tts import audio_tokenizer

    class FakeCodec(nn.Module):
        config = SimpleNamespace(sampling_rate=24000)

    calls: list[tuple[str, torch.dtype]] = []

    def fake_autocast(*, device_type: str, dtype: torch.dtype):
        calls.append((device_type, dtype))
        return nullcontext()

    monkeypatch.setattr(audio_tokenizer.torch, "autocast", fake_autocast)

    tokenizer = audio_tokenizer.MossTTSAudioTokenizer(
        FakeCodec(),
        device="cuda:7",
        compute_dtype=torch.bfloat16,
    )

    with tokenizer._autocast():
        pass

    assert tokenizer.dtype == torch.bfloat16
    assert calls == [("cuda", torch.bfloat16)]


def test_low_precision_encoder_wraps_quantizer_and_packed_stages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang_omni.models.moss_tts import audio_tokenizer

    config = SimpleNamespace(sampling_rate=24000)
    model = _TinyCodec(config)
    packed_encoder = nn.ModuleList([nn.Identity()])
    calls: list[tuple[str, torch.dtype | None]] = []

    monkeypatch.setattr(
        audio_tokenizer.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: config,
    )
    monkeypatch.setattr(
        audio_tokenizer.AutoModel,
        "from_config",
        lambda *args, **kwargs: model,
    )

    def load_component(module, *args, component, compute_dtype, **kwargs):
        del args, kwargs
        calls.append((component, compute_dtype))
        return packed_encoder if component == "encoder" else module

    monkeypatch.setattr(
        audio_tokenizer,
        "_load_mixed_dtype_component",
        load_component,
    )
    monkeypatch.setattr(
        audio_tokenizer,
        "load_module",
        lambda module, *args, **kwargs: module,
    )
    monkeypatch.setattr(
        "sglang_omni.models.moss_tts.vocoder_decoder.MossAudioTokenizerEncoder",
        lambda source: nn.ModuleList([source]),
    )

    loaded, runtime_dtype = audio_tokenizer._load_component_model(
        "model",
        component="encoder",
        device="cuda:0",
        dtype=torch.float32,
        compute_dtype=torch.float16,
    )

    assert calls == [("encoder", torch.float16)]
    assert isinstance(loaded.quantizer, audio_tokenizer._FP32Quantizer)
    assert loaded.encoder[0] is packed_encoder
    assert runtime_dtype is torch.float16
