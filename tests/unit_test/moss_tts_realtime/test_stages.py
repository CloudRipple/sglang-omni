# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import json
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from sglang_omni.models.moss_tts_realtime import request_builders, stages
from sglang_omni.models.moss_tts_realtime.engine_builder import (
    MossTTSRealtimeEngineBuilder,
)
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler


def _builder(**overrides: Any) -> MossTTSRealtimeEngineBuilder:
    values: dict[str, Any] = {
        "max_seq_len": 40960,
        "total_gpu_memory_fraction": 0.90,
        "max_sessions": 7,
        "max_held_sessions": 5,
        "max_active_turns": 3,
        "max_pending_text_tokens": 64,
        "max_pending_text_bytes": 2048,
        "max_input_updates": 32,
        "max_turn_frames": 40,
        "terminal_tombstone_limit": 77,
        "input_idle_timeout_s": 1.5,
        "turn_timeout_s": 2.5,
        "session_idle_ttl_s": 3.5,
    }
    values.update(overrides)
    return MossTTSRealtimeEngineBuilder(**values)


def test_load_processor_uses_checkpoint_auto_map(monkeypatch) -> None:
    import transformers

    calls: dict[str, Any] = {}
    processor = object()
    loaded_config = object()

    monkeypatch.setattr(stages, "resolve_moss_checkpoint", lambda _: "/resolved")
    monkeypatch.setattr(
        stages,
        "moss_transformers_processor_compat",
        contextlib.nullcontext,
    )
    monkeypatch.setattr(
        transformers.AutoProcessor,
        "from_pretrained",
        lambda checkpoint_dir, **kwargs: (
            calls.__setitem__("processor_checkpoint", checkpoint_dir),
            calls.__setitem__("processor_kwargs", kwargs),
            processor,
        )[-1],
    )
    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        lambda checkpoint_dir, **kwargs: (
            calls.__setitem__("config_checkpoint", checkpoint_dir),
            calls.__setitem__("config_kwargs", kwargs),
            loaded_config,
        )[-1],
    )
    monkeypatch.setattr(
        stages,
        "bind_moss_tts_realtime_processor_config",
        lambda config, processor: calls.__setitem__(
            "config_binding",
            (config, processor),
        ),
    )

    actual = stages.load_moss_tts_realtime_processor("model")

    assert actual is processor
    assert calls == {
        "processor_checkpoint": "/resolved",
        "processor_kwargs": {"trust_remote_code": True},
        "config_checkpoint": "/resolved",
        "config_kwargs": {"trust_remote_code": True},
        "config_binding": (loaded_config, processor),
    }


def test_engine_pre_infra_reuses_processor_model_config(monkeypatch) -> None:
    model_config = SimpleNamespace(
        language_config=SimpleNamespace(max_position_embeddings=2048),
        delay_tokens_len=12,
    )
    processor = SimpleNamespace(model_config=model_config)
    monkeypatch.setattr(
        stages,
        "load_moss_tts_realtime_processor",
        lambda checkpoint_dir: processor,
    )
    builder = _builder(max_seq_len=None, total_gpu_memory_fraction=None)

    builder.pre_infra_setup("checkpoint")

    assert builder.processor is processor
    assert builder.context_length == 2048


def test_audio_encoder_uses_codec_tensor_contract() -> None:
    calls: list[tuple[torch.Tensor, dict[str, Any]]] = []
    output = SimpleNamespace(audio_codes=torch.zeros((32, 1, 4), dtype=torch.long))

    class FakeCodec:
        config = SimpleNamespace(sampling_rate=24000)

        def encode(self, values: torch.Tensor, **kwargs: Any) -> Any:
            calls.append((values.detach().clone(), kwargs))
            return output

    encoder = stages.MossTTSRealtimeAudioEncoder(FakeCodec(), device="cpu")
    stereo = np.stack([np.ones(32, dtype=np.float32), np.zeros(32, dtype=np.float32)])

    result = encoder.encode(stereo)

    assert result is output
    assert len(calls) == 1
    values, kwargs = calls[0]
    assert values.shape == (1, 32)
    assert values.dtype == torch.float32
    assert torch.equal(values, torch.full((1, 32), 0.5))
    assert kwargs == {"return_dict": True}


def test_audio_encoder_normalizes_base64_mapping(monkeypatch) -> None:
    seen: dict[str, Any] = {}

    def fake_load_audio(source: Any, **kwargs: Any) -> np.ndarray:
        seen["source"] = source
        seen["kwargs"] = kwargs
        return np.arange(12, dtype=np.float32)

    class FakeCodec:
        config = SimpleNamespace(sampling_rate=24000)

        def encode(self, values: torch.Tensor, **kwargs: Any) -> Any:
            seen["values"] = values.detach().clone()
            seen["encode_kwargs"] = kwargs
            return SimpleNamespace(
                audio_codes=torch.zeros((32, 1, 1), dtype=torch.long)
            )

    monkeypatch.setattr(stages, "load_audio", fake_load_audio)
    encoder = stages.MossTTSRealtimeAudioEncoder(FakeCodec(), device="cpu")

    encoder.encode({"base64": "ZmFrZQ==", "media_type": "audio/flac"})

    assert seen["source"] == "data:audio/flac;base64,ZmFrZQ=="
    assert seen["kwargs"] == {
        "source_name": "MOSS-TTS-Realtime reference",
        "target_sample_rate": 24000,
        "mono": True,
    }
    assert seen["values"].shape == (1, 12)
    assert seen["encode_kwargs"] == {"return_dict": True}


@pytest.mark.parametrize(
    ("component", "expected_keys"),
    [
        ("encoder", {"encoder.weight", "quantizer.weight"}),
        ("decoder", {"decoder.weight", "quantizer.weight"}),
    ],
)
def test_codec_loader_reads_only_requested_component_weights(
    monkeypatch,
    tmp_path,
    component: str,
    expected_keys: set[str],
) -> None:
    import safetensors
    import transformers
    from safetensors.torch import save_file
    from torch import nn

    class TinyCodec(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Linear(2, 2, bias=False)
            self.decoder = nn.Linear(2, 2, bias=False)
            self.quantizer = nn.Linear(2, 2, bias=False)
            self.config = object()

    expected_weights = {
        "encoder.weight": torch.full((2, 2), 1.0),
        "decoder.weight": torch.full((2, 2), 2.0),
        "quantizer.weight": torch.full((2, 2), 3.0),
    }
    save_file(
        {
            "encoder.weight": expected_weights["encoder.weight"],
            "quantizer.weight": expected_weights["quantizer.weight"],
        },
        tmp_path / "model-00001-of-00002.safetensors",
    )
    save_file(
        {"decoder.weight": expected_weights["decoder.weight"]},
        tmp_path / "model-00002-of-00002.safetensors",
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "encoder.weight": "model-00001-of-00002.safetensors",
                    "quantizer.weight": "model-00001-of-00002.safetensors",
                    "decoder.weight": "model-00002-of-00002.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(stages, "resolve_moss_checkpoint", lambda _: tmp_path)
    monkeypatch.setattr(
        stages,
        "moss_transformers_processor_compat",
        contextlib.nullcontext,
    )
    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        transformers.AutoModel,
        "from_config",
        lambda *args, **kwargs: TinyCodec(),
    )

    loaded_keys: set[str] = set()
    real_safe_open = safetensors.safe_open

    class TrackingSafeOpen:
        def __init__(self, filename, *args, **kwargs) -> None:
            self._context = real_safe_open(filename, *args, **kwargs)
            self._reader = None

        def __enter__(self):
            self._reader = self._context.__enter__()
            return self

        def __exit__(self, *args):
            return self._context.__exit__(*args)

        def get_tensor(self, name: str) -> torch.Tensor:
            loaded_keys.add(name)
            return self._reader.get_tensor(name)

    monkeypatch.setattr(safetensors, "safe_open", TrackingSafeOpen)

    codec = stages.load_moss_tts_realtime_codec(
        "codec",
        component=component,
        device="cpu",
    )

    assert loaded_keys == expected_keys
    assert set(codec.state_dict()) == expected_keys
    for name, value in codec.state_dict().items():
        assert value.device.type == "cpu"
        assert torch.equal(value, expected_weights[name])


def test_codec_memory_estimate_scales_streaming_state_with_active_turns(
    monkeypatch,
) -> None:
    import transformers
    from torch import nn

    class StateModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self._streaming_state: Any | None = None

    class TinyCodec(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Linear(2, 2, bias=False)
            self.decoder = nn.Sequential(
                nn.Linear(2, 3, bias=False),
                StateModule(),
            )
            self.quantizer = nn.Linear(2, 2, bias=False)

        @contextlib.contextmanager
        def streaming(self, batch_size: int):
            cache = torch.empty(batch_size, 5, dtype=torch.float32)
            state = SimpleNamespace(
                cache=cache,
                cache_alias=cache,
                exec_mask=torch.empty(batch_size, dtype=torch.bool),
            )
            self.decoder[1]._streaming_state = state
            try:
                yield
            finally:
                self.decoder[1]._streaming_state = None

    monkeypatch.setattr(stages, "resolve_moss_checkpoint", lambda _: "/codec")
    monkeypatch.setattr(
        stages,
        "moss_transformers_processor_compat",
        contextlib.nullcontext,
    )
    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        transformers.AutoModel,
        "from_config",
        lambda *args, **kwargs: TinyCodec(),
    )

    decoder_bytes, state_bytes = stages.estimate_moss_tts_realtime_codec_memory(
        "codec",
        max_active_turns=4,
    )

    assert decoder_bytes == 40
    assert state_bytes == 84


def test_create_preprocessing_executor_wires_codec_and_cleanup(monkeypatch) -> None:
    calls: dict[str, Any] = {}
    processor = object()
    codec = object()
    encoder = object()

    def fake_load_processor(model_path: str) -> object:
        calls["processor_path"] = model_path
        return processor

    monkeypatch.setattr(
        stages,
        "load_moss_tts_realtime_processor",
        fake_load_processor,
    )

    def fake_load_codec(
        model_path: str,
        *,
        component: str,
        device: str,
    ) -> object:
        calls["codec"] = (model_path, component, device)
        return codec

    class FakeAudioEncoder:
        def __new__(cls, loaded_codec: Any, *, device: str) -> object:
            calls["encoder"] = (loaded_codec, device)
            return encoder

    def fake_set_context(*, processor: Any, audio_encoder: Any) -> None:
        calls["context"] = (processor, audio_encoder)

    monkeypatch.setattr(stages, "load_moss_tts_realtime_codec", fake_load_codec)
    monkeypatch.setattr(stages, "MossTTSRealtimeAudioEncoder", FakeAudioEncoder)
    monkeypatch.setattr(
        stages,
        "set_moss_tts_realtime_preprocessing_context",
        fake_set_context,
    )

    scheduler = stages.create_preprocessing_executor(
        "model",
        device=None,
        gpu_id=3,
        codec_model_path="codec",
        max_concurrency=6,
    )

    assert isinstance(scheduler, SimpleScheduler)
    assert scheduler._fn is request_builders.preprocess_moss_tts_realtime_payload
    assert (
        scheduler._abort_callback
        is request_builders.cleanup_prepared_moss_tts_realtime_request
    )
    assert scheduler._max_concurrency == 6
    assert calls == {
        "processor_path": "model",
        "codec": ("codec", "encoder", "cuda:3"),
        "encoder": (codec, "cuda:3"),
        "context": (processor, encoder),
    }


def test_engine_factory_builds_realtime_scheduler_and_wires_outbox(
    monkeypatch,
) -> None:
    from sglang_omni.models.moss_tts_realtime import model_runner, scheduler
    from sglang_omni.scheduling import bootstrap, engine_factory, sglang_backend
    from sglang_omni.utils import gpu_memory

    calls: dict[str, Any] = {}
    runners: list[Any] = []

    def fake_build_server_args(
        model_path: str, *, context_length: int, **kwargs: Any
    ) -> Any:
        assert calls["processor_loaded"] is True
        calls["server_args"] = {
            "model_path": model_path,
            "context_length": context_length,
            **kwargs,
        }
        return SimpleNamespace(
            model_path=model_path,
            context_length=context_length,
            **kwargs,
        )

    language_model = object()

    def init_frame_decode_graphs(batch_sizes: list[int]) -> None:
        calls["frame_decode_graphs"] = batch_sizes

    underlying_runner = SimpleNamespace(
        model=SimpleNamespace(
            language_model=language_model,
            config=SimpleNamespace(
                language_config=SimpleNamespace(max_position_embeddings=40960)
            ),
            _decode_input_embedding=torch.nn.Embedding(7, 4),
            init_frame_decode_graphs=init_frame_decode_graphs,
        ),
        init_device_graphs=lambda: calls.__setitem__("device_graphs", True),
    )
    worker = SimpleNamespace(
        gpu_id=2,
        model_runner=underlying_runner,
        model_config=SimpleNamespace(),
    )

    def fake_create_infrastructure(
        server_args: Any, gpu_id: int, **kwargs: Any
    ) -> tuple[Any, ...]:
        calls["infrastructure"] = (server_args, gpu_id, kwargs)
        return (
            worker,
            "tree-cache",
            "req-pool",
            "kv-pool",
            "prefill",
            "decode",
            worker.model_config,
        )

    class FakeRealtimeRunner:
        def __init__(self, model_worker: Any, output_processor: Any) -> None:
            self.model_worker = model_worker
            self.output_processor = output_processor
            self.stream_outbox = None
            runners.append(self)

        def set_stream_outbox(self, outbox: Any) -> None:
            self.stream_outbox = outbox

    class FakeScheduler:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            self.outbox = object()
            calls["scheduler"] = self

    request_builder = object()
    result_adapter = object()
    monkeypatch.setattr(
        MossTTSRealtimeEngineBuilder,
        "pre_infra_setup",
        lambda self, checkpoint_dir: (
            calls.__setitem__("processor_loaded", True),
            setattr(self, "context_length", 40960),
            setattr(self, "processor", "processor"),
            setattr(self, "minimum_codec_mem_reserve", 0.10),
        ),
    )
    monkeypatch.setattr(
        stages,
        "bind_moss_tts_realtime_processor_config",
        lambda config, processor: calls.__setitem__(
            "processor_binding",
            (config, processor),
        ),
    )
    monkeypatch.setattr(engine_factory, "_resolve_checkpoint", lambda path: path)
    monkeypatch.setattr(
        sglang_backend,
        "build_sglang_server_args",
        fake_build_server_args,
    )
    monkeypatch.setattr(
        bootstrap,
        "create_sglang_infrastructure_defer_cuda_graph",
        lambda server_args, gpu_id, **kwargs: (
            not bool(server_args.disable_cuda_graph),
            fake_create_infrastructure(server_args, gpu_id, **kwargs),
        ),
    )
    monkeypatch.setattr(
        sglang_backend,
        "SGLangOutputProcessor",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    monkeypatch.setattr(
        request_builders,
        "make_moss_tts_realtime_scheduler_adapters",
        lambda **_: (request_builder, result_adapter),
    )
    monkeypatch.setattr(
        model_runner,
        "MossTTSRealtimeModelRunner",
        FakeRealtimeRunner,
    )
    monkeypatch.setattr(scheduler, "MossTTSRealtimeScheduler", FakeScheduler)
    monkeypatch.setattr(gpu_memory, "get_process_gpu_memory_bytes", lambda _: 1024)

    built = _builder().build(
        "model",
        device="cuda:2",
        server_args_overrides={"enable_streaming_session": False},
    )

    assert built is calls["scheduler"]
    assert calls["server_args"]["context_length"] == 40960
    assert calls["server_args"]["max_running_requests"] == 7
    assert calls["server_args"]["mem_fraction_static"] == pytest.approx(0.80)
    assert "attention_backend" not in calls["server_args"]
    assert calls["server_args"]["enable_streaming_session"] is True
    assert calls["server_args"]["disable_cuda_graph"] is False
    assert calls["server_args"]["disable_overlap_schedule"] is True

    server_args, gpu_id, infra_kwargs = calls["infrastructure"]
    assert gpu_id == 2
    assert server_args.enable_streaming_session is True
    assert infra_kwargs == {
        "total_gpu_memory_fraction": pytest.approx(0.80),
        "model_arch_override": "MossTTSRealtimeSGLangModel",
    }
    assert worker.moss_tts_realtime_max_turn_frames == 40
    assert worker.moss_tts_realtime_max_active_turns == 3
    assert calls["processor_binding"] == (
        underlying_runner.model.config,
        "processor",
    )
    assert calls["device_graphs"] is True
    assert calls["frame_decode_graphs"] == [1, 2, 3]

    scheduler_kwargs = built.kwargs
    assert scheduler_kwargs["request_builder"] is request_builder
    assert scheduler_kwargs["result_adapter"] is result_adapter
    assert (
        scheduler_kwargs["abort_callback"]
        is request_builders.cleanup_prepared_moss_tts_realtime_request
    )
    assert scheduler_kwargs["enable_async_decode"] is False
    for key, value in _builder().limits.model_dump().items():
        assert scheduler_kwargs[key] == value
    assert len(runners) == 1
    assert runners[0].stream_outbox is built.outbox


def test_engine_factory_honors_disabled_cuda_graph(monkeypatch) -> None:
    from sglang_omni.models.moss_tts_realtime import model_runner, scheduler
    from sglang_omni.scheduling import bootstrap, engine_factory, sglang_backend

    calls: dict[str, Any] = {"device_graphs": 0, "frame_decode_graphs": 0}

    def fake_build_server_args(
        model_path: str, *, context_length: int, **kwargs: Any
    ) -> Any:
        return SimpleNamespace(
            model_path=model_path,
            context_length=context_length,
            **kwargs,
        )

    def init_device_graphs() -> None:
        calls["device_graphs"] += 1

    def init_frame_decode_graphs(_batch_sizes: list[int]) -> None:
        calls["frame_decode_graphs"] += 1

    underlying_runner = SimpleNamespace(
        model=SimpleNamespace(
            language_model=object(),
            _decode_input_embedding=torch.nn.Embedding(7, 4),
            config=SimpleNamespace(
                language_config=SimpleNamespace(max_position_embeddings=40960)
            ),
            init_frame_decode_graphs=init_frame_decode_graphs,
        ),
        init_device_graphs=init_device_graphs,
    )
    worker = SimpleNamespace(
        gpu_id=0,
        model_runner=underlying_runner,
        model_config=SimpleNamespace(),
    )

    def fake_deferred_infrastructure(
        server_args: Any, gpu_id: int, **kwargs: Any
    ) -> tuple[bool, tuple[Any, ...]]:
        del gpu_id, kwargs
        return (
            not bool(server_args.disable_cuda_graph),
            (
                worker,
                "tree-cache",
                "req-pool",
                "kv-pool",
                "prefill",
                "decode",
                worker.model_config,
            ),
        )

    class FakeRealtimeRunner:
        def __init__(self, model_worker: Any, output_processor: Any) -> None:
            del model_worker, output_processor

        def set_stream_outbox(self, outbox: Any) -> None:
            del outbox

    class FakeScheduler:
        def __init__(self, **kwargs: Any) -> None:
            self.outbox = object()
            self.kwargs = kwargs

    monkeypatch.setattr(
        MossTTSRealtimeEngineBuilder,
        "pre_infra_setup",
        lambda self, checkpoint_dir: (
            setattr(self, "context_length", 40960),
            setattr(self, "processor", "processor"),
        ),
    )
    monkeypatch.setattr(
        stages,
        "bind_moss_tts_realtime_processor_config",
        lambda config, processor: config,
    )

    monkeypatch.setattr(engine_factory, "_resolve_checkpoint", lambda path: path)
    monkeypatch.setattr(
        sglang_backend,
        "build_sglang_server_args",
        fake_build_server_args,
    )
    monkeypatch.setattr(
        bootstrap,
        "create_sglang_infrastructure_defer_cuda_graph",
        fake_deferred_infrastructure,
    )
    monkeypatch.setattr(
        sglang_backend,
        "SGLangOutputProcessor",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    monkeypatch.setattr(
        request_builders,
        "make_moss_tts_realtime_scheduler_adapters",
        lambda **_: (object(), object()),
    )
    monkeypatch.setattr(
        model_runner,
        "MossTTSRealtimeModelRunner",
        FakeRealtimeRunner,
    )
    monkeypatch.setattr(scheduler, "MossTTSRealtimeScheduler", FakeScheduler)

    _builder(total_gpu_memory_fraction=None).build(
        "model",
        server_args_overrides={"disable_cuda_graph": True},
    )

    assert calls == {"device_graphs": 0, "frame_decode_graphs": 0}


def test_create_vocoder_executor_threads_slot_limit(monkeypatch) -> None:
    calls: dict[str, Any] = {}
    codec = object()
    processor = SimpleNamespace(model_config=SimpleNamespace(rvq=16))
    scheduler = object()

    def fake_load_codec(
        model_path: str,
        *,
        component: str,
        device: str,
    ) -> object:
        calls["codec"] = (model_path, component, device)
        return codec

    def fake_scheduler(loaded_codec: Any, **kwargs: Any) -> object:
        calls["scheduler"] = (loaded_codec, kwargs)
        return scheduler

    monkeypatch.setattr(stages, "load_moss_tts_realtime_codec", fake_load_codec)
    monkeypatch.setattr(
        stages,
        "load_moss_tts_realtime_processor",
        lambda model_path: (
            calls.__setitem__("processor", model_path),
            processor,
        )[-1],
    )
    monkeypatch.setattr(
        stages,
        "MossTTSRealtimeStreamingVocoderScheduler",
        fake_scheduler,
    )

    result = stages.create_vocoder_executor(
        "model",
        device=None,
        gpu_id=2,
        codec_model_path="codec",
        stream_slots=4,
        max_batch_size=3,
        max_batch_wait_ms=7,
    )

    assert result is scheduler
    assert calls == {
        "processor": "model",
        "codec": ("codec", "decoder", "cuda:2"),
        "scheduler": (
            codec,
            {
                "n_vq": 16,
                "stream_slots": 4,
                "max_batch_size": 3,
                "max_batch_wait_ms": 7,
            },
        ),
    }
