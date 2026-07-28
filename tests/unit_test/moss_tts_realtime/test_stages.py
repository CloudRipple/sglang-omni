# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
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
        "codec_mem_reserve": 0.15,
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
    builder = _builder(max_seq_len=None)

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

    def fake_load_codec(model_path: str, *, device: str) -> object:
        calls["codec"] = (model_path, device)
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
        "codec": ("codec", "cuda:3"),
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
    assert calls["server_args"]["mem_fraction_static"] == pytest.approx(0.75)
    assert "attention_backend" not in calls["server_args"]
    assert calls["server_args"]["enable_streaming_session"] is True
    assert calls["server_args"]["disable_cuda_graph"] is False
    assert calls["server_args"]["disable_overlap_schedule"] is True

    server_args, gpu_id, infra_kwargs = calls["infrastructure"]
    assert gpu_id == 2
    assert server_args.enable_streaming_session is True
    assert infra_kwargs == {
        "total_gpu_memory_fraction": pytest.approx(0.75),
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

    def fake_load_codec(model_path: str, *, device: str) -> object:
        calls["codec"] = (model_path, device)
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
        "codec": ("codec", "cuda:2"),
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
