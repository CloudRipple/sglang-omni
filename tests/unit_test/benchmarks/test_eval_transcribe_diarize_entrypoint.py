from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

EVAL_SCRIPT_PATH = (
    Path(__file__).resolve().parents[3] / "benchmarks/eval/eval_transcribe_diarize.py"
)


def _load_eval_module():
    spec = importlib.util.spec_from_file_location(
        "eval_transcribe_diarize_entry", EVAL_SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_metrics_section_formats_order_and_precision() -> None:
    module = _load_eval_module()

    section = module._build_metrics_section(
        "diarization_metrics_percent",
        {
            "cp_cer": 0.0,
            "cer": 12.34567,
            "count": 3,
            "delta_cer": -7.89123,
        },
        ("cer", "cp_cer", "delta_cer", "count"),
    )

    assert "diarization_metrics_percent" in section
    assert "cer:" in section
    assert "12.35" in section
    assert "-7.89" in section
    assert "count:" in section


def test_parse_args_accepts_max_concurrency_alias(monkeypatch) -> None:
    module = _load_eval_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_transcribe_diarize.py",
            "--max-concurrency",
            "8",
            "--disable-tqdm",
            "--skip-gpu-cleanup",
        ],
    )

    args = module.parse_args()

    assert args.concurrency == 8
    assert args.disable_tqdm is True
    assert args.skip_gpu_cleanup is True
    assert args.max_running_requests is None
    assert args.cuda_graph_max_bs is None
    assert args.mem_fraction_static == module.DEFAULT_SERVER_MEM_FRACTION_STATIC


def test_use_existing_server_waits_for_health(monkeypatch, tmp_path: Path) -> None:
    module = _load_eval_module()
    calls = []
    payload = {
        "summary": {},
        "speed": {"failed_requests": 0},
        "diarization_metrics_percent": {},
    }

    async def fake_run_eval(*args, **kwargs):
        return [], 0.0

    monkeypatch.setattr(
        module,
        "wait_for_service",
        lambda base_url, timeout: calls.append((base_url, timeout)),
    )
    monkeypatch.setattr(module, "run_eval", fake_run_eval)
    monkeypatch.setattr(module, "_build_payload", lambda *args, **kwargs: payload)
    monkeypatch.setattr(
        module,
        "save_json_results",
        lambda _payload, _output_dir, filename: str(tmp_path / filename),
    )

    args = types.SimpleNamespace(
        use_existing_server=True,
        base_url="http://127.0.0.1:8123/",
        host="127.0.0.1",
        port=8000,
        server_timeout_s=42,
        model_path=module.MODEL_PATH,
        language=None,
        concurrency=4,
        warmup=0,
        request_rate=float("inf"),
        disable_tqdm=True,
        request_timeout_s=300,
        output_dir=str(tmp_path),
    )

    module._run_with_or_without_server(args, samples=[])

    assert calls == [("http://127.0.0.1:8123", 42)]


def test_managed_server_gets_memory_and_batch_args(monkeypatch, tmp_path: Path) -> None:
    module = _load_eval_module()
    calls = []
    payload = {
        "summary": {},
        "speed": {"failed_requests": 0},
        "diarization_metrics_percent": {},
    }

    class _ManagedServer:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, traceback):
            return False

    async def fake_run_eval(*args, **kwargs):
        return [], 0.0

    def fake_managed_omni_server(**kwargs):
        calls.append(kwargs)
        return _ManagedServer()

    monkeypatch.setattr(module, "managed_omni_server", fake_managed_omni_server)
    monkeypatch.setattr(module, "run_eval", fake_run_eval)
    monkeypatch.setattr(module, "_build_payload", lambda *args, **kwargs: payload)
    monkeypatch.setattr(
        module,
        "save_json_results",
        lambda _payload, _output_dir, filename: str(tmp_path / filename),
    )

    args = types.SimpleNamespace(
        use_existing_server=False,
        base_url=None,
        host="127.0.0.1",
        port=8000,
        server_timeout_s=42,
        model_path=module.MODEL_PATH,
        language=None,
        concurrency=16,
        max_running_requests=None,
        cuda_graph_max_bs=None,
        mem_fraction_static=module.DEFAULT_SERVER_MEM_FRACTION_STATIC,
        skip_gpu_cleanup=True,
        warmup=0,
        request_rate=float("inf"),
        disable_tqdm=True,
        request_timeout_s=300,
        output_dir=str(tmp_path),
    )

    module._run_with_or_without_server(args, samples=[])

    assert len(calls) == 1
    assert calls[0]["max_running_requests"] == 16
    assert calls[0]["cuda_graph_max_bs"] == 16
    assert calls[0]["mem_fraction_static"] == module.DEFAULT_SERVER_MEM_FRACTION_STATIC
    assert calls[0]["wait_for_gpu_release"] is False


def test_main_returns_nonzero_for_failed_requests(monkeypatch) -> None:
    module = _load_eval_module()
    payload = {
        "summary": {
            "total_samples": 1,
            "evaluated": 0,
            "skipped": 1,
            "exact_matches": 0,
            "mismatches": 0,
            "exact_match_rate": 0.0,
        },
        "speed": {
            "total_requests": 1,
            "completed_requests": 0,
            "failed_requests": 1,
        },
        "diarization_metrics_percent": {
            "cer": None,
            "cer_no_spk": None,
            "cp_cer": None,
            "cer_no_spk_cp_valid": None,
            "delta_cer": None,
            "cer_valid_samples": 0,
            "cp_cer_valid_samples": 0,
            "count": 0,
        },
    }
    args = types.SimpleNamespace(
        max_samples=1,
        repo_id=module.MOVIES800_REPO_ID,
        split="validation",
        audio_column="audio",
        expected_column="transcription",
        model_path=module.MODEL_PATH,
        concurrency=16,
    )

    monkeypatch.setattr(module, "parse_args", lambda: args)
    monkeypatch.setattr(module, "load_movies800_samples", lambda **kwargs: [])
    monkeypatch.setattr(
        module,
        "_run_with_or_without_server",
        lambda _args, _samples: (payload, "result.json"),
    )

    assert module.main() == 1
