from __future__ import annotations

import importlib.util
import sys
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
