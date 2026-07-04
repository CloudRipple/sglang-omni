from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[3] / "benchmarks/tasks/transcribe_diarize.py"
)


class _FakeDataset:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = rows
        self.column_names = list(rows[0].keys()) if rows else []
        self.selected_indices: list[int] | None = None

    def cast_column(self, _name: str, _audio_spec: object) -> "_FakeDataset":
        return self

    def select(self, indices: list[int]) -> "_FakeDataset":
        self.selected_indices = list(indices)
        return _FakeDataset([self._rows[index] for index in indices])

    def __iter__(self):
        return iter(self._rows)

    def __len__(self) -> int:
        return len(self._rows)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "eval_transcribe_diarize", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_load_movies800_samples_stages_audio_bytes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_module()
    dataset = _FakeDataset(
        [
            {
                "audio": {"bytes": b"audio-0", "path": None},
                "file_name": "validation/audio/val_000000.flac",
                "transcription": "[S1] Hello there.",
            }
        ]
    )

    monkeypatch.setitem(
        sys.modules,
        "datasets",
        types.SimpleNamespace(
            Audio=lambda **kwargs: ("Audio", kwargs),
            load_dataset=lambda repo_id, split: dataset,
        ),
    )
    monkeypatch.setattr(module.tempfile, "mkdtemp", lambda prefix: str(tmp_path))
    monkeypatch.setattr(module.atexit, "register", lambda *args, **kwargs: None)

    samples = module.load_movies800_samples(
        repo_id="zhaochenyang20/movies800",
        split="validation",
        audio_column="audio",
        expected_column="transcription",
        max_samples=1,
    )

    assert dataset.selected_indices == [0]
    assert len(samples) == 1
    assert samples[0].sample_id == "validation/audio/val_000000.flac"
    assert samples[0].expected_text == "[S1] Hello there."
    assert Path(samples[0].audio_path).read_bytes() == b"audio-0"


def test_load_movies800_samples_requires_full_800_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    dataset = _FakeDataset(
        [
            {
                "audio": {"bytes": b"audio-0", "path": None},
                "file_name": f"validation/audio/{index}.wav",
                "transcription": "[S1] sample",
            }
            for index in range(799)
        ]
    )

    monkeypatch.setitem(
        sys.modules,
        "datasets",
        types.SimpleNamespace(
            Audio=lambda **kwargs: ("Audio", kwargs),
            load_dataset=lambda repo_id, split: dataset,
        ),
    )
    monkeypatch.setattr(module.atexit, "register", lambda *args, **kwargs: None)

    with pytest.raises(ValueError, match="Expected 800 samples"):
        module.load_movies800_samples(
            repo_id="zhaochenyang20/movies800",
            split="validation",
            audio_column="audio",
            expected_column="transcription",
        )


def test_load_movies800_samples_allows_empty_transcription(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_module()
    dataset = _FakeDataset(
        [
            {
                "audio": {"bytes": b"audio-0", "path": None},
                "file_name": "validation/audio/val_000001.wav",
                "transcription": "",
            }
        ]
    )

    monkeypatch.setitem(
        sys.modules,
        "datasets",
        types.SimpleNamespace(
            Audio=lambda **kwargs: ("Audio", kwargs),
            load_dataset=lambda repo_id, split: dataset,
        ),
    )
    monkeypatch.setattr(module.tempfile, "mkdtemp", lambda prefix: str(tmp_path))
    monkeypatch.setattr(module.atexit, "register", lambda *args, **kwargs: None)

    samples = module.load_movies800_samples(
        repo_id="zhaochenyang20/movies800",
        split="validation",
        audio_column="audio",
        expected_column="transcription",
        max_samples=1,
    )

    assert len(samples) == 1
    assert samples[0].expected_text == ""


def test_extract_prediction_text_prefers_verbose_segments() -> None:
    module = _load_module()

    payload = {
        "text": "[0.00][S01]Hello there.[1.20]",
        "segments": [{"text": "[S01]Hello there."}, {"text": "[S02]Hi."}],
    }

    assert module.extract_prediction_text(payload) == "[S01]Hello there. [S02]Hi."


def test_normalize_transcribe_diarize_text_aligns_speaker_tags() -> None:
    module = _load_module()

    expected = "[0.12][S1] Hello there. [1.50][S02] Hi."
    actual = "[S01]hello there. [S2] hi."

    assert module.normalize_transcribe_diarize_text(
        expected
    ) == module.normalize_transcribe_diarize_text(actual)


def test_build_evaluation_payload_counts_matches_and_skips() -> None:
    module = _load_module()

    samples = [
        module.Movies800Sample(
            sample_id="sample-1",
            audio_path="/tmp/sample-1.wav",
            expected_text="[S1] Hello there.",
        ),
        module.Movies800Sample(
            sample_id="sample-2",
            audio_path="/tmp/sample-2.wav",
            expected_text="[S2] General Kenobi.",
        ),
    ]
    outputs = [
        module.RequestResult(
            request_id="sample-1",
            text="[S01] hello there.",
            is_success=True,
            latency_s=1.0,
            audio_duration_s=2.0,
            rtf=0.5,
        ),
        module.RequestResult(request_id="sample-2", error="boom", is_success=False),
    ]

    payload = module.build_evaluation_payload(
        samples=samples,
        outputs=outputs,
        wall_clock_s=2.0,
        model_path="OpenMOSS-Team/MOSS-Transcribe-Diarize",
        concurrency=4,
    )

    assert payload["summary"] == {
        "total_samples": 2,
        "evaluated": 1,
        "skipped": 1,
        "exact_matches": 1,
        "mismatches": 0,
        "exact_match_rate": 1.0,
    }
    assert payload["diarization_metrics"]["cer"] == 0.0
    assert payload["diarization_metrics"]["cp_cer"] == 0.0
    assert payload["diarization_metrics"]["delta_cer"] == 0.0
    assert "core3_metrics" not in payload
    assert payload["per_sample"][0]["is_exact_match"] is True
    assert payload["per_sample"][0]["cp_cer_valid"] is True
    assert payload["per_sample"][1]["error"] == "boom"


def test_build_evaluation_payload_reports_diarization_metrics_for_speaker_swap() -> (
    None
):
    module = _load_module()

    samples = [
        module.Movies800Sample(
            sample_id="sample-1",
            audio_path="/tmp/sample-1.wav",
            expected_text="[S1] hello [S2] world",
        )
    ]
    outputs = [
        module.RequestResult(
            request_id="sample-1",
            text="[S1] world [S2] hello",
            is_success=True,
            latency_s=1.0,
            audio_duration_s=2.0,
            rtf=0.5,
        )
    ]

    payload = module.build_evaluation_payload(
        samples=samples,
        outputs=outputs,
        wall_clock_s=2.0,
        model_path="OpenMOSS-Team/MOSS-Transcribe-Diarize",
        concurrency=4,
    )

    assert payload["diarization_metrics"]["cer"] is not None
    assert payload["diarization_metrics"]["cp_cer"] == pytest.approx(0.0)
    assert payload["diarization_metrics"]["cer"] > 0.0
    assert payload["diarization_metrics"]["delta_cer"] < 0.0
    assert payload["per_sample"][0]["cp_cer_valid"] is True
