from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from benchmarks.benchmarker.data import RequestResult
from benchmarks.dataset import prepare
from benchmarks.eval import benchmark_tts_voice_taxonomy as voice_taxonomy


def _write_wav(path: Path, duration_s: float, sample_rate: int = 100) -> None:
    frames = int(duration_s * sample_rate)
    sf.write(path, np.zeros(frames, dtype=np.float32), sample_rate, subtype="PCM_16")


def _config_from_cli(*args: str) -> voice_taxonomy.VoiceTaxonomyBenchmarkConfig:
    parser = voice_taxonomy._build_arg_parser()
    namespace = parser.parse_args(list(args))
    voice_taxonomy._validate_args(parser, namespace)
    return voice_taxonomy._config_from_args(namespace)


def test_voice_taxonomy_cli_defaults_match_metric_contract() -> None:
    config = _config_from_cli()

    assert config.model == "OpenMOSS-Team/MOSS-TTS-v1.5"
    assert config.meta == "OpenMOSS-Team/MOSS-TTS-Voice-Taxonomy-Eval"
    assert config.lang == "all"
    assert config.max_new_tokens == 16384
    assert config.tts_request_timeout_s == 1200
    assert config.repeat_count == 1
    assert config.repeat_aggregate == "mean"
    assert config.asr_model_path == "Qwen/Qwen3-ASR-1.7B"
    assert config.asr_max_new_tokens == 8192
    assert config.asr_max_running_requests == 8
    assert config.asr_chunk_batch_size == 4
    assert config.asr_chunk_seconds == 30.0
    assert config.sim_threshold_seconds == 60.0
    assert config.sim_window_seconds == 30.0

    generation_config = voice_taxonomy._seedtts_config_for_repeat(config, "en", 0)
    assert generation_config.voice_clone is True
    assert generation_config.ref_format == "references"
    assert generation_config.request_timeout_s == 1200
    assert generation_config.output_dir.endswith("voice-taxonomy-en")


def test_voice_taxonomy_dataset_is_downloadable_by_name() -> None:
    assert (
        prepare.DATASETS["moss-tts-voice-taxonomy"]
        == "OpenMOSS-Team/MOSS-TTS-Voice-Taxonomy-Eval"
    )


def test_voice_taxonomy_normalization_handles_pause_apostrophe_and_cjk() -> None:
    assert (
        voice_taxonomy.normalize_voice_taxonomy_text(
            "I'd [pause 0.5s] pay ＄5—today!",
            "en",
        )
        == "i'd pay 5 today"
    )
    assert (
        voice_taxonomy.normalize_voice_taxonomy_text(
            "rock ’ n ’ roll",
            "en",
        )
        == "rock n roll"
    )
    assert (
        voice_taxonomy.normalize_voice_taxonomy_text(
            "後臺，測試 [pause 2s]",
            "zh",
        )
        == "后 台 测 试"
    )


def test_voice_taxonomy_asr_chunks_use_non_overlapping_30s_windows(
    tmp_path: Path,
) -> None:
    wav_path = tmp_path / "long.wav"
    _write_wav(wav_path, 65.0)
    generated = [
        {
            "sample_id": "sample-1",
            "target_text": "hello",
            "wav_path": str(wav_path),
            "is_success": True,
            "audio_duration_s": 65.0,
        }
    ]

    samples, chunks_by_sample, failures = voice_taxonomy._prepare_asr_chunks(
        generated,
        repeat_index=0,
        chunk_dir=tmp_path / "chunks",
        chunk_seconds=30.0,
    )

    assert failures == {}
    assert len(samples) == 3
    chunks = chunks_by_sample["sample-1"]
    assert [chunk["chunk_index"] for chunk in chunks] == [0, 1, 2]
    assert [chunk["duration_s"] for chunk in chunks] == pytest.approx([30.0, 30.0, 5.0])


def test_voice_taxonomy_repeat_aggregation_supports_mean_and_best() -> None:
    rows = [
        {"id": "sample-1", "repeat_index": 0, "wer": 0.2},
        {"id": "sample-1", "repeat_index": 1, "wer": 0.4},
    ]

    mean_rows = voice_taxonomy._aggregate_repeat_rows(
        rows,
        metric="wer",
        aggregate="mean",
        best=min,
    )
    best_rows = voice_taxonomy._aggregate_repeat_rows(
        rows,
        metric="wer",
        aggregate="best",
        best=min,
    )

    assert mean_rows[0]["wer"] == pytest.approx(0.3)
    assert mean_rows[0]["wer_repeat_best"] == pytest.approx(0.2)
    assert mean_rows[0]["wer_repeat_var"] == pytest.approx(0.01)
    assert best_rows[0]["wer"] == pytest.approx(0.2)


def test_voice_taxonomy_similarity_uses_full_or_head_tail_policy(
    tmp_path: Path,
) -> None:
    ref_path = tmp_path / "ref.wav"
    short_path = tmp_path / "short.wav"
    long_path = tmp_path / "long.wav"
    _write_wav(ref_path, 2.0)
    _write_wav(short_path, 60.0)
    _write_wav(long_path, 65.0)
    generated = [
        {
            "sample_id": "short",
            "wav_path": str(short_path),
            "is_success": True,
        },
        {
            "sample_id": "long",
            "wav_path": str(long_path),
            "is_success": True,
        },
    ]

    segments, failures = voice_taxonomy._prepare_similarity_segments(
        generated,
        {"short": str(ref_path), "long": str(ref_path)},
        repeat_index=0,
        segment_dir=tmp_path / "segments",
        threshold_seconds=60.0,
        window_seconds=30.0,
    )

    assert failures == []
    assert [(segment["id"], segment["part"]) for segment in segments] == [
        ("short", "full"),
        ("long", "head"),
        ("long", "tail"),
    ]
    assert sf.info(segments[1]["segment_path"]).duration == pytest.approx(30.0)
    assert sf.info(segments[2]["segment_path"]).duration == pytest.approx(30.0)


def test_voice_taxonomy_similarity_restores_cosine_scale_and_averages_windows(
    tmp_path: Path,
) -> None:
    config = voice_taxonomy.VoiceTaxonomyBenchmarkConfig(
        output_dir=str(tmp_path),
        lang="en",
        disable_tqdm=True,
    )
    subset_dir = tmp_path / "voice-taxonomy-en"
    subset_dir.mkdir()
    ref_path = tmp_path / "ref.wav"
    short_path = tmp_path / "short.wav"
    long_path = tmp_path / "long.wav"
    _write_wav(ref_path, 2.0)
    _write_wav(short_path, 10.0)
    _write_wav(long_path, 65.0)
    (subset_dir / "generated.json").write_text(
        json.dumps(
            [
                {
                    "sample_id": "short",
                    "wav_path": str(short_path),
                    "is_success": True,
                },
                {
                    "sample_id": "long",
                    "wav_path": str(long_path),
                    "is_success": True,
                },
            ]
        )
    )

    class FakeScorer:
        def score_batch(self, ref_audio_paths, wav_paths):
            assert ref_audio_paths == [str(ref_path)] * 3
            assert len(wav_paths) == 3
            return [75.0, 80.0, 60.0]

    rows = voice_taxonomy._score_similarity_repeat(
        config,
        "en",
        0,
        scorer=FakeScorer(),
        ref_audio_by_id={"short": str(ref_path), "long": str(ref_path)},
    )
    by_id = {row["id"]: row for row in rows}

    assert by_id["short"]["sim"] == pytest.approx(0.75)
    assert by_id["short"]["sim_policy"] == "full"
    assert by_id["long"]["sim_head"] == pytest.approx(0.8)
    assert by_id["long"]["sim_tail"] == pytest.approx(0.6)
    assert by_id["long"]["sim"] == pytest.approx(0.7)


@pytest.mark.asyncio
async def test_voice_taxonomy_failed_asr_chunk_scores_as_empty_transcript(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = voice_taxonomy.VoiceTaxonomyBenchmarkConfig(
        output_dir=str(tmp_path),
        lang="en",
        disable_tqdm=True,
    )
    subset_dir = tmp_path / "voice-taxonomy-en"
    audio_dir = subset_dir / "audio"
    audio_dir.mkdir(parents=True)
    wav_path = audio_dir / "sample-1.wav"
    _write_wav(wav_path, 2.0)
    (subset_dir / "generated.json").write_text(
        json.dumps(
            [
                {
                    "sample_id": "sample-1",
                    "target_text": "hello world",
                    "wav_path": str(wav_path),
                    "is_success": True,
                    "audio_duration_s": 2.0,
                }
            ]
        )
    )

    async def fake_run_asr(samples, **kwargs):
        del kwargs
        return (
            [
                RequestResult(
                    request_id=sample.sample_id,
                    is_success=False,
                    error="synthetic ASR failure",
                )
                for sample in samples
            ],
            0.1,
        )

    monkeypatch.setattr(voice_taxonomy, "run_asr_transcription", fake_run_asr)

    rows, stats = await voice_taxonomy._score_wer_repeat(
        config,
        "en",
        0,
        asr_router_port=8000,
    )

    assert rows[0]["wer"] == pytest.approx(1.0)
    assert rows[0]["is_success"] is True
    assert rows[0]["asr_success"] is False
    assert "synthetic ASR failure" in rows[0]["error"]
    assert stats["successful_chunks"] == 0


@pytest.mark.asyncio
async def test_voice_taxonomy_overall_wer_is_sample_weighted_macro_mean(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = voice_taxonomy.VoiceTaxonomyBenchmarkConfig(
        output_dir=str(tmp_path),
        lang="all",
    )

    async def fake_score_repeat(config, lang, repeat_index, *, asr_router_port):
        del config, repeat_index, asr_router_port
        scores = [0.0, 1.0] if lang == "zh" else [0.0]
        return (
            [
                {
                    "id": f"{lang}-{idx}",
                    "repeat_index": 0,
                    "wer": score,
                }
                for idx, score in enumerate(scores)
            ],
            {
                "chunk_count": len(scores),
                "successful_chunks": len(scores),
                "asr_wall_time_s": 1.0,
            },
        )

    monkeypatch.setattr(voice_taxonomy, "_score_wer_repeat", fake_score_repeat)

    metrics = await voice_taxonomy.run_voice_taxonomy_wer(
        config,
        asr_router_port=8000,
    )

    assert metrics["voice-taxonomy-zh_wer"] == pytest.approx(0.5)
    assert metrics["voice-taxonomy-en_wer"] == pytest.approx(0.0)
    assert metrics["overall_wer"] == pytest.approx(1.0 / 3.0, abs=5e-5)


def test_voice_taxonomy_managed_asr_passes_stage_max_new_tokens(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    async def fake_run_wer(config, *, asr_router_port):
        captured.update({"config": config, "asr_router_port": asr_router_port})
        return {}

    monkeypatch.setattr(voice_taxonomy, "run_voice_taxonomy_wer", fake_run_wer)

    class FakeManagedServer:
        def __init__(self, **kwargs) -> None:
            captured["server_kwargs"] = kwargs

        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, traceback) -> None:
            del exc_type, exc, traceback

    monkeypatch.setattr(
        voice_taxonomy,
        "managed_omni_server",
        lambda **kwargs: FakeManagedServer(**kwargs),
    )
    config = voice_taxonomy.VoiceTaxonomyBenchmarkConfig(output_dir=str(tmp_path))
    args = SimpleNamespace(
        use_existing_asr_server=False,
        server_timeout=1200,
        skip_gpu_cleanup=False,
    )
    voice_taxonomy._run_wer_phase(config, args)

    server_kwargs = captured["server_kwargs"]
    assert isinstance(server_kwargs, dict)
    assert server_kwargs["extra_cli_args"] == [
        "--stages.asr.factory_args.max_new_tokens",
        "8192",
    ]
