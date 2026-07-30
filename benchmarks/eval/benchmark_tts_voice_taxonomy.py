# SPDX-License-Identifier: Apache-2.0
"""MOSS-TTS Voice Taxonomy benchmark: generation speed, WER/CER, and SIM.

This entry point follows the SeedTTS benchmark lifecycle while preserving the
metric contract documented for ``OpenMOSS-Team/MOSS-TTS-Voice-Taxonomy-Eval``:

* evaluate both the 241-sample English split and 243-sample Chinese split;
* transcribe generated audio with Qwen3-ASR in non-overlapping 30-second chunks;
* report sample-level macro WER (English) / CER (Chinese), not corpus WER;
* score speaker similarity against the prompt audio with the SeedTTS WavLM
  backend, using full generated audio up to 60 seconds and head/tail 30-second
  windows for longer generations;
* aggregate repeated generations per sample with ``mean`` or ``best``.

The generated-audio layout remains SeedTTS-compatible. Repeat zero is written
directly under each subset directory; later repeats use ``repeat_XX``::

    <output_dir>/
      voice-taxonomy-en/
        audio/*.wav
        generated.json
        repeat_01/audio/*.wav
        wer_results.json
        similarity_results.json
      voice-taxonomy-zh/
        ...
      voice-taxonomy-eval.json

Full standard run (TTS, then ASR, then offline SIM on one GPU)::

    python -m benchmarks.eval.benchmark_tts_voice_taxonomy \
        --model OpenMOSS-Team/MOSS-TTS-v1.5 \
        --max-concurrency 16 \
        --output-dir results/moss_voice_taxonomy

The dataset is private. Authenticate with Hugging Face before the first run, or
pre-warm the default cache with::

    python -m benchmarks.dataset.prepare --dataset moss-tts-voice-taxonomy
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import json
import logging
import math
import os
import re
import tempfile
import unicodedata
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import soundfile as sf
import torch
from jiwer import process_words
from opencc import OpenCC
from tqdm import tqdm

from benchmarks.benchmarker.data import RequestResult
from benchmarks.benchmarker.utils import managed_omni_server, save_json_results
from benchmarks.dataset.seedtts import SampleInput, load_seedtts_samples
from benchmarks.eval.benchmark_tts_seedtts import (
    DEFAULT_TTS_BENCHMARK_CONCURRENCY,
    TtsSeedttsBenchmarkConfig,
    run_tts_seedtts_benchmark,
)
from benchmarks.metrics.speaker_similarity import WavLMSpeakerSimilarity
from benchmarks.metrics.speaker_similarity_assets import (
    ensure_speaker_similarity_assets,
)
from benchmarks.tasks.asr import QWEN3_ASR_MODEL_PATH, run_asr_transcription
from benchmarks.tasks.tts import MOSS_TTS_TOKEN_COUNT_AUTO

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

VOICE_TAXONOMY_DATASET = "OpenMOSS-Team/MOSS-TTS-Voice-Taxonomy-Eval"
VOICE_TAXONOMY_MODEL = "OpenMOSS-Team/MOSS-TTS-v1.5"
VOICE_TAXONOMY_BENCHMARK = "moss-tts-voice-taxonomy"
VOICE_TAXONOMY_NORMALIZATION_VERSION = (
    "moss-voice-taxonomy-20260605-mixed-cjk-punct-space"
)
VOICE_TAXONOMY_SIM_POLICY = "long_safe_head_tail_30s"
VOICE_TAXONOMY_LANGS = ("zh", "en")
VOICE_TAXONOMY_ASR_CHUNK_SECONDS = 30.0
VOICE_TAXONOMY_SIM_THRESHOLD_SECONDS = 60.0
VOICE_TAXONOMY_SIM_WINDOW_SECONDS = 30.0
VOICE_TAXONOMY_SIM_BATCH_SIZE = 8
VOICE_TAXONOMY_ASR_CHUNK_BATCH_SIZE = 4
VOICE_TAXONOMY_ASR_MAX_RUNNING_REQUESTS = 8
VOICE_TAXONOMY_ASR_MAX_NEW_TOKENS = 8192
VOICE_TAXONOMY_TTS_MAX_NEW_TOKENS = 16384
VOICE_TAXONOMY_TTS_REQUEST_TIMEOUT_S = 1200

_PAUSE_MARKER_RE = re.compile(
    r"\[\s*pause\s+\d+(?:\.\d+)?s\s*\]",
    flags=re.IGNORECASE,
)
_APOSTROPHES = frozenset({"'", "‘", "’", "ʻ", "ʼ", "＇"})


@dataclass
class VoiceTaxonomyBenchmarkConfig:
    model: str = VOICE_TAXONOMY_MODEL
    meta: str = VOICE_TAXONOMY_DATASET
    base_url: str | None = None
    host: str = "localhost"
    port: int = 8000
    asr_port: int | None = None
    output_dir: str = "results/tts_voice_taxonomy"
    lang: str = "all"
    max_samples: int | None = None
    sample_offset: int = 0
    repeat_count: int = 1
    repeat_aggregate: str = "mean"
    response_format: str = "wav"
    max_new_tokens: int | None = VOICE_TAXONOMY_TTS_MAX_NEW_TOKENS
    token_count: int | str | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    repetition_penalty: float | None = None
    seed: int | None = None
    warmup: int = 1
    concurrency: int = DEFAULT_TTS_BENCHMARK_CONCURRENCY
    request_rate: float = float("inf")
    tts_request_timeout_s: int = VOICE_TAXONOMY_TTS_REQUEST_TIMEOUT_S
    stream: bool = False
    initial_codec_chunk_frames: int | None = None
    disable_tqdm: bool = False
    max_running_requests: int = 16
    cuda_graph_max_bs: int = 16
    device: str = "cuda:0"
    similarity_checkpoint: str | None = None
    asr_model_path: str = QWEN3_ASR_MODEL_PATH
    asr_max_new_tokens: int = VOICE_TAXONOMY_ASR_MAX_NEW_TOKENS
    asr_max_running_requests: int = VOICE_TAXONOMY_ASR_MAX_RUNNING_REQUESTS
    asr_chunk_batch_size: int = VOICE_TAXONOMY_ASR_CHUNK_BATCH_SIZE
    asr_chunk_seconds: float = VOICE_TAXONOMY_ASR_CHUNK_SECONDS
    sim_batch_size: int = VOICE_TAXONOMY_SIM_BATCH_SIZE
    sim_threshold_seconds: float = VOICE_TAXONOMY_SIM_THRESHOLD_SECONDS
    sim_window_seconds: float = VOICE_TAXONOMY_SIM_WINDOW_SECONDS


def _selected_languages(lang: str) -> tuple[str, ...]:
    if lang == "all":
        return VOICE_TAXONOMY_LANGS
    if lang not in VOICE_TAXONOMY_LANGS:
        raise ValueError(f"Unsupported Voice Taxonomy language: {lang}")
    return (lang,)


def _subset_name(lang: str) -> str:
    return f"voice-taxonomy-{lang}"


def _subset_dir(config: VoiceTaxonomyBenchmarkConfig, lang: str) -> Path:
    return Path(config.output_dir) / _subset_name(lang)


def _repeat_output_dir(
    config: VoiceTaxonomyBenchmarkConfig,
    lang: str,
    repeat_index: int,
) -> Path:
    subset_dir = _subset_dir(config, lang)
    if repeat_index == 0:
        return subset_dir
    return subset_dir / f"repeat_{repeat_index:02d}"


def _load_selected_samples(
    config: VoiceTaxonomyBenchmarkConfig,
    lang: str,
) -> list[SampleInput]:
    head = None
    if config.max_samples is not None:
        head = config.sample_offset + config.max_samples
    samples = load_seedtts_samples(config.meta, head, split=lang)
    return samples[config.sample_offset :]


def _seedtts_config_for_repeat(
    config: VoiceTaxonomyBenchmarkConfig,
    lang: str,
    repeat_index: int,
) -> TtsSeedttsBenchmarkConfig:
    return TtsSeedttsBenchmarkConfig(
        model=config.model,
        meta=config.meta,
        base_url=config.base_url,
        host=config.host,
        port=config.port,
        voice_clone=True,
        ref_format="references",
        response_format="pcm" if config.stream else config.response_format,
        output_dir=str(_repeat_output_dir(config, lang, repeat_index)),
        max_samples=config.max_samples,
        sample_offset=config.sample_offset,
        max_new_tokens=config.max_new_tokens,
        token_count=config.token_count,
        temperature=config.temperature,
        top_p=config.top_p,
        top_k=config.top_k,
        repetition_penalty=config.repetition_penalty,
        seed=config.seed,
        warmup=config.warmup,
        concurrency=config.concurrency,
        request_rate=config.request_rate,
        request_timeout_s=config.tts_request_timeout_s,
        stream=config.stream,
        initial_codec_chunk_frames=config.initial_codec_chunk_frames,
        disable_tqdm=config.disable_tqdm,
        max_running_requests=config.max_running_requests,
        cuda_graph_max_bs=config.cuda_graph_max_bs,
        lang=lang,
        device=config.device,
        similarity_checkpoint=config.similarity_checkpoint,
        asr_model_path=config.asr_model_path,
        asr_concurrency=config.asr_chunk_batch_size,
    )


def _public_config(config: VoiceTaxonomyBenchmarkConfig) -> dict[str, Any]:
    result = asdict(config)
    if math.isinf(config.request_rate):
        result["request_rate"] = "inf"
    result["benchmark"] = VOICE_TAXONOMY_BENCHMARK
    result["normalization_version"] = VOICE_TAXONOMY_NORMALIZATION_VERSION
    result["sim_policy"] = VOICE_TAXONOMY_SIM_POLICY
    return result


def _update_eval_results(
    config: VoiceTaxonomyBenchmarkConfig,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "voice-taxonomy-eval.json"
    if result_path.is_file():
        with result_path.open() as result_file:
            payload = json.load(result_file)
    else:
        payload = {
            "benchmark": VOICE_TAXONOMY_BENCHMARK,
            "config": _public_config(config),
        }
    payload["config"] = _public_config(config)
    payload.update(metrics)
    save_json_results(payload, config.output_dir, result_path.name)
    return payload


async def run_voice_taxonomy_generation(
    config: VoiceTaxonomyBenchmarkConfig,
) -> dict[str, Any]:
    """Generate all requested splits/repeats through the SeedTTS TTS harness."""
    results: dict[str, Any] = {}
    for lang in _selected_languages(config.lang):
        split_results: list[dict[str, Any]] = []
        for repeat_index in range(config.repeat_count):
            repeat_config = _seedtts_config_for_repeat(config, lang, repeat_index)
            logger.info(
                "Generating %s repeat %d/%d into %s",
                _subset_name(lang),
                repeat_index + 1,
                config.repeat_count,
                repeat_config.output_dir,
            )
            benchmark_result = await run_tts_seedtts_benchmark(repeat_config)
            split_results.append(
                {
                    "repeat_index": repeat_index,
                    "output_dir": repeat_config.output_dir,
                    "summary": benchmark_result["summary"],
                }
            )
        results[_subset_name(lang)] = split_results

    payload = {
        "benchmark": VOICE_TAXONOMY_BENCHMARK,
        "config": _public_config(config),
        "splits": results,
    }
    save_json_results(payload, config.output_dir, "generation_results.json")
    return payload


@functools.lru_cache(maxsize=1)
def _simplified_chinese_converter() -> OpenCC:
    return OpenCC("t2s")


def _is_cjk_ideograph(char: str) -> bool:
    codepoint = ord(char)
    return (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
    )


def _normalize_unicode_punctuation(text: str) -> str:
    text = unicodedata.normalize("NFKC", _PAUSE_MARKER_RE.sub(" ", text))
    normalized: list[str] = []
    for index, char in enumerate(text):
        if char in _APOSTROPHES:
            previous = text[index - 1] if index > 0 else ""
            following = text[index + 1] if index + 1 < len(text) else ""
            normalized.append(
                "'" if previous.isalnum() and following.isalnum() else " "
            )
            continue
        if unicodedata.category(char)[:1] in {"P", "S"}:
            normalized.append(" ")
        else:
            normalized.append(char)
    return " ".join("".join(normalized).split())


def normalize_voice_taxonomy_text(text: str, lang: str) -> str:
    """Apply the Voice Taxonomy mixed CJK/punctuation normalization contract."""
    normalized = _normalize_unicode_punctuation(text or "")
    if lang == "zh":
        normalized = _simplified_chinese_converter().convert(normalized)
        compact = "".join(normalized.split())
        return " ".join(compact)
    if lang != "en":
        raise ValueError(f"Unsupported Voice Taxonomy language: {lang}")

    pieces: list[str] = []
    for char in normalized.lower():
        if _is_cjk_ideograph(char):
            pieces.extend((" ", char, " "))
        else:
            pieces.append(char)
    return " ".join("".join(pieces).split())


def _load_generated_entries(
    config: VoiceTaxonomyBenchmarkConfig,
    lang: str,
    repeat_index: int,
) -> list[dict[str, Any]]:
    generated_path = _repeat_output_dir(config, lang, repeat_index) / "generated.json"
    if not generated_path.is_file():
        raise FileNotFoundError(
            f"Missing generated audio metadata for {_subset_name(lang)} repeat "
            f"{repeat_index}: {generated_path}"
        )
    with generated_path.open() as generated_file:
        return json.load(generated_file)


def _generation_failure(entry: dict[str, Any]) -> str | None:
    wav_path = entry.get("wav_path")
    if not entry.get("is_success"):
        return str(entry.get("error") or "generation reported is_success=False")
    if not isinstance(wav_path, str) or not wav_path:
        return "wav_path missing from generated.json entry"
    if not os.path.isfile(wav_path):
        return f"wav file not on disk: {wav_path}"
    return None


def _prepare_asr_chunks(
    generated: list[dict[str, Any]],
    *,
    repeat_index: int,
    chunk_dir: Path,
    chunk_seconds: float,
) -> tuple[list[SampleInput], dict[str, list[dict[str, Any]]], dict[str, str]]:
    chunk_dir.mkdir(parents=True, exist_ok=True)
    chunk_samples: list[SampleInput] = []
    chunks_by_sample: dict[str, list[dict[str, Any]]] = {}
    failures: dict[str, str] = {}

    for entry_index, entry in enumerate(generated):
        sample_id = str(entry.get("sample_id") or "")
        failure = _generation_failure(entry)
        if not sample_id:
            continue
        if failure is not None:
            failures[sample_id] = failure
            continue

        wav_path = str(entry["wav_path"])
        try:
            info = sf.info(wav_path)
        except (OSError, RuntimeError) as exc:
            failures[sample_id] = f"failed to inspect generated audio: {exc}"
            continue
        chunk_frames = max(1, int(round(chunk_seconds * info.samplerate)))
        descriptors: list[dict[str, Any]] = []

        if info.frames <= chunk_frames:
            chunk_paths = [(wav_path, int(info.frames))]
        else:
            chunk_paths: list[tuple[str, int]] = []
            with sf.SoundFile(wav_path) as source:
                chunk_index = 0
                while source.tell() < len(source):
                    audio = source.read(
                        chunk_frames,
                        dtype="float32",
                        always_2d=True,
                    )
                    if audio.shape[0] == 0:
                        break
                    chunk_path = chunk_dir / (
                        f"sample_{entry_index:04d}_chunk_{chunk_index:04d}.wav"
                    )
                    sf.write(
                        chunk_path,
                        audio,
                        source.samplerate,
                        subtype="PCM_16",
                    )
                    chunk_paths.append((str(chunk_path), int(audio.shape[0])))
                    chunk_index += 1

        for chunk_index, (chunk_path, frame_count) in enumerate(chunk_paths):
            chunk_id = f"repeat_{repeat_index:02d}:{sample_id}:chunk_{chunk_index:04d}"
            descriptor = {
                "chunk_id": chunk_id,
                "chunk_index": chunk_index,
                "wav_path": chunk_path,
                "duration_s": frame_count / info.samplerate,
            }
            descriptors.append(descriptor)
            chunk_samples.append(
                SampleInput(
                    sample_id=chunk_id,
                    ref_text="",
                    ref_audio=chunk_path,
                    target_text="",
                )
            )
        chunks_by_sample[sample_id] = descriptors

    return chunk_samples, chunks_by_sample, failures


def _wer_failure_row(
    entry: dict[str, Any],
    repeat_index: int,
    error: str,
) -> dict[str, Any]:
    return {
        "id": entry.get("sample_id"),
        "repeat_index": repeat_index,
        "target_text": entry.get("target_text", ""),
        "asr_text": "",
        "ref_norm": "",
        "hyp_norm": "",
        "wer": None,
        "substitutions": None,
        "deletions": None,
        "insertions": None,
        "hits": None,
        "audio_duration_s": entry.get("audio_duration_s", 0.0),
        "asr_latency_s": 0.0,
        "chunk_count": 0,
        "asr_success": False,
        "is_success": False,
        "error": error,
    }


async def _score_wer_repeat(
    config: VoiceTaxonomyBenchmarkConfig,
    lang: str,
    repeat_index: int,
    *,
    asr_router_port: int,
) -> tuple[list[dict[str, Any]], dict[str, float | int]]:
    generated = _load_generated_entries(config, lang, repeat_index)
    with tempfile.TemporaryDirectory(prefix=f"voice_taxonomy_{lang}_asr_") as tmpdir:
        chunk_samples, chunks_by_sample, failures = _prepare_asr_chunks(
            generated,
            repeat_index=repeat_index,
            chunk_dir=Path(tmpdir),
            chunk_seconds=config.asr_chunk_seconds,
        )
        if chunk_samples:
            chunk_outputs, wall_time_s = await run_asr_transcription(
                chunk_samples,
                host=config.host,
                port=asr_router_port,
                model_path=config.asr_model_path,
                lang=lang,
                concurrency=config.asr_chunk_batch_size,
                warmup=0,
                disable_tqdm=config.disable_tqdm,
            )
        else:
            chunk_outputs = []
            wall_time_s = 0.0

    output_by_id: dict[str, RequestResult] = {
        output.request_id: output for output in chunk_outputs
    }
    rows: list[dict[str, Any]] = []
    successful_chunks = 0

    for entry in generated:
        sample_id = str(entry.get("sample_id") or "")
        if sample_id in failures:
            rows.append(_wer_failure_row(entry, repeat_index, failures[sample_id]))
            continue

        descriptors = chunks_by_sample.get(sample_id)
        if not descriptors:
            rows.append(
                _wer_failure_row(
                    entry,
                    repeat_index,
                    "no ASR chunks were prepared for generated audio",
                )
            )
            continue

        chunk_texts: list[str] = []
        chunk_errors: list[str] = []
        asr_latency_s = 0.0
        for descriptor in descriptors:
            output = output_by_id.get(descriptor["chunk_id"])
            if output is None or not output.is_success:
                chunk_texts.append("")
                error = (output.error if output is not None else "") or "no response"
                chunk_errors.append(
                    f"chunk {descriptor['chunk_index']} transcription failed: {error}"
                )
                continue
            successful_chunks += 1
            asr_latency_s += output.latency_s
            chunk_texts.append(output.text.strip())

        target_text = str(entry.get("target_text") or "")
        hypothesis = " ".join(text for text in chunk_texts if text)
        ref_norm = normalize_voice_taxonomy_text(target_text, lang)
        hyp_norm = normalize_voice_taxonomy_text(hypothesis, lang)
        if not ref_norm:
            rows.append(
                _wer_failure_row(
                    entry,
                    repeat_index,
                    "empty reference after Voice Taxonomy normalization",
                )
            )
            continue

        measures = process_words(ref_norm, hyp_norm)
        rows.append(
            {
                "id": sample_id,
                "repeat_index": repeat_index,
                "target_text": target_text,
                "asr_text": hypothesis,
                "ref_norm": ref_norm,
                "hyp_norm": hyp_norm,
                "wer": float(measures.wer),
                "substitutions": int(measures.substitutions),
                "deletions": int(measures.deletions),
                "insertions": int(measures.insertions),
                "hits": int(measures.hits),
                "audio_duration_s": float(entry.get("audio_duration_s") or 0.0),
                "asr_latency_s": asr_latency_s,
                "chunk_count": len(descriptors),
                "asr_success": not chunk_errors,
                "is_success": True,
                "error": "; ".join(chunk_errors) or None,
            }
        )

    stats: dict[str, float | int] = {
        "chunk_count": len(chunk_samples),
        "successful_chunks": successful_chunks,
        "asr_wall_time_s": float(wall_time_s),
    }
    return rows, stats


def _aggregate_repeat_rows(
    rows: list[dict[str, Any]],
    *,
    metric: str,
    aggregate: str,
    best: Callable[[list[float]], float],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        sample_id = row.get("id")
        if isinstance(sample_id, str) and sample_id:
            grouped[sample_id].append(row)

    aggregated: list[dict[str, Any]] = []
    for sample_id, repeat_rows in grouped.items():
        repeat_rows.sort(key=lambda row: int(row.get("repeat_index", 0)))
        scores = [
            float(row[metric])
            for row in repeat_rows
            if isinstance(row.get(metric), (int, float))
            and math.isfinite(float(row[metric]))
        ]
        if scores:
            score_mean = float(np.mean(scores))
            score_best = float(best(scores))
            score_var = float(np.var(scores))
            selected = score_mean if aggregate == "mean" else score_best
        else:
            score_mean = score_best = score_var = selected = None

        aggregated.append(
            {
                "id": sample_id,
                metric: selected,
                f"{metric}_repeat_mean": score_mean,
                f"{metric}_repeat_best": score_best,
                f"{metric}_repeat_var": score_var,
                "repeat_count": len(scores),
                "is_success": selected is not None,
                "repeats": repeat_rows,
            }
        )
    return aggregated


async def run_voice_taxonomy_wer(
    config: VoiceTaxonomyBenchmarkConfig,
    *,
    asr_router_port: int,
) -> dict[str, Any]:
    """Compute Voice Taxonomy Qwen3-ASR macro WER/CER for saved audio."""
    flat_metrics: dict[str, Any] = {
        "wer_asr_backend": "qwen3_asr",
        "wer_normalization_version": VOICE_TAXONOMY_NORMALIZATION_VERSION,
        "wer_repeat_aggregate": config.repeat_aggregate,
    }
    all_selected_scores: list[float] = []
    all_sample_variances: list[float] = []

    for lang in _selected_languages(config.lang):
        repeat_rows: list[dict[str, Any]] = []
        asr_stats: list[dict[str, float | int]] = []
        for repeat_index in range(config.repeat_count):
            rows, stats = await _score_wer_repeat(
                config,
                lang,
                repeat_index,
                asr_router_port=asr_router_port,
            )
            repeat_rows.extend(rows)
            asr_stats.append(stats)

        per_sample = _aggregate_repeat_rows(
            repeat_rows,
            metric="wer",
            aggregate=config.repeat_aggregate,
            best=min,
        )
        selected_scores = [
            float(row["wer"])
            for row in per_sample
            if isinstance(row.get("wer"), (int, float))
        ]
        if not selected_scores:
            raise RuntimeError(f"No scoreable WER samples for {_subset_name(lang)}")
        variances = [
            float(row["wer_repeat_var"])
            for row in per_sample
            if isinstance(row.get("wer_repeat_var"), (int, float))
        ]
        subset = _subset_name(lang)
        subset_score = round(float(np.mean(selected_scores)), 4)
        flat_metrics[f"{subset}_wer"] = subset_score
        flat_metrics[f"{subset}_wer_repeat_count"] = config.repeat_count
        flat_metrics[f"{subset}_wer_evaluated"] = len(selected_scores)
        flat_metrics[f"{subset}_wer_total"] = len(per_sample)
        if config.repeat_count > 1:
            flat_metrics[f"{subset}_wer_var"] = round(float(np.mean(variances)), 4)

        save_json_results(
            {
                "summary": {
                    "wer": subset_score,
                    "language": lang,
                    "metric": "cer" if lang == "zh" else "wer",
                    "aggregation": "macro_sample_mean",
                    "repeat_count": config.repeat_count,
                    "repeat_aggregate": config.repeat_aggregate,
                    "evaluated": len(selected_scores),
                    "total_samples": len(per_sample),
                    "asr_chunk_count": sum(
                        int(stats["chunk_count"]) for stats in asr_stats
                    ),
                    "asr_successful_chunks": sum(
                        int(stats["successful_chunks"]) for stats in asr_stats
                    ),
                    "asr_wall_time_s": sum(
                        float(stats["asr_wall_time_s"]) for stats in asr_stats
                    ),
                    "asr_backend": "qwen3_asr",
                    "normalization_version": VOICE_TAXONOMY_NORMALIZATION_VERSION,
                },
                "config": _public_config(config),
                "per_sample": per_sample,
            },
            str(_subset_dir(config, lang)),
            "wer_results.json",
        )
        all_selected_scores.extend(selected_scores)
        all_sample_variances.extend(variances)
        print(
            f"{subset} {'CER' if lang == 'zh' else 'WER'}: {subset_score:.4f} "
            f"({len(selected_scores)}/{len(per_sample)} evaluated)"
        )

    flat_metrics["overall_wer"] = round(float(np.mean(all_selected_scores)), 4)
    flat_metrics["overall_wer_repeat_count"] = config.repeat_count
    flat_metrics["overall_wer_evaluated"] = len(all_selected_scores)
    if config.repeat_count > 1:
        flat_metrics["overall_wer_var"] = round(float(np.mean(all_sample_variances)), 4)
    _update_eval_results(config, flat_metrics)
    print(
        f"overall macro WER/CER: {flat_metrics['overall_wer']:.4f} "
        f"({len(all_selected_scores)} samples)"
    )
    return flat_metrics


def _write_audio_window(
    source_path: str,
    output_path: Path,
    *,
    start_frame: int,
    frame_count: int,
) -> None:
    with sf.SoundFile(source_path) as source:
        source.seek(start_frame)
        audio = source.read(frame_count, dtype="float32", always_2d=True)
        sf.write(output_path, audio, source.samplerate, subtype="PCM_16")


def _prepare_similarity_segments(
    generated: list[dict[str, Any]],
    ref_audio_by_id: dict[str, str],
    *,
    repeat_index: int,
    segment_dir: Path,
    threshold_seconds: float,
    window_seconds: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    segment_dir.mkdir(parents=True, exist_ok=True)
    segments: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for entry_index, entry in enumerate(generated):
        sample_id = str(entry.get("sample_id") or "")
        failure = _generation_failure(entry)
        ref_audio = ref_audio_by_id.get(sample_id)
        if failure is None and not ref_audio:
            failure = f"no reference audio for sample_id {sample_id!r}"
        if failure is None and not os.path.isfile(str(ref_audio)):
            failure = f"reference audio not on disk: {ref_audio}"
        if failure is not None:
            failures.append(
                {
                    "id": sample_id,
                    "repeat_index": repeat_index,
                    "sim": None,
                    "is_success": False,
                    "error": failure,
                }
            )
            continue

        wav_path = str(entry["wav_path"])
        try:
            info = sf.info(wav_path)
        except (OSError, RuntimeError) as exc:
            failures.append(
                {
                    "id": sample_id,
                    "repeat_index": repeat_index,
                    "sim": None,
                    "is_success": False,
                    "error": f"failed to inspect generated audio: {exc}",
                }
            )
            continue

        common = {
            "id": sample_id,
            "repeat_index": repeat_index,
            "ref_audio": str(ref_audio),
            "wav_path": wav_path,
            "pred_duration_s": float(info.duration),
        }
        if info.duration <= threshold_seconds:
            segments.append({**common, "part": "full", "segment_path": wav_path})
            continue

        window_frames = max(1, int(round(window_seconds * info.samplerate)))
        window_frames = min(window_frames, int(info.frames))
        head_path = segment_dir / f"sample_{entry_index:04d}_head.wav"
        tail_path = segment_dir / f"sample_{entry_index:04d}_tail.wav"
        _write_audio_window(
            wav_path,
            head_path,
            start_frame=0,
            frame_count=window_frames,
        )
        _write_audio_window(
            wav_path,
            tail_path,
            start_frame=max(0, int(info.frames) - window_frames),
            frame_count=window_frames,
        )
        segments.extend(
            [
                {**common, "part": "head", "segment_path": str(head_path)},
                {**common, "part": "tail", "segment_path": str(tail_path)},
            ]
        )
    return segments, failures


def _score_similarity_repeat(
    config: VoiceTaxonomyBenchmarkConfig,
    lang: str,
    repeat_index: int,
    *,
    scorer: WavLMSpeakerSimilarity,
    ref_audio_by_id: dict[str, str],
) -> list[dict[str, Any]]:
    generated = _load_generated_entries(config, lang, repeat_index)
    with tempfile.TemporaryDirectory(prefix=f"voice_taxonomy_{lang}_sim_") as tmpdir:
        segments, failures = _prepare_similarity_segments(
            generated,
            ref_audio_by_id,
            repeat_index=repeat_index,
            segment_dir=Path(tmpdir),
            threshold_seconds=config.sim_threshold_seconds,
            window_seconds=config.sim_window_seconds,
        )
        for start in tqdm(
            range(0, len(segments), config.sim_batch_size),
            desc=f"{_subset_name(lang)} SIM repeat {repeat_index}",
            disable=config.disable_tqdm,
        ):
            batch = segments[start : start + config.sim_batch_size]
            raw_scores = scorer.score_batch(
                [segment["ref_audio"] for segment in batch],
                [segment["segment_path"] for segment in batch],
            )
            if len(raw_scores) != len(batch):
                raise RuntimeError(
                    f"Speaker similarity returned {len(raw_scores)} scores for "
                    f"{len(batch)} segments"
                )
            for segment, raw_score in zip(batch, raw_scores):
                # The shared SeedTTS scorer exposes cosine * 100 for its legacy
                # result format. Voice Taxonomy reports the original cosine.
                score = float(raw_score) / 100.0
                if not math.isfinite(score):
                    raise RuntimeError(
                        f"Non-finite speaker similarity for {segment['id']} "
                        f"part={segment['part']}: {score}"
                    )
                segment["score"] = score

        by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for segment in segments:
            by_sample[segment["id"]].append(segment)

        rows: list[dict[str, Any]] = list(failures)
        for sample_id, sample_segments in by_sample.items():
            scores = {
                segment["part"]: float(segment["score"]) for segment in sample_segments
            }
            representative = sample_segments[0]
            if "full" in scores:
                sim = scores["full"]
                policy = "full"
            else:
                sim = float(np.mean([scores["head"], scores["tail"]]))
                policy = "head_tail_30s"
            rows.append(
                {
                    "id": sample_id,
                    "repeat_index": repeat_index,
                    "sim": sim,
                    "sim_head": scores.get("head"),
                    "sim_tail": scores.get("tail"),
                    "sim_policy": policy,
                    "sim_pred_duration_seconds": representative["pred_duration_s"],
                    "ref_audio": representative["ref_audio"],
                    "wav_path": representative["wav_path"],
                    "is_success": True,
                    "error": None,
                }
            )
        return rows


def run_voice_taxonomy_similarity(
    config: VoiceTaxonomyBenchmarkConfig,
) -> dict[str, Any]:
    """Compute Voice Taxonomy prompt-vs-generated WavLM speaker similarity."""
    if "cuda" in config.device:
        torch.cuda.set_device(config.device)
    assets = ensure_speaker_similarity_assets(
        finetune_checkpoint_override=config.similarity_checkpoint
    )
    scorer = WavLMSpeakerSimilarity(
        finetune_checkpoint=assets.finetune_checkpoint,
        wavlm_base=assets.wavlm_base,
        device=config.device,
    )

    flat_metrics: dict[str, Any] = {
        "sim_benchmark_policy": VOICE_TAXONOMY_SIM_POLICY,
        "sim_threshold_seconds": config.sim_threshold_seconds,
        "sim_window_seconds": config.sim_window_seconds,
        "sim_repeat_aggregate": config.repeat_aggregate,
    }
    for lang in _selected_languages(config.lang):
        ref_audio_by_id = {
            sample.sample_id: sample.ref_audio
            for sample in _load_selected_samples(config, lang)
        }
        repeat_rows: list[dict[str, Any]] = []
        for repeat_index in range(config.repeat_count):
            repeat_rows.extend(
                _score_similarity_repeat(
                    config,
                    lang,
                    repeat_index,
                    scorer=scorer,
                    ref_audio_by_id=ref_audio_by_id,
                )
            )

        per_sample = _aggregate_repeat_rows(
            repeat_rows,
            metric="sim",
            aggregate=config.repeat_aggregate,
            best=max,
        )
        selected_scores = [
            float(row["sim"])
            for row in per_sample
            if isinstance(row.get("sim"), (int, float))
        ]
        if not selected_scores:
            raise RuntimeError(f"No scoreable SIM samples for {_subset_name(lang)}")
        variances = [
            float(row["sim_repeat_var"])
            for row in per_sample
            if isinstance(row.get("sim_repeat_var"), (int, float))
        ]
        subset = _subset_name(lang)
        subset_score = round(float(np.mean(selected_scores)), 4)
        flat_metrics[f"{subset}_sim"] = subset_score
        flat_metrics[f"{subset}_sim_repeat_count"] = config.repeat_count
        flat_metrics[f"{subset}_sim_evaluated"] = len(selected_scores)
        flat_metrics[f"{subset}_sim_total"] = len(per_sample)
        if config.repeat_count > 1:
            flat_metrics[f"{subset}_sim_var"] = round(float(np.mean(variances)), 4)

        save_json_results(
            {
                "summary": {
                    "sim": subset_score,
                    "language": lang,
                    "aggregation": "macro_sample_mean",
                    "repeat_count": config.repeat_count,
                    "repeat_aggregate": config.repeat_aggregate,
                    "evaluated": len(selected_scores),
                    "total_samples": len(per_sample),
                    "sim_benchmark_policy": VOICE_TAXONOMY_SIM_POLICY,
                    "sim_threshold_seconds": config.sim_threshold_seconds,
                    "sim_window_seconds": config.sim_window_seconds,
                },
                "config": _public_config(config),
                "per_sample": per_sample,
            },
            str(_subset_dir(config, lang)),
            "similarity_results.json",
        )
        print(
            f"{subset} SIM: {subset_score:.4f} "
            f"({len(selected_scores)}/{len(per_sample)} evaluated)"
        )

    _update_eval_results(config, flat_metrics)
    return flat_metrics


def _parse_token_count(value: str) -> int | str:
    normalized = value.strip().lower()
    if normalized == MOSS_TTS_TOKEN_COUNT_AUTO:
        return MOSS_TTS_TOKEN_COUNT_AUTO
    try:
        token_count = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "token count must be a positive integer or 'auto'"
        ) from exc
    if token_count <= 0:
        raise argparse.ArgumentTypeError("token count must be positive")
    return token_count


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MOSS-TTS Voice Taxonomy generation, macro WER/CER, and SIM."
    )
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000, help="TTS server port.")
    parser.add_argument(
        "--asr-port",
        type=int,
        default=None,
        help="ASR server port. Defaults to --port for sequential managed phases.",
    )
    parser.add_argument("--model", type=str, default=VOICE_TAXONOMY_MODEL)
    parser.add_argument("--meta", type=str, default=VOICE_TAXONOMY_DATASET)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/tts_voice_taxonomy",
    )
    parser.add_argument("--lang", choices=["all", "en", "zh"], default="all")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples per selected language split.",
    )
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--repeat-count", type=int, default=1)
    parser.add_argument(
        "--repeat-aggregate",
        choices=["mean", "best"],
        default="mean",
    )
    parser.add_argument("--response-format", type=str, default="wav")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=VOICE_TAXONOMY_TTS_MAX_NEW_TOKENS,
    )
    parser.add_argument("--token-count", type=_parse_token_count, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--concurrency",
        "--max-concurrency",
        dest="concurrency",
        type=int,
        default=DEFAULT_TTS_BENCHMARK_CONCURRENCY,
    )
    parser.add_argument("--request-rate", type=float, default=float("inf"))
    parser.add_argument(
        "--tts-request-timeout-s",
        type=int,
        default=VOICE_TAXONOMY_TTS_REQUEST_TIMEOUT_S,
        help="Per-request TTS timeout; long Voice Taxonomy passages can exceed 300s.",
    )
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--initial-codec-chunk-frames", type=int, default=None)
    parser.add_argument("--disable-tqdm", action="store_true")
    parser.add_argument("--max-running-requests", type=int, default=16)
    parser.add_argument("--cuda-graph-max-bs", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--similarity-checkpoint", type=str, default=None)
    parser.add_argument("--asr-model-path", type=str, default=QWEN3_ASR_MODEL_PATH)
    parser.add_argument(
        "--asr-max-new-tokens",
        type=int,
        default=VOICE_TAXONOMY_ASR_MAX_NEW_TOKENS,
    )
    parser.add_argument(
        "--asr-max-running-requests",
        type=int,
        default=VOICE_TAXONOMY_ASR_MAX_RUNNING_REQUESTS,
    )
    parser.add_argument(
        "--asr-chunk-batch-size",
        type=int,
        default=VOICE_TAXONOMY_ASR_CHUNK_BATCH_SIZE,
    )
    parser.add_argument("--server-timeout", type=int, default=1200)
    parser.add_argument("--skip-gpu-cleanup", action="store_true")
    parser.add_argument("--use-existing-tts-server", action="store_true")
    parser.add_argument("--use-existing-asr-server", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--generate-only", action="store_true")
    mode.add_argument("--wer-only", action="store_true")
    mode.add_argument("--similarity-only", action="store_true")
    return parser


def _config_from_args(args: argparse.Namespace) -> VoiceTaxonomyBenchmarkConfig:
    return VoiceTaxonomyBenchmarkConfig(
        model=args.model,
        meta=args.meta,
        base_url=args.base_url,
        host=args.host,
        port=args.port,
        asr_port=args.asr_port,
        output_dir=args.output_dir,
        lang=args.lang,
        max_samples=args.max_samples,
        sample_offset=args.sample_offset,
        repeat_count=args.repeat_count,
        repeat_aggregate=args.repeat_aggregate,
        response_format=args.response_format,
        max_new_tokens=args.max_new_tokens,
        token_count=args.token_count,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed,
        warmup=args.warmup,
        concurrency=args.concurrency,
        request_rate=args.request_rate,
        tts_request_timeout_s=args.tts_request_timeout_s,
        stream=args.stream,
        initial_codec_chunk_frames=args.initial_codec_chunk_frames,
        disable_tqdm=args.disable_tqdm,
        max_running_requests=args.max_running_requests,
        cuda_graph_max_bs=args.cuda_graph_max_bs,
        device=args.device,
        similarity_checkpoint=args.similarity_checkpoint,
        asr_model_path=args.asr_model_path,
        asr_max_new_tokens=args.asr_max_new_tokens,
        asr_max_running_requests=args.asr_max_running_requests,
        asr_chunk_batch_size=args.asr_chunk_batch_size,
    )


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    positive_args = {
        "--port": args.port,
        "--repeat-count": args.repeat_count,
        "--concurrency": args.concurrency,
        "--tts-request-timeout-s": args.tts_request_timeout_s,
        "--max-running-requests": args.max_running_requests,
        "--cuda-graph-max-bs": args.cuda_graph_max_bs,
        "--asr-max-new-tokens": args.asr_max_new_tokens,
        "--asr-max-running-requests": args.asr_max_running_requests,
        "--asr-chunk-batch-size": args.asr_chunk_batch_size,
    }
    if args.asr_port is not None:
        positive_args["--asr-port"] = args.asr_port
    for name, value in positive_args.items():
        if value <= 0:
            parser.error(f"{name} must be positive")
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max-samples must be positive")
    if args.sample_offset < 0:
        parser.error("--sample-offset must be non-negative")
    if (
        args.initial_codec_chunk_frames is not None
        and args.initial_codec_chunk_frames < 0
    ):
        parser.error("--initial-codec-chunk-frames must be non-negative")

    run_generation = not (args.wer_only or args.similarity_only)
    run_wer = not (args.generate_only or args.similarity_only)
    asr_port = args.asr_port or args.port
    if (
        run_generation
        and run_wer
        and args.use_existing_tts_server
        and asr_port == args.port
    ):
        parser.error(
            "A persistent existing TTS server occupies --port during WER. "
            "Pass a different --asr-port and optionally --use-existing-asr-server."
        )


def _run_generation_phase(
    config: VoiceTaxonomyBenchmarkConfig,
    args: argparse.Namespace,
) -> None:
    if args.use_existing_tts_server:
        asyncio.run(run_voice_taxonomy_generation(config))
        return
    with managed_omni_server(
        model_path=config.model,
        port=config.port,
        host=config.host,
        max_running_requests=config.max_running_requests,
        cuda_graph_max_bs=config.cuda_graph_max_bs,
        log_file=Path(config.output_dir) / "server_logs" / "tts_server.log",
        timeout=args.server_timeout,
        wait_for_gpu_release=not args.skip_gpu_cleanup,
    ):
        asyncio.run(run_voice_taxonomy_generation(config))


def _run_wer_phase(
    config: VoiceTaxonomyBenchmarkConfig,
    args: argparse.Namespace,
) -> None:
    asr_port = config.asr_port or config.port
    if args.use_existing_asr_server:
        logger.warning(
            "Using an existing ASR server; it must be configured with "
            "max_new_tokens >= %d for the Voice Taxonomy metric contract.",
            config.asr_max_new_tokens,
        )
        asyncio.run(run_voice_taxonomy_wer(config, asr_router_port=asr_port))
        return

    with managed_omni_server(
        model_path=config.asr_model_path,
        port=asr_port,
        host=config.host,
        max_running_requests=config.asr_max_running_requests,
        cuda_graph_max_bs=config.asr_max_running_requests,
        extra_cli_args=[
            "--stages.asr.factory_args.max_new_tokens",
            str(config.asr_max_new_tokens),
        ],
        log_file=Path(config.output_dir) / "server_logs" / "asr_server.log",
        timeout=args.server_timeout,
        wait_for_gpu_release=not args.skip_gpu_cleanup,
    ):
        asyncio.run(run_voice_taxonomy_wer(config, asr_router_port=asr_port))


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    _validate_args(parser, args)
    config = _config_from_args(args)

    run_generation = not (args.wer_only or args.similarity_only)
    run_wer = not (args.generate_only or args.similarity_only)
    run_similarity = not (args.generate_only or args.wer_only)

    if run_generation:
        _run_generation_phase(config, args)
    if run_wer:
        _run_wer_phase(config, args)
    if run_similarity:
        run_voice_taxonomy_similarity(config)


if __name__ == "__main__":
    main()
