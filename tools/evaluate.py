"""Evaluate a local, annotated PCM16 WAV corpus."""

import argparse
import hashlib
import json
import math
import platform
import time
import wave
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import numpy as np
import psutil

from wakewordkit import AudioFormat, WakeWordDetector


def match_events(expected, detections, early, late):
    """Match each detection to the earliest eligible, unmatched annotation."""
    remaining = set(range(len(expected)))
    latencies, unmatched = [], 0
    for detection in sorted(detections, key=lambda item: item["timestamp"]):
        candidates = [
            index
            for index in remaining
            if expected[index]["name"] == detection["name"]
            and expected[index]["end_seconds"] - early
            <= detection["timestamp"]
            <= expected[index]["end_seconds"] + late
        ]
        if not candidates:
            unmatched += 1
            continue
        earliest = min(candidates, key=lambda index: expected[index]["end_seconds"])
        remaining.remove(earliest)
        latencies.append(detection["timestamp"] - expected[earliest]["end_seconds"])
    return {"missed": len(remaining), "unmatched": unmatched, "latencies": latencies}


def evaluate(manifest_path, *, words, threshold=0.5, cooldown=1.5, early=0.5, late=1.0):
    if not all(math.isfinite(value) and value >= 0 for value in (early, late)):
        raise ValueError("Matching tolerances must be finite and non-negative.")
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    if manifest.get("schema_version") != 1 or not manifest.get("recordings"):
        raise ValueError("Manifest needs schema_version=1 and non-empty recordings.")
    if not words:
        raise ValueError("Select at least one wake word.")
    started = time.perf_counter()
    detector = WakeWordDetector(*words, threshold=threshold, cooldown=cooldown)
    startup = time.perf_counter() - started
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    recordings, latencies, calls = [], [], []
    negative_seconds = total_seconds = cpu_seconds = 0.0
    false_activations = missed = positives = unmatched = 0
    for index, recording in enumerate(manifest["recordings"]):
        if index:
            detector.reset()
        path = manifest_path.parent / recording["path"]
        events = recording.get("activations", [])
        negative = recording.get("negative")
        if (
            not isinstance(negative, bool)
            or (negative and events)
            or (not negative and not events)
        ):
            raise ValueError(
                "Recording needs negative=true without events, "
                "or false with annotations."
            )
        detections = []
        with path.open("rb") as file:
            checksum = hashlib.file_digest(file, "sha256").hexdigest()
        with wave.open(str(path), "rb") as wav:
            if wav.getsampwidth() != 2 or wav.getcomptype() != "NONE":
                raise ValueError(f"{path}: expected uncompressed PCM16 WAV.")
            format = AudioFormat(wav.getframerate(), wav.getnchannels(), "pcm_s16le")
            duration = wav.getnframes() / format.sample_rate
            if duration <= 0:
                raise ValueError("Evaluation recordings must not be empty.")
            for event in events:
                end = event["end_seconds"]
                if (
                    event.get("name") not in detector.names
                    or not math.isfinite(end)
                    or not 0 <= end <= duration
                ):
                    raise ValueError(
                        "Annotation needs a selected model name "
                        "and a finite end within the WAV."
                    )

            def run(frame=None, *, format=format, detections=detections):
                nonlocal cpu_seconds, peak_rss
                cpu_start, start = time.process_time(), time.perf_counter()
                result = (
                    detector.finish_sync()
                    if frame is None
                    else detector.process_all_sync(frame, format=format)
                )
                calls.append((time.perf_counter() - start) * 1000)
                cpu_seconds += time.process_time() - cpu_start
                peak_rss = max(peak_rss, process.memory_info().rss)
                detections.extend(
                    [{**asdict(item), "timestamp": item.timestamp} for item in result]
                )

            while frame := wav.readframes(max(1, format.sample_rate // 50)):
                run(frame)
            run()
        matched = match_events(events, detections, early, late)
        total_seconds += duration
        positives += len(events)
        missed += matched["missed"]
        unmatched += matched["unmatched"]
        latencies.extend(matched["latencies"])
        if negative:
            negative_seconds += duration
            false_activations += len(detections)
        recordings.append(
            {
                "path": recording["path"],
                "sha256": checksum,
                "seconds": duration,
                "negative": negative,
                "detections": detections,
                "matching": matched,
            }
        )
    return {
        "schema_version": 1,
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "versions": {
            name: version(name)
            for name in ("wakewordkit", "openwakeword", "soxr", "onnxruntime")
        },
        "config": {
            "words": words,
            "threshold": threshold,
            "cooldown": cooldown,
            "early_seconds": early,
            "late_seconds": late,
        },
        "metrics": {
            "audio_seconds": total_seconds,
            "negative_hours": negative_seconds / 3600,
            "false_activations": false_activations,
            "false_activations_per_negative_hour": false_activations
            / (negative_seconds / 3600)
            if negative_seconds
            else None,
            "annotated_activations": positives,
            "missed_activations": missed,
            "miss_fraction": missed / positives if positives else None,
            "unmatched_detections_all_recordings": unmatched,
            "signed_model_latency_seconds": {
                "median": float(np.median(latencies)),
                "p95": float(np.percentile(latencies, 95)),
            }
            if latencies
            else None,
            "call_wall_ms": {
                "median": float(np.median(calls)),
                "p95": float(np.percentile(calls, 95)),
            },
            "inference_cpu_seconds": cpu_seconds,
            "cpu_percent_of_one_core_at_realtime": cpu_seconds / total_seconds * 100,
            "sampled_peak_rss_mib": peak_rss / 1024**2,
            "construction_seconds": startup,
        },
        "recordings": recordings,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--word", action="append", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--cooldown", type=float, default=1.5)
    parser.add_argument("--early", type=float, default=0.5)
    parser.add_argument("--late", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(
        args.manifest,
        words=args.word,
        threshold=args.threshold,
        cooldown=args.cooldown,
        early=args.early,
        late=args.late,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["metrics"], indent=2))


if __name__ == "__main__":
    main()
