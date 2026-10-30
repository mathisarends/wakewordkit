import importlib.util
import json
import wave
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "evaluate", Path(__file__).resolve().parents[1] / "tools" / "evaluate.py"
)
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)


def test_matching_does_not_reuse_annotations_or_confuse_labels():
    expected = [
        {"name": "alexa", "end_seconds": 1.0},
        {"name": "alexa", "end_seconds": 1.2},
        {"name": "hey_jarvis", "end_seconds": 2.0},
    ]
    detections = [
        {"name": "alexa", "timestamp": 1.15},
        {"name": "alexa", "timestamp": 2.0},
        {"name": "alexa", "timestamp": 2.1},
    ]
    result = evaluation.match_events(expected, detections, 0.5, 1.0)
    assert result["missed"] == 1
    assert result["unmatched"] == 1
    assert result["latencies"] == pytest.approx([0.15, 0.8])


def make_manifest(tmp_path, negative=True, activations=None):
    with wave.open(str(tmp_path / "audio.wav"), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(np.zeros(16007, np.int16).tobytes())
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "recordings": [
                    {
                        "path": "audio.wav",
                        "negative": negative,
                        "activations": activations or [],
                    }
                ],
            }
        )
    )
    return manifest


def test_evaluation_reports_misses_without_inventing_negative_exposure(
    fake_model, tmp_path
):
    path = make_manifest(tmp_path, False, [{"name": "alexa", "end_seconds": 0.8}])
    report = evaluation.evaluate(path, words=["alexa"])
    metrics = report["metrics"]
    assert metrics["miss_fraction"] == 1.0
    assert metrics["false_activations_per_negative_hour"] is None
    assert metrics["signed_model_latency_seconds"] is None
    assert len(report["recordings"][0]["sha256"]) == 64


@pytest.mark.integration
def test_evaluation_runs_real_inference_on_negative_smoke_audio(tmp_path):
    report = evaluation.evaluate(make_manifest(tmp_path), words=["alexa"])
    assert report["metrics"]["miss_fraction"] is None
    assert report["metrics"]["false_activations_per_negative_hour"] == 0
    assert report["metrics"]["audio_seconds"] == 16007 / 16000


def test_inconsistent_annotations_are_rejected(fake_model, tmp_path):
    manifest = make_manifest(tmp_path, True, [{"name": "alexa", "end_seconds": 0.8}])
    with pytest.raises(ValueError, match="Recording"):
        evaluation.evaluate(manifest, words=["alexa"])
