import asyncio

import numpy as np
import pytest

from wakewordkit import (
    AudioFormat,
    CustomWakeWord,
    Detection,
    WakeWord,
    WakeWordDetector,
)
from wakewordkit.detector import _StreamingNormalizer

ALEXA_KEY = WakeWord.ALEXA.path.stem
NEGATIVE = {ALEXA_KEY: 0.1}
POSITIVE = {ALEXA_KEY: 0.9}


def pcm(*samples):
    return np.asarray(samples, dtype="<i2").tobytes()


def detector_with_scores(fake_model, scores):
    detector = WakeWordDetector(WakeWord.ALEXA, threshold=0.5, cooldown=0)
    detector._model.score_sequence = list(scores)
    return detector


def test_validation_and_public_configuration(fake_model):
    with pytest.raises(ValueError, match="threshold"):
        WakeWordDetector(WakeWord.ALEXA, threshold=1.5)
    with pytest.raises(ValueError, match="cooldown"):
        WakeWordDetector(WakeWord.ALEXA, cooldown=-1)
    with pytest.raises(FileNotFoundError):
        WakeWordDetector(CustomWakeWord("missing", "missing.onnx"))

    detector = WakeWordDetector("hey_jarvis", threshold=0.7, cooldown=2)
    assert detector.names == ("hey_jarvis",)
    assert detector.threshold == 0.7
    assert detector.cooldown == 2


def test_process_converts_explicit_format_and_detects(fake_model):
    detector = detector_with_scores(fake_model, [POSITIVE])
    format = AudioFormat(24_000, 2, "pcm_s16le")
    frame = np.tile(np.array([100, 300], dtype=np.int16), 4800).tobytes()

    assert asyncio.run(detector.process(frame, format=format)) == Detection(
        "alexa", 0.9
    )
    assert detector._model.inputs[0].dtype == np.int16
    assert detector._model.inputs[0].size > 0


def test_process_returns_none_below_threshold(fake_model):
    detector = detector_with_scores(fake_model, [NEGATIVE])
    assert asyncio.run(detector.process(pcm(1))) is None


def test_process_applies_cooldown_and_resets_after_detection(fake_model):
    detector = WakeWordDetector(WakeWord.ALEXA, cooldown=60)
    detector._model.score_sequence = [POSITIVE, POSITIVE]

    assert asyncio.run(detector.process(pcm(1))) == Detection("alexa", 0.9)
    assert asyncio.run(detector.process(pcm(2))) is None
    assert detector._model.reset_calls == 1


def test_streaming_resampling_has_no_chunk_drift():
    format = AudioFormat(24_000, 1, "pcm_s16le")
    samples = np.arange(24_003, dtype=np.int16)
    chunked = _StreamingNormalizer(format)
    parts = [
        chunked.feed(part) for part in np.array_split(samples, [17, 131, 4099, 20_000])
    ]
    parts.append(chunked.finish())
    whole = _StreamingNormalizer(format)
    one_shot = np.concatenate([whole.feed(samples), whole.finish()])
    result = np.concatenate(parts)

    assert result.size == round(samples.size * 16_000 / 24_000)
    np.testing.assert_array_equal(result, one_shot)
