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

KEY = WakeWord.ALEXA.path.stem
POSITIVE = {KEY: 0.9}
BLOCK = np.zeros(1280, dtype=np.int16)


def detector(fake_model, **kwargs):
    result = WakeWordDetector("alexa", **kwargs)
    result._model.scores = POSITIVE
    return result


@pytest.mark.parametrize(
    "name,value",
    [
        ("threshold", 1.5),
        ("threshold", float("nan")),
        ("vad_threshold", -1),
        ("vad_threshold", float("nan")),
        ("vad_threshold", 1.1),
        ("cooldown", -1),
        ("cooldown", float("nan")),
        ("cooldown", float("inf")),
    ],
)
def test_invalid_configuration(fake_model, name, value):
    with pytest.raises(ValueError, match=name):
        WakeWordDetector("alexa", **{name: value})


def test_configuration_and_missing_model(fake_model):
    with pytest.raises(FileNotFoundError):
        WakeWordDetector(CustomWakeWord("missing", "missing.onnx"))
    result = WakeWordDetector("hey_jarvis", threshold=0.7, cooldown=2)
    assert (result.names, result.threshold, result.cooldown) == (
        ("hey_jarvis",),
        0.7,
        2,
    )


def test_async_and_sync_agree(fake_model):
    result = detector(fake_model, cooldown=0)
    assert asyncio.run(result.process(BLOCK)) == Detection("alexa", 0.9, 1280)
    assert result.process_sync(BLOCK) == Detection("alexa", 0.9, 2560)
    assert result.process_sync(BLOCK).timestamp == 0.24


def test_small_frames_are_buffered_without_loss(fake_model):
    result = detector(fake_model, cooldown=0)
    source = np.arange(2567, dtype=np.int16)
    events = []
    for part in np.split(source, [1, 321, 401, 800, 2000]):
        events.extend(result.process_all_sync(part))
    events.extend(result.finish_sync())
    actual = np.concatenate(result._model.inputs)
    np.testing.assert_array_equal(actual[: len(source)], source)
    assert not actual[len(source) :].any()
    assert [event.sample_index for event in events] == [1280, 2560, 2567]


def test_zero_cooldown_has_no_hidden_backend_reset(fake_model):
    result = detector(fake_model, cooldown=0)
    assert len(result.process_all_sync(np.tile(BLOCK, 8))) == 8
    assert result._model.reset_calls == 0


def test_cooldown_uses_audio_time_and_advances_on_negative_frames(fake_model):
    result = detector(fake_model, cooldown=0.16)
    assert [d.sample_index for d in result.process_all_sync(np.tile(BLOCK, 5))] == [
        1280,
        3840,
        6400,
    ]
    result._model.scores = {KEY: 0.1}
    assert result.process_sync(BLOCK) is None
    result._model.scores = POSITIVE
    assert result.process_sync(BLOCK).sample_index == 8960


@pytest.mark.parametrize("score", [0.1, float("nan"), float("inf"), -1.0, 1.1, 0.0])
def test_invalid_or_low_scores_cannot_trigger(fake_model, score):
    result = detector(fake_model)
    result._model.scores = {KEY: score}
    assert result.process_sync(BLOCK) is None


def test_format_change_requires_reset(fake_model):
    result = detector(fake_model)
    result.process_sync(BLOCK)
    changed = AudioFormat(48000, 1, "pcm_s16le")
    with pytest.raises(ValueError, match="reset"):
        result.process_sync(BLOCK, format=changed)
    result.reset(format=changed)
    assert result.process_sync(BLOCK) is None


def test_reset_recreates_backend_and_all_stream_state(fake_model):
    result = detector(fake_model)
    result.process_all_sync(np.ones(1400, dtype=np.int16))
    previous = result._model
    result.reset()
    assert result._model is not previous
    result._model.scores = POSITIVE
    assert result.process_sync(BLOCK) == Detection("alexa", 0.9, 1280)
    np.testing.assert_array_equal(result._model.inputs[0], BLOCK)


def test_finish_closes_stream_until_reset(fake_model):
    result = detector(fake_model)
    assert result.finish_sync() == ()
    with pytest.raises(RuntimeError, match="reset"):
        result.process_sync(BLOCK)
    with pytest.raises(RuntimeError, match="reset"):
        result.finish_sync()
    result.reset()
    assert result.process_sync(BLOCK) is None


def test_large_frame_convenience_api_returns_first_detection(fake_model):
    result = detector(fake_model, cooldown=0)
    assert result.process_sync(np.tile(BLOCK, 3)).sample_index == 1280
    assert len(result._model.inputs) == 3


def test_unexpected_backend_labels_require_reset(fake_model):
    result = detector(fake_model)
    result._model.scores = {"unexpected": 0.9}
    with pytest.raises(ValueError, match="labels"):
        result.process_sync(BLOCK)
    with pytest.raises(RuntimeError, match="reset"):
        result.process_sync(BLOCK)
