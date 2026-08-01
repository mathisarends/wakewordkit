import asyncio

import pytest

from wakewordkit import (
    AudioInput,
    CustomWakeWord,
    Detection,
    WakeWord,
    WakeWordDetector,
)

FRAME = b"\x00\x00"
ALEXA_KEY = WakeWord.ALEXA.path.stem
JARVIS_KEY = WakeWord.HEY_JARVIS.path.stem


class ListInput(AudioInput):
    def __init__(self, frames):
        self._frames = iter(frames)
        self.entered = False
        self.exited = False

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, *exc):
        self.exited = True

    def read(self):
        return next(self._frames)


class FixedClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


@pytest.fixture
def clock(monkeypatch):
    fake = FixedClock()
    monkeypatch.setattr("wakewordkit.detector.time.monotonic", fake)
    return fake


def test_rejects_out_of_range_threshold():
    with pytest.raises(ValueError, match="threshold"):
        WakeWordDetector(WakeWord.ALEXA, threshold=1.5)


def test_rejects_negative_cooldown():
    with pytest.raises(ValueError, match="cooldown"):
        WakeWordDetector(WakeWord.ALEXA, cooldown=-1.0)


def test_rejects_missing_custom_model():
    with pytest.raises(FileNotFoundError):
        WakeWordDetector(CustomWakeWord("nope", "does_not_exist.onnx"))


def test_defaults_to_all_builtin_wake_words(fake_model):
    detector = WakeWordDetector()
    assert detector.names == ("alexa", "hey_jarvis", "hey_mycroft", "hey_marvin")


def test_exposes_its_configuration(fake_model):
    detector = WakeWordDetector(WakeWord.ALEXA, threshold=0.7, cooldown=2.0)
    assert detector.threshold == 0.7
    assert detector.cooldown == 2.0


def test_custom_name_is_used_as_label(fake_model):
    detector = WakeWordDetector(
        WakeWord.ALEXA,
        CustomWakeWord("hey_computer", WakeWord.HEY_JARVIS.path),
    )
    assert detector.names == ("alexa", "hey_computer")


def test_process_returns_detection_above_threshold(fake_model):
    detector = WakeWordDetector(WakeWord.ALEXA, threshold=0.5)
    detector._model.scores = {ALEXA_KEY: 0.8}
    assert detector.process(FRAME) == Detection("alexa", 0.8)


def test_process_ignores_scores_below_threshold(fake_model):
    detector = WakeWordDetector(WakeWord.ALEXA, threshold=0.5)
    detector._model.scores = {ALEXA_KEY: 0.49}
    assert detector.process(FRAME) is None


def test_process_reports_highest_scoring_wake_word(fake_model):
    detector = WakeWordDetector(WakeWord.ALEXA, WakeWord.HEY_JARVIS, threshold=0.5)
    detector._model.scores = {ALEXA_KEY: 0.6, JARVIS_KEY: 0.91}
    assert detector.process(FRAME) == Detection("hey_jarvis", 0.91)


def test_stream_debounces_within_cooldown(fake_model, clock):
    detector = WakeWordDetector(WakeWord.ALEXA, threshold=0.5, cooldown=1.5)
    detector._model.scores = {ALEXA_KEY: 0.9}

    def source():
        for t in (0.0, 0.5, 1.0, 2.0):
            clock.now = t
            yield FRAME

    detections = list(detector.stream(source()))
    assert detections == [Detection("alexa", 0.9), Detection("alexa", 0.9)]
    assert detector._model.reset_calls == 2


def test_stream_without_cooldown_yields_every_positive_frame(fake_model, clock):
    detector = WakeWordDetector(WakeWord.ALEXA, threshold=0.5, cooldown=0.0)
    detector._model.scores = {ALEXA_KEY: 0.9}
    detections = list(detector.stream([FRAME, FRAME, FRAME]))
    assert len(detections) == 3


def test_stream_skips_negative_frames(fake_model, clock):
    detector = WakeWordDetector(WakeWord.ALEXA, threshold=0.5, cooldown=0.0)
    detector._model.scores = {ALEXA_KEY: 0.1}
    assert list(detector.stream([FRAME, FRAME])) == []
    assert detector._model.reset_calls == 0


def test_reset_clears_cooldown_state(fake_model, clock):
    detector = WakeWordDetector(WakeWord.ALEXA, threshold=0.5, cooldown=100.0)
    detector._model.scores = {ALEXA_KEY: 0.9}

    assert detector.process(FRAME) == Detection("alexa", 0.9)
    list(detector.stream([FRAME]))
    detector.reset()
    assert detector._last_trigger is None
    assert list(detector.stream([FRAME])) == [Detection("alexa", 0.9)]


def test_listen_consumes_and_closes_provided_source(fake_model):
    detector = WakeWordDetector(WakeWord.ALEXA, threshold=0.5, cooldown=0.0)
    detector._model.scores = {ALEXA_KEY: 0.9}
    source = ListInput([FRAME, FRAME])

    detections = list(detector.listen(source))

    assert len(detections) == 2
    assert source.entered and source.exited


def collect(agen):
    async def run():
        return [item async for item in agen]

    return asyncio.run(run())


async def frames(count):
    for _ in range(count):
        yield FRAME


def test_alisten_consumes_and_closes_provided_source(fake_model):
    detector = WakeWordDetector(WakeWord.ALEXA, threshold=0.5, cooldown=0.0)
    detector._model.scores = {ALEXA_KEY: 0.9}
    source = ListInput([FRAME, FRAME])

    detections = collect(detector.alisten(source))

    assert detections == [Detection("alexa", 0.9)] * 2
    assert source.entered and source.exited


def test_alisten_debounces_within_cooldown(fake_model, clock):
    detector = WakeWordDetector(WakeWord.ALEXA, threshold=0.5, cooldown=1.5)
    detector._model.scores = {ALEXA_KEY: 0.9}

    class Clocked(ListInput):
        def read(self):
            clock.now = next(self._times)
            return super().read()

    source = Clocked([FRAME] * 4)
    source._times = iter((0.0, 0.5, 1.0, 2.0))

    assert collect(detector.alisten(source)) == [Detection("alexa", 0.9)] * 2
    assert detector._model.reset_calls == 2


def test_astream_yields_detections_from_async_frames(fake_model, clock):
    detector = WakeWordDetector(WakeWord.ALEXA, threshold=0.5, cooldown=0.0)
    detector._model.scores = {ALEXA_KEY: 0.9}
    assert collect(detector.astream(frames(3))) == [Detection("alexa", 0.9)] * 3


def test_astream_skips_negative_frames(fake_model, clock):
    detector = WakeWordDetector(WakeWord.ALEXA, threshold=0.5, cooldown=0.0)
    detector._model.scores = {ALEXA_KEY: 0.1}
    assert collect(detector.astream(frames(2))) == []
    assert detector._model.reset_calls == 0
