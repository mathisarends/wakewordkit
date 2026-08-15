import asyncio
from contextlib import asynccontextmanager

import numpy as np
import pytest

from wakewordkit import (
    MODEL_AUDIO_FORMAT,
    Activation,
    AudioFormat,
    CapturedAudio,
    CustomWakeWord,
    Detection,
    WakeWord,
    WakeWordDetector,
)
from wakewordkit.detector import _StreamingNormalizer

ALEXA_KEY = WakeWord.ALEXA.path.stem
NEGATIVE = {ALEXA_KEY: 0.1}
POSITIVE = {ALEXA_KEY: 0.9}


class FakeDevice:
    def __init__(self, *sessions, format=MODEL_AUDIO_FORMAT):
        self._sessions = iter(sessions)
        self.format = format
        self.open_count = 0
        self.close_count = 0

    @asynccontextmanager
    async def open(self):
        frames = next(self._sessions)
        self.open_count += 1

        async def iterate_list():
            for frame in frames:
                yield frame
                await asyncio.sleep(0)

        source = frames if hasattr(frames, "__aiter__") else iterate_list()
        try:
            yield CapturedAudio(self.format, source)
        finally:
            self.close_count += 1


def pcm(*samples):
    return np.asarray(samples, dtype="<i2").tobytes()


async def collect(stream):
    async with stream:
        return [chunk async for chunk in stream]


def detector_with_scores(fake_model, scores, capture=None):
    detector = WakeWordDetector(
        WakeWord.ALEXA, threshold=0.5, cooldown=0, capture=capture
    )
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


def test_custom_device_owns_capture_configuration(fake_model):
    detector = detector_with_scores(fake_model, [], FakeDevice([]))

    async def run():
        with pytest.raises(ValueError, match="custom capture device"):
            await detector.listen(sample_rate=24_000)

    asyncio.run(run())


def test_process_converts_explicit_format_and_detects(fake_model):
    detector = detector_with_scores(fake_model, [POSITIVE])
    format = AudioFormat(24_000, 2, "pcm_s16le")
    frame = np.tile(np.array([100, 300], dtype=np.int16), 12).tobytes()

    assert detector.process(frame, format=format) == Detection("alexa", 0.9)
    assert detector._model.inputs[0].dtype == np.int16
    assert detector._model.inputs[0].size == 8


def test_process_returns_none_below_threshold(fake_model):
    detector = detector_with_scores(fake_model, [NEGATIVE])
    assert detector.process(pcm(1)) is None


def test_stream_is_async_and_does_not_own_source(fake_model):
    detector = detector_with_scores(fake_model, [NEGATIVE, POSITIVE])
    closed = False

    async def frames():
        nonlocal closed
        try:
            yield pcm(1)
            yield pcm(2)
        finally:
            closed = True

    async def run():
        return [item async for item in detector.stream(frames())]

    assert asyncio.run(run()) == [Detection("alexa", 0.9)]
    assert closed  # natural exhaustion, not ownership-driven close


def test_listen_hands_off_preroll_detection_and_live_without_gaps(fake_model):
    format = AudioFormat(10, 1, "pcm_s16le")
    device = FakeDevice([pcm(i) for i in range(1, 7)], format=format)
    detector = detector_with_scores(fake_model, [NEGATIVE, NEGATIVE, POSITIVE], device)

    async def run():
        activation = await detector.listen(preroll=0.2, max_buffer=2)
        assert isinstance(activation, Activation)
        assert activation.audio.format == format
        return activation, await collect(activation.audio)

    activation, chunks = asyncio.run(run())
    assert (activation.name, activation.score) == ("alexa", 0.9)
    assert chunks == [pcm(i) for i in range(1, 7)]
    assert device.open_count == device.close_count == 1


def test_preroll_zero_still_includes_detection_frame(fake_model):
    format = AudioFormat(10, 1, "pcm_s16le")
    device = FakeDevice([pcm(i) for i in range(1, 5)], format=format)
    detector = detector_with_scores(fake_model, [NEGATIVE, POSITIVE], device)

    async def run():
        activation = await detector.listen(preroll=0, max_buffer=1)
        return await collect(activation.audio)

    assert asyncio.run(run()) == [pcm(2), pcm(3), pcm(4)]


def test_delayed_consumer_gets_frames_captured_after_detection(fake_model):
    format = AudioFormat(100, 1, "pcm_s16le")
    device = FakeDevice([pcm(i) for i in range(20)], format=format)
    detector = detector_with_scores(fake_model, [POSITIVE], device)

    async def run():
        activation = await detector.listen(preroll=0, max_buffer=1)
        await asyncio.sleep(0.05)
        return await collect(activation.audio)

    assert asyncio.run(run()) == [pcm(i) for i in range(20)]


def test_slow_consumer_buffer_is_bounded_and_reports_drop_oldest(
    fake_model,
):
    format = AudioFormat(10, 1, "pcm_s16le")
    device = FakeDevice([pcm(i) for i in range(10)], format=format)
    detector = detector_with_scores(fake_model, [POSITIVE], device)

    async def run():
        activation = await detector.listen(preroll=0, max_buffer=0.3)
        await asyncio.sleep(0.05)
        stats = activation.audio.stats
        chunks = await collect(activation.audio)
        return activation, stats, chunks

    activation, stats, chunks = asyncio.run(run())
    assert activation.audio.overflow_policy == "drop_oldest"
    assert stats.buffered_bytes <= 6
    assert stats.dropped_chunks == 7
    assert chunks == [pcm(7), pcm(8), pcm(9)]


def test_waiting_reader_stops_when_stream_context_exits(fake_model):
    async def live_frames():
        yield pcm(42)
        await asyncio.Event().wait()

    device = FakeDevice(live_frames())
    detector = detector_with_scores(fake_model, [POSITIVE], device)

    async def run():
        activation = await detector.listen(preroll=0)
        async with activation.audio:
            first = await anext(activation.audio)
            pending = asyncio.create_task(anext(activation.audio))
            await asyncio.sleep(0)
        with pytest.raises(StopAsyncIteration):
            await pending
        return first

    assert asyncio.run(run()) == pcm(42)
    assert device.close_count == 1


def test_cancellation_cleans_up_and_detector_can_be_reused(fake_model):
    reading = asyncio.Event()

    async def blocking_frames():
        reading.set()
        await asyncio.Event().wait()
        yield  # pragma: no cover

    device = FakeDevice(blocking_frames(), [pcm(1)])
    detector = detector_with_scores(fake_model, [], device)

    async def run():
        task = asyncio.create_task(detector.listen())
        await reading.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        detector._model.score_sequence = [POSITIVE]
        activation = await detector.listen(preroll=0)
        return await collect(activation.audio)

    assert asyncio.run(run()) == [pcm(1)]
    assert device.open_count == device.close_count == 2


def test_capture_exception_cleans_up_and_detector_can_be_reused(
    fake_model,
):
    async def broken_frames():
        raise RuntimeError("capture failed")
        yield  # pragma: no cover

    device = FakeDevice(broken_frames(), [pcm(9)])
    detector = detector_with_scores(fake_model, [], device)

    async def run():
        with pytest.raises(RuntimeError, match="capture failed"):
            await detector.listen()
        detector._model.score_sequence = [POSITIVE]
        activation = await detector.listen(preroll=0)
        return await collect(activation.audio)

    assert asyncio.run(run()) == [pcm(9)]


def test_default_listen_constructs_and_opens_one_microphone(fake_model, monkeypatch):
    created = []
    device = FakeDevice([pcm(1)])

    def capture(**kwargs):
        created.append(kwargs)
        return device

    monkeypatch.setattr("wakewordkit.detector._SoundDeviceCapture", capture)
    detector = detector_with_scores(fake_model, [POSITIVE])

    async def run():
        activation = await detector.listen(device="default", preroll=0)
        return await collect(activation.audio)

    assert asyncio.run(run()) == [pcm(1)]
    assert len(created) == 1
    assert created[0]["device"] == "default"
    assert device.open_count == device.close_count == 1


def test_streaming_resampling_has_no_chunk_drift():
    format = AudioFormat(24_000, 1, "pcm_s16le")
    samples = np.arange(24_003, dtype=np.int16)
    chunked = _StreamingNormalizer(format)
    parts = [
        chunked.feed(part) for part in np.array_split(samples, [17, 131, 4099, 20_000])
    ]
    one_shot = _StreamingNormalizer(format).feed(samples)
    result = np.concatenate(parts)

    assert result.size == samples.size * 16_000 // 24_000
    np.testing.assert_array_equal(result, one_shot)
