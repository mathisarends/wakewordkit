import asyncio
import threading

import numpy as np
import pytest

from wakewordkit import Detection, WakeWord, WakeWordDetector

BLOCK = np.zeros(1280, np.int16)
KEY = WakeWord.ALEXA.path.stem


def blocked_detector(fake_model):
    detector = WakeWordDetector("alexa", cooldown=0)
    entered, release = threading.Event(), threading.Event()
    original = detector._model.predict
    detector._model.scores = {KEY: 0.9}

    def predict(audio):
        entered.set()
        assert release.wait(5), "test did not release inference"
        return original(audio)

    detector._model.predict = predict
    return detector, entered, release


def test_admitted_async_cancellation_delivers_result_and_rejects_overlaps(fake_model):
    detector, entered, release = blocked_detector(fake_model)

    async def run():
        task = asyncio.create_task(detector.process(BLOCK))
        assert await asyncio.to_thread(entered.wait, 5)
        try:
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
            for operation in [
                detector.reset,
                detector.finish_sync,
                lambda: detector.process_sync(BLOCK),
            ]:
                with pytest.raises(RuntimeError, match="Overlapping"):
                    operation()
            with pytest.raises(RuntimeError, match="Overlapping"):
                await detector.process(BLOCK)
            with pytest.raises(RuntimeError, match="Overlapping"):
                await detector.finish()
        finally:
            release.set()
        assert await task == Detection("alexa", 0.9, 1280)
        assert task.cancelling() == 0
        assert detector.process_sync(BLOCK).sample_index == 2560

    asyncio.run(run())


def test_cancel_before_admission_does_not_consume_audio(fake_model):
    detector = WakeWordDetector("alexa")

    async def run():
        task = asyncio.create_task(detector.process(BLOCK))
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not detector._model.inputs
        assert detector._samples_processed == 0

    asyncio.run(run())


@pytest.mark.parametrize("kind", ["array", "bytes"])
def test_async_owns_snapshot_of_mutable_input(fake_model, kind):
    detector, entered, release = blocked_detector(fake_model)
    source = np.ones(1280, np.int16)
    frame = source if kind == "array" else bytearray(source.tobytes())

    async def run():
        task = asyncio.create_task(detector.process(frame))
        assert await asyncio.to_thread(entered.wait, 5)
        try:
            if kind == "array":
                frame[:] = 9
            else:
                frame[:] = np.full(1280, 9, np.int16).tobytes()
        finally:
            release.set()
        await task
        np.testing.assert_array_equal(
            detector._model.inputs[0], np.ones(1280, np.int16)
        )

    asyncio.run(run())


def test_cancellation_preserves_backend_errors_and_releases_operation_lock(fake_model):
    detector, entered, release = blocked_detector(fake_model)

    def fail(audio):
        entered.set()
        assert release.wait(5)
        raise ValueError("inference failed")

    detector._model.predict = fail

    async def run():
        task = asyncio.create_task(detector.process(BLOCK))
        assert await asyncio.to_thread(entered.wait, 5)
        task.cancel()
        release.set()
        with pytest.raises(ValueError, match="inference failed"):
            await task
        detector.reset()
        assert detector.process_sync(BLOCK) is None

    asyncio.run(run())


def test_reset_failure_preserves_previous_stream(fake_model, monkeypatch):
    detector = WakeWordDetector("alexa")
    detector.process_sync(np.ones(17, np.int16))
    model = detector._model

    def fail():
        raise ValueError("model load failed")

    monkeypatch.setattr(detector, "_new_model", fail)
    with pytest.raises(ValueError, match="load failed"):
        detector.reset()
    assert detector._model is model
    detector.process_sync(np.zeros(1280 - 17, np.int16))
    assert detector._model.inputs[0][:17].sum() == 17
