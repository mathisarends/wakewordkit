import asyncio
import importlib.util
import sys
import threading
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_example(name):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "examples" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def microphone(monkeypatch):
    class Stream:
        fail_start = False
        fail_stop = False
        opened = []

        def __init__(self, **kwargs):
            self.closed = False
            self.opened.append(self)

        def start(self):
            if self.fail_start:
                raise RuntimeError("start failed")

        def stop(self):
            if self.fail_stop:
                raise RuntimeError("stop failed")

        def close(self):
            self.closed = True

    monkeypatch.setitem(
        sys.modules, "sounddevice", SimpleNamespace(RawInputStream=Stream)
    )
    return load_example("_microphone"), Stream


def test_microphone_start_failure_closes_allocated_stream(microphone):
    module, stream = microphone
    stream.fail_start = True
    source = module.MicrophoneFrames()

    async def run():
        with pytest.raises(RuntimeError, match="start failed"):
            async with source:
                pass
        assert stream.opened[0].closed
        assert source._stream is None
        assert source._closed

    asyncio.run(run())


def test_microphone_stop_failure_still_closes_stream(microphone):
    module, stream = microphone
    stream.fail_stop = True

    async def run():
        with pytest.raises(RuntimeError, match="stop failed"):
            async with module.MicrophoneFrames():
                pass
        assert stream.opened[0].closed

    asyncio.run(run())


def test_callback_backlog_is_bounded_before_event_loop_runs(microphone):
    module, _ = microphone

    async def run():
        async with module.MicrophoneFrames(queue_seconds=0.16) as source:
            notifications = []
            loop = source._loop
            source._loop = SimpleNamespace(
                call_soon_threadsafe=lambda callback: notifications.append(callback)
            )
            for n in range(1000):
                source._on_audio(
                    bytes([n % 256]), 1, None, SimpleNamespace(input_overflow=False)
                )
            assert len(notifications) == 1
            assert len(source._buffer) == 2
            assert source.dropped_frames == 998
            notifications.pop()()
            assert await source.read() == bytes([998 % 256])
            assert await source.read() == bytes([999 % 256])
            source._loop = loop

    asyncio.run(run())


def test_cancelled_microphone_entry_waits_then_closes(microphone):
    module, stream = microphone
    entered, release = threading.Event(), threading.Event()

    def slow_start(self):
        entered.set()
        assert release.wait(5)

    stream.start = slow_start

    async def run():
        source = module.MicrophoneFrames()
        task = asyncio.create_task(source.__aenter__())
        assert await asyncio.to_thread(entered.wait, 5)
        task.cancel()
        await asyncio.sleep(0)
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert stream.opened[0].closed
        assert source._stream is None

    asyncio.run(run())


@pytest.mark.parametrize("rate", [8000, 16000, 48000])
def test_wav_example_accepts_short_final_chunk(fake_model, tmp_path, capsys, rate):
    path = tmp_path / "short.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(np.zeros(1301, np.int16).tobytes())
    load_example("detect_from_wav").main(str(path))
    assert "No wake word detected" in capsys.readouterr().out
