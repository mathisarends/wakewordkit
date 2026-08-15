import asyncio
import sys

import numpy as np
import pytest

from wakewordkit import AudioFormat
from wakewordkit.audio import _import_sounddevice, _Microphone, _SoundDeviceCapture


class FakeStream:
    def __init__(self):
        self.stopped = False
        self.closed = False

    def read(self, frames):
        return np.zeros((frames, 1), dtype=np.int16), False

    def stop(self):
        self.stopped = True

    def close(self):
        self.closed = True


def test_microphone_maps_default_device_to_portaudio_default():
    assert _Microphone(device="default")._device is None


def test_microphone_read_before_open_raises():
    with pytest.raises(RuntimeError, match="context manager"):
        _Microphone().read()


def test_microphone_close_without_open_is_safe():
    _Microphone().close()


def test_microphone_read_flattens_stream_frames():
    mic = _Microphone(frame_size=1280)
    mic._stream = FakeStream()
    frame = mic.read()
    assert frame == np.zeros((1280, 1), dtype=np.int16).tobytes()


def test_microphone_close_tears_down_the_stream():
    mic = _Microphone()
    stream = FakeStream()
    mic._stream = stream
    mic.close()
    assert stream.stopped
    assert stream.closed
    assert mic._stream is None


def test_missing_sounddevice_raises_helpful_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "sounddevice", None)
    with pytest.raises(ImportError, match="mic"):
        _import_sounddevice()


def test_sounddevice_capture_implements_public_device_contract(monkeypatch):
    format = AudioFormat(24_000, 1, "pcm_s16le")

    class FakeMicrophone:
        def __init__(self, **options):
            self.format = AudioFormat(
                options["sample_rate"],
                options["channels"],
                options["sample_format"],
            )
            self.frames = iter([b"first", b"second"])
            self.open_count = 0
            self.close_count = 0

        def __enter__(self):
            self.open_count += 1
            return self

        def read(self):
            return next(self.frames)

        def close(self):
            self.close_count += 1

    created = []

    def microphone(**options):
        instance = FakeMicrophone(**options)
        created.append(instance)
        return instance

    monkeypatch.setattr("wakewordkit.audio._Microphone", microphone)
    device = _SoundDeviceCapture(
        sample_rate=24_000,
        channels=1,
        sample_format="pcm_s16le",
        frame_size=1_280,
        device="default",
    )

    async def run():
        async with device.open() as captured:
            return captured.format, [frame async for frame in captured.frames]

    captured_format, frames = asyncio.run(run())
    assert captured_format == format
    assert frames == [b"first", b"second"]
    assert created[0].open_count == created[0].close_count == 1
