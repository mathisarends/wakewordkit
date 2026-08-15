import sys

import numpy as np
import pytest

from wakewordkit.audio import _import_sounddevice, _Microphone


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
