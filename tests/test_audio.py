import sys

import numpy as np
import pytest

from wakewordkit.audio import AudioInput, CallableInput, Microphone, _import_sounddevice


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


class Frames(AudioInput):
    def __init__(self, frames):
        self._frames = iter(frames)

    def read(self):
        return next(self._frames)


def test_audio_input_cannot_be_instantiated():
    with pytest.raises(TypeError):
        AudioInput()


def test_audio_input_iterates_until_stop_iteration():
    source = Frames([b"a", b"b"])
    assert list(source) == [b"a", b"b"]


def test_audio_input_is_a_reusable_context_manager():
    source = Frames([b"a"])
    with source as entered:
        assert entered is source


def test_callable_input_passes_the_frame_size():
    source = CallableInput(lambda n: b"x" * n, 1280)
    assert source.read() == b"x" * 1280


def test_callable_input_omits_the_frame_size_when_unset():
    source = CallableInput(lambda: b"x")
    assert source.read() == b"x"


def test_callable_input_leaves_a_shared_source_open_by_default():
    with CallableInput(lambda: b"x") as source:
        assert source.read() == b"x"


def test_callable_input_closes_on_exit_when_given_a_closer():
    calls = []
    with CallableInput(lambda: b"x", close=lambda: calls.append(1)):
        pass
    assert calls == [1]


def test_microphone_is_an_audio_input():
    assert issubclass(Microphone, AudioInput)


def test_microphone_read_before_open_raises():
    with pytest.raises(RuntimeError, match="context manager"):
        Microphone().read()


def test_microphone_close_without_open_is_safe():
    Microphone().close()


def test_microphone_read_flattens_stream_frames():
    mic = Microphone(frame_size=1280)
    mic._stream = FakeStream()
    frame = mic.read()
    assert frame == np.zeros((1280, 1), dtype=np.int16).tobytes()


def test_microphone_close_tears_down_the_stream():
    mic = Microphone()
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
