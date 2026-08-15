from __future__ import annotations

import asyncio
import threading
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Self

import numpy as np

_SAMPLE_RATE = 16_000
_FRAME_SIZE = 1_280


@dataclass(frozen=True, slots=True)
class AudioFormat:
    """The contract of an interleaved PCM byte stream."""

    sample_rate: int
    channels: int
    sample_format: str

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive.")
        if self.channels <= 0:
            raise ValueError("channels must be positive.")
        if self.sample_format not in {"pcm_s16le", "pcm_f32le"}:
            raise ValueError("sample_format must be 'pcm_s16le' or 'pcm_f32le'.")

    @property
    def bytes_per_sample(self) -> int:
        return 2 if self.sample_format == "pcm_s16le" else 4

    @property
    def bytes_per_second(self) -> int:
        return self.sample_rate * self.channels * self.bytes_per_sample


MODEL_AUDIO_FORMAT = AudioFormat(16_000, 1, "pcm_s16le")


@dataclass(frozen=True, slots=True)
class AudioStreamStats:
    dropped_chunks: int = 0
    dropped_bytes: int = 0
    buffered_bytes: int = 0


class AudioStream:
    """A bounded continuation stream owned by an activation.

    The producer never blocks. If ``max_buffer`` is exhausted, the oldest
    complete chunks are dropped (``drop_oldest`` policy) and the loss is
    reported through :attr:`stats`.
    """

    overflow_policy = "drop_oldest"

    def __init__(
        self,
        format: AudioFormat,
        *,
        max_buffer_bytes: int,
        on_close: Callable[[], None],
    ) -> None:
        if max_buffer_bytes <= 0:
            raise ValueError("max_buffer_bytes must be positive.")
        self.format = format
        self._max_buffer_bytes = max_buffer_bytes
        self._on_close = on_close
        self._chunks: deque[bytes] = deque()
        self._buffered_bytes = 0
        self._dropped_chunks = 0
        self._dropped_bytes = 0
        self._condition = threading.Condition()
        self._finished = False
        self._closed = False
        self._error: BaseException | None = None

    @property
    def stats(self) -> AudioStreamStats:
        with self._condition:
            return AudioStreamStats(
                dropped_chunks=self._dropped_chunks,
                dropped_bytes=self._dropped_bytes,
                buffered_bytes=self._buffered_bytes,
            )

    @property
    def closed(self) -> bool:
        with self._condition:
            return self._closed

    def _put(self, chunk: bytes) -> None:
        if not chunk:
            return
        chunk = bytes(chunk)
        with self._condition:
            if self._finished or self._closed:
                return
            while self._chunks and (
                self._buffered_bytes + len(chunk) > self._max_buffer_bytes
            ):
                dropped = self._chunks.popleft()
                self._buffered_bytes -= len(dropped)
                self._dropped_chunks += 1
                self._dropped_bytes += len(dropped)
            if len(chunk) > self._max_buffer_bytes:
                alignment = self.format.channels * self.format.bytes_per_sample
                keep = self._max_buffer_bytes // alignment * alignment
                discarded = len(chunk) - keep
                chunk = chunk[-keep:] if keep else b""
                self._dropped_chunks += 1
                self._dropped_bytes += discarded
            if chunk:
                self._chunks.append(chunk)
                self._buffered_bytes += len(chunk)
                self._condition.notify()

    def _finish(self, error: BaseException | None = None) -> None:
        with self._condition:
            if self._finished:
                return
            self._error = error
            self._finished = True
            self._condition.notify_all()

    def _take(self) -> bytes | object:
        with self._condition:
            while not self._chunks and not self._finished and not self._closed:
                self._condition.wait()
            if self._chunks:
                chunk = self._chunks.popleft()
                self._buffered_bytes -= len(chunk)
                return chunk
            if self._error is not None:
                error, self._error = self._error, None
                raise error
            return _EOF

    def _close(self) -> None:
        callback: Callable[[], None] | None = None
        with self._condition:
            if not self._closed:
                self._closed = True
                self._chunks.clear()
                self._buffered_bytes = 0
                callback, self._on_close = self._on_close, lambda: None
                self._condition.notify_all()
        if callback is not None:
            callback()

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> bytes:
        try:
            item = await asyncio.to_thread(self._take)
        except asyncio.CancelledError:
            self._close()
            raise
        if item is _EOF:
            raise StopAsyncIteration
        return item

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        self._close()

    async def aclose(self) -> None:
        self._close()


_EOF = object()


class AudioInput(ABC):
    """Blocking audio input used internally by the default capture path."""

    @property
    def format(self) -> AudioFormat:
        return MODEL_AUDIO_FORMAT

    @abstractmethod
    def read(self) -> bytes | np.ndarray:
        """Return the next frame, or raise ``StopIteration`` at EOF."""

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def __iter__(self) -> Iterator[bytes | np.ndarray]:
        return self

    def __next__(self) -> bytes | np.ndarray:
        return self.read()


class CallableInput(AudioInput):
    def __init__(
        self,
        read: Callable[..., bytes | np.ndarray],
        frame_size: int | None = None,
        *,
        format: AudioFormat = MODEL_AUDIO_FORMAT,
        close: Callable[[], None] | None = None,
    ) -> None:
        self._read = read
        self._frame_size = frame_size
        self._format = format
        self._close = close

    @property
    def format(self) -> AudioFormat:
        return self._format

    def read(self) -> bytes | np.ndarray:
        if self._frame_size is None:
            return self._read()
        return self._read(self._frame_size)

    def __exit__(self, *exc: object) -> None:
        if self._close is not None:
            self._close()


class Microphone(AudioInput):
    def __init__(
        self,
        *,
        sample_rate: int = _SAMPLE_RATE,
        channels: int = 1,
        sample_format: str = "pcm_s16le",
        frame_size: int = _FRAME_SIZE,
        device: int | str | None = None,
        samplerate: int | None = None,
    ) -> None:
        # ``samplerate`` remains as a harmless alias for callers of 0.2.
        if samplerate is not None:
            sample_rate = samplerate
        self._format = AudioFormat(sample_rate, channels, sample_format)
        self._frame_size = frame_size
        # sounddevice selects PortAudio's configured default with ``None``.
        # WakewordKit also accepts the more self-explanatory public spelling.
        self._device = None if device == "default" else device
        self._stream = None

    @property
    def format(self) -> AudioFormat:
        return self._format

    def __enter__(self) -> Self:  # pragma: no cover - requires audio hardware
        sd = _import_sounddevice()
        dtype = "int16" if self.format.sample_format == "pcm_s16le" else "float32"
        self._stream = sd.InputStream(
            samplerate=self.format.sample_rate,
            blocksize=self._frame_size,
            device=self._device,
            channels=self.format.channels,
            dtype=dtype,
        )
        self._stream.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def read(self) -> bytes:
        if self._stream is None:
            raise RuntimeError("Microphone must be opened as a context manager.")
        data, overflowed = self._stream.read(self._frame_size)
        # PortAudio input overflow means samples were already lost. Surface it
        # instead of silently handing a discontinuous stream to the caller.
        if overflowed:
            raise OSError("Microphone input overflowed; captured audio was lost.")
        return np.asarray(data).tobytes()

    def close(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None


def _import_sounddevice():
    try:
        import sounddevice
    except ImportError as exc:
        raise ImportError(
            "Microphone input requires the 'mic' extra: pip install 'wakewordkit[mic]'"
        ) from exc
    return sounddevice
