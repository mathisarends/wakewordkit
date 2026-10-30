from __future__ import annotations

import asyncio
import math
import threading
from collections import deque
from collections.abc import Callable
from typing import Self

import sounddevice as sd

from wakewordkit import AudioFormat


async def _run_io(operation: Callable[[], None]) -> None:
    """Complete device work before propagating cancellation to the context manager."""
    worker = asyncio.get_running_loop().run_in_executor(None, operation)
    cancelled = None
    while True:
        try:
            await asyncio.shield(worker)
            break
        except asyncio.CancelledError as exc:
            if worker.cancelled():
                raise
            cancelled = exc
    if cancelled is not None:
        raise cancelled


class MicrophoneFrames:
    """Application-owned PCM source with a bounded callback-side buffer."""

    def __init__(
        self,
        *,
        sample_rate: int = 16_000,
        channels: int = 1,
        frame_size: int = 1_280,
        queue_seconds: float = 2.0,
        device: int | str | None = None,
    ) -> None:
        self.format = AudioFormat(sample_rate, channels, "pcm_s16le")
        if (
            isinstance(frame_size, bool)
            or not isinstance(frame_size, int)
            or frame_size <= 0
        ):
            raise ValueError("frame_size must be a positive integer.")
        if not math.isfinite(queue_seconds) or queue_seconds <= 0:
            raise ValueError("queue_seconds must be finite and positive.")
        self._capacity = max(1, math.ceil(queue_seconds * sample_rate / frame_size))
        self._frame_size = frame_size
        self._device = device
        self._buffer: deque[bytes] = deque()
        self._buffer_lock = threading.Lock()
        self._available = asyncio.Event()
        self._notify_pending = False
        self._closed = True
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stream: sd.RawInputStream | None = None
        self._dropped_frames = 0

    @property
    def dropped_frames(self) -> int:
        with self._buffer_lock:
            return self._dropped_frames

    @property
    def frame_size(self) -> int:
        return self._frame_size

    async def __aenter__(self) -> Self:
        if self._stream is not None:
            raise RuntimeError("Microphone is already open.")
        self._loop = asyncio.get_running_loop()
        try:
            await _run_io(self._open)
        except BaseException:
            await _run_io(self._close)
            raise
        return self

    async def __aexit__(self, *exc: object) -> None:
        await _run_io(self._close)

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> bytes:
        try:
            return await self.read()
        except EOFError:
            raise StopAsyncIteration from None

    async def read(self) -> bytes:
        while True:
            with self._buffer_lock:
                if self._buffer:
                    return self._buffer.popleft()
                if self._closed:
                    raise EOFError("Microphone is closed.")
            self._available.clear()
            await self._available.wait()

    def discard_buffered(self) -> None:
        with self._buffer_lock:
            self._buffer.clear()

    def _open(self) -> None:
        with self._buffer_lock:
            self._closed = False
        self._stream = sd.RawInputStream(
            samplerate=self.format.sample_rate,
            blocksize=self._frame_size,
            device=self._device,
            channels=self.format.channels,
            dtype="int16",
            callback=self._on_audio,
        )
        try:
            self._stream.start()
        except BaseException:
            # PortAudio may allocate a stream before start() fails.
            try:
                self._stream.close()
            finally:
                self._stream = None
            raise

    def _close(self) -> None:
        stream, self._stream = self._stream, None
        try:
            if stream is not None:
                try:
                    stream.stop()
                finally:
                    stream.close()
        finally:
            with self._buffer_lock:
                self._closed = True
                self._buffer.clear()
                self._schedule_notification()

    def _on_audio(
        self, data: object, frames: int, time: object, status: sd.CallbackFlags
    ) -> None:
        del frames, time
        frame = bytes(data)
        with self._buffer_lock:
            if self._closed:
                return
            if status.input_overflow:
                self._dropped_frames += 1
            if len(self._buffer) == self._capacity:
                self._buffer.popleft()
                self._dropped_frames += 1
            self._buffer.append(frame)
            self._schedule_notification()

    def _schedule_notification(self) -> None:
        # Called under _buffer_lock. At most one callback can await the event loop,
        # even when the event loop stalls for seconds during application work.
        if self._notify_pending or self._loop is None:
            return
        self._notify_pending = True
        try:
            self._loop.call_soon_threadsafe(self._notify)
        except RuntimeError:  # Event loop was closed by the application.
            self._notify_pending = False
            self._closed = True
            self._buffer.clear()

    def _notify(self) -> None:
        with self._buffer_lock:
            self._notify_pending = False
        self._available.set()
