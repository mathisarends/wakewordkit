import asyncio
import math
from typing import Self

import sounddevice as sd

from wakewordkit import AudioFormat


class MicrophoneFrames:
    """Application-owned asynchronous PCM source backed by sounddevice."""

    def __init__(
        self,
        *,
        sample_rate: int = 16_000,
        channels: int = 1,
        frame_size: int = 1_280,
        queue_seconds: float = 2.0,
        device: int | str | None = None,
    ) -> None:
        capacity = max(1, math.ceil(queue_seconds * sample_rate / frame_size))
        self.format = AudioFormat(sample_rate, channels, "pcm_s16le")
        self._frame_size = frame_size
        self._device = device
        self._queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=capacity)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stream: sd.RawInputStream | None = None
        self._dropped_frames = 0

    @property
    def dropped_frames(self) -> int:
        return self._dropped_frames

    @property
    def frame_size(self) -> int:
        return self._frame_size

    async def __aenter__(self) -> Self:
        self._loop = asyncio.get_running_loop()
        await asyncio.to_thread(self._open)
        return self

    async def __aexit__(self, *exc: object) -> None:
        await asyncio.to_thread(self._close)

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> bytes:
        return await self.read()

    async def read(self) -> bytes:
        return await self._queue.get()

    def discard_buffered(self) -> None:
        while not self._queue.empty():
            self._queue.get_nowait()

    def _open(self) -> None:
        self._stream = sd.RawInputStream(
            samplerate=self.format.sample_rate,
            blocksize=self._frame_size,
            device=self._device,
            channels=self.format.channels,
            dtype="int16",
            callback=self._on_audio,
        )
        self._stream.start()

    def _close(self) -> None:
        if self._stream is None:
            return
        self._stream.stop()
        self._stream.close()
        self._stream = None

    def _on_audio(
        self,
        data: object,
        frames: int,
        time: object,
        status: sd.CallbackFlags,
    ) -> None:
        del frames, time
        if status.input_overflow:
            self._dropped_frames += 1
        assert self._loop is not None
        self._loop.call_soon_threadsafe(self._put, bytes(data))

    def _put(self, frame: bytes) -> None:
        if self._queue.full():
            self._queue.get_nowait()
            self._dropped_frames += 1
        self._queue.put_nowait(frame)
