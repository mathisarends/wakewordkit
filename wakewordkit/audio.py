from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from typing import Self

import numpy as np

_SAMPLE_RATE = 16_000
_FRAME_SIZE = 1_280


class AudioInput(ABC):
    @abstractmethod
    def read(self) -> bytes | np.ndarray:
        """Return the next audio frame as 16 kHz mono int16 samples.

        Raise StopIteration to end the stream.
        """

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def __iter__(self) -> Iterator[bytes | np.ndarray]:
        return self

    def __next__(self) -> bytes | np.ndarray:
        return self.read()


class CallableInput(AudioInput):
    """Adapt an existing read function into an `AudioInput`.

    Use this to plug a microphone you already own into a detector without
    subclassing. `frame_size` is passed to `read` when given, so both
    `read()` and `read(1280)` style callables work. `close` is only called on
    context-manager exit if you provide it, so a shared stream stays open.

    Example:
        ```python
        source = CallableInput(my_microphone.read, frame_size=1280)

        with source:
            frame = source.read()  # calls my_microphone.read(1280)
        ```
    """

    def __init__(
        self,
        read: Callable[..., bytes | np.ndarray],
        frame_size: int | None = None,
        *,
        close: Callable[[], None] | None = None,
    ) -> None:
        self._read = read
        self._frame_size = frame_size
        self._close = close

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
        samplerate: int = _SAMPLE_RATE,
        frame_size: int = _FRAME_SIZE,
        device: int | str | None = None,
    ) -> None:
        self._samplerate = samplerate
        self._frame_size = frame_size
        self._device = device
        self._stream = None

    def __enter__(self) -> Self:  # pragma: no cover - requires audio hardware
        sd = _import_sounddevice()
        self._stream = sd.InputStream(
            samplerate=self._samplerate,
            blocksize=self._frame_size,
            device=self._device,
            channels=1,
            dtype="int16",
        )
        self._stream.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def read(self) -> np.ndarray:
        if self._stream is None:
            raise RuntimeError("Microphone must be opened as a context manager.")
        data, _ = self._stream.read(self._frame_size)
        return data.reshape(-1)

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
