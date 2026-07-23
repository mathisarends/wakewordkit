import warnings
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import numpy as np
from openwakeword.model import Model

from .audio import AudioInput, Microphone
from .models import CustomWakeWord, WakeWord

type WakeWordSpec = WakeWord | CustomWakeWord


@dataclass(frozen=True, slots=True)
class Detection:
    name: str
    score: float


class WakeWordDetector:
    def __init__(
        self,
        *wake_words: WakeWordSpec | Iterable[WakeWordSpec],
        threshold: float = 0.5,
        vad_threshold: float = 0.0,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0.")

        specs = _flatten(wake_words) or list(WakeWord)

        self._threshold = threshold
        self._labels: dict[str, str] = {}
        paths: list[str] = []
        for spec in specs:
            path = spec.path
            if not path.exists():
                raise FileNotFoundError(f"Wake word model not found: {path}")
            self._labels[path.stem] = spec.label
            paths.append(str(path))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model = Model(wakeword_model_paths=paths, vad_threshold=vad_threshold)

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._labels.values())

    def process(self, frame: bytes | np.ndarray) -> Detection | None:
        scores = self._model.predict(_as_int16(frame))
        key, score = max(scores.items(), key=lambda item: item[1])
        if score < self._threshold:
            return None
        return Detection(self._labels.get(key, key), float(score))

    def stream(self, source: Iterable[bytes | np.ndarray]) -> Iterator[Detection]:
        for frame in source:
            detection = self.process(frame)
            if detection is not None:
                yield detection
                self._model.reset()

    def listen(self, source: AudioInput | None = None) -> Iterator[Detection]:
        with source or Microphone() as audio:
            yield from self.stream(audio)

    def reset(self) -> None:
        self._model.reset()


def _flatten(
    items: Iterable[WakeWordSpec | Iterable[WakeWordSpec]],
) -> list[WakeWordSpec]:
    result: list[WakeWordSpec] = []
    for item in items:
        if isinstance(item, WakeWord | CustomWakeWord):
            result.append(item)
        else:
            result.extend(item)
    return result


def _as_int16(frame: bytes | np.ndarray) -> np.ndarray:
    if isinstance(frame, bytes | bytearray | memoryview):
        return np.frombuffer(frame, dtype=np.int16)
    array = np.asarray(frame)
    if array.dtype == np.int16:
        return array
    if np.issubdtype(array.dtype, np.floating):
        return (array * np.iinfo(np.int16).max).astype(np.int16)
    return array.astype(np.int16)
