from __future__ import annotations

import threading
import time
import warnings
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from openwakeword.model import Model

from wakewordkit.audio import MODEL_AUDIO_FORMAT, AudioFormat
from wakewordkit.models import CustomWakeWord, WakeWord

type WakeWordSpec = WakeWord | CustomWakeWord | str


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
        cooldown: float = 1.5,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0.")
        if cooldown < 0.0:
            raise ValueError("cooldown must be non-negative.")

        specs = [_resolve_spec(spec) for spec in _flatten(wake_words)]
        specs = specs or list(WakeWord)
        self._threshold = threshold
        self._cooldown = cooldown
        self._last_trigger: float | None = None
        self._labels: dict[str, str] = {}
        self._model_lock = threading.RLock()
        self._process_normalizer: _StreamingNormalizer | None = None
        self._process_format: AudioFormat | None = None

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
    def cooldown(self) -> float:
        return self._cooldown

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._labels.values())

    def process(
        self,
        frame: bytes | np.ndarray,
        *,
        format: AudioFormat = MODEL_AUDIO_FORMAT,
    ) -> Detection | None:
        """Process one frame without taking ownership of its source."""
        with self._model_lock:
            if self._process_normalizer is None or self._process_format != format:
                self._process_normalizer = _StreamingNormalizer(format)
                self._process_format = format
            return self._detect(self._process_normalizer.feed(frame))

    def reset(self) -> None:
        """Reset frame-processing state (normally not needed by applications)."""
        self._reset_runtime()
        self._process_normalizer = None
        self._process_format = None

    def _detect(self, normalized: np.ndarray) -> Detection | None:
        detection = self._apply_cooldown(self._predict(normalized))
        if detection is not None:
            self._model.reset()
        return detection

    def _predict(self, normalized: np.ndarray) -> Detection | None:
        if normalized.size == 0:
            return None
        scores = self._model.predict(normalized)
        if not scores:
            return None
        key, score = max(scores.items(), key=lambda item: item[1])
        if score < self._threshold:
            return None
        return Detection(self._labels.get(key, key), float(score))

    def _apply_cooldown(self, detection: Detection | None) -> Detection | None:
        if detection is None:
            return None
        now = time.monotonic()
        if self._last_trigger is not None and now - self._last_trigger < self._cooldown:
            return None
        self._last_trigger = now
        return detection

    def _reset_runtime(self) -> None:
        with self._model_lock:
            self._model.reset()
            self._last_trigger = None


class _StreamingNormalizer:
    """Stateful PCM decoder/downmixer/resampler with rational sample timing."""

    def __init__(self, format: AudioFormat) -> None:
        self.format = format
        self._buffer = np.empty(0, dtype=np.float64)
        self._buffer_start = 0
        self._seen = 0
        self._produced = 0

    def feed(self, frame: bytes | np.ndarray) -> np.ndarray:
        samples = _decode_mono(frame, self.format)
        if samples.size == 0:
            return np.empty(0, dtype=np.int16)
        self._buffer = np.concatenate((self._buffer, samples))
        self._seen += samples.size
        target_total = (
            self._seen * MODEL_AUDIO_FORMAT.sample_rate // self.format.sample_rate
        )
        if target_total == self._produced:
            return np.empty(0, dtype=np.int16)

        output_indexes = np.arange(self._produced, target_total, dtype=np.int64)
        numerators = output_indexes * self.format.sample_rate
        left_global = numerators // MODEL_AUDIO_FORMAT.sample_rate
        fractions = (
            numerators % MODEL_AUDIO_FORMAT.sample_rate
        ) / MODEL_AUDIO_FORMAT.sample_rate
        left = left_global - self._buffer_start
        right = np.minimum(left + 1, self._buffer.size - 1)
        output = (
            self._buffer[left] * (1.0 - fractions) + self._buffer[right] * fractions
        )
        self._produced = int(target_total)

        next_numerator = self._produced * self.format.sample_rate
        keep_global = next_numerator // MODEL_AUDIO_FORMAT.sample_rate
        discard = max(0, int(keep_global - self._buffer_start))
        # Retain one interpolation neighbour when it still exists.
        discard = min(discard, max(0, self._buffer.size - 1))
        if discard:
            self._buffer = self._buffer[discard:]
            self._buffer_start += discard
        return np.clip(np.rint(output * 32768.0), -32768, 32767).astype(np.int16)


def _decode_mono(frame: bytes | np.ndarray, format: AudioFormat) -> np.ndarray:
    if isinstance(frame, bytes | bytearray | memoryview):
        dtype = "<i2" if format.sample_format == "pcm_s16le" else "<f4"
        array = np.frombuffer(frame, dtype=dtype)
        integer_input = format.sample_format == "pcm_s16le"
    else:
        array = np.asarray(frame).reshape(-1)
        integer_input = np.issubdtype(array.dtype, np.integer)
    usable = array.size // format.channels * format.channels
    array = array[:usable]
    if format.channels > 1:
        array = array.reshape(-1, format.channels).astype(np.float64).mean(axis=1)
    else:
        array = array.astype(np.float64)
    if integer_input:
        array /= 32768.0
    return array


def _resolve_spec(spec: WakeWordSpec) -> WakeWord | CustomWakeWord:
    if not isinstance(spec, str):
        return spec
    for builtin in WakeWord:
        if spec in {builtin.label, builtin.name.lower(), builtin.value}:
            return builtin
    raise ValueError(f"Unknown built-in wake word: {spec!r}")


def _flatten(
    items: Iterable[WakeWordSpec | Iterable[WakeWordSpec]],
) -> list[WakeWordSpec]:
    result: list[WakeWordSpec] = []
    for item in items:
        if isinstance(item, WakeWord | CustomWakeWord | str):
            result.append(item)
        else:
            result.extend(item)
    return result
