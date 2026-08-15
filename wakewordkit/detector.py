from __future__ import annotations

import asyncio
import threading
import time
import warnings
from collections import deque
from collections.abc import AsyncIterable, AsyncIterator, Iterable
from dataclasses import dataclass

import numpy as np
from openwakeword.model import Model

from wakewordkit.audio import (
    MODEL_AUDIO_FORMAT,
    AudioFormat,
    AudioInput,
    AudioStream,
    Microphone,
)
from wakewordkit.models import CustomWakeWord, WakeWord

type WakeWordSpec = WakeWord | CustomWakeWord | str


@dataclass(frozen=True, slots=True)
class Detection:
    name: str
    score: float


@dataclass(frozen=True, slots=True)
class Activation:
    name: str
    score: float
    audio: AudioStream


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
        self._session_guard = threading.Lock()
        self._session: _ListenSession | None = None

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
            normalized = self._process_normalizer.feed(frame)
            return self._predict(normalized)

    async def stream(
        self,
        source: AsyncIterable[bytes | np.ndarray],
        *,
        format: AudioFormat = MODEL_AUDIO_FORMAT,
    ) -> AsyncIterator[Detection]:
        """Detect wake words in a caller-owned asynchronous source."""
        normalizer = _StreamingNormalizer(format)
        try:
            async for frame in source:
                # Model work is moved off the event loop; source ownership stays
                # with the caller and the iterator is never closed by us.
                detection = await asyncio.to_thread(
                    self._process_stream_frame, normalizer, frame
                )
                if detection is not None:
                    yield detection
        finally:
            await asyncio.to_thread(self._reset_runtime)

    async def listen(
        self,
        *,
        device: int | str | None = None,
        sample_rate: int = 16_000,
        channels: int = 1,
        sample_format: str = "pcm_s16le",
        preroll: float = 0.2,
        max_buffer: float = 10.0,
        frame_size: int = 1_280,
    ) -> Activation:
        """Open one input and await exactly one activation.

        ``max_buffer`` (seconds, default 10) bounds a slow continuation
        consumer. Overflow drops oldest chunks and is visible in
        ``activation.audio.stats``.
        """
        session = self._start_session(
            device=device,
            sample_rate=sample_rate,
            channels=channels,
            sample_format=sample_format,
            preroll=preroll,
            max_buffer=max_buffer,
            frame_size=frame_size,
        )
        try:
            return await asyncio.to_thread(session.wait)
        except BaseException:
            session.stop()
            await asyncio.to_thread(session.join)
            raise

    def reset(self) -> None:
        """Reset frame-processing state (normally not needed by applications)."""
        self._reset_runtime()
        self._process_normalizer = None
        self._process_format = None

    def _start_session(self, **kwargs: object) -> _ListenSession:
        with self._session_guard:
            if self._session is not None and not self._session.done:
                raise RuntimeError(
                    "This detector already has an active listen operation."
                )
            session = _ListenSession(self, **kwargs)
            self._session = session
        session.start()
        return session

    def _session_done(self, session: _ListenSession) -> None:
        with self._session_guard:
            if self._session is session:
                self._session = None

    def _process_stream_frame(
        self, normalizer: _StreamingNormalizer, frame: bytes | np.ndarray
    ) -> Detection | None:
        with self._model_lock:
            detection = self._apply_cooldown(self._predict(normalizer.feed(frame)))
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


class _ListenSession:
    def __init__(
        self,
        detector: WakeWordDetector,
        *,
        source: AudioInput | None = None,
        device: int | str | None = None,
        sample_rate: int = 16_000,
        channels: int = 1,
        sample_format: str = "pcm_s16le",
        preroll: float = 0.2,
        max_buffer: float = 10.0,
        frame_size: int = 1_280,
    ) -> None:
        if preroll < 0:
            raise ValueError("preroll must be non-negative.")
        if max_buffer <= 0:
            raise ValueError("max_buffer must be positive.")
        self.detector = detector
        self.source = (
            source
            if source is not None
            else Microphone(
                device=device,
                sample_rate=sample_rate,
                channels=channels,
                sample_format=sample_format,
                frame_size=frame_size,
            )
        )
        self.format = self.source.format
        self.preroll_bytes = int(preroll * self.format.bytes_per_second)
        alignment = self.format.channels * self.format.bytes_per_sample
        self.preroll_bytes = self.preroll_bytes // alignment * alignment
        configured_max = int(max_buffer * self.format.bytes_per_second)
        detection_frame_bytes = frame_size * alignment
        self.max_buffer_bytes = max(
            configured_max,
            self.preroll_bytes + detection_frame_bytes,
            alignment,
        )
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="wakewordkit-capture", daemon=True
        )
        self._activation: Activation | None = None
        self._error: BaseException | None = None
        self._stream: AudioStream | None = None

    @property
    def done(self) -> bool:
        return not self._thread.is_alive() and self._thread.ident is not None

    def start(self) -> None:
        self._thread.start()

    def wait(self) -> Activation:
        self._ready.wait()
        if self._activation is not None:
            return self._activation
        if self._error is not None:
            raise self._error
        raise EOFError("Audio input ended before a wake word was detected.")

    def stop(self) -> None:
        self._stop.set()
        close = getattr(self.source, "close", None)
        if close is not None:
            close()

    def join(self) -> None:
        if threading.current_thread() is not self._thread:
            self._thread.join()

    def _run(self) -> None:
        preroll: deque[bytes] = deque()
        preroll_size = 0
        normalizer = _StreamingNormalizer(self.format)
        activated = False
        try:
            self.detector._reset_runtime()
            with self.source as source:
                while not self._stop.is_set():
                    try:
                        raw = source.read()
                    except StopIteration:
                        break
                    chunk = _frame_as_bytes(raw, self.format)
                    if activated:
                        assert self._stream is not None
                        self._stream._put(chunk)
                        continue

                    with self.detector._model_lock:
                        detection = self.detector._predict(normalizer.feed(raw))
                    if detection is None:
                        if self.preroll_bytes:
                            preroll.append(chunk)
                            preroll_size += len(chunk)
                            while preroll and preroll_size > self.preroll_bytes:
                                excess = preroll_size - self.preroll_bytes
                                first = preroll[0]
                                if excess >= len(first):
                                    preroll.popleft()
                                    preroll_size -= len(first)
                                else:
                                    preroll[0] = first[excess:]
                                    preroll_size -= excess
                        continue

                    stream = AudioStream(
                        self.format,
                        max_buffer_bytes=self.max_buffer_bytes,
                        on_close=self.stop,
                    )
                    # Pre-roll contains only frames before detection. The
                    # detection frame itself is always handed off, including
                    # when preroll is zero.
                    for buffered in preroll:
                        stream._put(buffered)
                    stream._put(chunk)
                    self._stream = stream
                    self._activation = Activation(
                        detection.name, detection.score, stream
                    )
                    activated = True
                    self._ready.set()
        except BaseException as exc:
            if self._stream is not None:
                self._stream._finish(exc)
            else:
                self._error = exc
        finally:
            self.detector._reset_runtime()
            if self._stream is not None:
                self._stream._finish()
            self._ready.set()
            self.detector._session_done(self)


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


def _frame_as_bytes(frame: bytes | np.ndarray, format: AudioFormat) -> bytes:
    if isinstance(frame, bytes | bytearray | memoryview):
        return bytes(frame)
    array = np.asarray(frame)
    if format.sample_format == "pcm_s16le":
        if np.issubdtype(array.dtype, np.floating):
            array = np.clip(np.rint(array * 32768.0), -32768, 32767)
        return array.astype("<i2", copy=False).tobytes()
    if np.issubdtype(array.dtype, np.integer):
        array = array.astype(np.float32) / 32768.0
    return array.astype("<f4", copy=False).tobytes()


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


def _as_int16(frame: bytes | np.ndarray) -> np.ndarray:
    """Compatibility helper for callers that used the old private utility."""
    if isinstance(frame, bytes | bytearray | memoryview):
        return np.frombuffer(frame, dtype=np.int16)
    array = np.asarray(frame)
    if array.dtype == np.int16:
        return array
    if np.issubdtype(array.dtype, np.floating):
        return (array * np.iinfo(np.int16).max).astype(np.int16)
    return array.astype(np.int16)
