from __future__ import annotations

import asyncio
import math
import threading
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ._audio import _StreamingNormalizer
from ._backend import Backend, create_model
from .audio import MODEL_AUDIO_FORMAT, AudioFormat, AudioFrame
from .models import CustomWakeWord, WakeWord

type WakeWordSpec = WakeWord | CustomWakeWord | str
_MODEL_BLOCK = 1280
_SAMPLE_RATE = MODEL_AUDIO_FORMAT.sample_rate


@dataclass(frozen=True, slots=True)
class Detection:
    """A score and the exclusive end sample of its model window at 16 kHz."""

    name: str
    score: float
    sample_index: int | None = None

    @property
    def timestamp(self) -> float | None:
        return None if self.sample_index is None else self.sample_index / _SAMPLE_RATE


class WakeWordDetector:
    """One instance processes one ordered stream; overlapping calls are rejected."""

    def __init__(
        self,
        *wake_words: WakeWordSpec,
        threshold: float = 0.5,
        vad_threshold: float = 0.0,
        cooldown: float = 1.5,
        format: AudioFormat | None = None,
    ) -> None:
        self._threshold = _number("threshold", threshold, maximum=1.0)
        self._vad_threshold = _number("vad_threshold", vad_threshold, maximum=1.0)
        self._cooldown = _number("cooldown", cooldown)
        if format is not None and not isinstance(format, AudioFormat):
            raise TypeError("format must be an AudioFormat.")
        self._initial_format = format
        self._labels: dict[str, str] = {}
        self._paths: list[str] = []
        paths: set[str] = set()
        stems: set[str] = set()
        labels: set[str] = set()
        for item in wake_words or tuple(WakeWord):
            spec = _resolve_spec(item)
            path = spec.path.resolve()
            if not path.is_file():
                raise FileNotFoundError(f"Wake word model not found: {path}")
            canonical, stem, label = (
                str(path).casefold(),
                path.stem.casefold(),
                spec.label.casefold(),
            )
            if canonical in paths or stem in stems or label in labels:
                raise ValueError(
                    "Model paths, file stems and public labels must be unique."
                )
            paths.add(canonical)
            stems.add(stem)
            labels.add(label)
            self._paths.append(str(path))
            self._labels[path.stem] = spec.label
        self._operation_lock = threading.Lock()
        self._model = self._new_model()
        self._clear_stream(format)

    def _new_model(self) -> Backend:
        return create_model(self._paths, self._vad_threshold)

    def _clear_stream(self, format: AudioFormat | None) -> None:
        self._format = format
        self._normalizer: _StreamingNormalizer | None = None
        self._pending = np.empty(0, dtype=np.int16)
        self._samples_processed = 0
        self._last_trigger: int | None = None
        self._finished = False
        self._failed = False

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def cooldown(self) -> float:
        return self._cooldown

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._labels.values())

    def _acquire(self) -> None:
        if not self._operation_lock.acquire(blocking=False):
            raise RuntimeError(
                "Overlapping detector calls are not allowed; use one ordered stream."
            )

    def _ensure_open(self) -> None:
        if self._finished or self._failed:
            raise RuntimeError(
                "Stream is finished or failed; call reset() before processing."
            )

    def process_sync(
        self, frame: AudioFrame, *, format: AudioFormat | None = None
    ) -> Detection | None:
        """Return the first detection; use process_all_sync for bulk input."""
        detections = self.process_all_sync(frame, format=format)
        return detections[0] if detections else None

    def process_all_sync(
        self, frame: AudioFrame, *, format: AudioFormat | None = None
    ) -> tuple[Detection, ...]:
        """Return all detections from complete model blocks in audio order."""
        self._acquire()
        try:
            return self._process_frame(frame, format)
        finally:
            self._operation_lock.release()

    def _process_frame(
        self, frame: AudioFrame, format: AudioFormat | None
    ) -> tuple[Detection, ...]:
        self._ensure_open()
        effective = format if format is not None else self._format or MODEL_AUDIO_FORMAT
        if not isinstance(effective, AudioFormat):
            raise TypeError("format must be an AudioFormat.")
        if self._format is not None and self._format != effective:
            raise ValueError(
                "AudioFormat cannot change within a stream; call reset(format=...)."
            )
        normalizer = self._normalizer or _StreamingNormalizer(effective)
        normalized = normalizer.feed(frame)
        self._normalizer, self._format = normalizer, effective
        return self._consume(normalized)

    async def process(
        self, frame: AudioFrame, *, format: AudioFormat | None = None
    ) -> Detection | None:
        """Return the first detection, deferring cancellation of accepted work."""
        detections = await self.process_all(frame, format=format)
        return detections[0] if detections else None

    async def process_all(
        self, frame: AudioFrame, *, format: AudioFormat | None = None
    ) -> tuple[Detection, ...]:
        """Return every detection; reject overlaps before submitting worker work."""
        self._acquire()
        try:
            # Capture mutable input before handing it to another thread.
            snapshot = _snapshot(frame)
            return await self._run_reserved(
                lambda: self._process_frame(snapshot, format)
            )
        finally:
            self._operation_lock.release()

    async def _run_reserved(
        self, operation: Callable[[], tuple[Detection, ...]]
    ) -> tuple[Detection, ...]:
        # An executor Future is not an independently cancellable asyncio Task.
        # Shield it, drain it even on repeated caller cancellation, and deliver
        # its result before allowing any subsequent operation on this stream.
        worker = asyncio.get_running_loop().run_in_executor(None, operation)
        while True:
            try:
                return await asyncio.shield(worker)
            except asyncio.CancelledError:
                if worker.cancelled():
                    raise
                task = asyncio.current_task()
                if task is not None:
                    task.uncancel()
                if worker.done():
                    return worker.result()

    def finish_sync(self) -> tuple[Detection, ...]:
        """Flush the resampler, zero-pad the last model block, then close."""
        self._acquire()
        try:
            return self._finish_stream()
        finally:
            self._operation_lock.release()

    def _finish_stream(self) -> tuple[Detection, ...]:
        self._ensure_open()
        detections = list(
            self._consume(
                self._normalizer.finish()
                if self._normalizer is not None
                else np.empty(0, dtype=np.int16)
            )
        )
        if self._pending.size:
            valid = self._pending.size
            block = np.pad(self._pending, (0, _MODEL_BLOCK - valid))
            self._pending = np.empty(0, dtype=np.int16)
            detection = self._predict(block, valid)
            if detection is not None:
                detections.append(detection)
        self._finished = True
        return tuple(detections)

    async def finish(self) -> tuple[Detection, ...]:
        """Finish off the event loop, deferring cancellation of accepted work."""
        self._acquire()
        try:
            return await self._run_reserved(self._finish_stream)
        finally:
            self._operation_lock.release()

    def reset(self, *, format: AudioFormat | None = None) -> None:
        """Start a fresh stream including feature, VAD and cooldown state."""
        if format is not None and not isinstance(format, AudioFormat):
            raise TypeError("format must be an AudioFormat.")
        self._acquire()
        try:
            # Model.reset() only clears predictions in openWakeWord 0.4.0.
            # Recreate the backend before touching the existing stream state.
            model = self._new_model()
            self._model = model
            self._clear_stream(format if format is not None else self._initial_format)
        finally:
            self._operation_lock.release()

    def _consume(self, normalized: NDArray[np.int16]) -> tuple[Detection, ...]:
        samples = (
            np.concatenate((self._pending, normalized))
            if self._pending.size
            else normalized
        )
        complete = samples.size // _MODEL_BLOCK * _MODEL_BLOCK
        detections: list[Detection] = []
        for offset in range(0, complete, _MODEL_BLOCK):
            detection = self._predict(
                samples[offset : offset + _MODEL_BLOCK], _MODEL_BLOCK
            )
            if detection is not None:
                detections.append(detection)
        self._pending = samples[complete:].copy()
        return tuple(detections)

    def _predict(
        self, block: NDArray[np.int16], valid_samples: int
    ) -> Detection | None:
        try:
            scores = self._model.predict(block)
            unknown = scores.keys() - self._labels.keys()
            if unknown:
                raise ValueError(
                    f"Backend returned unsupported model labels: {unknown}."
                )
        except Exception:
            self._failed = True
            raise
        self._samples_processed += valid_samples
        candidates = [
            (key, float(score))
            for key, score in scores.items()
            if math.isfinite(score) and 0.0 < score <= 1.0 and score >= self._threshold
        ]
        if not candidates:
            return None
        if self._last_trigger is not None and (
            (self._samples_processed - self._last_trigger) / _SAMPLE_RATE
            < self._cooldown
        ):
            return None
        key, score = max(candidates, key=lambda item: item[1])
        self._last_trigger = self._samples_processed
        return Detection(self._labels[key], score, self._samples_processed)


def _number(name: str, value: float, *, maximum: float | None = None) -> float:
    try:
        valid = not isinstance(value, bool) and math.isfinite(value) and value >= 0
        valid = valid and (maximum is None or value <= maximum)
    except (TypeError, OverflowError):
        valid = False
    if not valid:
        bound = (
            f" between 0.0 and {maximum}"
            if maximum is not None
            else " and non-negative"
        )
        raise ValueError(f"{name} must be finite{bound}.")
    return float(value)


def _resolve_spec(spec: WakeWordSpec) -> WakeWord | CustomWakeWord:
    if isinstance(spec, WakeWord | CustomWakeWord):
        return spec
    if not isinstance(spec, str):
        raise TypeError(
            "Pass wake words as positional arguments; use *words for a non-empty list."
        )
    for builtin in WakeWord:
        if spec in {builtin.label, builtin.name.lower(), builtin.value}:
            return builtin
    raise ValueError(f"Unknown built-in wake word: {spec!r}")


def _snapshot(frame: AudioFrame) -> AudioFrame:
    if isinstance(frame, np.ndarray):
        return frame.copy()
    if isinstance(frame, memoryview) and not frame.c_contiguous:
        raise ValueError("PCM bytes must be contiguous.")
    if isinstance(frame, bytes | bytearray | memoryview):
        return bytes(frame)
    raise TypeError("frame must be PCM bytes, bytearray, memoryview or a NumPy array.")
