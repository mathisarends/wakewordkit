"""Validated PCM decoding and stateful, anti-aliased sample-rate conversion."""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from .audio import MODEL_AUDIO_FORMAT, AudioFormat, AudioFrame


def _decode_frame(frame: AudioFrame, format: AudioFormat) -> NDArray[Any]:
    dtype = np.dtype("<i2" if format.sample_format == "pcm_s16le" else "<f4")
    if isinstance(frame, bytes | bytearray | memoryview):
        view = memoryview(frame)
        alignment = format.channels * dtype.itemsize
        if not view.c_contiguous or view.nbytes % alignment:
            raise ValueError(
                "PCM bytes must contain complete, contiguous channel frames."
            )
        array = np.frombuffer(view, dtype=dtype).reshape(-1, format.channels)
    elif isinstance(frame, np.ndarray):
        if frame.dtype != dtype:
            raise ValueError(f"Array dtype must be {dtype} for {format.sample_format}.")
        if frame.ndim == 1 and format.channels == 1:
            array = frame.reshape(-1, 1)
        elif frame.ndim == 2 and frame.shape[1] == format.channels:
            array = frame
        else:
            raise ValueError(
                "Array shape must be (samples, channels), or (samples,) mono."
            )
    else:
        raise TypeError(
            "frame must be PCM bytes, bytearray, memoryview or a NumPy array."
        )
    if format.sample_format == "pcm_f32le" and (
        not np.isfinite(array).all() or np.any(np.abs(array) > 1.0)
    ):
        raise ValueError("Float PCM samples must be finite and within [-1.0, 1.0].")
    return array


def _pcm16(samples: NDArray[Any]) -> NDArray[np.int16]:
    return np.clip(np.rint(samples * 32768.0), -32768, 32767).astype(np.int16)


class _StreamingNormalizer:
    """SoXR retains filter history, phase and delayed output across feed calls."""

    def __init__(self, format: AudioFormat) -> None:
        self.format = format
        self._resampler: Any = None
        if format.sample_rate != MODEL_AUDIO_FORMAT.sample_rate:
            import soxr

            self._resampler = soxr.ResampleStream(
                format.sample_rate, 16_000, 1, dtype="float64", quality="HQ"
            )

    def feed(self, frame: AudioFrame) -> NDArray[np.int16]:
        array = _decode_frame(frame, self.format)
        if array.size == 0:
            return np.empty(0, dtype=np.int16)
        if self.format == MODEL_AUDIO_FORMAT:
            # Preserve PCM16 exactly and avoid float conversion for the common case.
            return array[:, 0]
        mono = array.astype(np.float64).mean(axis=1)
        if self.format.sample_format == "pcm_s16le":
            mono /= 32768.0
        if self._resampler is not None:
            mono = self._resampler.resample_chunk(mono)
        return _pcm16(mono)

    def finish(self) -> NDArray[np.int16]:
        if self._resampler is None:
            return np.empty(0, dtype=np.int16)
        return _pcm16(
            self._resampler.resample_chunk(np.empty(0, dtype=np.float64), last=True)
        )
