"""Audio formats and the public PCM input contract."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

type SampleFormat = Literal["pcm_s16le", "pcm_f32le"]
type AudioFrame = (
    bytes | bytearray | memoryview | NDArray[np.int16] | NDArray[np.float32]
)


@dataclass(frozen=True, slots=True)
class AudioFormat:
    """PCM bytes are interleaved; arrays use (samples, channels), or mono (samples,)."""

    sample_rate: int
    channels: int
    sample_format: SampleFormat

    def __post_init__(self) -> None:
        for name in ("sample_rate", "channels"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if self.sample_format not in {"pcm_s16le", "pcm_f32le"}:
            raise ValueError("sample_format must be 'pcm_s16le' or 'pcm_f32le'.")

    @property
    def bytes_per_sample(self) -> int:
        return 2 if self.sample_format == "pcm_s16le" else 4

    @property
    def bytes_per_second(self) -> int:
        return self.sample_rate * self.channels * self.bytes_per_sample


MODEL_AUDIO_FORMAT = AudioFormat(16_000, 1, "pcm_s16le")
