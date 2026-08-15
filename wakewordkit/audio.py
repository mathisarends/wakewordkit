from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AudioFormat:
    """The contract of an interleaved PCM frame source."""

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
