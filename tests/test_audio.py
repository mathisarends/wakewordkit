import pytest

from wakewordkit import MODEL_AUDIO_FORMAT, AudioFormat


def test_audio_format_exposes_byte_sizes():
    format = AudioFormat(24_000, 2, "pcm_f32le")

    assert format.bytes_per_sample == 4
    assert format.bytes_per_second == 192_000
    assert MODEL_AUDIO_FORMAT == AudioFormat(16_000, 1, "pcm_s16le")


@pytest.mark.parametrize(
    ("format", "message"),
    [
        (AudioFormat(1, 1, "pcm_s16le"), None),
        ((0, 1, "pcm_s16le"), "sample_rate"),
        ((16_000, 0, "pcm_s16le"), "channels"),
        ((16_000, 1, "mp3"), "sample_format"),
    ],
)
def test_audio_format_validation(format, message):
    if message is None:
        assert format.sample_rate == 1
        return
    with pytest.raises(ValueError, match=message):
        AudioFormat(*format)
