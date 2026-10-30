import numpy as np
import pytest
import soxr

from wakewordkit import MODEL_AUDIO_FORMAT, AudioFormat
from wakewordkit._audio import _StreamingNormalizer


def normalize(samples, rate, cuts=()):
    normalizer = _StreamingNormalizer(AudioFormat(rate, 1, "pcm_s16le"))
    parts = [normalizer.feed(part) for part in np.split(samples, cuts)]
    return np.concatenate([*parts, normalizer.finish()])


@pytest.mark.parametrize("rate", [8000, 16000, 22050, 24000, 44100, 48000])
def test_resampling_is_independent_of_chunk_boundaries(rate):
    samples = np.random.default_rng(42).integers(-12000, 12000, rate + 3, np.int16)
    whole = normalize(samples, rate)
    chunked = normalize(samples, rate, [1, 17, 131, 4099, rate - 1])
    np.testing.assert_array_equal(chunked, whole)
    assert len(whole) == round(len(samples) * 16000 / rate)
    if rate != 16000:
        reference = soxr.resample(samples.astype(np.float64) / 32768, rate, 16000)
        reference = np.clip(np.rint(reference * 32768), -32768, 32767).astype(np.int16)
        np.testing.assert_array_equal(whole, reference)


def test_downsampling_rejects_aliases_and_preserves_speech_band():
    t = np.arange(48000) / 48000

    def rms(frequency):
        source = np.rint(20000 * np.sin(2 * np.pi * frequency * t)).astype(np.int16)
        output = normalize(source, 48000)[1000:-1000].astype(float)
        return np.sqrt(np.mean(output**2))

    assert rms(12000) < rms(1000) * 0.001
    assert rms(1000) == pytest.approx(20000 / np.sqrt(2), rel=0.01)


def test_pcm16_fast_path_is_exact_and_shares_memory():
    samples = np.array([-32768, -1, 0, 1, 32767], dtype=np.int16)
    result = _StreamingNormalizer(MODEL_AUDIO_FORMAT).feed(samples)
    np.testing.assert_array_equal(result, samples)
    assert np.shares_memory(result, samples)


@pytest.mark.parametrize("channels", [1, 2])
def test_float_and_integer_pcm_agree(channels):
    ints = np.array([[-32768], [0], [16384]], dtype=np.int16).repeat(channels, axis=1)
    floats = (ints.astype(np.float32) / 32768).astype(np.float32)

    def feed(frame, kind):
        return _StreamingNormalizer(AudioFormat(16000, channels, kind)).feed(frame)

    np.testing.assert_array_equal(feed(ints, "pcm_s16le"), feed(floats, "pcm_f32le"))
    np.testing.assert_array_equal(feed(ints.tobytes(), "pcm_s16le"), ints[:, 0])


@pytest.mark.parametrize(
    "frame,format",
    [
        (b"\x00", MODEL_AUDIO_FORMAT),
        (np.zeros(3, np.int16).tobytes(), AudioFormat(16000, 2, "pcm_s16le")),
        (np.zeros((2, 10), np.int16), AudioFormat(16000, 2, "pcm_s16le")),
        (np.zeros(4, np.int16), AudioFormat(16000, 2, "pcm_s16le")),
        (np.zeros(4, np.int32), MODEL_AUDIO_FORMAT),
        (np.zeros(4, np.float64), AudioFormat(16000, 1, "pcm_f32le")),
        (np.array([np.nan], np.float32), AudioFormat(16000, 1, "pcm_f32le")),
        (np.array([np.inf], np.float32), AudioFormat(16000, 1, "pcm_f32le")),
        (np.array([1.01], np.float32), AudioFormat(16000, 1, "pcm_f32le")),
    ],
)
def test_invalid_pcm_is_rejected(frame, format):
    with pytest.raises(ValueError):
        _StreamingNormalizer(format).feed(frame)


@pytest.mark.parametrize("kind", [bytearray, memoryview])
def test_mutable_byte_buffers_are_supported(kind):
    source = np.arange(20, dtype=np.int16)
    output = _StreamingNormalizer(MODEL_AUDIO_FORMAT).feed(kind(source.tobytes()))
    np.testing.assert_array_equal(output, source)
