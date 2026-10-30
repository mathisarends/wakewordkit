"""Exercise the pinned backend, never the FakeModel fixture."""

import numpy as np
import pytest

from wakewordkit import AudioFormat, CustomWakeWord, WakeWord, WakeWordDetector

pytestmark = pytest.mark.integration


def test_real_inference_accepts_20ms_frames_and_retains_remainders():
    detector = WakeWordDetector("alexa")
    for _ in range(15):
        assert detector.process_sync(np.zeros(320, np.int16)) is None
    assert len(detector._model.preprocessor.raw_data_buffer) == 3840
    assert detector._pending.size == 960
    detector.finish_sync()
    assert len(detector._model.preprocessor.raw_data_buffer) == 5120


@pytest.mark.parametrize("rate", [8000, 16000, 48000])
def test_real_backend_state_is_independent_of_input_chunking(rate):
    source = np.random.default_rng(5).integers(-10000, 10000, rate * 2 + 3, np.int16)
    format = AudioFormat(rate, 1, "pcm_s16le")
    whole, split = WakeWordDetector("alexa"), WakeWordDetector("alexa")
    expected = whole.process_all_sync(source, format=format) + whole.finish_sync()
    actual = []
    for part in np.split(source, [1, 321, 400, 900, 4321, rate + 1]):
        actual.extend(split.process_all_sync(part, format=format))
    actual.extend(split.finish_sync())
    assert tuple(actual) == expected
    for name in ("melspectrogram_buffer", "feature_buffer"):
        np.testing.assert_array_equal(
            getattr(split._model.preprocessor, name),
            getattr(whole._model.preprocessor, name),
        )
    assert split._model.prediction_buffer == whole._model.prediction_buffer


def test_real_reset_matches_fresh_instance_including_vad():
    detector = WakeWordDetector("alexa", vad_threshold=0.5)
    source = np.random.default_rng(8).integers(-20000, 20000, 16007, np.int16)
    detector.process_all_sync(source)
    detector.reset()
    fresh = WakeWordDetector("alexa", vad_threshold=0.5)
    for name in ("raw_data_buffer", "melspectrogram_buffer", "feature_buffer"):
        np.testing.assert_array_equal(
            getattr(detector._model.preprocessor, name),
            getattr(fresh._model.preprocessor, name),
        )
    for name in ("_h", "_c", "prediction_buffer"):
        np.testing.assert_array_equal(
            getattr(detector._model.vad, name), getattr(fresh._model.vad, name)
        )
    assert detector.process_all_sync(source) == fresh.process_all_sync(source)
    np.testing.assert_array_equal(
        detector._model.preprocessor.feature_buffer,
        fresh._model.preprocessor.feature_buffer,
    )


def test_custom_binary_model_uses_public_label():
    model = CustomWakeWord("my_alexa", WakeWord.ALEXA.path)
    detector = WakeWordDetector(model)
    assert detector.names == ("my_alexa",)
    assert detector.process_sync(np.zeros(1280, np.int16)) is None


def test_corrupt_onnx_has_clear_load_error(tmp_path):
    path = tmp_path / "invalid.onnx"
    path.write_bytes(b"not an ONNX graph")
    with pytest.raises(ValueError, match="binary openWakeWord"):
        WakeWordDetector(CustomWakeWord("invalid", path))


def test_incompatible_feature_model_is_rejected():
    path = WakeWord.ALEXA.path.parent / "embedding_model.onnx"
    with pytest.raises(ValueError, match="binary openWakeWord"):
        WakeWordDetector(CustomWakeWord("features", path))


def test_all_bundled_models_load_and_run_together():
    detector = WakeWordDetector()
    assert len(detector.names) == 4
    assert detector.process_all_sync(np.zeros(1280 * 8, np.int16)) == ()
    assert detector.finish_sync() == ()
