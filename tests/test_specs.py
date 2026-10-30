import subprocess
import sys

import pytest

from wakewordkit import CustomWakeWord, WakeWord, WakeWordDetector


def test_builtin_assets_and_custom_labels():
    for word in WakeWord:
        assert word.path.is_file()
        assert word.path.suffix == ".onnx"
    custom = CustomWakeWord("hey_computer", "models/hey_computer.onnx")
    assert custom.label == "hey_computer"
    assert custom.path.name == "hey_computer.onnx"


@pytest.mark.parametrize(
    "name,model",
    [("", "x.onnx"), ("  ", "x.onnx"), (" x", "x.onnx"), ("x", ""), ("x", "x.tflite")],
)
def test_custom_spec_validation(name, model):
    with pytest.raises(ValueError):
        CustomWakeWord(name, model)


@pytest.mark.parametrize("selection", [([],), ([WakeWord.ALEXA],), ("unknown",)])
def test_invalid_selection_is_explicit(fake_model, selection):
    with pytest.raises((ValueError, TypeError)):
        WakeWordDetector(*selection)


def test_duplicate_paths_stems_and_labels_are_rejected(fake_model, tmp_path):
    first, second = tmp_path / "a", tmp_path / "b"
    first.mkdir()
    second.mkdir()
    one, two = first / "same.onnx", second / "same.onnx"
    one.touch()
    two.touch()
    other = second / "other.onnx"
    other.touch()
    for models in [
        (CustomWakeWord("one", one), CustomWakeWord("two", one)),
        (CustomWakeWord("one", one), CustomWakeWord("two", two)),
        (CustomWakeWord("same", one), CustomWakeWord("same", other)),
    ]:
        with pytest.raises(ValueError, match="unique"):
            WakeWordDetector(*models)


def test_importing_public_types_does_not_import_backend():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from wakewordkit import AudioFormat, CustomWakeWord, Detection, WakeWord
assert 'openwakeword' not in sys.modules
assert 'scipy' not in sys.modules
assert 'sklearn' not in sys.modules
assert 'onnxruntime' not in sys.modules
assert 'soxr' not in sys.modules
assert WakeWord.ALEXA.path.is_file()
assert 'openwakeword' not in sys.modules
""",
        ],
        check=True,
    )
