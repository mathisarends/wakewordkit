from wakewordkit import CustomWakeWord, WakeWord
from wakewordkit.detector import _flatten


def test_builtin_label_and_path():
    assert WakeWord.HEY_JARVIS.label == "hey_jarvis"
    assert WakeWord.HEY_JARVIS.path.exists()
    assert WakeWord.HEY_JARVIS.path.suffix == ".onnx"


def test_every_builtin_model_file_ships():
    for wake_word in WakeWord:
        assert wake_word.path.exists(), wake_word


def test_custom_label_and_path():
    custom = CustomWakeWord("hey_computer", "models/hey_computer.onnx")
    assert custom.label == "hey_computer"
    assert custom.path.name == "hey_computer.onnx"


def test_flatten_passes_through_single_specs():
    assert _flatten([WakeWord.ALEXA, WakeWord.HEY_MYCROFT]) == [
        WakeWord.ALEXA,
        WakeWord.HEY_MYCROFT,
    ]


def test_flatten_expands_iterables():
    assert _flatten([[WakeWord.ALEXA, WakeWord.HEY_JARVIS]]) == [
        WakeWord.ALEXA,
        WakeWord.HEY_JARVIS,
    ]


def test_flatten_mixes_specs_and_iterables():
    custom = CustomWakeWord("x", "x.onnx")
    assert _flatten([WakeWord.ALEXA, [custom]]) == [WakeWord.ALEXA, custom]


def test_flatten_empty():
    assert _flatten([]) == []
