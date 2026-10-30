from dataclasses import dataclass
from enum import Enum
from importlib.util import find_spec
from pathlib import Path


class WakeWord(Enum):
    ALEXA = "alexa_v0.1"
    HEY_JARVIS = "hey_jarvis_v0.1"
    HEY_MYCROFT = "hey_mycroft_v0.1"
    HEY_MARVIN = "hey_marvin_v0.1"

    @property
    def label(self) -> str:
        return self.name.lower()

    @property
    def path(self) -> Path:
        # Locate package resources without importing the inference stack.
        spec = find_spec("openwakeword")
        if spec is None or spec.origin is None:
            raise ImportError("Install wakewordkit with its openwakeword dependency.")
        return Path(spec.origin).parent / "resources" / "models" / f"{self.value}.onnx"


@dataclass(frozen=True, slots=True)
class CustomWakeWord:
    """A binary openWakeWord-compatible ONNX classifier, not a general ONNX model."""

    name: str
    model: str | Path

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Custom model name must be a non-empty string.")
        if self.name != self.name.strip():
            raise ValueError("Custom model name must not have surrounding whitespace.")
        if not isinstance(self.model, str | Path) or not str(self.model).strip():
            raise ValueError("Custom model must be a non-empty ONNX path.")
        if self.path.suffix.lower() != ".onnx":
            raise ValueError(
                "Custom model must be an openWakeWord-compatible .onnx file."
            )

    @property
    def label(self) -> str:
        return self.name

    @property
    def path(self) -> Path:
        return Path(self.model)
