from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import openwakeword

_RESOURCES = Path(openwakeword.__file__).parent / "resources" / "models"


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
        return _RESOURCES / f"{self.value}.onnx"


@dataclass(frozen=True, slots=True)
class CustomWakeWord:
    name: str
    model: str | Path

    @property
    def label(self) -> str:
        return self.name

    @property
    def path(self) -> Path:
        return Path(self.model)
