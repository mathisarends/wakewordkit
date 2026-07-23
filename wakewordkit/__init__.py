from .audio import AudioInput, Microphone
from .detector import Detection, WakeWordDetector, WakeWordSpec
from .models import CustomWakeWord, WakeWord

__version__ = "0.1.0"

__all__ = [
    "AudioInput",
    "CustomWakeWord",
    "Detection",
    "Microphone",
    "WakeWord",
    "WakeWordDetector",
    "WakeWordSpec",
]
