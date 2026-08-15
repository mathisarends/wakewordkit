from .audio import MODEL_AUDIO_FORMAT, AudioFormat
from .detector import Detection, WakeWordDetector, WakeWordSpec
from .models import CustomWakeWord, WakeWord

__version__ = "0.4.0"

__all__ = [
    "AudioFormat",
    "CustomWakeWord",
    "Detection",
    "MODEL_AUDIO_FORMAT",
    "WakeWord",
    "WakeWordDetector",
    "WakeWordSpec",
]
