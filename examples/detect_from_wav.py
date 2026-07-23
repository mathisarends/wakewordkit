"""Run detection over a WAV file by implementing a custom AudioInput.

No microphone required — great for testing. The WAV must be 16 kHz mono PCM.

    python examples/detect_from_wav.py path/to/recording.wav
"""

import sys
import wave

from wakewordkit import AudioInput, WakeWordDetector

_FRAME_SIZE = 1_280


class WavInput(AudioInput):
    def __init__(self, path: str) -> None:
        self._wav = wave.open(path, "rb")
        if (self._wav.getframerate(), self._wav.getnchannels()) != (16_000, 1):
            raise ValueError("WAV must be 16 kHz mono PCM.")

    def read(self) -> bytes:
        frame = self._wav.readframes(_FRAME_SIZE)
        if not frame:
            raise StopIteration
        return frame

    def __exit__(self, *exc: object) -> None:
        self._wav.close()


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: python examples/detect_from_wav.py <recording.wav>")

    detector = WakeWordDetector()  # all built-in wake words
    with WavInput(sys.argv[1]) as source:
        detections = list(detector.stream(source))

    if not detections:
        print("No wake word detected.")
    for detection in detections:
        print(f"detected {detection.name} ({detection.score:.2f})")


if __name__ == "__main__":
    main()
