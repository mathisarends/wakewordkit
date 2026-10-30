"""Process caller-owned WAV audio offline, including the final short frame."""

import sys
import wave

from wakewordkit import AudioFormat, Detection, WakeWordDetector

_FRAME_SIZE = 1_280


def report(detection: Detection) -> None:
    print(
        f"Detected {detection.name} ({detection.score:.2f}) "
        f"at {detection.timestamp:.3f}s"
    )


def main(path: str) -> None:
    detector = WakeWordDetector()
    with wave.open(path, "rb") as wav:
        if wav.getsampwidth() != 2 or wav.getcomptype() != "NONE":
            raise ValueError("WAV must contain uncompressed 16-bit PCM audio.")
        format = AudioFormat(wav.getframerate(), wav.getnchannels(), "pcm_s16le")
        found = False
        while frame := wav.readframes(_FRAME_SIZE):
            for detection in detector.process_all_sync(frame, format=format):
                found = True
                report(detection)
        for detection in detector.finish_sync():
            found = True
            report(detection)
        if not found:
            print("No wake word detected.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python examples/detect_from_wav.py <recording.wav>")
    main(sys.argv[1])
