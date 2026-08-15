"""Run detection over caller-owned WAV frames."""

import asyncio
import sys
import wave

from wakewordkit import AudioFormat, WakeWordDetector

_FRAME_SIZE = 1_280


async def main(path: str) -> None:
    detector = WakeWordDetector()
    with wave.open(path, "rb") as wav:
        if wav.getsampwidth() != 2 or wav.getcomptype() != "NONE":
            raise ValueError("WAV must contain uncompressed 16-bit PCM audio.")
        format = AudioFormat(wav.getframerate(), wav.getnchannels(), "pcm_s16le")

        found = False
        while frame := wav.readframes(_FRAME_SIZE):
            detection = await detector.process(frame, format=format)
            if detection is not None:
                found = True
                print(f"Detected {detection.name} ({detection.score:.2f})")
        if not found:
            print("No wake word detected.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python examples/detect_from_wav.py <recording.wav>")
    asyncio.run(main(sys.argv[1]))
