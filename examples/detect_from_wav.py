"""Run detection over a caller-owned WAV stream."""

import asyncio
import sys
import wave

from wakewordkit import AudioFormat, WakeWordDetector

_FRAME_SIZE = 1_280


async def main(path: str) -> None:
    detector = WakeWordDetector()
    with wave.open(path, "rb") as wav:
        format = AudioFormat(wav.getframerate(), wav.getnchannels(), "pcm_s16le")

        async def frames():
            while frame := wav.readframes(_FRAME_SIZE):
                yield frame
                await asyncio.sleep(0)

        found = False
        async for detection in detector.stream(frames(), format=format):
            found = True
            print(f"Detected {detection.name} ({detection.score:.2f})")
        if not found:
            print("No wake word detected.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python examples/detect_from_wav.py <recording.wav>")
    asyncio.run(main(sys.argv[1]))
