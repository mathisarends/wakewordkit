"""Detect one wake word from an application-owned microphone stream.

Install the optional example dependencies first:

    pip install 'wakewordkit[examples]'
"""

import asyncio

from _microphone import MicrophoneFrames

from wakewordkit import WakeWordDetector


async def main() -> None:
    detector = WakeWordDetector("hey_jarvis")

    async with MicrophoneFrames() as microphone:
        print('Say "Hey Jarvis"...')
        while True:
            frame = await microphone.read()
            detection = await asyncio.to_thread(
                detector.process,
                frame,
                format=microphone.format,
            )
            if detection is not None:
                print(f"Detected {detection.name} ({detection.score:.2f})")
                break

    print(f"Application capture dropped {microphone.dropped_frames} frame(s).")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
