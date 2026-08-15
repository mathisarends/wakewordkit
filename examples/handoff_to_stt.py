"""Listen on the default microphone from an asyncio application.

`alisten()` reads in the default executor, so the event loop stays free. While
the body of the `async for` runs, no frames are read — handling a detection
pauses detection without pausing the capture.

pip install 'wakewordkit[mic]'
python examples/listen_async.py
"""

import asyncio

from wakewordkit import Detection, WakeWord, WakeWordDetector


async def handle(detection: Detection) -> None:
    print(f"  detected {detection.name} ({detection.score:.2f})")
    await asyncio.sleep(1.0)  # stand-in for recording, transcribing, ...


async def main() -> None:
    detector = WakeWordDetector(WakeWord.ALEXA)
    print(f'Listening for "{detector.names[0]}"... (Ctrl+C to stop)')
    async for detection in detector.alisten():
        await handle(detection)


if __name__ == "__main__":
    asyncio.run(main())
