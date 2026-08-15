"""Detect once, then consume the same microphone stream without audio loss.

pip install 'wakewordkit[mic]'
python examples/listen_microphone.py
"""

import asyncio

from wakewordkit import WakeWordDetector


async def main() -> None:
    detector = WakeWordDetector("hey_jarvis")
    print('Say "Hey Jarvis"...')

    activation = await detector.listen()
    print(f"Detected {activation.name} ({activation.score:.2f})")
    print(f"Handoff format: {activation.audio.format}")

    # Replace this loop body with await stt.send_audio(chunk). The stream starts
    # with pre-roll + the detection frame, then continues live from the same mic.
    byte_count = 0
    async with activation.audio:
        async for chunk in activation.audio:
            byte_count += len(chunk)
            print(f"\rForwarded {byte_count:,} bytes to STT", end="", flush=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped; microphone closed.")
