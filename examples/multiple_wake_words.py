"""Wait once for any of several built-in wake words."""

import asyncio

from wakewordkit import WakeWord, WakeWordDetector


async def main() -> None:
    detector = WakeWordDetector(
        WakeWord.HEY_JARVIS,
        WakeWord.HEY_MYCROFT,
        WakeWord.ALEXA,
        threshold=0.5,
    )
    print(f"Listening for: {', '.join(detector.names)}")
    activation = await detector.listen()
    print(f"Detected {activation.name} ({activation.score:.2f})")

    async with activation.audio:
        async for chunk in activation.audio:
            # Send the continuation to the handler chosen by activation.name.
            print(f"\rReceived {len(chunk)} live audio bytes", end="", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
