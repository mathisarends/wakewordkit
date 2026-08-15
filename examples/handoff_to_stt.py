"""Configured capture and a minimal asynchronous STT handoff."""

import asyncio

from wakewordkit import WakeWordDetector


class ExampleSttClient:
    async def send_audio(self, chunk: bytes) -> None:
        await asyncio.sleep(0)  # replace with the real network send


async def main() -> None:
    detector = WakeWordDetector("hey_jarvis")
    stt = ExampleSttClient()

    activation = await detector.listen(
        device="default",
        sample_rate=24_000,
        channels=1,
        sample_format="pcm_s16le",
        preroll=0.2,
    )
    print(f"Detected {activation.name}; starting STT...")
    await asyncio.sleep(1)  # startup is safe: incoming audio remains buffered

    async with activation.audio:
        async for chunk in activation.audio:
            await stt.send_audio(chunk)


if __name__ == "__main__":
    asyncio.run(main())
