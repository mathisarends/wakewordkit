"""Use a custom openWakeWord model and retain the post-detection audio."""

import asyncio
import sys

from wakewordkit import CustomWakeWord, WakeWord, WakeWordDetector


async def main(model_path: str) -> None:
    detector = WakeWordDetector(
        WakeWord.ALEXA,
        CustomWakeWord("hey_computer", model_path),
        threshold=0.6,
    )
    print(f"Listening for: {', '.join(detector.names)}")
    activation = await detector.listen()
    print(f"Detected {activation.name} ({activation.score:.2f})")

    async with activation.audio:
        async for chunk in activation.audio:
            print(f"\rHandoff chunk: {len(chunk)} bytes", end="", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python examples/custom_wake_word.py <model.onnx>")
    asyncio.run(main(sys.argv[1]))
