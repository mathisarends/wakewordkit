"""Plug in a custom-trained openWakeWord model alongside a built-in one.

Pass the path to your own .onnx model:

    python examples/custom_wake_word.py path/to/hey_computer.onnx
"""

import sys

from wakewordkit import CustomWakeWord, WakeWord, WakeWordDetector


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: python examples/custom_wake_word.py <model.onnx>")

    detector = WakeWordDetector(
        WakeWord.ALEXA,
        CustomWakeWord("hey_computer", sys.argv[1]),
        threshold=0.6,
    )
    print(f"Listening for: {', '.join(detector.names)} (Ctrl+C to stop)")
    for detection in detector.listen():
        print(f"  detected {detection.name} ({detection.score:.2f})")


if __name__ == "__main__":
    main()
