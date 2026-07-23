"""Listen on the default microphone for a single built-in wake word.

pip install 'wakewordkit[mic]'
python examples/listen_microphone.py
"""

from wakewordkit import WakeWord, WakeWordDetector


def main() -> None:
    detector = WakeWordDetector(WakeWord.ALEXA)
    print(f'Listening for "{detector.names[0]}"... (Ctrl+C to stop)')
    for detection in detector.listen():
        print(f"  detected {detection.name} ({detection.score:.2f})")


if __name__ == "__main__":
    main()
