"""Listen for several built-in wake words at once and react per detection.

python examples/multiple_wake_words.py
"""

from wakewordkit import WakeWord, WakeWordDetector


def main() -> None:
    detector = WakeWordDetector(
        WakeWord.HEY_JARVIS,
        WakeWord.HEY_MYCROFT,
        WakeWord.ALEXA,
        threshold=0.5,
    )
    print(f"Listening for: {', '.join(detector.names)} (Ctrl+C to stop)")
    for detection in detector.listen():
        match detection.name:
            case "hey_jarvis":
                print("  Jarvis here.")
            case "hey_mycroft":
                print("  Mycroft here.")
            case name:
                print(f"  {name} ({detection.score:.2f})")


if __name__ == "__main__":
    main()
