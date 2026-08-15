# wakewordkit

Lightweight wake-word detection for application-owned PCM audio, powered by
[openWakeWord](https://github.com/dscripka/openWakeWord).

WakewordKit deliberately does not open microphones, buffer application audio,
or hand streams to speech-to-text services. The application owns capture and
routing; WakewordKit only converts frames to the model format and reports
detections.

Requires Python 3.12+.

## Install

```bash
pip install wakewordkit
```

## Process individual frames

Use `process()` when your application already has a capture loop:

```python
from wakewordkit import AudioFormat, WakeWordDetector

detector = WakeWordDetector("hey_jarvis")
audio_format = AudioFormat(
    sample_rate=48_000,
    channels=2,
    sample_format="pcm_s16le",
)

while chunk := microphone.read():
    detection = detector.process(chunk, format=audio_format)
    if detection is not None:
        print(detection.name, detection.score)
```

`process()` never takes ownership of the source. Audio capture, buffering,
pre-roll, STT handoff, and device shutdown remain the application's
responsibility.

## Asynchronous applications

The application also owns iteration and scheduling. Move synchronous model work
off the event loop when appropriate:

```python
while True:
    frame = await microphone.read()
    detection = await asyncio.to_thread(
        detector.process,
        frame,
        format=microphone.format,
    )
    if detection is not None:
        await handle(detection)
```

Input may be interleaved 16-bit PCM bytes, 32-bit float PCM bytes, or NumPy
arrays. `AudioFormat` describes its sample rate, channel count, and encoding.
Downmixing and conversion to the model's 16 kHz mono format are internal and
stateful across frame boundaries.

## Application-owned STT handoff

A voice application normally keeps one capture loop and routes its frames to
the active consumer:

```text
microphone -> application buffer -> wakewordkit
                               \-> VAD / recorder / STT
```

Keep any pre-roll ring buffer beside the capture loop. When WakewordKit reports
a detection, the application can give that buffered prefix and subsequent live
frames to its recorder or STT client without transferring device ownership.
This also keeps follow-up turns and barge-in on the same audio lifecycle.

## Models

Built-in and custom models can be combined:

```python
from wakewordkit import CustomWakeWord, WakeWord, WakeWordDetector

WakeWordDetector("hey_jarvis")
WakeWordDetector(WakeWord.ALEXA, WakeWord.HEY_MYCROFT)
WakeWordDetector(CustomWakeWord("hey_computer", "models/hey_computer.onnx"))
WakeWordDetector()  # all bundled models
```

Bundled names are `alexa`, `hey_jarvis`, `hey_mycroft`, and `hey_marvin`.

## Development

```bash
uv sync --dev
uv run pytest
```

See [`examples/detect_from_wav.py`](examples/detect_from_wav.py) for a complete
caller-owned source example.

### Real microphone and STT/TTS examples

Hardware and provider SDKs are deliberately excluded from WakewordKit's core
dependencies. Install the separate examples extra to run them:

```bash
uv sync --extra examples
uv run python examples/external_microphone.py
uv run python examples/stt_tts_assistant.py
```

- [`external_microphone.py`](examples/external_microphone.py) reads a
  `sounddevice` stream owned by the example and passes each frame to
  `detector.process()`.
- [`stt_tts_assistant.py`](examples/stt_tts_assistant.py) keeps one microphone
  source and its pre-roll in the application, calls `detector.process()`,
  records through silence, transcribes the resulting WAV, and plays a generated
  spoken response. It uses the request-based
  [OpenAI audio APIs](https://developers.openai.com/api/docs/guides/audio), requires
  `OPENAI_API_KEY` (the example also loads it from `.env`), and is intentionally
  half-duplex.

The energy threshold in the compact STT/TTS example is suitable for trying the
ownership pattern, not a production VAD. A real assistant should replace it
with its own VAD/turn detector and add echo cancellation or explicit barge-in.

## License

MIT — see [LICENSE](LICENSE).
