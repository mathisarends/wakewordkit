# wakewordkit

Async-first wake-word detection with lossless microphone handoff, powered by
[openWakeWord](https://github.com/dscripka/openWakeWord). WakewordKit opens the
microphone once, detects a wake word, and hands the very same audio stream to
your speech-to-text client with pre-roll and startup buffering included.

Requires Python 3.12+.

## Install

```bash
pip install wakewordkit          # externally managed audio
pip install 'wakewordkit[mic]'   # default microphone support
```

## Microphone to STT

```python
from wakewordkit import WakeWordDetector

detector = WakeWordDetector("hey_jarvis")
activation = await detector.listen()

print(activation.name, activation.score)

async with activation.audio:
    async for chunk in activation.audio:
        await stt.send_audio(chunk)
```

`listen()` is a one-shot async operation. It opens the input device exactly
once and returns one `Activation`. Its audio stream begins with the configured
pre-roll and the detection frame, then continues with live audio without a gap
or duplicate. Audio captured while the STT connection starts is retained.
Leaving the async context closes the continuation and releases the microphone.

Capture can be configured at the call site:

```python
activation = await detector.listen(
    device="default",
    sample_rate=24_000,
    channels=1,
    sample_format="pcm_s16le",
    preroll=0.2,
    max_buffer=10.0,
)
```

The handoff bytes remain in the capture format, exposed as
`activation.audio.format`. Model conversion to 16 kHz mono PCM is internal and
stateful across chunk boundaries.

## Buffering and overflow

`max_buffer` bounds continuation memory in seconds. A slow consumer never
blocks capture or detection. If the buffer fills, WakewordKit drops the oldest
complete chunks. The policy and loss are observable:

```python
print(activation.audio.overflow_policy)  # "drop_oldest"
print(activation.audio.stats.dropped_chunks)
print(activation.audio.stats.dropped_bytes)
```

The effective minimum capacity is the requested pre-roll plus one capture
frame, because those bytes form the mandatory lossless handoff prefix.

`preroll=0` removes only audio before detection. The detection frame and every
subsequent frame are still handed off.

## Externally managed async audio

For a source your application owns, pass an async iterable and its explicit
format. `stream()` yields detections but does not close or continue the source.

```python
from wakewordkit import AudioFormat

audio_format = AudioFormat(
    sample_rate=48_000,
    channels=2,
    sample_format="pcm_s16le",
)

async for detection in detector.stream(frames, format=audio_format):
    await handle(detection)
```

For individual frames, `process()` is the one immediate, non-owning operation:

```python
detection = detector.process(frame, format=audio_format)
if detection is not None:
    handle(detection)
```

There are deliberately no parallel `alisten`/`astream` names and no synchronous
capture API. `listen()` and `stream()` are async; `process()` handles the
single-frame use case directly.

## Models

Built-ins may be selected by short string or enum:

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

Runnable examples are in [`examples`](examples), including a visible
microphone-to-STT handoff and detection from an externally managed WAV stream.

## License

MIT — see [LICENSE](LICENSE).
