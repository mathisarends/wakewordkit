# wakewordkit

Lightweight, developer-friendly wake word detection built on
[openWakeWord](https://github.com/dscripka/openWakeWord).

Ships with every default openWakeWord wake word and lets you plug in your own
trained models through the same tiny API. Requires Python 3.12+.

## Install

```bash
pip install wakewordkit          # core
pip install 'wakewordkit[mic]'   # + microphone input
```

## Quick start

```python
from wakewordkit import WakeWordDetector, WakeWord

detector = WakeWordDetector(WakeWord.ALEXA)

for detection in detector.listen():
    print(f"{detection.name} ({detection.score:.2f})")
```

`listen()` opens the default microphone (requires the `mic` extra) and yields a
`Detection` every time a wake word is heard.

## Built-in wake words

```python
WakeWord.ALEXA
WakeWord.HEY_JARVIS
WakeWord.HEY_MYCROFT
WakeWord.HEY_MARVIN
```

Listen for several at once, or leave it empty to listen for all of them:

```python
WakeWordDetector(WakeWord.HEY_JARVIS, WakeWord.HEY_MYCROFT)
WakeWordDetector()  # all built-in wake words
```

## Custom wake words

Point `CustomWakeWord` at any openWakeWord-compatible `.onnx` model:

```python
from wakewordkit import WakeWordDetector, WakeWord, CustomWakeWord

detector = WakeWordDetector(
    WakeWord.ALEXA,
    CustomWakeWord("hey_computer", "models/hey_computer.onnx"),
    threshold=0.6,
)

for detection in detector.listen():
    print(detection.name)  # "alexa" or "hey_computer"
```

## Bring your own audio

Skip the built-in microphone and feed raw 16 kHz mono PCM yourself. Frames may
be `bytes` (int16) or a NumPy array, ideally 1280 samples (80 ms) each.

```python
detector = WakeWordDetector(WakeWord.HEY_JARVIS)

# Frame at a time
if detection := detector.process(frame):
    handle(detection)

# Or stream from any iterable of frames
for detection in detector.stream(my_audio_source):
    handle(detection)
```

## asyncio

Every method has an async twin: `alisten()` for a blocking `AudioInput` (reads
run in the default executor) and `astream()` for your own async frames.

```python
detector = WakeWordDetector(WakeWord.HEY_MYCROFT)

async for detection in detector.alisten():
    await handle(detection)
```

No frames are read while the body of the `async for` runs, so handling a
detection pauses detection without pausing the underlying capture — useful when
the microphone is shared with a recorder.

Already have a microphone object? `CallableInput` adapts it without subclassing:

```python
from wakewordkit import CallableInput

source = CallableInput(my_mic.read, 1280)  # calls my_mic.read(1280)

async for detection in detector.alisten(source):
    await handle(detection)
```

`CallableInput` never closes the underlying source unless you pass
`close=my_mic.close`, so sharing a stream is safe.

## Bring your own audio source

`Microphone` is only one implementation of the `AudioInput` interface. Subclass
it to plug in any source — a socket, a file, another capture library — and pass
it to `listen()`:

```python
from wakewordkit import AudioInput, WakeWordDetector, WakeWord


class FileInput(AudioInput):
    def __init__(self, path):
        self._reader = open_pcm(path)

    def read(self):
        frame = self._reader.read(1280)
        if not frame:
            raise StopIteration
        return frame


detector = WakeWordDetector(WakeWord.HEY_MYCROFT)
for detection in detector.listen(FileInput("recording.pcm")):
    handle(detection)
```

Override `__enter__` / `__exit__` for setup and teardown; `read()` is the only
required method.

## API

| Object | Purpose |
| --- | --- |
| `WakeWord` | Enum of built-in wake words. |
| `CustomWakeWord(name, model)` | Wraps a custom `.onnx` model under a chosen name. |
| `WakeWordDetector(*wake_words, threshold=0.5, vad_threshold=0.0, cooldown=1.5)` | The detector. |
| `WakeWordDetector.listen(source=None)` | Iterate detections from an `AudioInput` (defaults to `Microphone`). |
| `WakeWordDetector.stream(source)` | Iterate detections from your own frames. |
| `WakeWordDetector.alisten(source=None)` | `listen()` for asyncio; reads in the default executor. |
| `WakeWordDetector.astream(source)` | `stream()` for asyncio; takes an async iterable of frames. |
| `WakeWordDetector.process(frame)` | Score a single frame, returns `Detection` or `None`. |
| `WakeWordDetector.reset()` | Clear detection state. |
| `Detection(name, score)` | A single detection result. |
| `AudioInput` | Abstract audio source; implement `read()`. |
| `CallableInput(read, frame_size=None, *, close=None)` | Wrap an existing read function as an `AudioInput`. |
| `Microphone(*, samplerate=16000, frame_size=1280, device=None)` | Context-managed 16 kHz mic (`AudioInput`). |

`vad_threshold` (0 disables) turns on openWakeWord's Silero voice-activity
filter to suppress non-speech false triggers. `cooldown` is the minimum number
of seconds between two detections, so a single utterance triggers only once.

## Development

```bash
uv sync --dev
uv run pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full workflow.

## License

MIT — see [LICENSE](LICENSE).
