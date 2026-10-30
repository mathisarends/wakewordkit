# wakewordkit

Wake-word detection for application-owned PCM audio, powered by
[openWakeWord](https://github.com/dscripka/openWakeWord/tree/v0.4.0).
Requires Python 3.12+.

The application owns capture, pre-roll, recording and STT routing. WakewordKit
handles a small detection pipeline:

```text
PCM -> validated decoder/downmixer -> streaming resampler -> 80 ms model blocks
    -> openWakeWord -> audio-time cooldown -> detections
```

## Install

```bash
pip install wakewordkit
```

The dependency is pinned to openWakeWord 0.4.0 because that wheel includes the
ONNX models and exposes the API used here. Detector construction uses local
models and does not download files. Importing the public types does not load
the inference backend. Construction and `reset()` load models synchronously;
create/reset detectors outside a latency-sensitive event loop or worker.

## Process application-owned audio

```python
from wakewordkit import AudioFormat, WakeWordDetector

audio_format = AudioFormat(48_000, 2, "pcm_s16le")
detector = WakeWordDetector("hey_jarvis", format=audio_format)

while chunk := await microphone.read():
    detection = await detector.process(chunk)
    if detection is not None:
        print(detection.name, detection.score, detection.timestamp)

for detection in await detector.finish():
    print(detection)
```

`process()` performs inference in a worker thread. `process_sync()` provides the
same behavior for synchronous capture loops, WAV files and existing workers.
Both return the first detection in the call, or `None`. For large frames or
lossless offline evaluation, use `process_all()` / `process_all_sync()` instead:
they return every detection as a tuple in audio order. Both variants process all
audio passed to them; the single-result variants discard additional detections
from that call. At most one winning model is reported per 80 ms model block.

## PCM contract

`AudioFormat(sample_rate, channels, sample_format)` requires positive integer
rates and channel counts. The supported encodings are:

| Encoding | Byte buffers | NumPy arrays | Values |
| --- | --- | --- | --- |
| `pcm_s16le` | little-endian signed 16-bit PCM | `int16` | -32768 to 32767 |
| `pcm_f32le` | little-endian IEEE float32 PCM | `float32` | finite, normalized [-1, 1] |

Byte buffers may be `bytes`, `bytearray` or contiguous `memoryview`. They must
contain complete interleaved channel frames. Arrays must have shape
`(samples, channels)`, or `(samples,)` for mono; no flattening, orientation guessing
or dtype conversion is performed. The array dtype must match `sample_format`.
Stereo is downmixed by channel averaging. NaN, infinity, incomplete channel
frames and unsupported dtypes/layouts raise an error before inference.

Any number of complete input sample frames is accepted, including 20 ms frames
and empty frames. The detector retains incomplete 1,280-sample model blocks.
16 kHz mono PCM16 bypasses float conversion and resampling. Other rates use the
[SoXR streaming HQ resampler](https://python-soxr.readthedocs.io/en/latest/soxr.html),
which preserves filter history and phase, suppresses aliasing and may delay
output until enough filter context is available. Live latency includes that
rate-dependent filter delay and up to 80 ms of model-block accumulation.

`finish()` / `finish_sync()` explicitly ends the stream: it flushes the resampler
and zero-pads the final short model block. It returns all final detections.
Padding does not advance detection timestamps beyond the real resampled audio.
An empty stream produces no inference. Further processing or another finish
requires `reset()`. If no finish is requested, buffered tail samples remain
pending until more input arrives or reset discards them.

## Stream state, reset and cancellation

One detector represents one ordered stream. Await each call before submitting
the next. Overlapping sync/async processing, finish and reset calls raise
`RuntimeError`; they are not queued or reordered. Async calls snapshot mutable
input before dispatch; sync callers must keep their input stable during a call.

Configure `format` at construction or on the first process call. Further calls
can omit it; an explicit format change raises `ValueError`. Call
`reset(format=new_format)` to begin a new stream with a new format. `reset()`
without a format uses the constructor format, or allows the next call to choose
when none was configured initially.

Reset recreates the entire backend, including raw audio, mel features,
embeddings, prediction history and VAD, and clears the resampler, pending model
block, sample clock and cooldown. It has the same initialization cost and backend
warmup as a new detector (up to five 80 ms blocks). Successful detections do not
reset the backend. After an inference error, reset before continuing.

An async call cancelled before its coroutine is admitted consumes no audio.
Once admitted, **cancellation is deferred and suppressed**: the worker completes
and its result or exception is delivered before the call releases the stream.
Repeated cancellation cannot leave background inference modifying the detector
after the call ends. Consequently `asyncio.wait_for()` / `asyncio.timeout()` are
not hard inference deadlines. Stop submitting frames between calls when stopping
a stream. Device/source cleanup remains the application's responsibility.

`cooldown` is measured in processed audio seconds, even when a WAV is processed
faster than real time. It applies globally across all selected wake words.
`cooldown=0` permits consecutive positive model blocks. There is no extra
post-trigger suppression phase.

## Detection results

A `Detection` contains `name`, `score` and optional `sample_index`; `timestamp`
converts the index to seconds at 16 kHz. Detector-produced indices mark the
exclusive end of the evaluated model block, relative to the current stream.
They describe model-window time, not the exact end of the spoken wake word or
the wall time when a delayed resampler emits its samples. Manually constructed
`Detection(name, score)` objects retain `None` for index/time.

Scores are model outputs, not guaranteed calibrated probabilities. `threshold`
and `vad_threshold` must be finite in [0, 1]; `cooldown` must be finite and
non-negative. Non-finite and out-of-range scores cannot trigger a detection.

## Models

```python
from wakewordkit import CustomWakeWord, WakeWord, WakeWordDetector

WakeWordDetector("hey_jarvis")
WakeWordDetector(WakeWord.ALEXA, WakeWord.HEY_MYCROFT)
WakeWordDetector(CustomWakeWord("hey_computer", "models/hey_computer.onnx"))
WakeWordDetector()  # all four bundled models
words = [WakeWord.ALEXA, WakeWord.HEY_JARVIS]
WakeWordDetector(*words)
```

Bundled names are `alexa`, `hey_jarvis`, `hey_mycroft` and `hey_marvin` (English).
Pass individual model arguments; list/iterable arguments, including explicit
empty lists, raise an error. No arguments selects all built-ins.
Model paths, file stems and public labels must each be unique.

Custom models must be binary classifiers trained for the openWakeWord 0.4.0
96-dimensional embedding pipeline: one float32 input `[batch, frames, 96]`, with a
fixed frame count from 1 to 116 (the backend's initial feature context), and one
float32 output `[batch, 1]`. Batch size must be 1 or dynamic. General ONNX and
multiclass models are unsupported. Shape validation cannot prove training or
recognition quality. Custom names must be non-empty without surrounding
whitespace; model loading and compatibility failures are reported explicitly.
Backend initialization warnings remain visible.

## Examples

```bash
uv sync --locked --dev
uv run python examples/detect_from_wav.py recording.wav

uv sync --locked --dev --extra examples
uv run python examples/external_microphone.py
uv run python examples/stt_tts_assistant.py
```

The WAV example processes all detections and the final short frame offline.
The microphone source is owned by the example, closes on failed/cancelled
startup, and bounds both buffered audio and pending event-loop wakeups. When
its buffer fills it drops the oldest frames and increments `dropped_frames`;
device overflow indications also increment that count. A discontinuity means
lost audio; reset your detector if continuous context is required afterwards.

The half-duplex STT/TTS example keeps pre-roll and utterance recording in the
application and requires `OPENAI_API_KEY` (also loaded from `.env`). Its energy
threshold demonstrates the ownership pattern; production applications should
supply their own VAD/turn detection and echo handling.

## Development and evaluation

See [CONTRIBUTING.md](CONTRIBUTING.md) for lint, strict typing, unit/integration
tests and isolated wheel checks. See [docs/EVALUATION.md](docs/EVALUATION.md) for
reproducible latency/CPU/memory benchmarks and annotated recognition evaluation.
The [review checklist](docs/REVIEW_CHECKLIST.md) maps the 26 review items to changes.

## Code and model licenses

WakewordKit's code is **MIT**; see [LICENSE](LICENSE). This does not relicense
dependency code or model weights. openWakeWord's code is Apache-2.0, while its
bundled pretrained models are **CC BY-NC-SA 4.0**, including a non-commercial
restriction; see the [upstream 0.4.0 license statement](https://github.com/dscripka/openWakeWord/blob/v0.4.0/README.md#license)
and [model license](https://creativecommons.org/licenses/by-nc-sa/4.0/).
SoXR/Python-SoXR is [LGPL-2.1-or-later](https://github.com/dofuuz/python-soxr#credit-and-license).
Custom model licenses are determined by their respective authors and training
assets; selecting a custom classifier does not remove licenses on shared
upstream feature models.
