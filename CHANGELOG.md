# Changelog

## 0.6.0 (unreleased)

WakewordKit 0.6.0 makes PCM processing and detector state independent of caller
frame boundaries. It adds a public synchronous API and explicit stream lifecycle.
The release includes intentional contract changes from 0.5.0.

### Audio and state correctness

- Accept short input frames, including 20 ms PCM. Accumulate exact 1,280-sample
  model blocks internally and retain incomplete blocks across calls.
- Use a stateful SoXR HQ resampler for anti-aliased downsampling and consistent
  upsampling. Flush delayed samples at stream end.
- Reset the entire backend, including raw audio, features, prediction history,
  VAD, resampler, pending samples, sample clock and cooldown.
- Measure the global cooldown in audio time. Successful detections no longer
  reset the backend or introduce a second suppression period.
- Restore the required openWakeWord 0.4.0 dependency and its bundled ONNX assets;
  add SoXR as a runtime dependency.

### Public API

- Add `process_sync()` and `process_all_sync()` for synchronous callers.
- Add `process_all()` for all detections in a frame. The single-result APIs
  return the first detection and discard additional detections from that call.
- Add `finish()` / `finish_sync()` to flush delayed output and zero-pad a final
  short model block. Processing after finish requires reset.
- Add optional `Detection.sample_index` and derived `timestamp`, identifying
  the model-window end on the 16 kHz stream clock, rather than an exact speech end.
- Add constructor-level `format` configuration and `reset(format=...)`.
- Export typed `AudioFrame` and `SampleFormat` aliases and defer backend imports
  until detector construction.

### Contract changes and migration from 0.5.0

1. Pass models as individual positional arguments. Replace
   `WakeWordDetector(words)` with `WakeWordDetector(*words)` for a non-empty list.
   Explicit list arguments, including `[]`, raise an error. No arguments still
   selects all four built-in models.
2. Match arrays to `AudioFormat.sample_format`: `int16` for `pcm_s16le`, or finite
   normalized `float32` in [-1, 1] for `pcm_f32le`. Use `(samples, channels)` arrays
   or `(samples,)` for mono. Other dtypes, flattenable/transposed layouts and
   incomplete interleaved byte/channel frames now raise errors.
3. Keep one ordered stream per detector and await each async call. Overlapping
   process, finish and reset operations raise `RuntimeError`. Async calls copy
   mutable input before dispatch.
4. Cancellation before async admission consumes nothing. After admission,
   cancellation is deferred and suppressed so the result or inference exception
   is delivered before the stream is released. Async timeouts are not hard
   inference deadlines.
5. Changing audio format requires `reset(format=new_format)`. Reset discards
   pending audio, starts a new sample clock, reloads models synchronously and
   repeats backend warmup. It is intended for a new stream or a discontinuity.
6. Use `process_all[_sync]()` for exhaustive batch/offline processing and call
   `finish[_sync]()` at end of input to process delayed and trailing samples.
7. Thresholds, VAD threshold, cooldown and audio samples now require valid finite
   values. Model paths, file stems and public labels must be unique. Custom
   classifiers must satisfy the documented binary openWakeWord ONNX contract;
   general and multiclass models are rejected.

### Examples, validation and licensing

- Process the final short WAV frame and use audio-time cooldown offline.
- Close microphone streams on failed or cancelled startup; bound captured audio
  and event-loop notification backlog at the callback boundary.
- Add real-backend inference/state/reset tests, resampling-quality tests, strict
  public API typing, and isolated wheel installation checks. CI covers Python
  3.12-3.14 on Linux and Windows.
- Add reproducible latency/CPU/memory benchmarking and an annotated corpus
  evaluator. Recognition-quality claims still require representative recordings;
  the included engineering smoke benchmark does not establish those rates.
- Keep backend initialization warnings visible. Separate the MIT code license
  from upstream pretrained model licensing, including CC BY-NC-SA 4.0 restrictions.

See [README.md](README.md) for the full contract and
[docs/RELEASING.md](docs/RELEASING.md) for the release procedure.
