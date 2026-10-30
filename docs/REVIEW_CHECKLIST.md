# Review implementation checklist

All 26 numbered findings/recommendations from the supplied review are addressed.
The recognition corpus limitation is tracked separately below.

| Item | Result and verification |
| --- | --- |
| 1 | Pinned backend restored; lock consistent; isolated built-wheel installation and real inference in `tools/wheel_smoke.py`. |
| 2 | Exact 1,280-sample backend blocks with retained tails; short-frame unit and real integration tests. |
| 3 | Stateful SoXR HQ anti-alias filtering; passband, alias rejection and chunk tests. |
| 4 | Reset recreates backend, features, raw audio and VAD; compared with a fresh real instance. |
| 5 | No backend reset on detection; zero cooldown permits consecutive positive blocks. |
| 6 | Misaligned byte/channel groups raise a clear error. |
| 7 | Stateful filtered upsampling waits for context; chunked output matches one-shot output after finish. |
| 8 | Sample-clock cooldown, global across models; offline tests run without wall-clock waits. |
| 9 | Admitted async calls finish and return results despite cancellation; pre-admission cancellation consumes nothing. |
| 10 | All operations use one nonblocking admission lock; overlaps rejected; reset is atomic. |
| 11 | Format binds to a stream; changes require explicit reset. |
| 12 | Paths, stems and public labels validated for uniqueness. |
| 13 | Explicit matching int16/float32 dtype, array layout, byte order and value range validation. |
| 14 | Finite thresholds/cooldown/audio/scores; strictly positive integer format parameters. |
| 15 | Binary openWakeWord ONNX shape/type contract; corrupt, incompatible and multiclass models tested. |
| 16 | Initialization warnings remain visible, with a regression test. |
| 17 | MIT code license separated from upstream model/dependency licenses in README. |
| 18 | Real inference, complete reset/VAD, chunk partitions, resampling quality and installed-wheel tests; CI on Linux/Windows, Python 3.12-3.14. |
| 19 | Public `process_sync` / `process_all_sync`; async adapters; synchronous construction/reset documented. |
| 20 | Native 16 kHz mono PCM16 direct path; exact samples and shared-memory unit test. |
| 21 | Backend/SoXR lazy imports; public type import tested in a fresh process. |
| 22 | Positional `*wake_words`; iterable/empty-list arguments rejected, no-argument default retained. |
| 23 | Optional model sample index and timestamp, with precise window semantics; score calibration caveat. |
| 24 | Literal encoding and typed NumPy/byte-buffer inputs, exported aliases, strict mypy public API consumer and CI. |
| 25 | One Hatch version source, current Python/examples setup and matching pinned Ruff versions. |
| 26 | Offline WAV finish, stream cleanup on failed/cancelled start, bounded callback-side buffering and one pending wakeup; mocked device tests. |

Additional evaluation infrastructure: `tools/benchmark.py` records startup,
CPU, memory and sync/async latency; `tools/evaluate.py` records false activations,
misses and annotated model-window latency. See `EVALUATION.md`.

Pending external evidence: collect/select a licensed positive and long negative
recording corpus and run the evaluation on defined target hardware. No corpus
was provided; synthetic engineering tests are not presented as recognition
quality measurements.
