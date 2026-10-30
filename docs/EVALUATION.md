# Evaluation

Engineering correctness and recognition quality need separate evidence. Real
backend tests cover stream state, framing, reset and resampling. The performance
smoke benchmark below uses silence; it cannot establish false-activation rates,
miss rates or robustness to language, speakers, music and background noise.

## Reproducible performance smoke

```bash
uv sync --locked --dev
uv run python tools/benchmark.py --iterations 150 --output benchmark.json
```

The report records Python/platform/CPU identity and dependency versions. Each
one-model/four-model, sync/async scenario runs in a fresh process, warms eight
80 ms blocks, then measures 150 calls. It reports construction and full-reset
cost, median/p95 wall and process CPU time, CPU cost as a fraction of one core
at real time, and sampled peak RSS. Public import time is measured in a fresh
process. Cold construction includes lazy backend imports; reset reuses imported
libraries while recreating model sessions and audio state. File caches and OS scheduling still affect results; record power mode,
CPU model, RAM and competing workloads when comparing machines.

RSS is sampled at construction, calls and reset, so it may miss transient peaks.
It includes Python and backend libraries. CPU timer granularity can be coarse,
particularly for individual short Windows calls. These measurements are for
16 kHz input; they do not measure SoXR filter delay, capture/queueing latency,
STT or playback. Measure capture-to-observed-trigger time in the target app for
end-to-end live latency. [benchmark-smoke.json](benchmark-smoke.json) is one local
Windows run, not a performance guarantee.

## Recognition corpus

Create a private/appropriately licensed corpus with:

- Positive recordings from multiple held-out speakers, distances, microphones,
  levels, accents and rooms; annotate each expected phrase's end in seconds.
- Several hours of negative conversations, music, TV, silence and household
  noise from the intended environment, including words similar to each keyword.
- Separate threshold-tuning and held-out evaluation sets. Keep recordings,
  splits and hardware fixed when comparing changes.

No such recognition corpus was supplied with the review. Therefore there is no
new recognition-quality baseline or claim of superiority in this change. Audio
is not downloaded or captured by the evaluation tool.

Use a JSON manifest (paths are relative to the manifest):

```json
{
  "schema_version": 1,
  "recordings": [
    {
      "path": "positive/speaker01.wav",
      "negative": false,
      "activations": [{"name": "hey_jarvis", "end_seconds": 2.4}]
    },
    {"path": "negative/kitchen.wav", "negative": true, "activations": []}
  ]
}
```

WAV files must contain uncompressed PCM16, at any supported rate/channel count.
Positive files require annotations; negative files must have none.

```bash
uv run python tools/evaluate.py corpus/manifest.json --word hey_jarvis   --threshold 0.5 --cooldown 1.5 --early 0.5 --late 1.0 --output results.json
```

Pass `--word` repeatedly to select multiple models. A fresh stream is used for
each recording, including normal backend warmup; include realistic lead-in
rather than placing every wake word at sample zero. Every detection is retained,
and finish flushes delayed output and pads the final short model block.

Matching uses labels and an explicit interval around each annotated phrase end.
In time order, a detection matches the earliest eligible unmatched annotation.
An annotation can match only once; extra triggers remain unmatched. Report:

- False activations per **negative** audio hour; positive-file background time is
  not silently counted as known-negative exposure.
- Missed/annotated activations and miss fraction; unmatched detections across
  all files are also reported to expose duplicate triggers.
- Signed model-window latency relative to the annotation (may be negative),
  median/p95 processing-call time, CPU, sampled RSS and construction time.

Timestamps describe evaluated audio windows, not an exact speech boundary or
observed live latency. Matching tolerances are part of the evaluation protocol;
do not retune them on the held-out set. Zero denominators produce JSON null,
not fabricated zero error rates. Reports retain the manifest and WAV SHA256
hashes, versions, selected models and thresholds for reproducibility. Report
exposure, sample counts and confidence intervals when interpreting rates; a
short silence smoke with zero triggers is not evidence of a low real-world rate.
