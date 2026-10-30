"""Reproducible smoke benchmark; each scenario uses a fresh process."""

import argparse
import asyncio
import json
import platform
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import psutil

from wakewordkit import WakeWord, WakeWordDetector


def percentiles(samples):
    return {
        "median": float(np.median(samples)),
        "p95": float(np.percentile(samples, 95)),
    }


def measure(models, asynchronous, iterations):
    process = psutil.Process()
    started = time.perf_counter_ns()
    detector = WakeWordDetector(*list(WakeWord)[:models])
    construction_ms = (time.perf_counter_ns() - started) / 1e6
    block = np.zeros(1280, np.int16)
    wall, cpu = [], []
    rss = process.memory_info().rss

    def record(start, cpu_start):
        nonlocal rss
        wall.append((time.perf_counter_ns() - start) / 1e6)
        cpu.append((time.process_time_ns() - cpu_start) / 1e6)
        rss = max(rss, process.memory_info().rss)

    if asynchronous:

        async def run():
            for _ in range(8):
                await detector.process(block)
            for _ in range(iterations):
                cpu_start, start = time.process_time_ns(), time.perf_counter_ns()
                await detector.process(block)
                record(start, cpu_start)

        asyncio.run(run())
    else:
        for _ in range(8):
            detector.process_sync(block)
        for _ in range(iterations):
            cpu_start, start = time.process_time_ns(), time.perf_counter_ns()
            detector.process_sync(block)
            record(start, cpu_start)
    started = time.perf_counter_ns()
    detector.reset()
    reset_ms = (time.perf_counter_ns() - started) / 1e6
    rss = max(rss, process.memory_info().rss)
    return {
        "models": models,
        "api": "async" if asynchronous else "sync",
        "construction_ms": construction_ms,
        "reset_ms": reset_ms,
        "call_wall_ms": percentiles(wall),
        "call_cpu_ms": percentiles(cpu),
        "cpu_percent_of_one_core_at_realtime": sum(cpu) / (iterations * 80) * 100,
        "sampled_peak_rss_mib": rss / 1024**2,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=150)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", type=int, choices=[1, 4])
    parser.add_argument("--async-api", action="store_true")
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    if args.worker:
        print(json.dumps(measure(args.worker, args.async_api, args.iterations)))
        return
    public_import = subprocess.check_output(
        [
            sys.executable,
            "-c",
            "import time; start=time.perf_counter(); import wakewordkit; "
            "print((time.perf_counter()-start)*1000)",
        ],
        text=True,
    )
    results = []
    for models in (1, 4):
        for asynchronous in (False, True):
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                str(models),
                "--iterations",
                str(args.iterations),
            ]
            if asynchronous:
                command.append("--async-api")
            results.append(json.loads(subprocess.check_output(command, text=True)))
    report = {
        "kind": "engineering smoke benchmark; silence, not recognition quality",
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpus": psutil.cpu_count(),
        "versions": {
            name: version(name)
            for name in ("wakewordkit", "openwakeword", "numpy", "soxr", "onnxruntime")
        },
        "public_import_ms": float(public_import),
        "sample_rate": 16000,
        "block_samples": 1280,
        "warmup_blocks": 8,
        "iterations": args.iterations,
        "scenarios": results,
    }
    output = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
