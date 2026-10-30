"""Static checks for the installed public API's supported input and result types."""

from typing import assert_type

import numpy as np

from wakewordkit import AudioFormat, Detection, WakeWordDetector


def process(detector: WakeWordDetector) -> None:
    assert_type(detector.process_sync(np.zeros(320, dtype=np.int16)), Detection | None)
    assert_type(detector.process_sync(bytearray(640)), Detection | None)
    assert_type(
        detector.process_all_sync(memoryview(bytes(640))), tuple[Detection, ...]
    )
    detector.process_sync(
        np.zeros((320, 2), dtype=np.float32),
        format=AudioFormat(48000, 2, "pcm_f32le"),
    )
    assert_type(detector.finish_sync(), tuple[Detection, ...])


async def process_async(detector: WakeWordDetector) -> None:
    assert_type(await detector.process(bytes(640)), Detection | None)
    assert_type(await detector.process_all(bytes(640)), tuple[Detection, ...])
    assert_type(await detector.finish(), tuple[Detection, ...])
