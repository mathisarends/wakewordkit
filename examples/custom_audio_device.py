"""Use a WAV file as a WakewordKit-owned audio device with full handoff.

python examples/custom_audio_device.py path/to/recording.wav
"""

import asyncio
import sys
import wave
from contextlib import asynccontextmanager

from wakewordkit import AudioFormat, CapturedAudio, WakeWordDetector


class WavAudioDevice:
    def __init__(self, path: str, *, frame_size: int = 1_280) -> None:
        self._path = path
        self._frame_size = frame_size

    @asynccontextmanager
    async def open(self):
        wav = await asyncio.to_thread(wave.open, self._path, "rb")
        if wav.getsampwidth() != 2 or wav.getcomptype() != "NONE":
            wav.close()
            raise ValueError("WAV must contain uncompressed 16-bit PCM audio.")

        async def frames():
            while chunk := await asyncio.to_thread(wav.readframes, self._frame_size):
                yield chunk

        try:
            yield CapturedAudio(
                format=AudioFormat(wav.getframerate(), wav.getnchannels(), "pcm_s16le"),
                frames=frames(),
            )
        finally:
            await asyncio.to_thread(wav.close)


async def main(path: str) -> None:
    detector = WakeWordDetector("hey_jarvis", capture=WavAudioDevice(path))
    activation = await detector.listen(preroll=0.2)
    print(f"Detected {activation.name} ({activation.score:.2f})")

    byte_count = 0
    async with activation.audio:
        async for chunk in activation.audio:
            byte_count += len(chunk)
    print(f"Lossless handoff contained {byte_count:,} bytes")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(
            "usage: python examples/custom_audio_device.py <recording.wav>"
        )
    asyncio.run(main(sys.argv[1]))
