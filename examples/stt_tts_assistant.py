"""Half-duplex wake word -> STT -> TTS example with application-owned audio.

This example intentionally keeps capture, buffering, utterance boundaries, API
calls, and playback outside WakewordKit. Set OPENAI_API_KEY before running it.

    pip install 'wakewordkit[examples]'
    python examples/stt_tts_assistant.py
"""

import asyncio
import io
import math
import time
import wave
from collections import deque

import numpy as np
import sounddevice as sd
from _microphone import MicrophoneFrames
from dotenv import load_dotenv
from openai import AsyncOpenAI

from wakewordkit import Detection, WakeWordDetector

_PREROLL_SECONDS = 0.24
_MIN_UTTERANCE_SECONDS = 0.8
_END_SILENCE_SECONDS = 1.0
_MAX_UTTERANCE_SECONDS = 15.0
_SPEECH_RMS_THRESHOLD = 500


async def main() -> None:
    load_dotenv()
    detector = WakeWordDetector("hey_jarvis")
    client = AsyncOpenAI()

    try:
        async with MicrophoneFrames() as microphone:
            while True:
                microphone.discard_buffered()
                detector.reset()
                print('Say "Hey Jarvis", followed by a short sentence...')

                detection, audio = await _record_utterance(microphone, detector)
                print(f"Detected {detection.name} ({detection.score:.2f})")

                transcript = await client.audio.transcriptions.create(
                    model="gpt-4o-mini-transcribe",
                    file=("utterance.wav", audio),
                    language="de",
                )
                text = transcript.text.strip()
                if not text:
                    print("No speech transcribed.")
                    continue
                print(f"Heard: {text}")

                speech = await client.audio.speech.create(
                    model="gpt-4o-mini-tts",
                    voice="marin",
                    input=f"Ich habe verstanden: {text}",
                    response_format="wav",
                )
                await asyncio.to_thread(_play_wav, speech.content)
    finally:
        await client.close()


async def _record_utterance(
    microphone: MicrophoneFrames,
    detector: WakeWordDetector,
) -> tuple[Detection, bytes]:
    frame_seconds = microphone.frame_size / microphone.format.sample_rate
    preroll: deque[bytes] = deque(
        maxlen=max(1, math.ceil(_PREROLL_SECONDS / frame_seconds))
    )

    while True:
        frame = await microphone.read()
        preroll.append(frame)
        detection = await asyncio.to_thread(
            detector.process,
            frame,
            format=microphone.format,
        )
        if detection is not None:
            break

    frames = list(preroll)
    started_at = time.monotonic()
    last_speech_at = started_at

    while True:
        frame = await microphone.read()
        frames.append(frame)
        now = time.monotonic()
        if _rms(frame) >= _SPEECH_RMS_THRESHOLD:
            last_speech_at = now

        duration = now - started_at
        if (
            duration >= _MIN_UTTERANCE_SECONDS
            and now - last_speech_at >= _END_SILENCE_SECONDS
        ):
            break
        if duration >= _MAX_UTTERANCE_SECONDS:
            break

    return detection, _as_wav(frames, microphone)


def _rms(frame: bytes) -> float:
    samples = np.frombuffer(frame, dtype="<i2").astype(np.float32)
    return float(np.sqrt(np.mean(samples * samples)))


def _as_wav(frames: list[bytes], microphone: MicrophoneFrames) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(microphone.format.channels)
        wav.setsampwidth(microphone.format.bytes_per_sample)
        wav.setframerate(microphone.format.sample_rate)
        wav.writeframes(b"".join(frames))
    return output.getvalue()


def _play_wav(audio: bytes) -> None:
    with wave.open(io.BytesIO(audio), "rb") as wav:
        if wav.getsampwidth() != 2 or wav.getcomptype() != "NONE":
            raise ValueError("TTS response must be uncompressed 16-bit WAV audio.")
        samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
        if wav.getnchannels() > 1:
            samples = samples.reshape(-1, wav.getnchannels())
        sample_rate = wav.getframerate()

    sd.play(samples, samplerate=sample_rate)
    sd.wait()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
