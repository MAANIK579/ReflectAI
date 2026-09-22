"""
voice_engine/whisper_test.py — Test Faster-Whisper (Whisper-Base) Speech-to-Text.

Usage:
  python voice_engine/whisper_test.py                # records 4 seconds from mic and transcribes
  python voice_engine/whisper_test.py <audio.wav>    # transcribes specified audio file
"""

import sys
import time
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000
RECORD_SECONDS = 4


def test_file(file_path: str):
    print(f"Loading Whisper base.en model...")
    model = WhisperModel("base.en", device="cpu", compute_type="int8")
    print(f"Transcribing audio file: {file_path}")
    t0 = time.time()
    segments, info = model.transcribe(file_path, beam_size=1, language="en")
    text = " ".join(s.text for s in segments).strip()
    elapsed = time.time() - t0
    print(f"\nResult ({elapsed:.2f}s): \"{text}\"")
    print(f"Language: {info.language} (probability: {info.language_probability:.2f})")


def test_mic():
    print(f"Loading Whisper base.en model...")
    model = WhisperModel("base.en", device="cpu", compute_type="int8")

    print(f"\nRecording {RECORD_SECONDS} seconds from microphone... Speak now!")
    audio = sd.rec(int(RECORD_SECONDS * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()
    print("Recording complete! Transcribing with Faster-Whisper...")

    t0 = time.time()
    audio_flat = audio.flatten()
    segments, info = model.transcribe(audio_flat, beam_size=1, language="en")
    text = " ".join(s.text for s in segments).strip()
    elapsed = time.time() - t0

    print(f"\nYou said ({elapsed:.2f}s): \"{text}\"")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_file(sys.argv[1])
    else:
        test_mic()
