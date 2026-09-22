"""
tts_test.py — Step 5: confirm text-to-speech works.

Uses Piper (a standalone offline TTS engine) to convert text into
spoken audio and play it back directly, using sounddevice (already
installed) instead of needing ffmpeg/ffplay as an extra dependency.
"""

import subprocess
import sys

import numpy as np
import sounddevice as sd

PIPER_EXE = "piper.exe"          # piper.exe, in this same folder (or on PATH)
VOICE_MODEL = "en_US-lessac-medium.onnx"  # downloaded alongside its .json config
SAMPLE_RATE = 22050


def speak(text):
    result = subprocess.run(
        [PIPER_EXE, "--model", VOICE_MODEL, "--output-raw"],
        input=text.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if result.returncode != 0:
        print("Piper error:", result.stderr.decode(errors="ignore"))
        return

    audio = np.frombuffer(result.stdout, dtype=np.int16)
    sd.play(audio, samplerate=SAMPLE_RATE)
    sd.wait()


if __name__ == "__main__":
    text = " ".join(sys.argv[1:]) or "Hello, this is a test of the text to speech system."
    speak(text)
