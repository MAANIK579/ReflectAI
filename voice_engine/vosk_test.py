"""
vosk_test.py — Step 1: confirm speech-to-text works on your machine.

Listens to your microphone continuously and prints out whatever it
transcribes, in real time. Talk normally and watch the terminal.
Press Ctrl+C to stop.
"""

import json
import queue
import sys

import sounddevice as sd
from vosk import Model, KaldiRecognizer

MODEL_PATH = "model"  # folder where you unzipped the Vosk model
SAMPLE_RATE = 16000

q = queue.Queue()


def audio_callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    q.put(bytes(indata))


def main():
    print("Loading model... (this can take a few seconds)")
    model = Model(MODEL_PATH)
    recognizer = KaldiRecognizer(model, SAMPLE_RATE)

    print("Listening... speak into your microphone. Ctrl+C to stop.")

    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=8000,
        dtype="int16",
        channels=1,
        callback=audio_callback,
    ):
        while True:
            data = q.get()
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                text = result.get("text", "")
                if text:
                    print(f"You said: {text}")
            else:
                # Partial (in-progress) result — optional to show
                partial = json.loads(recognizer.PartialResult())
                partial_text = partial.get("partial", "")
                if partial_text:
                    print(f"...{partial_text}", end="\r")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
