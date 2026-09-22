"""
wakeword_test.py — Step 2: confirm wake word detection works.

Uses a pretrained openWakeWord model to listen for a wake phrase.
We start with a built-in phrase ("hey jarvis") to prove the pipeline
works, before training a custom "Hey Mirror" wake word later.

Say "hey jarvis" and watch the terminal. Press Ctrl+C to stop.
"""

import numpy as np
import sounddevice as sd
from openwakeword.model import Model

SAMPLE_RATE = 16000
FRAME_SIZE = 1280  # openWakeWord expects 80ms chunks at 16kHz
THRESHOLD = 0.5

# Built-in pretrained wake word to test the pipeline first.
# Options include: "hey_jarvis", "alexa", "hey_mycroft"
# inference_framework="onnx" avoids needing tflite-runtime, which has poor
# Windows support on newer Python versions.
oww_model = Model(wakeword_models=["hey_jarvis"], inference_framework="onnx")

print("Loading model...")
print("Listening for wake word 'hey jarvis'... Ctrl+C to stop.")


def audio_callback(indata, frames, time, status):
    if status:
        print(status)

    audio = np.frombuffer(indata, dtype=np.int16)
    prediction = oww_model.predict(audio)

    for wakeword, score in prediction.items():
        if score > THRESHOLD:
            print(f"Wake word detected! ({wakeword}, score={score:.2f})")


def main():
    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=FRAME_SIZE,
        dtype="int16",
        channels=1,
        callback=audio_callback,
    ):
        while True:
            sd.sleep(1000)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")