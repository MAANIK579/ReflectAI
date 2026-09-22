"""
assistant.py — full assistant loop using the small local LLM instead of
rigid phrase matching, with silence-based recording (stops automatically
once you stop talking, instead of a fixed time window).

Uses ONE continuous microphone stream that switches between "listening
for wake word" and "recording a command" internally — opening two
separate streams at once is unreliable on Windows, so we avoid that.

Say "hey jarvis" -> speak your question -> it stops once you go quiet
-> Vosk transcribes -> the LLM (grounded with real time/date/weather)
replies -> Piper speaks it.
"""

import json
import queue
import subprocess
import threading
import time
from datetime import datetime

import numpy as np
import requests
import sounddevice as sd
from llama_cpp import Llama
from openwakeword.model import Model

try:
    from faster_whisper import WhisperModel
    _HAS_WHISPER = True
except ImportError:
    _HAS_WHISPER = False

try:
    from vosk import KaldiRecognizer, Model as VoskModel
    _HAS_VOSK = True
except ImportError:
    _HAS_VOSK = False

import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------- Config ----------
SAMPLE_RATE = 16000
WAKE_FRAME_SIZE = 1280       # openWakeWord expects 80ms chunks at 16kHz
WAKE_THRESHOLD = 0.5

WAKE_WORD_MODEL = "hey_jarvis"

VOSK_MODEL_PATH = os.path.join(SCRIPT_DIR, "model")

def get_best_llm_model():
    """
    Selects the highest capability model available in voice_engine:
    1. Qwen 2.5 - 3B (if present)
    2. Qwen 2.5 - 1.5B (optimal for Pi 4 4GB & PC)
    3. Qwen 2.5 - 0.5B (fallback)
    """
    custom = os.getenv("LLM_MODEL_PATH")
    if custom and os.path.exists(custom):
        return custom

    candidates = [
        os.path.join(SCRIPT_DIR, "qwen2.5-3b-instruct-q4_k_m.gguf"),
        os.path.join(SCRIPT_DIR, "qwen2.5-1.5b-instruct-q4_k_m.gguf"),
        os.path.join(SCRIPT_DIR, "qwen2.5-0.5b-instruct-q4_k_m.gguf"),
    ]
    for c in candidates:
        if os.path.exists(c) and os.path.getsize(c) > 100 * 1024 * 1024:
            return c
    return candidates[-1]

LLM_MODEL_PATH = get_best_llm_model()
WEATHER_API_URL = "http://localhost:5000/api/weather"
VOICE_EVENT_URL = "http://localhost:5000/api/voice/event"
ACTIVE_USER_URL = "http://localhost:5000/api/status"  # we use /api/status now to get active user and schedule


def notify_ui(state_name, text="", reply=""):
    """
    Sends voice assistant state transitions and transcripts to the mirror UI.
    Runs asynchronously in a background daemon thread so it never delays the audio loop.
    """
    def _post():
        try:
            requests.post(
                VOICE_EVENT_URL,
                json={"state": state_name, "text": text, "reply": reply},
                timeout=0.5,
            )
        except Exception:
            pass  # Mirror UI might not be running; assistant continues working offline

    threading.Thread(target=_post, daemon=True).start()


PIPER_EXE = os.path.join(SCRIPT_DIR, "piper.exe")
VOICE_MODEL = os.path.join(SCRIPT_DIR, "en_US-lessac-medium.onnx")
ESPEAK_DATA = os.path.join(SCRIPT_DIR, "espeak-ng-data")
TTS_SAMPLE_RATE = 22050

# Silence & Speech detection tuning (values are on raw int16 amplitude)
SPEECH_THRESHOLD = 350      # amplitude required to detect user started speaking
SILENCE_THRESHOLD = 250     # amplitude below which is considered quiet/silence
SILENCE_DURATION = 1.3      # seconds of quiet AFTER speech before processing command
MAX_RECORD_SECONDS = 25     # maximum command length once speaking has started
NO_SPEECH_TIMEOUT = 60      # max seconds to wait in listening mode if nobody speaks at all
MIN_RECORD_SECONDS = 0.5    # minimum speaking time before silence check engages

STOP_PHRASES = [
    "stop", "cancel", "never mind", "nevermind",
    "exit", "quit", "shut up", "bye", "stop listening",
    "go to sleep"
]


def is_stop_command(text):
    if not text:
        return False
    clean = text.strip().lower()
    words = clean.split()
    return clean in STOP_PHRASES or any(w in STOP_PHRASES for w in words)


# ---------- Setup ----------
print("Loading models... (this can take a bit)")

oww_model = Model(wakeword_models=[WAKE_WORD_MODEL], inference_framework="onnx")
print(f"  Wake word model: {WAKE_WORD_MODEL}")

# Speech-to-Text: Faster-Whisper (with Vosk fallback)
whisper_model = None
vosk_model = None
use_whisper = False

if _HAS_WHISPER:
    try:
        whisper_model = WhisperModel("base.en", device="cpu", compute_type="int8")
        use_whisper = True
        print("  STT Engine: Faster-Whisper (base.en, int8)")
    except Exception as e:
        print(f"  Faster-Whisper init failed ({e}), falling back to Vosk...")

if not use_whisper and _HAS_VOSK and os.path.exists(VOSK_MODEL_PATH):
    vosk_model = VoskModel(VOSK_MODEL_PATH)
    print("  STT Engine: Vosk")

llm = Llama(model_path=LLM_MODEL_PATH, n_ctx=1024, n_threads=4, verbose=False)
print(f"  LLM Model: {os.path.basename(LLM_MODEL_PATH)}")

audio_queue = queue.Queue()

# state.mode: "wake" or "command"
state = {
    "mode": "wake",
    "buffer": [],
    "pre_speech_buffer": [],
    "speech_detected": False,
    "silence_start": None,
    "record_start": None,
}


def audio_callback(indata, frames, time_info, status):
    audio_queue.put(bytes(indata))


def speak(text):
    print(f"Assistant: {text}")
    result = subprocess.run(
        [
            PIPER_EXE,
            "--model", VOICE_MODEL,
            "--espeak-ng-data", ESPEAK_DATA,
            "--output-raw",
        ],
        input=text.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        print("Piper error:", result.stderr.decode(errors="ignore"))
        return
    audio = np.frombuffer(result.stdout, dtype=np.int16)
    sd.play(audio, samplerate=TTS_SAMPLE_RATE)
    sd.wait()


def get_context_facts():
    now = datetime.now()
    facts = [
        f"Current time: {now.strftime('%I:%M %p')}",
        f"Current date: {now.strftime('%A, %B %d, %Y')}",
    ]
    try:
        res = requests.get(WEATHER_API_URL, timeout=3)
        data = res.json()
        if "temperature" in data:
            facts.append(
                f"Current weather: {data['temperature']}°, {data['condition']}, "
                f"in {data['city']}"
            )
        else:
            facts.append("Current weather: unavailable")
    except Exception:
        facts.append("Current weather: unavailable")

    # Fetch active user profile for personalized responses
    try:
        res = requests.get(ACTIVE_USER_URL, timeout=2)
        user_data = res.json()
        user = user_data.get("user", {})
        name = user.get("name", "User")
        user_id = user.get("id", "guest")
        facts.append(f"Active user: {name}")
        
        cal_res = requests.get(f"http://localhost:5000/api/calendar?user_id={user_id}", timeout=2)
        cal_data = cal_res.json()
        events = cal_data.get("events", [])
        if events:
            cal_str = "; ".join([f"{e['time']}: {e['title']}" for e in events if e['title']])
            facts.append(f"User's schedule today: {cal_str}")

        # Fetch live camera clothing detection and stylist advice
        try:
            stylist_res = requests.get(f"http://localhost:5000/api/mirror/clothing/status?user_id={user_id}", timeout=2)
            stylist_data = stylist_res.json()
            if stylist_data and stylist_data.get("status") == "ok":
                detected = stylist_data.get("detected", {})
                advice = stylist_data.get("advice", "")
                if detected.get("name"):
                    facts.append(f"Camera detected user is wearing: {detected['name']}.")
                if advice:
                    facts.append(f"AI Stylist clothing advice: {advice}")
        except Exception:
            pass

        # Fetch live camera grooming & appearance analysis
        try:
            groom_res = requests.get(f"http://localhost:5000/api/mirror/grooming/status?user_id={user_id}", timeout=2)
            groom_data = groom_res.json()
            if groom_data and groom_data.get("status") == "ok":
                hair = groom_data.get("hair", "")
                beard = groom_data.get("facial_hair", "")
                skin = groom_data.get("skin", "")
                groom_advice = groom_data.get("advice", "")
                facts.append(f"User grooming & appearance: hair: {hair}; facial hair: {beard}; skin: {skin}.")
                if groom_advice:
                    facts.append(f"Grooming & wellness advice: {groom_advice}")
        except Exception:
            pass
    except Exception:
        pass  # Mirror server may not be running

    return "\n".join(facts)


def build_system_prompt():
    facts = get_context_facts()
    return (
        "You are a helpful voice assistant embedded in a smart mirror named ReflectAI. "
        "Keep every answer to one or two short sentences. Be direct and "
        "conversational, no bullet points, no markdown.\n\n"
        "Use the facts below when asked about time, date, weather, the user's schedule, "
        "what they are wearing, or how their hair, beard, skin, and grooming look. "
        "If asked something these facts don't cover, answer normally from general knowledge.\n\n"
        f"{facts}"
    )


def ask_llm(user_text):
    try:
        response = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": build_system_prompt()},
                {"role": "user", "content": user_text},
            ],
            max_tokens=120,
            temperature=0.7,
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"LLM error: {e}")
        return "Sorry, I had trouble thinking of a reply."


def transcribe(raw_frames):
    if not raw_frames:
        return ""

    if use_whisper and whisper_model is not None:
        try:
            raw_bytes = b"".join(raw_frames)
            audio_int16 = np.frombuffer(raw_bytes, dtype=np.int16)
            audio_float32 = audio_int16.astype(np.float32) / 32768.0
            segments, _ = whisper_model.transcribe(audio_float32, beam_size=1, language="en")
            text = " ".join(s.text for s in segments).strip()
            if text:
                return text
        except Exception as e:
            print(f"Faster-Whisper transcription error: {e}")

    # Fallback to Vosk
    if vosk_model is not None:
        recognizer = KaldiRecognizer(vosk_model, SAMPLE_RATE)
        for chunk in raw_frames:
            recognizer.AcceptWaveform(chunk)
        result = json.loads(recognizer.FinalResult())
        return result.get("text", "")

    return ""


def handle_command(text):
    print(f"You said: {text}")
    if text:
        notify_ui("processing", text=text)
        reply = ask_llm(text)
    else:
        reply = "Sorry, I didn't catch that."

    notify_ui("speaking", text=text, reply=reply)
    speak(reply)
    notify_ui("idle")
    print("\nReady. Say 'hey jarvis' to start.")


def main():
    print("Ready. Say 'hey jarvis' to start.")

    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=WAKE_FRAME_SIZE,  # small enough blocks to also work for wake word
        dtype="int16",
        channels=1,
        callback=audio_callback,
    ):
        while True:
            data = audio_queue.get()

            if state["mode"] == "wake":
                audio = np.frombuffer(data, dtype=np.int16)
                prediction = oww_model.predict(audio)
                triggered = any(
                    score > WAKE_THRESHOLD for score in prediction.values()
                )
                if triggered:
                    print("Wake word detected! Listening...")
                    notify_ui("listening")
                    state["mode"] = "command"
                    state["buffer"] = []
                    state["pre_speech_buffer"] = []
                    state["speech_detected"] = False
                    state["silence_start"] = None
                    state["record_start"] = time.time()

            elif state["mode"] == "command":
                chunk = np.frombuffer(data, dtype=np.int16)
                volume = np.abs(chunk).mean() if len(chunk) else 0

                # PHASE 1: User hasn't started speaking yet ("user says nothing")
                # Keep listening patiently — never cut off while the user is thinking/silent.
                if not state["speech_detected"]:
                    state["pre_speech_buffer"].append(data)
                    if len(state["pre_speech_buffer"]) > 15:
                        state["pre_speech_buffer"].pop(0)

                    if volume >= SPEECH_THRESHOLD:
                        print("Speech detected! Recording command...")
                        state["speech_detected"] = True
                        state["buffer"] = list(state["pre_speech_buffer"])
                        state["silence_start"] = None
                        state["record_start"] = time.time()
                    else:
                        if time.time() - state["record_start"] > NO_SPEECH_TIMEOUT:
                            print("No speech detected (timeout). Returning to wake mode.")
                            notify_ui("idle")
                            state["mode"] = "wake"
                        continue

                # PHASE 2: User has spoken — record until user stops talking (silence) or says stop
                state["buffer"].append(data)
                elapsed = time.time() - state["record_start"]

                if volume < SILENCE_THRESHOLD and elapsed > MIN_RECORD_SECONDS:
                    if state["silence_start"] is None:
                        state["silence_start"] = time.time()
                    elif time.time() - state["silence_start"] > SILENCE_DURATION:
                        notify_ui("processing")
                        text = transcribe(state["buffer"]).strip()

                        # If Vosk didn't pick up words (background noise/cough), keep listening!
                        if not text:
                            print("No words recognized yet, continuing to listen...")
                            state["speech_detected"] = False
                            state["silence_start"] = None
                            state["buffer"] = []
                            notify_ui("listening")
                            continue

                        # If user explicitly said "stop" / "cancel", stop listening immediately
                        if is_stop_command(text):
                            print(f"Stop command received: '{text}'. Stopping.")
                            notify_ui("idle")
                            state["mode"] = "wake"
                            continue

                        state["mode"] = "wake"
                        handle_command(text)
                        continue
                else:
                    state["silence_start"] = None

                if elapsed > MAX_RECORD_SECONDS:
                    notify_ui("processing")
                    text = transcribe(state["buffer"]).strip()
                    if not text:
                        state["speech_detected"] = False
                        state["silence_start"] = None
                        state["buffer"] = []
                        notify_ui("listening")
                        continue

                    if is_stop_command(text):
                        print(f"Stop command received: '{text}'. Stopping.")
                        notify_ui("idle")
                        state["mode"] = "wake"
                        continue

                    state["mode"] = "wake"
                    handle_command(text)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
