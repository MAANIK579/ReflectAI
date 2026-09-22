"""
app/assistant/concierge.py — Multimodal Voice Concierge & Morning Briefing Service.

Aggregates:
- User identity & preferences (UserRepository)
- Live weather & forecast (WeatherService)
- Schedule & reminders (CalendarService / ReminderRepository)
- Live camera styling & grooming (MirrorStylist / GroomingClassifier / OutfitEngine)
- ESP32 indoor telemetry (temperature, humidity, motion)

Synthesizes voice briefing offline using Piper TTS and streams audio
or plays through mirror speakers.
"""

import io
import logging
import os
import shutil
import subprocess
import threading
import time
import wave
from datetime import datetime
from typing import Any, Callable, Dict, Optional

from app.calendar import CalendarService
from app.config.settings import settings
from app.database.repositories import SettingsRepository, UserRepository
from app.recommendations import OutfitEngine
from app.weather import WeatherService

logger = logging.getLogger("reflectai.concierge")


def get_piper_binary() -> Optional[str]:
    """Finds the Piper TTS executable on Windows or Linux."""
    candidates = [
        settings.PROJECT_ROOT / "voice_engine" / "piper.exe",
        settings.PROJECT_ROOT / "voice_engine" / "piper",
    ]
    for c in candidates:
        if c.exists() and os.access(str(c), os.X_OK | os.R_OK):
            return str(c)

    # Check system PATH
    found = shutil.which("piper") or shutil.which("piper.exe")
    if found:
        return found
    return None


def get_voice_model() -> Optional[str]:
    """Returns absolute path to the Piper ONNX voice model."""
    model_path = settings.PROJECT_ROOT / "voice_engine" / "en_US-lessac-medium.onnx"
    return str(model_path) if model_path.exists() else None


def get_espeak_data() -> Optional[str]:
    """Returns espeak-ng-data directory path for phonemization."""
    data_path = settings.PROJECT_ROOT / "voice_engine" / "espeak-ng-data"
    return str(data_path) if data_path.exists() else None


class VoiceConciergeService:
    _lock = threading.Lock()
    _is_playing = False
    _stop_requested = False

    @classmethod
    def get_time_greeting(cls, now: Optional[datetime] = None) -> str:
        if now is None:
            now = datetime.now()
        hour = now.hour
        if 4 <= hour < 12:
            return "Good morning"
        elif 12 <= hour < 17:
            return "Good afternoon"
        else:
            return "Good evening"

    @classmethod
    def build_briefing(
        cls,
        user_id: Optional[str] = None,
        weather_data: Optional[Dict[str, Any]] = None,
        esp32_data: Optional[Dict[str, Any]] = None,
        stylist_data: Optional[Dict[str, Any]] = None,
        grooming_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Builds a comprehensive, natural conversational briefing for the given user.
        """
        if not user_id:
            user_id = SettingsRepository.get("active_user", default="guest")

        user = UserRepository.get(user_id) if user_id else None
        user_name = user["name"] if user else (user_id.replace("_", " ").title() if user_id != "guest" else "there")

        now = datetime.now()
        time_greeting = cls.get_time_greeting(now)
        date_str = now.strftime("%A, %B %d")

        # 1. Greeting section
        greeting_text = f"{time_greeting}, {user_name}! Today is {date_str}."

        # 2. Outdoor Weather section
        if weather_data is None:
            try:
                weather_data = WeatherService.get_weather()
            except Exception as e:
                logger.warning(f"Could not load weather for briefing: {e}")
                weather_data = {}

        weather_text = ""
        temp = weather_data.get("temperature")
        cond = weather_data.get("condition", "Clear")
        city = weather_data.get("city", "your area")
        high = weather_data.get("high")
        low = weather_data.get("low")

        if temp is not None:
            weather_text = f"Right now in {city}, it's {int(round(float(temp)))} degrees and {cond.lower()}."
            if high is not None and low is not None:
                weather_text += f" You can expect a high of {int(round(float(high)))} and a low of {int(round(float(low)))} degrees."
        else:
            weather_text = "Weather data is currently updating."

        # 3. Schedule & Reminders section
        schedule_text = ""
        try:
            events = CalendarService.get_events_for_user(user_id)
            valid_events = [e for e in events if e.get("title") and "no more upcoming" not in e.get("title", "").lower()]
            if valid_events:
                if len(valid_events) == 1:
                    ev = valid_events[0]
                    schedule_text = f"On your agenda today, you have {ev['title']} at {ev['time']}."
                else:
                    ev1, ev2 = valid_events[0], valid_events[1]
                    schedule_text = f"You have {len(valid_events)} items scheduled today: {ev1['title']} at {ev1['time']}, followed by {ev2['title']} at {ev2['time']}."
            else:
                schedule_text = "You have a completely clear schedule today."
        except Exception as e:
            logger.warning(f"Could not load calendar events: {e}")
            schedule_text = "Your schedule is clear for today."

        # 4. Wardrobe & Mirror Stylist Advice
        style_text = ""
        if stylist_data and stylist_data.get("advice"):
            style_text = f"Mirror Stylist recommends: {stylist_data['advice']}"
        else:
            try:
                rec = OutfitEngine.recommend(user_id)
                items = rec.get("items", [])
                top_item = next((i for i in items if i.get("label") == "Top"), None)
                bot_item = next((i for i in items if i.get("label") == "Bottom"), None)
                if top_item and bot_item:
                    style_text = f"Today's recommended outfit is your {top_item['name']} paired with {bot_item['name']}."
                elif top_item:
                    style_text = f"For today's outfit, I suggest your {top_item['name']}."
                elif rec.get("reason"):
                    style_text = f"Styling tip: {rec.get('reason')}."
            except Exception as e:
                logger.warning(f"Could not load outfit recommendation: {e}")

        # Grooming compliment/check if present
        if grooming_data and grooming_data.get("status") == "ok":
            groom_advice = grooming_data.get("advice")
            if groom_advice:
                style_text = f"{style_text} {groom_advice}".strip()

        # 5. ESP32 Room Climate
        indoor_text = ""
        if esp32_data and esp32_data.get("temperature") is not None:
            in_temp = esp32_data.get("temperature")
            in_hum = esp32_data.get("humidity")
            if in_hum is not None:
                indoor_text = f"Your room is currently {float(in_temp):.1f} degrees with {int(round(float(in_hum)))} percent humidity."
            else:
                indoor_text = f"Your room temperature is {float(in_temp):.1f} degrees."

        # 6. Assemble complete conversational script
        parts = [greeting_text, weather_text, schedule_text]
        if style_text:
            parts.append(style_text)
        if indoor_text:
            parts.append(indoor_text)
        parts.append("Have a fantastic and productive day ahead!")

        full_text = " ".join(p.strip() for p in parts if p.strip())

        return {
            "status": "ok",
            "user_id": user_id,
            "user_name": user_name,
            "greeting": time_greeting,
            "full_text": full_text,
            "sections": {
                "greeting": greeting_text,
                "weather": weather_text,
                "schedule": schedule_text,
                "style": style_text,
                "indoor": indoor_text,
            },
            "timestamp": time.time(),
        }

    @classmethod
    def synthesize_to_wav(cls, text: str) -> bytes:
        """
        Synthesizes text into high quality 22050Hz 16-bit Mono WAV bytes using Piper TTS.
        Falls back to empty silent WAV if Piper engine is missing.
        """
        piper_bin = get_piper_binary()
        model_file = get_voice_model()
        espeak_dir = get_espeak_data()

        if not piper_bin or not model_file or not os.path.exists(piper_bin) or not os.path.exists(model_file):
            logger.warning("Piper TTS binary or model not available; generating silent fallback WAV.")
            return cls._create_silent_wav()

        cmd = [piper_bin, "--model", model_file, "--output-raw"]
        if espeak_dir and os.path.exists(espeak_dir):
            cmd.extend(["--espeak-ng-data", espeak_dir])

        try:
            res = subprocess.run(
                cmd,
                input=text.encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=15,
            )
            if res.returncode != 0:
                logger.error(f"Piper TTS synthesis failed: {res.stderr.decode('utf-8', errors='ignore')}")
                return cls._create_silent_wav()

            raw_pcm = res.stdout
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(22050)
                wf.writeframes(raw_pcm)
            return buf.getvalue()
        except Exception as e:
            logger.error(f"Exception during Piper synthesis: {e}")
            return cls._create_wav_from_silence()

    @classmethod
    def _create_silent_wav(cls, duration_sec: float = 0.5) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(22050)
            num_samples = int(22050 * duration_sec)
            wf.writeframes(b"\x00\x00" * num_samples)
        return buf.getvalue()

    @classmethod
    def play_on_mirror_async(
        cls,
        text: str,
        user_id: Optional[str] = None,
        on_start: Optional[Callable[[str], None]] = None,
        on_finish: Optional[Callable[[], None]] = None,
    ):
        """
        Synthesizes and plays audio asynchronously on local speakers using sounddevice.
        Updates state hooks on start and finish.
        """
        def _runner():
            with cls._lock:
                cls._is_playing = True
                cls._stop_requested = False

            if on_start:
                try:
                    on_start(text)
                except Exception:
                    pass

            try:
                wav_bytes = cls.synthesize_to_wav(text)
                if not cls._stop_requested:
                    import numpy as np
                    import sounddevice as sd

                    buf = io.BytesIO(wav_bytes)
                    with wave.open(buf, "rb") as wf:
                        framerate = wf.getframerate()
                        raw_data = wf.readframes(wf.getnframes())

                    audio = np.frombuffer(raw_data, dtype=np.int16)
                    sd.play(audio, samplerate=framerate)
                    sd.wait()
            except Exception as e:
                logger.error(f"Mirror speaker audio playback error: {e}")
            finally:
                with cls._lock:
                    cls._is_playing = False
                    cls._stop_requested = False
                if on_finish:
                    try:
                        on_finish()
                    except Exception:
                        pass

        thread = threading.Thread(target=_runner, daemon=True)
        thread.start()

    @classmethod
    def stop_playback(cls):
        """Stops any active audio playback on mirror."""
        with cls._lock:
            cls._stop_requested = True
        try:
            import sounddevice as sd
            sd.stop()
        except Exception:
            pass
        with cls._lock:
            cls._is_playing = False
