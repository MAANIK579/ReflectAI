# ReflectAI (AAINA) — AI Smart Mirror

ReflectAI is a Raspberry Pi–powered smart mirror. It recognizes the person standing in front of it, greets them, and surfaces weather, calendar, reminders, and outfit/grooming suggestions on a full-screen "mirror" dashboard — with an offline voice assistant and a physical ESP32 sensor unit (temperature/humidity/light/motion + buttons) feeding it in real time.

## Dashboard

![ReflectAI dashboard](docs/screenshot-dashboard.png)

*Add a real screenshot at `docs/screenshot-dashboard.png` (create the `docs/` folder if it doesn't exist) — GitHub will render it here automatically once the file is committed.*

## Features

- **Face-recognition login** — OpenCV YuNet (detection) + SFace (128-D embeddings) recognize a registered user and switch the active profile automatically.
- **Full-screen mirror UI** — clock, personalized greeting, active user/avatar, live weather, calendar, reminders, and system status.
- **ESP32 sensor bridge** — receives temperature, humidity, light, and PIR motion telemetry, plus physical button presses (user switch, privacy toggle) from an Arduino-based sensor unit.
- **Privacy mode** — instantly hides personal info, triggerable from the dashboard or the physical ESP32 button.
- **Wardrobe manager & outfit recommendations** — upload/browse a personal wardrobe; an AI clothing classifier tags category/color and an outfit engine suggests what to wear.
- **Live mirror AI overlays** — real-time clothing and grooming/hair/skin analysis from the camera feed.
- **Offline voice concierge** — spoken morning briefings built from weather, calendar, and outfit data, played back on the mirror.
- **Reminders** — simple per-user reminder CRUD.

## Tech stack

| Layer | Tech |
|---|---|
| Backend | Python, Flask, SQLite |
| Face recognition | OpenCV (YuNet detector + SFace recognizer) |
| Fashion / grooming AI | PyTorch + torchvision (ResNet-18) and open_clip — checkpoints under `models/` |
| Voice assistant | openWakeWord → Vosk (STT) → llama-cpp-python (local LLM) → Piper (TTS), fully offline |
| Hardware bridge | ESP32 (Arduino), `firmware/esp32_firmware/esp32_firmware.ino`, POSTs sensor JSON to the Flask API over Wi-Fi |
| Target hardware | Raspberry Pi 4 (dev happens on a regular PC/Mac first, with mock hardware flags) |

## Project structure

```
ReflectAI/
├── app/
│   ├── assistant/        # voice concierge / briefing builder
│   ├── calendar/         # calendar integration
│   ├── camera/           # camera capture helpers
│   ├── config/           # settings.py (env config), logging_config.py
│   ├── dashboard/        # Flask app — dashboard.py, templates/, assets/
│   ├── database/         # SQLite schema + repositories
│   ├── esp32/            # ESP32 telemetry handling
│   ├── news/
│   ├── recommendations/  # outfit_engine, clothing_classifier, grooming_classifier, mirror_stylist
│   ├── reminders/
│   ├── weather/
│   └── main.py           # bootstrap: DB init, default users, privacy mode
├── data/
│   ├── embeddings/       # registered face embeddings (.npy) + avatar crops
│   ├── wardrobe/         # per-user wardrobe photos
│   └── profiles.json
├── face_engine/          # YuNet + SFace models, register_user.py, download_face_models.py
├── firmware/esp32_firmware/   # Arduino sketch for the sensor unit
├── models/
│   ├── clothing/fashion_classifier.pt
│   └── grooming/grooming_classifier.pt
├── scripts/sync_to_pi.ps1     # syncs the project to a Raspberry Pi over the network
├── tests/
├── voice_engine/          # offline voice pipeline + its own requirements.txt
├── requirements.txt
├── requirements-pi.txt    # Pi-only extras (on top of requirements.txt)
├── run.py                 # bootstrap only (Phase 1 style check)
├── run_dashboard.py       # starts the Flask dashboard
└── .env.example
```

## Getting started

### Prerequisites

- Python 3.9+ and pip
- git
- (optional) a webcam, if you want to test face login/registration locally

### Installation

```bash
git clone https://github.com/MAANIK579/ReflectAI.git
cd ReflectAI

python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pip install -r face_engine/requirements.txt   # needed for face detection/recognition

cp .env.example .env
```

Open `.env` and fill in whatever you need — everything has a sane default for local development:

| Variable | Purpose |
|---|---|
| `ESP32_MOCK_MODE` / `CAMERA_MOCK_MODE` | `true` for dev on a laptop (no real hardware), `false` on the Pi with real hardware |
| `DATABASE_PATH` | SQLite file path (default `data/reflectai.db`) |
| `ESP32_HOST` / `ESP32_PORT` | Only used when `ESP32_MOCK_MODE=false` |
| `WEATHER_API_KEY`, `WEATHER_CITY`, `WEATHER_LAT`, `WEATHER_LON`, `WEATHER_UNITS` | Weather panel |
| `NEWS_API_KEY` | Reserved for the news panel |
| `LOG_LEVEL` | Logging verbosity |
| `WAKE_PHRASE` | Voice assistant wake phrase |

The face-detection models (`face_engine/models/*.onnx`) are already committed to the repo. If they're ever missing, fetch them with:

```bash
python face_engine/download_face_models.py
```

### Running it

```bash
python run_dashboard.py
```

Then open **http://localhost:5000** in a browser (go fullscreen for the actual mirror look). The wardrobe manager lives at **http://localhost:5000/wardrobe**.

This is equivalent to running `python -m app.dashboard.dashboard` directly, which is what the Pi deployment uses.

### Registering a face

With the dashboard already running locally:

```bash
python face_engine/register_user.py
```

Follow the prompts — it captures a face, generates the embedding, and creates the user in ReflectAI's database.

### Voice assistant (optional, fully offline)

```bash
pip install -r voice_engine/requirements.txt
python voice_engine/download_llm.py
python voice_engine/download_models.py
```

### Tests

```bash
python -m unittest discover tests -v
```

## Deploying to a Raspberry Pi

1. Install the extra hardware-dependent packages on the Pi:
   ```bash
   pip install -r requirements-pi.txt
   ```
2. From your dev machine, sync the project over to the Pi:
   ```powershell
   ./scripts/sync_to_pi.ps1 -PiHost <pi-ip> -PiUser <pi-username>
   ```
3. On the Pi, set `ESP32_MOCK_MODE=false` and `CAMERA_MOCK_MODE=false` in `.env`, then start it:
   ```bash
   python -m app.dashboard.dashboard
   ```
   (Wire this up to a systemd service if you want it to start on boot.)

## ESP32 sensor unit

`firmware/esp32_firmware/esp32_firmware.ino` is the Arduino sketch for the companion sensor board (DHT22 temperature/humidity, an LDR light sensor, a PIR motion sensor, and user-select/privacy buttons). It POSTs JSON telemetry to the dashboard's `/api/esp32/state` endpoint over Wi-Fi.
