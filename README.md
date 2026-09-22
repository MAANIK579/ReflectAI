# ReflectAI (AAINA) — Phase 1 + Phase 2

Phase 1: config, logging, database, basic launcher.
Phase 2: full-screen dashboard UI (clock, user profile, system status).

## Setup (on your MacBook)

```bash
cd ReflectAI
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Run Phase 1 only (foundation check)

```bash
python run.py
```

## Run the dashboard (Phase 2)

```bash
python run_dashboard.py
```

Then open **http://localhost:5000** in a browser. You should see the
full-screen mirror layout: clock/date top-left, greeting + active user
name, system status (ESP32/camera mock mode) bottom-left, and a privacy
banner that appears automatically if privacy mode is toggled on in the
database.

For the real mirror-glass look, open it in your browser and go
fullscreen (F11 on most browsers) — true kiosk auto-launch on boot is a
Phase 12 concern once real hardware is involved.

Weather and calendar panels are intentionally empty placeholders right
now (`#weather-slot`, `#calendar-slot` in the HTML) — they're Phase 8,
not built yet.

## Test

```bash
python -m unittest tests.test_foundation -v
```

## What's next (Phase 3)

ESP32 API: receive JSON from the ESP32 (or mock data), display
temperature/humidity/light/motion, receive user-selection and privacy
button events.

