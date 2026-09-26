"""
run_dashboard.py — Phase 2: launch the dashboard server.

Run this, then open http://localhost:5000 in a browser to see the
full-screen mirror UI. On the Mac, just open it in a normal browser
window for now — kiosk mode (auto-launching fullscreen on boot) is a
Phase 12 concern, once real hardware is involved.
"""

import os
from app.main import bootstrap
from app.dashboard.dashboard import run_dashboard

if __name__ == "__main__":
    bootstrap()
    # Default to False on Raspberry Pi / production so Werkzeug reloader doesn't
    # fork processes and lock hardware serial ports (/dev/ttyUSB*).
    debug_mode = os.getenv("FLASK_DEBUG", "false").strip().lower() in ("1", "true", "yes")
    run_dashboard(debug=debug_mode)

