"""
run_dashboard.py — Phase 2: launch the dashboard server.

Run this, then open http://localhost:5000 in a browser to see the
full-screen mirror UI. On the Mac, just open it in a normal browser
window for now — kiosk mode (auto-launching fullscreen on boot) is a
Phase 12 concern, once real hardware is involved.
"""

from app.main import bootstrap
from app.dashboard.dashboard import run_dashboard

if __name__ == "__main__":
    bootstrap()
    run_dashboard(debug=True)
