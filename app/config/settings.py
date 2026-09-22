"""
app/config/settings.py — single source of truth for configuration.

Every other module reads config from here, never from os.environ directly
and never hard-codes a value that belongs in .env. This is what makes the
Mac (mock) vs Raspberry Pi (real hardware) switch a config change instead
of a code change.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Project root = two levels up from this file (app/config/settings.py -> app -> root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _bool(env_var: str, default: bool) -> bool:
    val = os.getenv(env_var)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


class Settings:
    # --- Paths ---
    PROJECT_ROOT = PROJECT_ROOT
    DATA_DIR = PROJECT_ROOT / "data"
    LOGS_DIR = PROJECT_ROOT / "logs"
    MODELS_DIR = PROJECT_ROOT / "models"

    # --- Mode flags ---
    ESP32_MOCK_MODE = _bool("ESP32_MOCK_MODE", default=True)
    CAMERA_MOCK_MODE = _bool("CAMERA_MOCK_MODE", default=True)

    # --- Database ---
    DATABASE_PATH = PROJECT_ROOT / os.getenv("DATABASE_PATH", "data/reflectai.db")

    # --- ESP32 ---
    ESP32_HOST = os.getenv("ESP32_HOST", "")
    ESP32_PORT = os.getenv("ESP32_PORT", "")

    # --- External services ---
    WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "")
    WEATHER_CITY = os.getenv("WEATHER_CITY", "New Delhi")
    WEATHER_LAT = float(os.getenv("WEATHER_LAT", "28.6139"))
    WEATHER_LON = float(os.getenv("WEATHER_LON", "77.2090"))
    WEATHER_UNITS = os.getenv("WEATHER_UNITS", "celsius")
    NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")

    # --- App ---
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    WAKE_PHRASE = os.getenv("WAKE_PHRASE", "Hey Aaina")


settings = Settings()
