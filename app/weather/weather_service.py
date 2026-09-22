"""
app/weather/weather_service.py — Weather data fetching and caching.

Uses Open-Meteo's free weather API (no API key required) as primary source,
with in-memory TTL caching and graceful offline fallback.
"""

import json
import logging
import time
import urllib.request
from typing import Any, Dict, Optional

from app.config.settings import settings

logger = logging.getLogger("reflectai.weather")

# WMO Weather interpretation codes (WW) mapped to (Description, Icon ID)
WMO_CODE_MAP = {
    0: ("Clear sky", "clear"),
    1: ("Mainly clear", "clear"),
    2: ("Partly cloudy", "partly-cloudy"),
    3: ("Overcast", "cloudy"),
    45: ("Foggy", "fog"),
    48: ("Depositing rime fog", "fog"),
    51: ("Light drizzle", "drizzle"),
    53: ("Moderate drizzle", "drizzle"),
    55: ("Dense drizzle", "drizzle"),
    56: ("Light freezing drizzle", "drizzle"),
    57: ("Dense freezing drizzle", "drizzle"),
    61: ("Slight rain", "rain"),
    63: ("Moderate rain", "rain"),
    65: ("Heavy rain", "rain"),
    66: ("Freezing rain", "rain"),
    67: ("Heavy freezing rain", "rain"),
    71: ("Slight snow fall", "snow"),
    73: ("Moderate snow fall", "snow"),
    75: ("Heavy snow fall", "snow"),
    77: ("Snow grains", "snow"),
    80: ("Slight rain showers", "showers"),
    81: ("Moderate rain showers", "showers"),
    82: ("Violent rain showers", "showers"),
    85: ("Slight snow showers", "snow"),
    86: ("Heavy snow showers", "snow"),
    95: ("Thunderstorm", "thunderstorm"),
    96: ("Thunderstorm with slight hail", "thunderstorm"),
    99: ("Thunderstorm with heavy hail", "thunderstorm"),
}

CACHE_TTL_SECONDS = 900  # 15 minutes


class WeatherService:
    _cached_data: Optional[Dict[str, Any]] = None
    _last_fetch_time: float = 0.0

    @classmethod
    def get_weather(cls, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Returns the current weather. Uses cached data if within TTL.
        Falls back to realistic mock data if network or service is unreachable.
        """
        now = time.time()
        if not force_refresh and cls._cached_data and (now - cls._last_fetch_time < CACHE_TTL_SECONDS):
            return cls._cached_data

        try:
            data = cls._fetch_from_open_meteo(settings.WEATHER_LAT, settings.WEATHER_LON)
            cls._cached_data = data
            cls._last_fetch_time = now
            logger.info("Successfully fetched live weather from Open-Meteo")
            return data
        except Exception as e:
            logger.warning(f"Failed to fetch live weather: {e}. Using fallback.")
            if cls._cached_data:
                # Return expired cache if available
                return cls._cached_data
            return cls._get_offline_fallback()

    @classmethod
    def _fetch_from_open_meteo(cls, lat: float, lon: float) -> Dict[str, Any]:
        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}&"
            f"current=temperature_2m,relative_humidity_2m,apparent_temperature,is_day,weather_code,wind_speed_10m&"
            f"daily=weather_code,temperature_2m_max,temperature_2m_min&"
            f"timezone=auto"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "ReflectAI-SmartMirror/1.0"})
        with urllib.request.urlopen(req, timeout=4) as response:
            payload = json.loads(response.read().decode("utf-8"))

        current = payload.get("current", {})
        daily = payload.get("daily", {})

        weather_code = current.get("weather_code", 0)
        condition_desc, icon_type = WMO_CODE_MAP.get(weather_code, ("Clear sky", "clear"))

        high_list = daily.get("temperature_2m_max", [])
        low_list = daily.get("temperature_2m_min", [])
        high = round(high_list[0]) if high_list else None
        low = round(low_list[0]) if low_list else None

        temp = round(current.get("temperature_2m", 25))
        apparent_temp = round(current.get("apparent_temperature", temp))
        humidity = round(current.get("relative_humidity_2m", 50))
        is_day = current.get("is_day", 1)

        return {
            "city": settings.WEATHER_CITY,
            "temperature": temp,
            "unit": "°C",
            "condition": condition_desc,
            "icon": icon_type,
            "high": high,
            "low": low,
            "humidity": humidity,
            "apparent_temperature": apparent_temp,
            "is_day": is_day,
            "mock": False,
            "updated_at": time.strftime("%H:%M"),
        }

    @classmethod
    def _get_offline_fallback(cls) -> Dict[str, Any]:
        return {
            "city": settings.WEATHER_CITY,
            "temperature": 26,
            "unit": "°C",
            "condition": "Partly cloudy",
            "icon": "partly-cloudy",
            "high": 29,
            "low": 21,
            "humidity": 48,
            "apparent_temperature": 27,
            "is_day": 1,
            "mock": True,
            "updated_at": time.strftime("%H:%M"),
        }
