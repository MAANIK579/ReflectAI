"""
tests/test_weather_calendar.py — Tests for Weather & Calendar widgets.

Run with: python -m unittest tests.test_weather_calendar -v
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestWeatherCalendar(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        os.environ["DATABASE_PATH"] = os.path.join(self.temp_dir, "test.db")

        # Refresh modules
        for mod in list(sys.modules.keys()):
            if mod.startswith("app."):
                del sys.modules[mod]

        from app.database.database import init_db
        init_db()

        from app.dashboard.dashboard import app
        self.app = app
        self.client = app.test_client()

    def test_weather_service_offline_fallback(self):
        from app.weather.weather_service import WeatherService

        WeatherService._cached_data = None
        WeatherService._last_fetch_time = 0

        # Simulate network failure
        with patch.object(WeatherService, "_fetch_from_open_meteo", side_effect=Exception("Network error")):
            data = WeatherService.get_weather(force_refresh=True)
            self.assertIsNotNone(data)
            self.assertIn("temperature", data)
            self.assertIn("condition", data)
            self.assertTrue(data.get("mock"))

    def test_weather_service_caching(self):
        from app.weather.weather_service import WeatherService

        mock_payload = {
            "city": "Test City",
            "temperature": 22,
            "unit": "°C",
            "condition": "Clear sky",
            "icon": "clear",
            "high": 25,
            "low": 18,
            "humidity": 40,
            "apparent_temperature": 22,
            "is_day": 1,
            "mock": False,
            "updated_at": "12:00",
        }

        with patch.object(WeatherService, "_fetch_from_open_meteo", return_value=mock_payload) as mock_fetch:
            # First fetch
            res1 = WeatherService.get_weather(force_refresh=True)
            self.assertEqual(res1["temperature"], 22)
            self.assertEqual(mock_fetch.call_count, 1)

            # Second fetch without force_refresh within TTL
            res2 = WeatherService.get_weather(force_refresh=False)
            self.assertEqual(res2["temperature"], 22)
            self.assertEqual(mock_fetch.call_count, 1)  # Cached, no second API call

    def test_calendar_service_profiles_and_guest(self):
        from app.calendar.calendar_service import CalendarService

        # Test guest user schedule
        guest_events = CalendarService.get_events_for_user("guest")
        self.assertIsInstance(guest_events, list)
        self.assertGreater(len(guest_events), 0)
        self.assertIn("title", guest_events[0])
        self.assertIn("time", guest_events[0])

        # Test registered user with profiles.json (maanik)
        maanik_events = CalendarService.get_events_for_user("maanik")
        self.assertIsInstance(maanik_events, list)
        self.assertGreater(len(maanik_events), 0)
        titles = [e["title"] for e in maanik_events]
        self.assertTrue(any("Project Review" in t for t in titles))

    def test_calendar_service_with_reminders(self):
        from app.calendar.calendar_service import CalendarService
        from app.database.repositories import ReminderRepository, UserRepository

        UserRepository.create_or_update("testuser", "Test User")
        ReminderRepository.create("testuser", "Submit Report", "2026-09-20T16:00:00")

        events = CalendarService.get_events_for_user("testuser")
        self.assertIsInstance(events, list)
        titles = [e["title"] for e in events]
        self.assertIn("Submit Report", titles)

    def test_api_weather_endpoint(self):
        res = self.client.get("/api/weather")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("temperature", data)
        self.assertIn("condition", data)
        self.assertIn("city", data)
        self.assertIn("icon", data)

    def test_api_calendar_endpoint(self):
        res = self.client.get("/api/calendar?user_id=maanik")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data.get("user_id"), "maanik")
        self.assertIsInstance(data.get("events"), list)

    def test_api_status_includes_user_id(self):
        res = self.client.get("/api/status")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("id", data["user"])
        self.assertIn("name", data["user"])


if __name__ == "__main__":
    unittest.main()
