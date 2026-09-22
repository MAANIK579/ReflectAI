"""
tests/test_concierge.py — Unit tests for Multimodal Voice Concierge & Briefing feature.

Run with: python -m unittest tests.test_concierge -v
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestVoiceConcierge(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        os.environ["DATABASE_PATH"] = os.path.join(self.temp_dir, "test.db")

        from app.database.database import init_db
        init_db()

        from app.database.repositories import UserRepository
        UserRepository.create_or_update("testuser", "Test User")

        from app.dashboard.dashboard import app
        self.app = app
        self.client = app.test_client()

    def test_build_briefing_structure(self):
        from app.assistant import VoiceConciergeService

        briefing = VoiceConciergeService.build_briefing(
            user_id="testuser",
            weather_data={
                "temperature": 23.5,
                "condition": "Partly cloudy",
                "city": "London",
                "high": 25.0,
                "low": 17.0,
            },
            esp32_data={
                "temperature": 21.8,
                "humidity": 45,
            },
            stylist_data={
                "advice": "Navy blazer looks sharp with grey trousers."
            },
        )

        self.assertEqual(briefing["status"], "ok")
        self.assertEqual(briefing["user_name"], "Test User")
        self.assertIn("Test User", briefing["full_text"])
        self.assertIn("London", briefing["full_text"])
        self.assertIn("24 degrees", briefing["full_text"])
        self.assertIn("Navy blazer", briefing["full_text"])
        self.assertIn("21.8 degrees", briefing["full_text"])

        # Check section keys
        for key in ["greeting", "weather", "schedule", "style", "indoor"]:
            self.assertIn(key, briefing["sections"])

    def test_synthesize_to_wav(self):
        from app.assistant import VoiceConciergeService

        wav_bytes = VoiceConciergeService.synthesize_to_wav("Hello from ReflectAI test.")
        self.assertIsInstance(wav_bytes, bytes)
        self.assertGreater(len(wav_bytes), 44)
        self.assertEqual(wav_bytes[:4], b"RIFF")
        self.assertEqual(wav_bytes[8:12], b"WAVE")

    def test_api_voice_briefing(self):
        res = self.client.get("/api/voice/briefing?user_id=testuser")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data.get("status"), "ok")
        self.assertIn("full_text", data)
        self.assertIn("sections", data)

    def test_api_voice_briefing_audio(self):
        res = self.client.get("/api/voice/briefing/audio?user_id=testuser")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.mimetype, "audio/wav")
        self.assertGreater(len(res.data), 44)
        self.assertEqual(res.data[:4], b"RIFF")

    def test_api_voice_briefing_play_and_stop(self):
        # Target phone
        res_phone = self.client.post(
            "/api/voice/briefing/play",
            json={"user_id": "testuser", "target": "phone"}
        )
        self.assertEqual(res_phone.status_code, 200)
        data_phone = res_phone.get_json()
        self.assertEqual(data_phone.get("status"), "ok")
        self.assertEqual(data_phone.get("target"), "phone")

        # Stop command
        res_stop = self.client.post("/api/voice/stop")
        self.assertEqual(res_stop.status_code, 200)
        data_stop = res_stop.get_json()
        self.assertEqual(data_stop.get("status"), "ok")

    def test_auto_greeting_on_motion_and_face_recognition(self):
        # 1. ESP32 detects motion
        res_esp = self.client.post("/api/esp32/state", json={"motion": True, "temperature": 22.0})
        self.assertEqual(res_esp.status_code, 200)

        # 2. Camera confirms face recognition for registered user
        res_face = self.client.post("/api/user/active", json={"userId": "testuser", "name": "Test User"})
        self.assertEqual(res_face.status_code, 200)
        data_face = res_face.get_json()
        self.assertEqual(data_face.get("status"), "ok")
        self.assertTrue(data_face.get("greeted"))

        # 3. Subsequent immediate face detection respects cooldown
        res_face_cooldown = self.client.post("/api/user/active", json={"userId": "testuser", "name": "Test User"})
        self.assertFalse(res_face_cooldown.get_json().get("greeted"))


if __name__ == "__main__":
    unittest.main()
