"""
tests/test_voice_brain.py — Comprehensive Unit Tests for VoiceBrain & Omnipresent Voice Assistant.

Verifies:
- User profile & identity queries ("Who am I", "Who is registered", "Switch user to...")
- Reminders voice CRUD actions (add, list, complete, delete)
- Wardrobe & clothing queries (item counts, wear frequency, last worn, most worn, outfit recommendations, outfit log)
- ESP32 indoor climate queries (temperature, humidity, motion)
- Privacy mode toggling and status checking
- Camera vision queries (detected clothing, grooming scan)
- REST API endpoints: POST /api/voice/ask, GET /api/voice/context
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestVoiceBrain(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from app.config.settings import settings
        from app.database.database import init_db
        settings.DATABASE_PATH = settings.DATA_DIR / "test_voice_brain.db"
        if settings.DATABASE_PATH.exists():
            settings.DATABASE_PATH.unlink()
        init_db()

        from app.dashboard.dashboard import app
        cls.app = app
        cls.client = app.test_client()

    @classmethod
    def tearDownClass(cls):
        from app.config.settings import settings
        if settings.DATABASE_PATH.exists():
            try:
                settings.DATABASE_PATH.unlink()
            except Exception:
                pass

    def setUp(self):
        from app.database.database import get_connection
        from app.database.repositories import (
            OutfitLogRepository,
            ReminderRepository,
            SettingsRepository,
            UserRepository,
            WardrobeRepository,
        )

        with get_connection() as conn:
            conn.execute("DELETE FROM wardrobe")
            conn.execute("DELETE FROM outfit_log")
            conn.execute("DELETE FROM reminders")
            conn.execute("DELETE FROM users")
            conn.execute("DELETE FROM settings")

        # Seed test user and settings
        UserRepository.create_or_update("alex", "Alex", preferred_style="Smart Casual", hair_preference="Side Part", temperature_unit="C")
        UserRepository.create_or_update("sarah", "Sarah", preferred_style="Streetwear", hair_preference="Ponytail", temperature_unit="C")
        SettingsRepository.set("active_user", "alex")
        SettingsRepository.set("privacy_mode", "false")

        # Seed wardrobe
        self.top_id = WardrobeRepository.add_item("alex", "Navy Oxford Shirt", category="top", color="Navy")
        WardrobeRepository.update_item(self.top_id, times_worn=5, last_worn="2026-09-24")

        self.bot_id = WardrobeRepository.add_item("alex", "Khaki Chinos", category="bottom", color="Beige")
        WardrobeRepository.update_item(self.bot_id, times_worn=3, last_worn="2026-09-24")

        self.jacket_id = WardrobeRepository.add_item("alex", "Charcoal Blazer", category="outerwear", color="Grey")
        WardrobeRepository.update_item(self.jacket_id, times_worn=8, last_worn="2026-09-20")

        # Seed outfit log
        OutfitLogRepository.log_outfit("alex", date="2026-09-24", top_id=self.top_id, bottom_id=self.bot_id, weather_condition="Clear")

        # Seed reminders
        ReminderRepository.create("alex", "Team Sprint Meeting", "10:00 AM")

    def test_user_profile_and_switching(self):
        from app.assistant import VoiceBrain
        from app.database.repositories import SettingsRepository

        # Who am I
        res = VoiceBrain.process_query("Who am I?", user_id="alex")
        self.assertEqual(res["status"], "ok")
        self.assertIn("Alex", res["reply"])
        self.assertIn("Smart Casual", res["reply"])
        self.assertEqual(res["action"], "get_user_profile")

        # Who is registered
        res_users = VoiceBrain.process_query("Who is registered on this mirror?", user_id="alex")
        self.assertEqual(res_users["status"], "ok")
        self.assertIn("Alex", res_users["reply"])
        self.assertIn("Sarah", res_users["reply"])
        self.assertEqual(res_users["action"], "list_users")

        # Switch user
        res_switch = VoiceBrain.process_query("Switch user to Sarah", user_id="alex")
        self.assertEqual(res_switch["status"], "ok")
        self.assertIn("Switched active user to Sarah", res_switch["reply"])
        self.assertEqual(res_switch["action"], "switch_user")
        self.assertEqual(SettingsRepository.get("active_user"), "sarah")

    def test_reminders_voice_crud(self):
        from app.assistant import VoiceBrain
        from app.database.repositories import ReminderRepository

        # 1. List reminders
        res_list = VoiceBrain.process_query("What are my reminders?", user_id="alex")
        self.assertEqual(res_list["status"], "ok")
        self.assertIn("Team Sprint Meeting", res_list["reply"])
        self.assertEqual(res_list["action"], "list_reminders")

        # 2. Add reminder via voice
        res_add = VoiceBrain.process_query("Add reminder to buy groceries at 6 PM", user_id="alex")
        self.assertEqual(res_add["status"], "ok")
        self.assertIn("buy groceries", res_add["reply"].lower())
        self.assertEqual(res_add["action"], "add_reminder")

        # Verify added to database
        db_rems = ReminderRepository.list_for_user("alex", include_completed=False)
        self.assertTrue(any("groceries" in r["title"].lower() for r in db_rems))

        # 3. Complete reminder
        res_comp = VoiceBrain.process_query("Mark reminder buy groceries as done", user_id="alex")
        self.assertEqual(res_comp["status"], "ok")
        self.assertEqual(res_comp["action"], "complete_reminder")

        # Verify completed in DB
        pending = ReminderRepository.list_for_user("alex", include_completed=False)
        self.assertFalse(any("groceries" in r["title"].lower() for r in pending))

        # 4. Delete reminder
        res_del = VoiceBrain.process_query("Delete reminder Team Sprint Meeting", user_id="alex")
        self.assertEqual(res_del["status"], "ok")
        self.assertEqual(res_del["action"], "delete_reminder")

        all_rems = ReminderRepository.list_for_user("alex", include_completed=True)
        self.assertFalse(any("Team Sprint Meeting" in r["title"] for r in all_rems))

    def test_wardrobe_queries(self):
        from app.assistant import VoiceBrain

        # 1. List wardrobe
        res = VoiceBrain.process_query("What clothes do I have?", user_id="alex")
        self.assertEqual(res["status"], "ok")
        self.assertIn("3 items", res["reply"])
        self.assertIn("Navy Oxford Shirt", res["reply"])

        # 2. Count shirts
        res_shirts = VoiceBrain.process_query("How many shirts do I have?", user_id="alex")
        self.assertEqual(res_shirts["status"], "ok")
        self.assertIn("1 top", res_shirts["reply"])
        self.assertIn("Navy Oxford Shirt", res_shirts["reply"])

        # 3. Item wear count
        res_wear = VoiceBrain.process_query("How many times have I worn my Charcoal Blazer?", user_id="alex")
        self.assertEqual(res_wear["status"], "ok")
        self.assertIn("8 times", res_wear["reply"])
        self.assertIn("2026-09-20", res_wear["reply"])

        # 4. Last worn lookup
        res_last = VoiceBrain.process_query("When did I last wear my Khaki Chinos?", user_id="alex")
        self.assertEqual(res_last["status"], "ok")
        self.assertIn("2026-09-24", res_last["reply"])

        # 5. Most worn item
        res_most = VoiceBrain.process_query("What is my most worn item?", user_id="alex")
        self.assertEqual(res_most["status"], "ok")
        self.assertIn("Charcoal Blazer", res_most["reply"])
        self.assertIn("8 times", res_most["reply"])

        # 6. Outfit log history
        res_hist = VoiceBrain.process_query("What did I wear yesterday?", user_id="alex")
        self.assertEqual(res_hist["status"], "ok")
        self.assertIn("Navy Oxford Shirt", res_hist["reply"])

    def test_esp32_indoor_climate(self):
        from app.assistant import VoiceBrain

        esp_data = {"temperature": 23.4, "humidity": 52, "motion": True}

        res_temp = VoiceBrain.process_query("What is the room temperature?", user_id="alex", esp32_data=esp_data)
        self.assertEqual(res_temp["status"], "ok")
        self.assertIn("23.4", res_temp["reply"])
        self.assertEqual(res_temp["action"], "room_temperature")

        res_hum = VoiceBrain.process_query("What is the room humidity?", user_id="alex", esp32_data=esp_data)
        self.assertEqual(res_hum["status"], "ok")
        self.assertIn("52 percent", res_hum["reply"])

        res_motion = VoiceBrain.process_query("Is there motion detected?", user_id="alex", esp32_data=esp_data)
        self.assertEqual(res_motion["status"], "ok")
        self.assertIn("Motion is currently detected", res_motion["reply"])

    def test_privacy_mode_voice_control(self):
        from app.assistant import VoiceBrain
        from app.database.repositories import SettingsRepository

        # Turn on
        res_on = VoiceBrain.process_query("Turn on privacy mode", user_id="alex")
        self.assertEqual(res_on["status"], "ok")
        self.assertTrue(res_on["data"]["privacy_mode"])
        self.assertEqual(SettingsRepository.get("privacy_mode"), "true")

        # Check status
        res_check = VoiceBrain.process_query("Is privacy mode on?", user_id="alex")
        self.assertIn("enabled", res_check["reply"])

        # Turn off
        res_off = VoiceBrain.process_query("Turn off privacy mode", user_id="alex")
        self.assertEqual(res_off["status"], "ok")
        self.assertFalse(res_off["data"]["privacy_mode"])
        self.assertEqual(SettingsRepository.get("privacy_mode"), "false")

    def test_camera_vision_queries(self):
        from app.assistant import VoiceBrain

        stylist = {
            "status": "ok",
            "detected": {"name": "Navy Oxford Shirt", "category": "top"},
            "advice": "Complements your style well."
        }
        grooming = {
            "status": "ok",
            "hair": "neat side part",
            "facial_hair": "clean shaven",
            "skin": "clear",
            "advice": "Looking sharp and ready."
        }

        res_clothes = VoiceBrain.process_query("What am I wearing?", user_id="alex", stylist_data=stylist)
        self.assertEqual(res_clothes["status"], "ok")
        self.assertIn("Navy Oxford Shirt", res_clothes["reply"])

        res_groom = VoiceBrain.process_query("How is my grooming?", user_id="alex", grooming_data=grooming)
        self.assertEqual(res_groom["status"], "ok")
        self.assertIn("neat side part", res_groom["reply"])
        self.assertIn("clean shaven", res_groom["reply"])

    def test_api_voice_endpoints(self):
        # 1. POST /api/voice/ask
        res = self.client.post("/api/voice/ask", json={"query": "Who am I?", "user_id": "alex"})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("Alex", data["reply"])
        self.assertEqual(data["action"], "get_user_profile")

        # 2. Empty query returns 400
        res_bad = self.client.post("/api/voice/ask", json={"query": ""})
        self.assertEqual(res_bad.status_code, 400)

        # 3. GET /api/voice/context returns full knowledge snapshot
        res_ctx = self.client.get("/api/voice/context?user_id=alex")
        self.assertEqual(res_ctx.status_code, 200)
        data_ctx = res_ctx.get_json()
        self.assertEqual(data_ctx["status"], "ok")
        ctx = data_ctx["context"]
        self.assertEqual(ctx["active_user"]["id"], "alex")
        self.assertEqual(ctx["wardrobe"]["total_items"], 3)
        self.assertIn("Alex", ctx["registered_users"])
        self.assertIn("Sarah", ctx["registered_users"])


if __name__ == "__main__":
    unittest.main()
