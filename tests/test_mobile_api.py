"""
tests/test_mobile_api.py — Unit tests for ReflectAI Mobile Companion endpoints.
"""

import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config.settings import settings
from app.dashboard.dashboard import app
from app.database.database import init_db
from app.database.repositories import UserRepository, ReminderRepository, SettingsRepository


class TestMobileAPI(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.old_db = settings.DATABASE_PATH
        self.old_data_dir = settings.DATA_DIR
        settings.DATABASE_PATH = Path(self.temp_dir) / "test.db"
        settings.DATA_DIR = Path(self.temp_dir) / "data"
        settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
        init_db()
        self.client = app.test_client()

    def tearDown(self):
        settings.DATABASE_PATH = self.old_db
        settings.DATA_DIR = self.old_data_dir
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_routes_accessible(self):
        res1 = self.client.get("/wardrobe")
        self.assertEqual(res1.status_code, 200)
        self.assertIn(b"ReflectAI", res1.data)

        res2 = self.client.get("/app")
        self.assertEqual(res2.status_code, 200)
        self.assertIn(b"ReflectAI", res2.data)

    def test_user_list_api(self):
        UserRepository.create_or_update("sam", "Sam", preferred_style="Streetwear")
        res = self.client.get("/api/users")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("users", data)
        user_ids = [u["id"] for u in data["users"]]
        self.assertIn("sam", user_ids)

    def test_user_register_and_delete_api(self):
        # 1. Register without photo
        res = self.client.post("/api/user/register", data={
            "name": "Jordan Lee",
            "preferred_style": "Minimalist",
            "hair_preference": "Buzz Cut",
            "switch_now": "true",
        })
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["user_id"], "jordan_lee")
        self.assertFalse(data["has_face"])

        # Check DB
        user = UserRepository.get("jordan_lee")
        self.assertIsNotNone(user)
        self.assertEqual(user["name"], "Jordan Lee")

        # Check active user switched
        self.assertEqual(SettingsRepository.get("active_user"), "jordan_lee")

        # 2. Delete user
        del_res = self.client.post("/api/user/delete/jordan_lee")
        self.assertEqual(del_res.status_code, 200)
        self.assertIsNone(UserRepository.get("jordan_lee"))

    def test_avatar_svg_fallback(self):
        UserRepository.create_or_update("alex", "Alex")
        res = self.client.get("/avatar/alex")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.headers.get("Content-Type"), "image/svg+xml; charset=utf-8")
        self.assertIn(b"<svg", res.data)
        self.assertIn(b"A", res.data)

    def test_privacy_toggle_api(self):
        SettingsRepository.set("privacy_mode", "false")
        res = self.client.post("/api/mirror/privacy/toggle")
        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.get_json()["privacy_mode"])

        res2 = self.client.post("/api/mirror/privacy/toggle")
        self.assertEqual(res2.status_code, 200)
        self.assertFalse(res2.get_json()["privacy_mode"])

    def test_reminders_crud_api(self):
        # Create reminder
        res = self.client.post("/api/reminders", json={
            "user_id": "test_user",
            "title": "Evening Workout",
            "datetime": "06:00 PM"
        })
        self.assertEqual(res.status_code, 200)

        # List reminders
        res_list = self.client.get("/api/reminders?user_id=test_user")
        self.assertEqual(res_list.status_code, 200)
        reminders = res_list.get_json()["reminders"]
        self.assertEqual(len(reminders), 1)
        self.assertEqual(reminders[0]["title"], "Evening Workout")

        # Delete reminder
        rem_id = reminders[0]["id"]
        res_del = self.client.delete(f"/api/reminders/{rem_id}")
        self.assertEqual(res_del.status_code, 200)

        # Confirm deleted
        res_list2 = self.client.get("/api/reminders?user_id=test_user")
        self.assertEqual(len(res_list2.get_json()["reminders"]), 0)


if __name__ == "__main__":
    unittest.main()
