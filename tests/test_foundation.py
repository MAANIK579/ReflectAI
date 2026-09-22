"""
tests/test_foundation.py — Phase 1 tests.

Run with: python -m pytest tests/test_foundation.py -v
(or just: python -m unittest tests.test_foundation)
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestFoundation(unittest.TestCase):
    def setUp(self):
        from app.config.settings import settings
        self.old_db_path = settings.DATABASE_PATH
        self.temp_dir = tempfile.mkdtemp()
        settings.DATABASE_PATH = Path(self.temp_dir) / "test.db"

        from app.database.database import init_db
        init_db()

    def tearDown(self):
        from app.config.settings import settings
        settings.DATABASE_PATH = self.old_db_path
        if hasattr(self, "temp_dir") and os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_database_initializes(self):
        from app.database.database import get_connection
        with get_connection() as conn:
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            table_names = {row["name"] for row in tables}

        expected = {
            "users", "preferences", "face_embeddings", "wardrobe",
            "hairstyles", "clothing", "outfits", "reminders", "settings",
        }
        self.assertTrue(expected.issubset(table_names))

    def test_user_create_and_get(self):
        from app.database.repositories import UserRepository
        UserRepository.create_or_update("user1", "User 1", preferred_style="casual")
        user = UserRepository.get("user1")
        self.assertIsNotNone(user)
        self.assertEqual(user["name"], "User 1")
        self.assertEqual(user["preferred_style"], "casual")

    def test_privacy_mode_setting(self):
        from app.database.repositories import SettingsRepository
        self.assertIsNone(SettingsRepository.get("privacy_mode"))
        SettingsRepository.set("privacy_mode", "true")
        self.assertEqual(SettingsRepository.get("privacy_mode"), "true")

    def test_reminder_lifecycle(self):
        from app.database.repositories import ReminderRepository, UserRepository
        UserRepository.create_or_update("user1", "User 1")
        ReminderRepository.create("user1", "Buy groceries", "2026-09-20T10:00:00")

        reminders = ReminderRepository.list_for_user("user1")
        self.assertEqual(len(reminders), 1)
        self.assertEqual(reminders[0]["title"], "Buy groceries")

        ReminderRepository.complete(reminders[0]["id"])
        active = ReminderRepository.list_for_user("user1")
        self.assertEqual(len(active), 0)


if __name__ == "__main__":
    unittest.main()
