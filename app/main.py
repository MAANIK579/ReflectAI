"""
app/main.py — application entry point.

Phase 1 scope only: verify configuration loads, logging works, and the
database initializes correctly. Later phases will add the dashboard,
camera, ESP32 connection, recommendations, and voice assistant on top
of this foundation.
"""

import logging

from app.config.logging_config import setup_logging
from app.config.settings import settings
from app.database.database import init_db
from app.database.repositories import SettingsRepository, UserRepository

logger = logging.getLogger("reflectai")


def bootstrap():
    setup_logging()
    logger.info("=" * 50)
    logger.info("ReflectAI (AAINA) starting up")
    logger.info("=" * 50)

    logger.info(f"Project root: {settings.PROJECT_ROOT}")
    logger.info(f"ESP32 mock mode: {settings.ESP32_MOCK_MODE}")
    logger.info(f"Camera mock mode: {settings.CAMERA_MOCK_MODE}")
    logger.info(f"Database path: {settings.DATABASE_PATH}")

    init_db()
    logger.info("Database initialized")

    # Seed a default privacy state if none exists yet, so downstream
    # phases (ESP32, camera) always have a defined value to check.
    if SettingsRepository.get("privacy_mode") is None:
        SettingsRepository.set("privacy_mode", "false")
        logger.info("Privacy mode initialized to: false")

    # Seed the three default users from the spec if they don't exist yet.
    default_users = [
        ("user1", "User 1"),
        ("user2", "User 2"),
        ("guest", "Guest"),
    ]
    for user_id, name in default_users:
        if UserRepository.get(user_id) is None:
            UserRepository.create_or_update(user_id, name)
            logger.info(f"Created default user: {name}")

    logger.info("Phase 1 foundation ready.")
    logger.info(f"Current privacy mode: {SettingsRepository.get('privacy_mode')}")
    logger.info(f"Registered users: {[u['name'] for u in UserRepository.list_all()]}")


if __name__ == "__main__":
    bootstrap()
