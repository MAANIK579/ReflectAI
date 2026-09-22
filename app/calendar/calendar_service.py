"""
app/calendar/calendar_service.py — Calendar & schedule resolution.

Resolves today's schedule for the active recognized user by combining:
1. Profiles defined in data/profiles.json (from face registration)
2. Active reminders in SQLite (reminders table)
3. Smart default agenda for guest users
"""

import json
import logging
import os
import re
from typing import Any, Dict, List

from app.config.settings import settings
from app.database.repositories import ReminderRepository

logger = logging.getLogger("reflectai.calendar")

DEFAULT_GUEST_SCHEDULE = [
    {"time": "10:30 AM", "title": "ReflectAI Live Demo"},
    {"time": "02:00 PM", "title": "AI & Smart Mirror Showcase"},
    {"time": "05:30 PM", "title": "Project Evaluation"},
]


class CalendarService:
    @staticmethod
    def _load_profiles() -> Dict[str, Any]:
        profiles_path = settings.DATA_DIR / "profiles.json"
        if profiles_path.exists():
            try:
                with open(profiles_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not read profiles.json: {e}")
        return {}

    @classmethod
    def get_events_for_user(cls, user_id: str) -> List[Dict[str, str]]:
        events: List[Dict[str, str]] = []
        user_key = (user_id or "").lower().strip()

        # 1. Look up user in profiles.json
        profiles = cls._load_profiles()
        user_profile = profiles.get(user_key)

        if user_profile and "calendar" in user_profile:
            raw_entries = user_profile["calendar"]
            for entry in raw_entries:
                parsed = cls._parse_event_string(entry)
                if parsed:
                    events.append(parsed)

        # 2. Look up reminders in database
        try:
            db_reminders = ReminderRepository.list_for_user(user_key, include_completed=False)
            for reminder in db_reminders:
                events.append({
                    "time": cls._format_time_or_default(reminder.get("datetime")),
                    "title": reminder.get("title", ""),
                })
        except Exception as e:
            logger.warning(f"Error querying reminders for {user_key}: {e}")

        # 3. If no events found or user is guest, provide default agenda
        if not events:
            if user_key in ("guest", "user1", "user2", ""):
                events = list(DEFAULT_GUEST_SCHEDULE)
            else:
                events = [
                    {"time": "Today", "title": "No more upcoming events"},
                ]

        return events

    @staticmethod
    def _parse_event_string(entry: str) -> Dict[str, str]:
        """Parses strings like '10:30 AM · Project Review' or '4:00 PM * Gym'"""
        if not entry:
            return {"time": "Today", "title": ""}

        # Split on ·, *, -, or |
        parts = re.split(r"\s*[·\*\-\|]\s*", entry, maxsplit=1)
        if len(parts) == 2:
            return {"time": parts[0].strip(), "title": parts[1].strip()}

        return {"time": "Today", "title": entry.strip()}

    @staticmethod
    def _format_time_or_default(datetime_str: str) -> str:
        if not datetime_str:
            return "Reminder"
        # If ISO format (e.g. 2026-09-20T14:30:00), extract time
        if "T" in datetime_str:
            time_part = datetime_str.split("T")[1][:5]
            return time_part
        return datetime_str
