"""
app/database/repositories.py — repository layer.

Per the spec: "Keep database access behind a service/repository layer."
Other modules call these functions; nothing else touches sqlite directly.
"""

from app.database.database import get_connection


class UserRepository:
    @staticmethod
    def create_or_update(user_id: str, name: str, preferred_style: str = None,
                          hair_preference: str = None, temperature_unit: str = "C"):
        with get_connection() as conn:
            conn.execute(
                """
                INSERT INTO users (id, name, preferred_style, hair_preference, temperature_unit)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name=excluded.name,
                    preferred_style=excluded.preferred_style,
                    hair_preference=excluded.hair_preference,
                    temperature_unit=excluded.temperature_unit
                """,
                (user_id, name, preferred_style, hair_preference, temperature_unit),
            )

    @staticmethod
    def get(user_id: str):
        with get_connection() as conn:
            row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
            return dict(row) if row else None

    @staticmethod
    def list_all():
        with get_connection() as conn:
            rows = conn.execute("SELECT * FROM users").fetchall()
            return [dict(r) for r in rows]


class SettingsRepository:
    """Simple key-value store for system state, e.g. privacy mode."""

    @staticmethod
    def set(key: str, value: str):
        with get_connection() as conn:
            conn.execute(
                """
                INSERT INTO settings (key, value) VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value
                """,
                (key, value),
            )

    @staticmethod
    def get(key: str, default: str = None):
        with get_connection() as conn:
            row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
            return row["value"] if row else default


class ReminderRepository:
    @staticmethod
    def create(user_id: str, title: str, datetime_str: str):
        with get_connection() as conn:
            conn.execute(
                "INSERT INTO reminders (user_id, title, datetime) VALUES (?, ?, ?)",
                (user_id, title, datetime_str),
            )

    @staticmethod
    def list_for_user(user_id: str, include_completed: bool = False):
        query = "SELECT * FROM reminders WHERE user_id = ?"
        if not include_completed:
            query += " AND completed = 0"
        with get_connection() as conn:
            rows = conn.execute(query, (user_id,)).fetchall()
            return [dict(r) for r in rows]

    @staticmethod
    def complete(reminder_id: int):
        with get_connection() as conn:
            conn.execute("UPDATE reminders SET completed = 1 WHERE id = ?", (reminder_id,))

    @staticmethod
    def delete(reminder_id: int):
        with get_connection() as conn:
            conn.execute("DELETE FROM reminders WHERE id = ?", (reminder_id,))


class WardrobeRepository:
    """CRUD for user wardrobe items — their actual, photographed clothes."""

    @staticmethod
    def add_item(user_id: str, name: str, category: str, color: str = None,
                 weather: str = "any", temperature_min: float = None,
                 temperature_max: float = None, occasion: str = "casual",
                 image_filename: str = None) -> int:
        with get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO wardrobe
                    (user_id, name, category, color, weather, temperature_min,
                     temperature_max, occasion, image_filename)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (user_id, name, category, color, weather, temperature_min,
                 temperature_max, occasion, image_filename),
            )
            return cursor.lastrowid

    @staticmethod
    def get_item(item_id: int):
        with get_connection() as conn:
            row = conn.execute("SELECT * FROM wardrobe WHERE id = ?", (item_id,)).fetchone()
            return dict(row) if row else None

    @staticmethod
    def list_for_user(user_id: str, category: str = None, active_only: bool = True):
        query = "SELECT * FROM wardrobe WHERE user_id = ?"
        params = [user_id]
        if active_only:
            query += " AND active = 1"
        if category:
            query += " AND category = ?"
            params.append(category)
        query += " ORDER BY created_at DESC"
        with get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]

    @staticmethod
    def update_item(item_id: int, **fields):
        allowed = {"name", "category", "color", "weather", "temperature_min",
                    "temperature_max", "occasion", "image_filename", "active"}
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            return
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [item_id]
        with get_connection() as conn:
            conn.execute(f"UPDATE wardrobe SET {set_clause} WHERE id = ?", values)

    @staticmethod
    def mark_worn(item_id: int, date: str):
        with get_connection() as conn:
            # Only increment if it hasn't already been marked worn on this exact date
            conn.execute(
                """
                UPDATE wardrobe 
                SET times_worn = times_worn + 1, last_worn = ? 
                WHERE id = ? AND (last_worn IS NULL OR last_worn != ?)
                """,
                (date, item_id, date),
            )

    @staticmethod
    def deactivate(item_id: int):
        with get_connection() as conn:
            conn.execute("UPDATE wardrobe SET active = 0 WHERE id = ?", (item_id,))

    @staticmethod
    def activate(item_id: int):
        with get_connection() as conn:
            conn.execute("UPDATE wardrobe SET active = 1 WHERE id = ?", (item_id,))

    @staticmethod
    def delete_item(item_id: int):
        with get_connection() as conn:
            conn.execute("DELETE FROM wardrobe WHERE id = ?", (item_id,))


class OutfitLogRepository:
    """Tracks which outfit was recommended each day to avoid repeats."""

    @staticmethod
    def log_outfit(user_id: str, date: str, top_id: int = None,
                   bottom_id: int = None, footwear_id: int = None,
                   outerwear_id: int = None, temperature: float = None,
                   weather_condition: str = None):
        with get_connection() as conn:
            # Replace if already logged today
            conn.execute("DELETE FROM outfit_log WHERE user_id = ? AND date = ?",
                         (user_id, date))
            conn.execute(
                """
                INSERT INTO outfit_log
                    (user_id, date, top_id, bottom_id, footwear_id,
                     outerwear_id, temperature, weather_condition)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (user_id, date, top_id, bottom_id, footwear_id,
                 outerwear_id, temperature, weather_condition),
            )

    @staticmethod
    def get_today(user_id: str):
        from datetime import date
        today = date.today().isoformat()
        with get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM outfit_log WHERE user_id = ? AND date = ?",
                (user_id, today),
            ).fetchone()
            return dict(row) if row else None

    @staticmethod
    def get_recent(user_id: str, days: int = 7):
        from datetime import date, timedelta
        cutoff = (date.today() - timedelta(days=days)).isoformat()
        with get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM outfit_log WHERE user_id = ? AND date >= ? ORDER BY date DESC",
                (user_id, cutoff),
            ).fetchall()
            return [dict(r) for r in rows]

