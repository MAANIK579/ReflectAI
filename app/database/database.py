"""
app/database/database.py — SQLite connection + schema setup.

Per the spec: keep database access behind a service/repository layer
(see repositories.py) — nothing outside this module should write raw SQL.
"""

import sqlite3
from contextlib import contextmanager

from app.config.settings import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    preferred_style TEXT,
    hair_preference TEXT,
    temperature_unit TEXT DEFAULT 'C',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS preferences (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

CREATE TABLE IF NOT EXISTS face_embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    embedding BLOB NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

CREATE TABLE IF NOT EXISTS wardrobe (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    name TEXT NOT NULL,
    category TEXT NOT NULL,
    color TEXT,
    weather TEXT DEFAULT 'any',
    temperature_min REAL,
    temperature_max REAL,
    occasion TEXT DEFAULT 'casual',
    image_filename TEXT,
    last_worn TEXT,
    times_worn INTEGER DEFAULT 0,
    active INTEGER DEFAULT 1,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

CREATE TABLE IF NOT EXISTS hairstyles (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    length TEXT,
    texture TEXT,
    occasion TEXT,
    face_shapes TEXT,
    maintenance TEXT,
    description TEXT
);

CREATE TABLE IF NOT EXISTS clothing (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT,
    color TEXT,
    weather TEXT,
    temperature_min REAL,
    temperature_max REAL,
    occasion TEXT,
    season TEXT,
    style TEXT
);

CREATE TABLE IF NOT EXISTS outfits (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT,
    top_id TEXT,
    bottom_id TEXT,
    footwear_id TEXT,
    outerwear_id TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS outfit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    date TEXT NOT NULL,
    top_id INTEGER,
    bottom_id INTEGER,
    footwear_id INTEGER,
    outerwear_id INTEGER,
    temperature REAL,
    weather_condition TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

CREATE TABLE IF NOT EXISTS reminders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    title TEXT NOT NULL,
    datetime TEXT NOT NULL,
    completed INTEGER DEFAULT 0,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""


def init_db():
    settings.DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.DATABASE_PATH)
    conn.executescript(SCHEMA)
    conn.commit()
    conn.close()


@contextmanager
def get_connection():
    """Usage: with get_connection() as conn: conn.execute(...)"""
    conn = sqlite3.connect(settings.DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()
