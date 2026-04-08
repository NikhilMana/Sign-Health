"""
Database and User models for the SignHealth telehealth platform.

Uses SQLite with context-managed connections to prevent resource leaks.
"""

import sqlite3
import bcrypt
import logging
from pathlib import Path
from contextlib import contextmanager
from datetime import datetime

logger = logging.getLogger(__name__)


class Database:
    """Thin wrapper around SQLite with safe connection management."""

    def __init__(self, db_path):
        self.db_path = db_path
        self._init_db()

    @contextmanager
    def get_connection(self):
        """Context manager that guarantees connection cleanup."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self):
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL,
                    full_name TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS consultations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_id INTEGER NOT NULL,
                    doctor_id INTEGER NOT NULL,
                    transcript TEXT,
                    notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (patient_id) REFERENCES users(id),
                    FOREIGN KEY (doctor_id) REFERENCES users(id)
                )
            """)

            conn.commit()


class User:
    """Flask-Login compatible user model."""

    def __init__(self, id, email, role, full_name):
        self.id = id
        self.email = email
        self.role = role
        self.full_name = full_name
        self.is_authenticated = True
        self.is_active = True
        self.is_anonymous = False

    def get_id(self):
        return str(self.id)

    # ── password helpers ─────────────────────────

    @staticmethod
    def hash_password(password):
        return bcrypt.hashpw(
            password.encode("utf-8"), bcrypt.gensalt()
        ).decode("utf-8")

    @staticmethod
    def check_password(password, password_hash):
        return bcrypt.checkpw(
            password.encode("utf-8"), password_hash.encode("utf-8")
        )

    # ── CRUD ─────────────────────────────────────

    @staticmethod
    def create_user(db, email, password, full_name, role):
        """Create a new user.  Returns ``None`` if email already exists."""
        with db.get_connection() as conn:
            cursor = conn.cursor()
            pw_hash = User.hash_password(password)
            try:
                cursor.execute(
                    "INSERT INTO users (email, password_hash, role, full_name) VALUES (?, ?, ?, ?)",
                    (email, pw_hash, role, full_name),
                )
                conn.commit()
                return User(cursor.lastrowid, email, role, full_name)
            except sqlite3.IntegrityError:
                logger.warning("Duplicate email registration attempt: %s", email)
                return None

    @staticmethod
    def get_by_email(db, email):
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
            row = cursor.fetchone()
        if row:
            return User(row["id"], row["email"], row["role"], row["full_name"])
        return None

    @staticmethod
    def get_by_id(db, user_id):
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
            row = cursor.fetchone()
        if row:
            return User(row["id"], row["email"], row["role"], row["full_name"])
        return None

    @staticmethod
    def authenticate(db, email, password):
        """Verify credentials and return a ``User`` or ``None``."""
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
            row = cursor.fetchone()
        if row and User.check_password(password, row["password_hash"]):
            return User(row["id"], row["email"], row["role"], row["full_name"])
        return None
