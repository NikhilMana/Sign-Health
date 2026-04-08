"""Tests for the Database and User models."""

import pytest
import sys
import tempfile
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "webapp"))

from models.user import Database, User


@pytest.fixture
def db(tmp_path):
    """Create a fresh in-memory-like database for each test."""
    db_path = tmp_path / "test.db"
    return Database(str(db_path))


class TestDatabase:
    def test_creates_tables(self, db):
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = {row["name"] for row in cursor.fetchall()}
        assert "users" in tables
        assert "consultations" in tables

    def test_context_manager_closes_connection(self, db):
        conn_ref = None
        with db.get_connection() as conn:
            conn_ref = conn
            conn.execute("SELECT 1")
        # After context exit, the connection should be closed
        # Attempting to use it should raise
        with pytest.raises(Exception):
            conn_ref.execute("SELECT 1")


class TestUser:
    def test_create_user(self, db):
        user = User.create_user(db, "test@example.com", "password123", "Test User", "patient")
        assert user is not None
        assert user.email == "test@example.com"
        assert user.role == "patient"
        assert user.full_name == "Test User"

    def test_duplicate_email_returns_none(self, db):
        User.create_user(db, "test@example.com", "password123", "Test User", "patient")
        duplicate = User.create_user(db, "test@example.com", "other", "Other", "doctor")
        assert duplicate is None

    def test_authenticate_valid(self, db):
        User.create_user(db, "test@example.com", "password123", "Test User", "patient")
        user = User.authenticate(db, "test@example.com", "password123")
        assert user is not None
        assert user.email == "test@example.com"

    def test_authenticate_wrong_password(self, db):
        User.create_user(db, "test@example.com", "password123", "Test User", "patient")
        user = User.authenticate(db, "test@example.com", "wrongpassword")
        assert user is None

    def test_authenticate_nonexistent_email(self, db):
        user = User.authenticate(db, "nobody@example.com", "password123")
        assert user is None

    def test_get_by_id(self, db):
        created = User.create_user(db, "test@example.com", "password123", "Test User", "patient")
        found = User.get_by_id(db, created.id)
        assert found is not None
        assert found.email == "test@example.com"

    def test_get_by_email(self, db):
        User.create_user(db, "test@example.com", "password123", "Test User", "patient")
        found = User.get_by_email(db, "test@example.com")
        assert found is not None
        assert found.full_name == "Test User"

    def test_get_by_id_nonexistent(self, db):
        assert User.get_by_id(db, 9999) is None

    def test_password_hashing(self):
        hashed = User.hash_password("my_secret")
        assert User.check_password("my_secret", hashed)
        assert not User.check_password("wrong", hashed)

    def test_flask_login_interface(self, db):
        user = User.create_user(db, "test@example.com", "password123", "Test User", "patient")
        assert user.get_id() == str(user.id)
        assert user.is_authenticated is True
        assert user.is_active is True
        assert user.is_anonymous is False
