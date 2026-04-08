"""Authentication routes — login, register, logout."""

from flask import Blueprint, render_template, request, redirect, url_for, flash
from flask_login import login_user, logout_user, login_required, current_user

auth_bp = Blueprint("auth", __name__)

# The ``db`` reference is injected from app context at import time via init_app.
_db = None
_doctor_emails = []


def init_auth(db, doctor_emails):
    """Called once by the application factory to inject dependencies."""
    global _db, _doctor_emails
    _db = db
    _doctor_emails = doctor_emails


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    from models.user import User

    if current_user.is_authenticated:
        return redirect(url_for("dashboard.index"))

    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        user = User.authenticate(_db, email, password)

        if user:
            login_user(user)
            return redirect(url_for("dashboard.index"))
        else:
            flash("Invalid email or password", "error")

    return render_template("login.html")


@auth_bp.route("/register", methods=["GET", "POST"])
def register():
    from models.user import User

    if current_user.is_authenticated:
        return redirect(url_for("dashboard.index"))

    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        full_name = request.form.get("full_name")

        role = "doctor" if email in _doctor_emails else "patient"
        user = User.create_user(_db, email, password, full_name, role)

        if user:
            flash("Registration successful! Please login.", "success")
            return redirect(url_for("auth.login"))
        else:
            flash("Email already exists", "error")

    return render_template("register.html")


@auth_bp.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("auth.login"))
