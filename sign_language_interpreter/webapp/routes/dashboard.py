"""Dashboard routes — patient and doctor views."""

from flask import Blueprint, render_template, redirect, url_for
from flask_login import login_required, current_user

dashboard_bp = Blueprint("dashboard", __name__)


@dashboard_bp.route("/")
def index():
    if current_user.is_authenticated:
        if current_user.role == "doctor":
            return redirect(url_for("dashboard.doctor_dashboard"))
        return redirect(url_for("dashboard.patient_dashboard"))
    return redirect(url_for("auth.login"))


@dashboard_bp.route("/patient/dashboard")
@login_required
def patient_dashboard():
    if current_user.role != "patient":
        return redirect(url_for("dashboard.index"))
    return render_template("patient_dashboard.html", user=current_user)


@dashboard_bp.route("/doctor/dashboard")
@login_required
def doctor_dashboard():
    if current_user.role != "doctor":
        return redirect(url_for("dashboard.index"))
    return render_template("doctor_dashboard.html", user=current_user)
