"""
Heart Disease Prediction - Flask web application
=================================================
Run:  python app.py        then open http://127.0.0.1:5000

This single file holds every web route. It works together with:

    database.py   SQLite storage (users + predictions)
    ml_model.py   loads the saved model, validates input, predicts
    train_model.py  retrains the models (run once before first use)

The prediction flow, which is the heart of the project:

    form  ->  validate_input()  ->  prepare_input()  ->  saved scaler
          ->  saved model  ->  probability  ->  risk level
          ->  result page  ->  saved to SQLite
"""

import json
import os
import re
import secrets
from datetime import timedelta
from functools import wraps

from flask import (Flask, render_template, request, redirect, url_for,
                   session, flash, abort, Response)

import database
import ml_model

# ---------------------------------------------------------------
# Application setup
# ---------------------------------------------------------------
app = Flask(__name__)
app.secret_key = os.environ.get("HDP_SECRET_KEY", "dev-only-key-change-me")
app.permanent_session_lifetime = timedelta(hours=12)

# Create the database tables (and the seed admin account) on startup,
# then load the scaler and the trained model once.
database.init_db()
ml_model.init()

ROLE_PATIENT = database.ROLE_PATIENT
ROLE_ADMIN = database.ROLE_ADMIN


# ---------------------------------------------------------------
# Small helpers: sign-in checks, CSRF protection
# ---------------------------------------------------------------
def current_user():
    """The signed-in user's details from the session, or None."""
    return session.get("user")


def login_required(view):
    """Block a page for visitors who are not signed in."""
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not current_user():
            flash("Please sign in first.", "warning")
            return redirect(url_for("login"))
        return view(*args, **kwargs)
    return wrapped


def admin_required(view):
    """Block a page for everyone except Admin accounts."""
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not current_user():
            flash("Please sign in first.", "warning")
            return redirect(url_for("login"))
        if current_user().get("role") != ROLE_ADMIN:
            abort(403)
        return view(*args, **kwargs)
    return wrapped


def csrf_token():
    """One random token per session, embedded in every form."""
    if "_csrf_token" not in session:
        session["_csrf_token"] = secrets.token_hex(32)
    return session["_csrf_token"]


@app.before_request
def check_csrf():
    """Reject any form POST whose token does not match the session."""
    if request.method == "POST":
        expected = session.get("_csrf_token")
        supplied = request.form.get("csrf_token")
        if not expected or not supplied \
                or not secrets.compare_digest(expected, supplied):
            abort(400, description="Security token missing or invalid. "
                                   "Please go back and try again.")


@app.context_processor
def template_helpers():
    """Values every template can use without being passed in."""
    return {
        "csrf_token": csrf_token,
        "current_user": current_user(),
        "ROLE_PATIENT": ROLE_PATIENT,
        "ROLE_ADMIN": ROLE_ADMIN,
    }


@app.after_request
def no_cache(response):
    """
    Stop the browser from reusing cached pages when the user presses
    Back. Without this, after a new assessment the dashboard or history
    could show the old cached numbers instead of the new ones.
    """
    response.headers["Cache-Control"] = "no-store, max-age=0"
    return response


# ---------------------------------------------------------------
# Authentication pages
# ---------------------------------------------------------------
@app.route("/")
def index():
    """Send visitors to the dashboard, or to login if signed out."""
    if current_user():
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    """Show the sign-in form and process it."""
    if current_user():
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        username = (request.form.get("username") or "").strip()
        password = request.form.get("password") or ""

        if not username or not password:
            flash("Please enter both a username and a password.", "warning")
            return render_template("login.html")

        user, status = database.validate_login(username, password)
        if status == "ok":
            session.clear()          # fresh session on every sign-in
            session["user"] = user
            session.permanent = True
            flash(f"Welcome back, {user['fullname'] or user['username']}.", "success")
            return redirect(url_for("dashboard"))
        elif status == "banned":
            flash("This account has been suspended. Contact an administrator.", "danger")
        else:
            flash("Invalid username or password.", "danger")

    return render_template("login.html")


def validate_registration(username, fullname, email, password, confirm):
    """Rules for the sign-up form. Returns a list of problems (empty = OK)."""
    errors = []
    if not re.fullmatch(r"[A-Za-z0-9_]{3,20}", username or ""):
        errors.append("Username must be 3-20 characters: letters, numbers "
                      "and underscores only.")
    if not (fullname and 2 <= len(fullname) <= 60):
        errors.append("Please enter your full name (up to 60 characters).")
    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[A-Za-z]{2,}", email or ""):
        errors.append("Please enter a valid email address.")
    if len(password or "") < 8 or not re.search(r"[A-Za-z]", password) \
            or not re.search(r"\d", password):
        errors.append("Password must be at least 8 characters and mix "
                      "letters with numbers.")
    if password != confirm:
        errors.append("The two passwords do not match.")
    if not errors and database.username_exists(username):
        errors.append("That username is already taken.")
    if not errors and database.email_exists(email):
        errors.append("An account already exists with that email address.")
    return errors


@app.route("/register", methods=["GET", "POST"])
def register():
    """Public sign-up. Creates Patient accounts only."""
    if current_user():
        return redirect(url_for("dashboard"))

    form = {}
    if request.method == "POST":
        form = {
            "username": (request.form.get("username") or "").strip(),
            "fullname": (request.form.get("fullname") or "").strip(),
            "email": (request.form.get("email") or "").strip(),
        }
        password = request.form.get("password") or ""
        confirm = request.form.get("confirm_password") or ""

        errors = validate_registration(form["username"], form["fullname"],
                                       form["email"], password, confirm)
        if errors:
            for message in errors:
                flash(message, "warning")
        else:
            _, error = database.register_user(
                form["username"], password, ROLE_PATIENT,
                form["fullname"], form["email"])
            if error == "taken":
                flash("That username is already taken.", "warning")
            else:
                flash("Account created. You can sign in now.", "success")
                return redirect(url_for("login"))

    return render_template("register.html", form=form)


@app.route("/logout")
@login_required
def logout():
    """Clear the session and return to the login page."""
    session.clear()
    flash("You have been signed out.", "info")
    return redirect(url_for("login"))


# ---------------------------------------------------------------
# Patient pages: dashboard, prediction, history, report, profile
# ---------------------------------------------------------------
def _scope_user_id(user):
    """
    Patients only ever see their own records; an admin sees everything.
    Returns the user id to filter by, or None for 'no filter'.
    """
    return None if user["role"] == ROLE_ADMIN else user["id"]


@app.route("/dashboard")
@login_required
def dashboard():
    """The landing page: totals, last result and a risk trend chart."""
    user = current_user()
    predictions = database.get_predictions(user_id=_scope_user_id(user))

    high = sum(1 for p in predictions if p["predicted_class"] == 1)
    low = len(predictions) - high
    last = predictions[0] if predictions else None

    results = ml_model.get_results()
    best_model = results.get("best_model")
    best_accuracy = results.get("models", {}).get(best_model, {}).get("accuracy")

    # risk scores for the trend chart, oldest first
    trend = [{"label": (p["timestamp"] or "")[:16],
              "value": round(p["probability"] * 100, 1)}
             for p in list(reversed(predictions[:10]))]

    return render_template("dashboard.html",
                           predictions=predictions[:10],
                           total=len(predictions), high=high, low=low,
                           last=last, trend=trend,
                           best_model=best_model, best_accuracy=best_accuracy,
                           engine_ready=ml_model.is_ready())


@app.route("/diagnosis", methods=["GET", "POST"])
@login_required
def diagnosis_form():
    """
    The core of the application.

    GET  shows the empty form.
    POST validates the input, prepares the 13 features, runs the saved
         model, saves the result to the database and shows the result page.
    """
    if request.method == "GET":
        if not ml_model.is_ready():
            flash("No trained model found. Run:  python train_model.py", "danger")
        return render_template("predict.html", form={},
                               engine_ready=ml_model.is_ready())

    # -- POST: run one assessment --
    if not ml_model.is_ready():
        flash("No trained model found. Run:  python train_model.py", "danger")
        return redirect(url_for("diagnosis_form"))

    form = request.form.to_dict()

    # 1. server-side validation (the browser check is only a convenience)
    ok, errors = ml_model.validate_input(form)
    if not ok:
        for message in errors:
            flash(message, "warning")
        return render_template("predict.html", form=form, engine_ready=True)

    # 2. turn the 11 answers into the 13 model features
    vector, readable = ml_model.prepare_input(form)

    # 3. scale and run the saved model
    try:
        outcome = ml_model.predict(vector)
    except Exception:
        app.logger.exception("Prediction failed")
        flash("Something went wrong while predicting.", "danger")
        return redirect(url_for("diagnosis_form"))

    # 4. save to the database (a failure here does not lose the result)
    user = current_user()
    saved_id = None
    try:
        saved_id = database.add_prediction(
            user["id"], readable, outcome["predicted_class"],
            outcome["probability"], outcome["model_used"],
            votes=outcome["votes"])
    except Exception:
        app.logger.exception("Could not save prediction")
        flash("The result is valid but could not be saved.", "warning")

    # 5. show the result page
    risk = ml_model.get_risk_level(outcome["probability"])
    return render_template("result.html",
                           outcome=outcome, risk=risk,
                           tips=ml_model.get_tips(risk["label"]),
                           bmi_category=ml_model.get_bmi_category(readable["bmi"]),
                           readable=readable,
                           level_labels=ml_model.LEVEL_LABELS,
                           saved_id=saved_id)


@app.route("/history")
@login_required
def history():
    """The signed-in user's assessment history, with optional filters."""
    user = current_user()
    predictions = database.get_predictions(user_id=_scope_user_id(user))

    verdict = request.args.get("verdict", "all")
    if verdict == "high":
        predictions = [p for p in predictions if p["predicted_class"] == 1]
    elif verdict == "low":
        predictions = [p for p in predictions if p["predicted_class"] == 0]

    search = (request.args.get("q") or "").strip().lower()
    if search:
        predictions = [p for p in predictions
                       if search in str(p.get("model_used", "")).lower()]

    rows = [{**p, "risk": ml_model.get_risk_level(p["probability"])}
            for p in predictions]

    return render_template("history.html", rows=rows, verdict=verdict,
                           search=search,
                           is_personal=(user["role"] != ROLE_ADMIN))


@app.route("/report/<int:pred_id>")
@login_required
def report(pred_id):
    """Download one saved assessment as a formatted text report."""
    user = current_user()
    record = database.get_prediction_by_id(pred_id)
    if not record:
        abort(404)
    # Ownership check: your own record, or an admin
    if record["user_id"] != user["id"] and user["role"] != ROLE_ADMIN:
        abort(403)

    risk = ml_model.get_risk_level(record["probability"])
    bmi = round(record["weight"] / ((record["height"] / 100) ** 2), 2)
    labels = ml_model.LEVEL_LABELS

    text = f"""==================================================
HEART DISEASE PREDICTION - RISK ASSESSMENT REPORT
==================================================
Report ID  : {record['id']}
Date/Time  : {record['timestamp']}
Account    : {record.get('account_name') or record.get('account')}

CLINICAL INDICATORS:
--------------------
Age              : {int(record['age'])} years
Gender           : {'Male' if record['gender'] == 1 else 'Female'}
Height           : {int(record['height'])} cm
Weight           : {record['weight']} kg
BMI              : {bmi}
Systolic BP      : {int(record['ap_hi'])} mmHg
Diastolic BP     : {int(record['ap_lo'])} mmHg
Pulse Pressure   : {int(record['ap_hi'] - record['ap_lo'])} mmHg
Cholesterol      : {labels.get(record['cholesterol'], record['cholesterol'])}
Glucose          : {labels.get(record['gluc'], record['gluc'])}
Smoker           : {'Yes' if record['smoke'] else 'No'}
Alcohol Use      : {'Yes' if record['alco'] else 'No'}
Physically Active: {'Yes' if record['active'] else 'No'}

AI PREDICTION:
--------------
Model Used      : {record['model_used']}
Risk Probability: {record['probability']:.2%}
Risk Level      : {risk['label']}
Interpretation  : {risk['note']}

==================================================
IMPORTANT - PLEASE READ
==================================================
This is a SCREENING TOOL, NOT A MEDICAL DIAGNOSIS.
It estimates risk from statistical patterns in population
data and can be wrong in both directions. Always consult
a qualified doctor before acting on any result.
==================================================
Report generated by Heart Disease Prediction (FYP 2026)
=================================================="""

    return Response(
        text,
        mimetype="text/plain",
        headers={"Content-Disposition":
                 f"attachment; filename=heart_report_{pred_id}.txt"},
    )


def _build_pdf(lines):
    """
    Build a one-page A4 PDF from simple text lines, using ONLY the
    Python standard library - so the project still works fully offline.

    `lines` is a list of (bold, size, text) tuples. Returns PDF bytes.
    """
    content = []
    y = 800                                   # points from the bottom
    for bold, size, text in lines:
        text = str(text).encode("latin-1", "replace").decode("latin-1")
        text = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        content.append(f"BT /F{2 if bold else 1} {size} Tf 50 {y} Td ({text}) Tj ET")
        y -= size + 5
    stream = "\n".join(content).encode("latin-1", "replace")

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
        b"/Resources << /Font << /F1 4 0 R /F2 5 0 R >> >> /Contents 6 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n"
        + stream + b"\nendstream",
    ]

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref_at = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode() + b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_at}\n%%EOF").encode()
    return bytes(out)


@app.route("/report-pdf/<int:pred_id>")
@login_required
def report_pdf(pred_id):
    """Download one saved assessment as a PDF report.

    Shows BOTH results: the ensemble verdict (what the app displayed)
    and each single model's probability. The per-model probabilities
    were stored with the prediction, so the PDF always shows the exact
    numbers the user saw at assessment time.
    """
    user = current_user()
    record = database.get_prediction_by_id(pred_id)
    if not record:
        abort(404)
    if record["user_id"] != user["id"] and user["role"] != ROLE_ADMIN:
        abort(403)

    risk = ml_model.get_risk_level(record["probability"])
    bmi = round(record["weight"] / ((record["height"] / 100) ** 2), 2)
    labels = ml_model.LEVEL_LABELS
    try:
        votes = json.loads(record["votes_json"]) if record["votes_json"] else {}
    except (TypeError, ValueError):
        votes = {}

    lines = [
        (True, 15, "HEART DISEASE PREDICTION - RISK REPORT"),
        (False, 10, "Screening tool - educational use only, NOT a medical diagnosis."),
        (False, 8, ""),
        (True, 12, "Report details"),
        (False, 10, f"Report ID : {record['id']}    Date/Time: {record['timestamp']}"),
        (False, 10, f"Account   : {record.get('account_name') or record.get('account')}"),
        (False, 8, ""),
        (True, 12, "Clinical indicators"),
        (False, 10, f"Age: {int(record['age'])} years    Gender: "
                    f"{'Male' if record['gender'] == 1 else 'Female'}"),
        (False, 10, f"Height: {int(record['height'])} cm    "
                    f"Weight: {record['weight']} kg    BMI: {bmi}"),
        (False, 10, f"Systolic BP: {int(record['ap_hi'])} mmHg    "
                    f"Diastolic BP: {int(record['ap_lo'])} mmHg"),
        (False, 10, f"Pulse pressure: {int(record['ap_hi'] - record['ap_lo'])} mmHg"),
        (False, 10, f"Cholesterol: {labels.get(record['cholesterol'], record['cholesterol'])}"
                    f"    Glucose: {labels.get(record['gluc'], record['gluc'])}"),
        (False, 10, f"Smoker: {'Yes' if record['smoke'] else 'No'}    "
                    f"Alcohol: {'Yes' if record['alco'] else 'No'}    "
                    f"Active: {'Yes' if record['active'] else 'No'}"),
        (False, 8, ""),
        (True, 12, "AI prediction"),
        (False, 11, f"Ensemble (average of the models): "
                    f"{record['probability']:.1%}  ->  {risk['label']}"),
        (False, 10, f"Model used: {record['model_used']}"),
    ]
    if votes:
        lines.append((True, 11, "Single model breakdown"))
        for name, prob in votes.items():
            verdict = "High Risk" if prob >= 0.5 else "Low Risk"
            lines.append((False, 10, f"   {name}: {prob:.1%}  ({verdict})"))
    lines += [
        (False, 8, ""),
        (True, 10, "IMPORTANT - PLEASE READ"),
        (False, 9, "This is a SCREENING TOOL, NOT A MEDICAL DIAGNOSIS. It estimates risk"),
        (False, 9, "from statistical patterns in population data and can be wrong in both"),
        (False, 9, "directions. Always consult a qualified doctor before acting on any result."),
    ]

    return Response(_build_pdf(lines), mimetype="application/pdf",
                    headers={"Content-Disposition":
                             f"attachment; filename=heart_report_{pred_id}.pdf"})


@app.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    """View and update the signed-in user's own details."""
    user = current_user()

    if request.method == "POST":
        fullname = (request.form.get("fullname") or "").strip()
        email = (request.form.get("email") or "").strip()
        new_password = request.form.get("new_password") or ""
        confirm = request.form.get("confirm_password") or ""

        errors = []
        if not fullname:
            errors.append("Your full name is required.")
        if "@" not in email:
            errors.append("A valid email address is required.")
        if new_password:
            if len(new_password) < 8:
                errors.append("The new password must be at least 8 characters.")
            if new_password != confirm:
                errors.append("The two passwords do not match.")
        if errors:
            for message in errors:
                flash(message, "warning")
            return redirect(url_for("profile"))

        database.update_user_profile(user["id"], fullname, email,
                                     new_password or None)
        # refresh the session copy so the sidebar shows the new name
        refreshed = dict(session["user"])
        refreshed["fullname"], refreshed["email"] = fullname, email
        session["user"] = refreshed

        flash("Profile updated.", "success")
        return redirect(url_for("profile"))

    record = database.get_user_by_id(user["id"]) or user
    return render_template("profile.html", record=record)


# ---------------------------------------------------------------
# Admin pages: users, all predictions, model performance
# ---------------------------------------------------------------
@app.route("/admin/users")
@admin_required
def admin_users():
    """Every account, with prediction counts and management actions."""
    users = database.get_all_users()
    counts = database.count_predictions_by_user()
    rows = [{**u, "count": counts.get(u["id"], 0)} for u in users]
    return render_template("users.html", rows=rows)


@app.route("/admin/users/<int:user_id>/role", methods=["POST"])
@admin_required
def admin_change_role(user_id):
    """Promote a patient to Admin, or demote an Admin to Patient."""
    actor = current_user()
    new_role = request.form.get("role", "")
    target = database.get_user_by_id(user_id)

    if new_role not in (ROLE_PATIENT, ROLE_ADMIN):
        flash("Unknown role.", "warning")
    elif not target:
        flash("That user no longer exists.", "warning")
    elif user_id == actor["id"]:
        flash("You cannot change your own role.", "danger")
    elif (target["role"] == ROLE_ADMIN and new_role != ROLE_ADMIN
            and database.count_active_admins() <= 1):
        flash("This is the only admin - promote someone else first.", "danger")
    else:
        database.update_user_role(user_id, new_role)
        flash(f"{target['username']} is now a {new_role}.", "success")
    return redirect(url_for("admin_users"))


@app.route("/admin/users/<int:user_id>/ban", methods=["POST"])
@admin_required
def admin_ban(user_id):
    """Suspend or reinstate an account."""
    actor = current_user()
    target = database.get_user_by_id(user_id)

    if not target:
        flash("That user no longer exists.", "warning")
    elif user_id == actor["id"]:
        flash("You cannot suspend your own account.", "danger")
    elif target["is_banned"]:
        database.unban_user(user_id)
        flash(f"{target['username']} has been reinstated.", "success")
    elif (target["role"] == ROLE_ADMIN
            and database.count_active_admins() <= 1):
        flash("This is the only admin - promote someone else first.", "danger")
    else:
        database.ban_user(user_id)
        flash(f"{target['username']} has been suspended.", "success")
    return redirect(url_for("admin_users"))


@app.route("/admin/users/<int:user_id>/delete", methods=["POST"])
@admin_required
def admin_delete_user(user_id):
    """Permanently delete an account (and its predictions)."""
    actor = current_user()
    target = database.get_user_by_id(user_id)

    if not target:
        flash("That user no longer exists.", "warning")
    elif user_id == actor["id"]:
        flash("You cannot delete your own account.", "danger")
    elif (target["role"] == ROLE_ADMIN
            and database.count_active_admins() <= 1):
        flash("This is the only admin - promote someone else first.", "danger")
    else:
        database.delete_user(user_id)
        flash(f"Account '{target['username']}' has been deleted.", "success")
    return redirect(url_for("admin_users"))


@app.route("/admin/predictions")
@admin_required
def admin_predictions():
    """Every saved assessment in the system, with a verdict filter."""
    records = database.get_predictions()
    verdict = request.args.get("verdict", "all")
    if verdict == "high":
        records = [r for r in records if r["predicted_class"] == 1]
    elif verdict == "low":
        records = [r for r in records if r["predicted_class"] == 0]

    rows = [{**r, "risk": ml_model.get_risk_level(r["probability"])}
            for r in records]
    return render_template("predictions.html", rows=rows, verdict=verdict)


@app.route("/admin/predictions/<int:pred_id>/delete", methods=["POST"])
@admin_required
def admin_delete_prediction(pred_id):
    """Remove one saved assessment."""
    database.delete_prediction(pred_id)
    flash(f"Prediction #{pred_id} deleted.", "success")
    return redirect(url_for("admin_predictions"))


@app.route("/admin/models")
@admin_required
def admin_models():
    """Model comparison page: metrics from the last training run."""
    results = ml_model.get_results()
    models = results.get("models", {})
    rows = [{"name": name, **data} for name, data in models.items()]
    # Best ROC AUC first; when two models tie, the better CV AUC ranks first
    rows.sort(key=lambda r: (-r.get("roc_auc", 0), -r.get("cv_auc", 0)))

    # Find the row of the selected model, for the confusion matrix panel
    best = results.get("best_model")
    best_row = None
    for r in rows:
        if r["name"] == best:
            best_row = r
    if best_row is None and rows:
        best_row = rows[0]

    return render_template("models.html",
                           rows=rows,
                           best=best,
                           best_row=best_row,
                           rows_used=results.get("rows_used"),
                           trained_at=results.get("trained_at"),
                           importances=ml_model.get_feature_importances())


# ---------------------------------------------------------------
# Error pages
# ---------------------------------------------------------------
@app.errorhandler(400)
def bad_request(e):
    return render_template("errors/400.html",
                           message=getattr(e, "description", None)), 400


@app.errorhandler(403)
def forbidden(e):
    return render_template("errors/403.html"), 403


@app.errorhandler(404)
def not_found(e):
    return render_template("errors/404.html"), 404


@app.errorhandler(500)
def server_error(e):
    app.logger.error(f"Internal error on {request.path}: {e}", exc_info=True)
    return render_template("errors/500.html"), 500


# ---------------------------------------------------------------
# Start the development server
# ---------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 56)
    print("  Heart Disease Prediction")
    print("  Running at  : http://127.0.0.1:5000")
    print("  Default user: admin / admin123   (role: Admin)")
    print("  Press CTRL+C to stop")
    print("=" * 56)
    app.run(host="127.0.0.1", port=5000, debug=True)
