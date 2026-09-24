"""
Heart Disease Prediction - simple test suite
=============================================
Run with:  python test_project.py       (train the models first)

Plain assert-style checks - no testing framework, easy to read and easy
to explain in the FYP viva.

Covers:
  1. the shared feature function (used by BOTH training and the app)
  2. input validation
  3. model prediction end to end
  4. the database layer (passwords, users, predictions)
  5. the web pages (register, login, predict, history, report, admin)
"""

import os
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

passed = 0
failed = 0
failures = []


def check(name, condition, detail=""):
    """Record one test result and print it immediately."""
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        failures.append(name)
        print(f"  FAIL  {name}   {detail}")


# ══════════════════════════════════════════════════════════════════
# [1] The shared feature function
# ══════════════════════════════════════════════════════════════════
print("\n[1] FEATURE FUNCTION")

import ml_model
import train_model

check("training imports the SAME function the app uses",
      train_model.compute_derived_features is ml_model.compute_derived_features)

bmi, pp = ml_model.compute_derived_features(170, 70, 120, 80)
check("BMI of 170cm / 70kg is 24.22", bmi == 24.22, f"got {bmi}")
check("Pulse pressure of 120/80 is 40", pp == 40, f"got {pp}")
check("BMI is clipped at the top (70)",
      ml_model.compute_derived_features(100, 200, 120, 80)[0] == 70)
check("BMI is clipped at the bottom (10)",
      ml_model.compute_derived_features(250, 20, 120, 80)[0] == 10)


# ══════════════════════════════════════════════════════════════════
# [2] Input validation
# ══════════════════════════════════════════════════════════════════
print("\n[2] INPUT VALIDATION")

good = {"age": "45", "gender": "1", "height": "170", "weight": "70",
        "ap_hi": "120", "ap_lo": "80", "cholesterol": "1", "gluc": "1",
        "smoke": "0", "alco": "0", "active": "1"}

ok, errors = ml_model.validate_input(good)
check("a normal form passes validation", ok, str(errors))
check("no error messages for a good form", errors == [], str(errors))

check("age below 18 is rejected",
      ml_model.validate_input(dict(good, age="15"))[0] is False)
check("systolic below diastolic is rejected",
      ml_model.validate_input(dict(good, ap_hi="80", ap_lo="120"))[0] is False)
check("cholesterol code 7 is rejected",
      ml_model.validate_input(dict(good, cholesterol="7"))[0] is False)
missing = dict(good)
del missing["weight"]
check("a missing field is rejected", ml_model.validate_input(missing)[0] is False)
check("an impossible BMI is rejected",
      ml_model.validate_input(dict(good, height="250", weight="20"))[0] is False)


# ══════════════════════════════════════════════════════════════════
# [3] Model prediction (end to end, using the real saved model)
# ══════════════════════════════════════════════════════════════════
print("\n[3] MODEL PREDICTION")

ml_model.init()
check("model engine is ready", ml_model.is_ready())

for fname in ["scaler.pkl", "logistic_regression.pkl", "random_forest.pkl",
              "xgboost.pkl", "features.json", "results.json"]:
    check(f"models/{fname} exists",
          os.path.exists(os.path.join(MODELS_DIR, fname)))

results = ml_model.get_results()
check("results.json names a best model",
      results.get("best_model") in ml_model.MODEL_FILES,
      str(results.get("best_model")))

vector, readable = ml_model.prepare_input(good)
check("feature vector has 13 values", len(vector) == 13, f"got {len(vector)}")

with open(os.path.join(MODELS_DIR, "features.json")) as f:
    saved_features = json.load(f)
check("feature order matches models/features.json",
      saved_features == ml_model.FEATURE_ORDER)

healthy = dict(good, age="30", gender="0", height="165", weight="55",
               ap_hi="110", ap_lo="70", cholesterol="1",
               smoke="0", alco="0", active="1")
risky = dict(good, age="65", gender="1", height="170", weight="95",
             ap_hi="160", ap_lo="95", cholesterol="3",
             smoke="1", alco="1", active="0")

low_outcome = ml_model.predict(ml_model.prepare_input(healthy)[0])
high_outcome = ml_model.predict(ml_model.prepare_input(risky)[0])

check("prediction returns a probability",
      0.0 <= low_outcome["probability"] <= 1.0)
check("healthy sample scores LOWER than the risky sample",
      low_outcome["probability"] < high_outcome["probability"],
      f"{low_outcome['probability']:.3f} vs {high_outcome['probability']:.3f}")
check("predicted_class follows the 0.5 threshold",
      high_outcome["predicted_class"] == (1 if high_outcome["probability"] >= 0.5 else 0))
check("all three models produced a vote",
      set(high_outcome["votes"]) == set(ml_model.MODEL_FILES))
check("ensemble probability is the average of the votes",
      abs(high_outcome["probability"]
          - sum(high_outcome["votes"].values()) / len(high_outcome["votes"])) < 1e-9)
check("the ensemble name is recorded with the result",
      high_outcome["model_used"].startswith("Ensemble"))

risk = ml_model.get_risk_level(low_outcome["probability"])
check("low probability maps to LOW RISK", risk["label"] == "LOW RISK")
risk = ml_model.get_risk_level(0.9)
check("high probability maps to HIGH RISK", risk["label"] == "HIGH RISK")


# ══════════════════════════════════════════════════════════════════
# [4] Database layer
# ══════════════════════════════════════════════════════════════════
print("\n[4] DATABASE")

import database

database.init_db()   # create the tables if this is the first run

hashed = database.hash_password("secret123")
check("the hash never contains the plain password", "secret123" not in hashed)
check("the correct password verifies",
      database.verify_password("secret123", hashed))
check("a wrong password fails",
      not database.verify_password("wrongpass", hashed))

TEST_USER = "test_user_934"
# remove the same user from any previous test run
for u in database.get_all_users():
    if u["username"] in (TEST_USER, "web_test_934"):
        database.delete_user(u["id"])

new_id, err = database.register_user(
    TEST_USER, "test1234", database.ROLE_PATIENT, "Test User", "test934@example.com")
check("user registration works", err is None and new_id is not None)
check("a duplicate username is rejected",
      database.register_user(
          TEST_USER, "x1234567", database.ROLE_PATIENT, "Dup", "dup@example.com")[1] == "taken")

user, status = database.validate_login(TEST_USER, "test1234")
check("login with the correct password works",
      status == "ok" and user["username"] == TEST_USER)
check("login never returns the password hash",
      "password_hash" not in (user or {}))
_, status = database.validate_login(TEST_USER, "wrongpass")
check("login with a wrong password fails", status == "invalid")

pred_id = database.add_prediction(new_id, readable, 0, 0.25, "Random Forest")
rows = database.get_predictions(user_id=new_id)
check("a prediction is saved for the user",
      len(rows) == 1 and rows[0]["probability"] == 0.25)
rec = database.get_prediction_by_id(pred_id)
check("a single prediction can be fetched",
      rec is not None and rec["account"] == TEST_USER)
database.delete_prediction(pred_id)
check("the prediction is deleted", database.get_predictions(user_id=new_id) == [])
database.delete_user(new_id)
check("the test user is deleted", database.get_user_by_id(new_id) is None)


# ══════════════════════════════════════════════════════════════════
# [5] Web pages (Flask test client - a real browser session)
# ══════════════════════════════════════════════════════════════════
print("\n[5] WEB PAGES")

from app import app as flask_app

client = flask_app.test_client()
DEPLOYED_MODEL = ml_model.predict(ml_model.prepare_input(good)[0])["model_used"]


def post(url, data):
    """POST with a matching CSRF token.

    The token is OVERWRITTEN every time, because a successful login
    clears the session (correct behaviour) and later GETs create a fresh
    random token that a stored value would no longer match.
    """
    with client.session_transaction() as s:
        s["_csrf_token"] = "test-token"
    return client.post(url, data={**data, "csrf_token": "test-token"})


r = client.get("/")
check("a signed-out visitor is sent to the login page",
      r.status_code == 302 and "/login" in r.headers["Location"])

r = post("/login", {"username": "admin", "password": "wrong"})
check("a wrong password is refused", r.status_code == 200 and b"Invalid" in r.data)

# register a fresh patient through the web form
WEB_USER = "web_test_934"
r = post("/register", {"username": WEB_USER, "fullname": "Web Test",
                       "email": "web934@example.com",
                       "password": "web12345", "confirm_password": "web12345"})
check("registration accepts a valid form", r.status_code == 302)

r = post("/login", {"username": WEB_USER, "password": "web12345"})
check("login works after registration", r.status_code == 302)

r = client.get("/dashboard")
check("the dashboard loads", r.status_code == 200)

r = client.get("/diagnosis")
check("the assessment form loads",
      r.status_code == 200 and b"predictForm" in r.data)

r = post("/diagnosis", good)
check("the prediction page returns a result",
      r.status_code == 200 and b"RISK" in r.data.upper())

r = client.get("/history")
check("history lists the new record",
      r.status_code == 200 and DEPLOYED_MODEL.encode() in r.data)

all_records = database.get_predictions()
mine = [p for p in all_records if p.get("account") == WEB_USER]
check("the prediction was stored in the database", len(mine) >= 1)

r = client.get(f"/report/{mine[0]['id']}")
check("the report downloads",
      r.status_code == 200 and b"RISK ASSESSMENT REPORT" in r.data)

r = client.get(f"/report-pdf/{mine[0]['id']}")
check("the PDF report downloads (single + ensemble results)",
      r.status_code == 200 and r.data[:4] == b"%PDF")

r = client.get("/admin/users")
check("a patient is blocked from admin pages (403)", r.status_code == 403)

client.get("/logout")
r = post("/login", {"username": "admin", "password": "admin123"})
check("the admin account can sign in", r.status_code == 302)

r = client.get("/admin/users")
check("the admin users page loads", r.status_code == 200)
r = client.get("/admin/predictions")
check("the admin predictions page loads", r.status_code == 200)
r = client.get("/admin/models")
check("the admin model page loads", r.status_code == 200 and b"ROC AUC" in r.data)

# CSRF protection must reject a POST without a token
fresh_client = flask_app.test_client()
r = fresh_client.post("/login", data={"username": "admin", "password": "admin123"})
check("a POST without a CSRF token is rejected (400)", r.status_code == 400)

# clean up the web test user (their prediction is removed too, by cascade)
for u in database.get_all_users():
    if u["username"] == WEB_USER:
        database.delete_user(u["id"])
check("the web test user is cleaned up", not database.username_exists(WEB_USER))


# ══════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(f"  {passed} passed, {failed} failed")
if failures:
    print("  Failed checks:")
    for name in failures:
        print(f"    - {name}")
print("=" * 60)
