"""
Heart Disease Prediction - machine learning module
===================================================
Everything to do with making a prediction:

    validate_input(form)  -> (True/False, list of error messages)
    prepare_input(form)   -> (feature vector, readable values dict)
    predict(vector)       -> {"probability": ..., "predicted_class": ...,
                              "model_used": ..., "votes": ...}

The scaler and ALL the trained models are loaded ONCE by init() when the
application starts, so a prediction costs no disk access.

The final probability is the AVERAGE of the three models' probabilities
(an ensemble). The individual probabilities are returned too, in
"votes", so the result page can show what each model said.

IMPORTANT - the feature order below is a CONTRACT with models/scaler.pkl.
The scaler matches values by POSITION, not by name, so this order must
never change without retraining (run train_model.py).
"""

import os
import json
import pickle

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Decision threshold: a probability at or above this is "High risk".
# 0.50 is the standard cut-off, so the app behaves exactly like the
# metrics printed by train_model.py.
RISK_THRESHOLD = 0.50

# The 13 features, in the exact order the scaler expects.
# 11 come from the form; the last 2 (bmi, pulse_pressure) are calculated
# by compute_derived_features() below.
FEATURE_ORDER = [
    "age", "gender", "height", "weight", "ap_hi", "ap_lo",
    "cholesterol", "gluc", "smoke", "alco", "active",
    "bmi", "pulse_pressure",
]

# Which pickle file belongs to which trained model
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.pkl",
}

# XGBoost is required for local training, but is excluded from the Vercel
# runtime so the serverless function does not install the training stack.
RUNTIME_MODEL_FILES = (
    {name: filename for name, filename in MODEL_FILES.items()
     if name != "XGBoost"}
    if os.environ.get("VERCEL")
    else MODEL_FILES.copy()
)

# Labels for the 1-3 clinical scale used by cholesterol and glucose
LEVEL_LABELS = {1: "Normal", 2: "Above Normal", 3: "Well Above Normal"}

# Module-level cache, filled once by init()
_scaler = None
_models = {}          # every trained model, e.g. {"Random Forest": model, ...}
_best_name = ""       # best single model, from results.json (for importances)
_results = {}


# ══════════════════════════════════════════════════════════════════
# STARTUP
# ══════════════════════════════════════════════════════════════════
def init():
    """
    Load models/features.json, models/scaler.pkl and every saved model.

    The prediction later AVERAGES the models (an ensemble).
    """
    global _scaler, _models, _best_name, _results

    # -- evaluation results (written by train_model.py) --
    _results = _read_json(os.path.join(MODELS_DIR, "results.json"), {})

    # -- feature list (the order is the contract with the scaler) --
    feature_names = _read_json(os.path.join(MODELS_DIR, "features.json"),
                               FEATURE_ORDER)

    # -- scaler --
    try:
        with open(os.path.join(MODELS_DIR, "scaler.pkl"), "rb") as f:
            _scaler = pickle.load(f)
    except FileNotFoundError:
        _scaler = None
        print("WARNING: models/scaler.pkl not found - run train_model.py first.")
        return

    # -- all the trained models --
    for name, filename in RUNTIME_MODEL_FILES.items():
        try:
            with open(os.path.join(MODELS_DIR, filename), "rb") as f:
                _models[name] = pickle.load(f)
        except FileNotFoundError:
            print(f"WARNING: models/{filename} not found - run train_model.py first.")

    _best_name = _results.get("best_model", "Random Forest")
    if _models:
        print(f"ML engine ready: scaler loaded, {len(_models)} models "
              f"({', '.join(_models)}); prediction = their average.")


def is_ready():
    """True when the scaler and at least one model are available."""
    return _scaler is not None and len(_models) > 0


def get_feature_names():
    """The feature names in the exact order the scaler expects."""
    return list(FEATURE_ORDER)


def get_results():
    """Evaluation metrics produced by train_model.py (may be empty)."""
    return _results


def _read_json(path, default):
    """Read a JSON file, returning `default` instead of raising."""
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default


# ══════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ------------------------------------------------------------------
# This ONE function is also imported by train_model.py, so training and
# prediction calculate BMI and pulse pressure in exactly the same way -
# the two implementations cannot drift apart.
# ══════════════════════════════════════════════════════════════════
def compute_derived_features(height, weight, ap_hi, ap_lo):
    """
    Build the 2 derived features from the raw measurements.

    Returns: (bmi, pulse_pressure)

    bmi            = weight (kg) / height (m) squared, clipped to 10-70
    pulse_pressure = systolic - diastolic blood pressure
    """
    bmi = weight / ((height / 100) ** 2)
    bmi = round(max(10, min(70, bmi)), 2)   # clip to the training range

    pulse_pressure = ap_hi - ap_lo
    return bmi, pulse_pressure


# ══════════════════════════════════════════════════════════════════
# SERVER-SIDE VALIDATION - never trust the browser
# ══════════════════════════════════════════════════════════════════
def validate_input(form):
    """
    Check every submitted value before it reaches the model.

    The browser (predict.js) checks the same things first so the user
    gets instant feedback, but a form can always be posted directly,
    so THIS is the real gate.

    Returns: (is_valid, list_of_error_messages)
    """
    errors = []
    numbers = {}

    # -- each measurement present, numeric and inside its range --
    ranges = {
        "age":    (18, 120, "Age"),
        "height": (100, 250, "Height"),
        "weight": (20, 200, "Weight"),
        "ap_hi":  (60, 250, "Systolic blood pressure"),
        "ap_lo":  (40, 200, "Diastolic blood pressure"),
    }
    for field, (low, high, label) in ranges.items():
        raw = str(form.get(field, "")).strip()
        if raw == "":
            errors.append(f"{label} is required.")
            continue
        try:
            value = float(raw)
        except ValueError:
            errors.append(f"{label} must be a number.")
            continue
        if not (low <= value <= high):
            errors.append(f"{label} must be between {low} and {high}.")
            continue
        numbers[field] = value

    # age, height and both blood pressures are whole numbers in practice
    for field, label in [("age", "Age"), ("height", "Height"),
                         ("ap_hi", "Systolic blood pressure"),
                         ("ap_lo", "Diastolic blood pressure")]:
        if field in numbers and numbers[field] != int(numbers[field]):
            errors.append(f"{label} must be a whole number.")

    # -- dropdowns must hold one of their allowed values --
    choices = {
        "gender": (0, 1),          # 0 = female, 1 = male
        "cholesterol": (1, 2, 3),  # 1 normal, 2 above, 3 well above
        "gluc": (1, 2, 3),
        "smoke": (0, 1),
        "alco": (0, 1),
        "active": (0, 1),
    }
    for field, allowed in choices.items():
        try:
            value = int(float(str(form.get(field, "")).strip()))
        except (TypeError, ValueError):
            errors.append("Please choose a value for every question.")
            continue
        if value not in allowed:
            errors.append("Please choose a value for every question.")

    if errors:
        return False, errors

    # -- systolic must exceed diastolic in a living patient --
    if numbers["ap_hi"] <= numbers["ap_lo"]:
        errors.append("Systolic blood pressure must be higher than diastolic "
                      "- the two readings may be swapped.")

    # -- BMI outside 12-60 means height or weight was mistyped --
    bmi = numbers["weight"] / ((numbers["height"] / 100) ** 2)
    if bmi < 12 or bmi > 60:
        errors.append(f"The height and weight give a BMI of {bmi:.1f}, "
                      "which is outside the plausible range - "
                      "please double-check them.")

    return (len(errors) == 0), errors


# ══════════════════════════════════════════════════════════════════
# INPUT PREPARATION
# ══════════════════════════════════════════════════════════════════
def prepare_input(form):
    """
    Turn the 11 validated form values into the 13 model features.

    Returns: (vector, readable)
      vector   the 13 numbers, in FEATURE_ORDER - this feeds the model
      readable the same values as a dict - shown on the result page
    """
    age = int(float(form["age"]))
    gender = int(float(form["gender"]))
    height = int(float(form["height"]))
    weight = float(form["weight"])
    ap_hi = int(float(form["ap_hi"]))
    ap_lo = int(float(form["ap_lo"]))
    cholesterol = int(float(form["cholesterol"]))
    gluc = int(float(form["gluc"]))
    smoke = int(float(form["smoke"]))
    alco = int(float(form["alco"]))
    active = int(float(form["active"]))

    bmi, pulse_pressure = compute_derived_features(height, weight, ap_hi, ap_lo)

    # Order must match models/features.json exactly - the scaler matches
    # values by POSITION, not by name.
    vector = [age, gender, height, weight, ap_hi, ap_lo,
              cholesterol, gluc, smoke, alco, active,
              bmi, pulse_pressure]

    readable = {
        "age": age, "gender": gender, "height": height, "weight": weight,
        "ap_hi": ap_hi, "ap_lo": ap_lo, "cholesterol": cholesterol,
        "gluc": gluc, "smoke": smoke, "alco": alco, "active": active,
        "bmi": bmi, "pulse_pressure": pulse_pressure,
    }
    return vector, readable


# ══════════════════════════════════════════════════════════════════
# PREDICTION
# ══════════════════════════════════════════════════════════════════
def predict(vector):
    """
    Scale the feature vector and predict with EVERY saved model.

    The final probability is the AVERAGE of the models' probabilities
    (an ensemble). The individual probabilities are returned in "votes"
    so the result page can show what each model said.

    The scaler is REUSED, never refitted - it is the exact object fitted
    on the training data, so the patient is measured against the same
    population the models learned from.

    Returns: {"probability": 0-1,        the average of the models
              "predicted_class": 0 or 1, at the 0.50 threshold
              "model_used": name,        e.g. "Ensemble (3 models)"
              "votes": {model name: probability}}
    """
    if not is_ready():
        raise RuntimeError("No trained model available. Run: python train_model.py")

    scaled = _scaler.transform(np.asarray([vector], dtype=float))

    votes = {}
    for name, model in _models.items():
        # predict_proba returns [P(no disease), P(disease)]; index [1]
        # is the probability that disease is PRESENT.
        votes[name] = float(model.predict_proba(scaled)[0][1])

    probability = sum(votes.values()) / len(votes)
    return {
        "probability": probability,
        "predicted_class": 1 if probability >= RISK_THRESHOLD else 0,
        "model_used": f"Ensemble ({len(votes)} models)",
        "votes": votes,
    }


# ══════════════════════════════════════════════════════════════════
# PRESENTATION HELPERS
# ══════════════════════════════════════════════════════════════════
def get_risk_level(probability):
    """Turn a probability into a risk level, a CSS class and a sentence.

    "icon" is the name of the SVG icon the result page shows:
    a check mark for low risk, a warning triangle for high risk.
    """
    if probability < RISK_THRESHOLD:
        return {"label": "LOW RISK", "css": "risk-low", "icon": "check",
                "note": "No significant cardiovascular risk pattern detected."}
    return {"label": "HIGH RISK", "css": "risk-high", "icon": "alert",
            "note": "Several risk indicators are present. "
                    "Please consider consulting a doctor."}


def get_bmi_category(bmi):
    """Standard World Health Organisation BMI bands."""
    if bmi < 18.5:
        return {"label": "Underweight"}
    if bmi < 25:
        return {"label": "Normal weight"}
    if bmi < 30:
        return {"label": "Overweight"}
    return {"label": "Obese"}


def get_tips(risk_label):
    """General, non-personalised advice matched to the risk level."""
    if risk_label == "HIGH RISK":
        return [
            "Consider seeing a doctor for a full cardiovascular check-up.",
            "Monitor your blood pressure regularly and record the readings.",
            "If you smoke, stopping is the single largest risk reduction available.",
            "Aim for regular gentle exercise and a diet low in salt and fat.",
        ]
    return [
        "Keep up a healthy routine - the indicators look good.",
        "Stay physically active - about 150 minutes of activity per week.",
        "Keep salt and saturated fat low, and avoid smoking.",
        "A routine blood pressure and cholesterol check is still worthwhile.",
    ]


def get_feature_importances():
    """
    Feature importances of the best single model (for the model page).

    Reads them straight from the trained model (no extra computation):
    tree models expose feature_importances_, Logistic Regression
    exposes coef_. Returns a list of {feature, importance} sorted high
    to low, normalised to sum to 1.
    """
    if not _models:
        return []

    model = _models.get(_best_name)
    if model is None:
        model = next(iter(_models.values()))

    values = None
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        values = [abs(c) for c in model.coef_[0]]

    if values is None or len(values) != len(FEATURE_ORDER):
        return []

    total = sum(values) or 1
    rows = [{"feature": name, "importance": float(value) / total}
            for name, value in zip(FEATURE_ORDER, values)]
    rows.sort(key=lambda r: r["importance"], reverse=True)
    return rows
