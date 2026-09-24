# Heart Disease Prediction — Flask + Machine Learning (FYP)

A Flask web application that estimates cardiovascular disease risk from
eleven routine health measurements. The data is cleaned and used to train
three machine-learning models; the web app predicts with the average of
the three (an ensemble). Results are stored in SQLite so each user can
review their own history.

Final Year Project — 2026.

> **Educational tool, not a medical diagnosis.**
> The app estimates risk from statistical patterns in population data and
> can be wrong. Always consult a qualified doctor before acting on any
> result.

---

## Documentation

| File | What it explains |
|---|---|
| [docs/architecture.md](docs/architecture.md) | How the system is put together, request by request |
| [docs/file_guide.md](docs/file_guide.md) | What every file does |
| [docs/ml_pipeline.md](docs/ml_pipeline.md) | The machine-learning pipeline, step by step |
| [docs/VIVA_QUESTIONS.md](docs/VIVA_QUESTIONS.md) | 34 viva questions with answers |
| [CHANGES.md](CHANGES.md) | The log of every change made to the project |

## How it works, in one picture

```
TRAINING (run once)                       WEB APPLICATION (run every day)
────────────────────                      ────────────────────────────────
data/heart.csv                            user fills the form (11 inputs)
   ↓                                         ↓
clean the data                            validate_input()   (ml_model.py)
   ↓                                         ↓
13 features (11 + BMI + pulse pressure)   prepare_input()    (ml_model.py)
   ↓                                         ↓
80/20 train/test split                    saved scaler transforms the values
   ↓                                         ↓
StandardScaler (fit on train only)        3 saved models predict (ensemble)
   ↓                                         ↓
train 3 models, compare them              probability → LOW or HIGH RISK
   ↓                                         ↓
save scaler + models + results.json       result page + saved to SQLite
```

---

## Technologies

| Layer | Choice | Why |
|---|---|---|
| Web framework | Flask 3.1 | Small and explicit — every route is code you wrote |
| Templates | Jinja2 (ships with Flask) | Normal server-rendered HTML pages |
| Database | SQLite (one file) | No server to install, easy to show in a viva |
| Machine learning | scikit-learn 1.9, XGBoost 3.4 | Standard tools for tabular data |
| Charts | Chart.js (bundled locally) | One risk-trend chart on the dashboard |
| Styling | Hand-written CSS | No framework needed |
| JavaScript | Vanilla, ~350 lines | Live BMI display, form checks, sidebar |

---

## Requirements

- Python 3.10 or newer (developed on 3.13)
- The packages in `requirements.txt`: Flask, numpy, and scikit-learn
- For local model training, also install `requirements-train.txt` (pandas and xgboost)

## Install and run

```bash
# 1. open the project folder
cd "Heart_Disease_Prediction_using_flask_and_machine_learning - Copy"

# 2. (recommended) create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 3. install the dependencies
pip install -r requirements-train.txt

# 4. train the models (about 1 minute — needed once)
python train_model.py

# 5. start the web application
python app.py
```

Then open <http://127.0.0.1:5000> in a browser.

- The database `heart_disease.db` is created automatically on first run.
- Step 4 creates everything in `models/`. If you skip it, the app runs but
  shows a clear "no trained model" message until you train once.

## Default login

| Username | Password | Role | Created by |
|---|---|---|---|
| `admin` | `admin123` | Admin | automatically, on first run |

Patient accounts are created on the public **Register** page — no role
choice is offered, the server always creates Patients. An admin can
promote a user to Admin on the Users page.

Demo credentials are for local development only. Change them before
showing the app on a network.

---

## Folder structure

```
├── app.py               Flask application: every route, session, CSRF, error pages
├── database.py          SQLite storage: users + predictions, password hashing
├── ml_model.py          Loads the saved model; validate / prepare / predict
├── train_model.py       Training pipeline: clean → features → train → compare → save
├── test_project.py      57 plain-assert checks:  python test_project.py
├── requirements.txt     Flask, pandas, numpy, scikit-learn, xgboost
├── data/
│   └── heart.csv        The dataset (70,000 rows)
├── models/              Created by train_model.py
│   ├── scaler.pkl                    fitted on the training split only
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl             ← best single model (Model Performance page)
│   ├── xgboost.pkl
│   ├── features.json                the 13 feature names, in order
│   └── results.json                 metrics + best model name
├── templates/           base, login, register, dashboard, predict, result,
│                        history, profile, users, predictions, models, errors/
├── static/
    ├── css/style.css    the whole theme
    ├── js/main.js       sidebar, flash messages, confirm dialogs
    ├── js/predict.js    live BMI + blood-pressure checks on the form
    ├── js/register.js   instant sign-up feedback
    ├── js/charts.js     the dashboard risk-trend chart
    └── js/chart.min.js  Chart.js, bundled so no internet is needed
├── docs/               architecture.md, file_guide.md, ml_pipeline.md,
│                        VIVA_QUESTIONS.md - documentation for the project
├── CHANGES.md          the log of every change made to the project
└── README.md           this file
```

---

## The dataset

**Cardiovascular Disease dataset** — Svetlana Ulianova, Kaggle:
<https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset>

70,000 patient records: objective measurements (age, height, weight, blood
pressure), examination results (cholesterol, glucose), self-reported
lifestyle answers (smoking, alcohol, physical activity), and a binary
`cardio` label (1 = disease present).

Two quirks of the raw file matter (both fixed in `train_model.py`):

1. **`age` is stored in days**, not years — converted by dividing by 365.25.
2. **`cholesterol` and `gluc` are stored 0-indexed** (0, 1, 2) while the
   clinical scale — and the web form — use 1, 2, 3. The pipeline shifts
   them by +1 *before* filtering, otherwise every "normal cholesterol"
   patient would be silently thrown away.

## The 13 features

Eleven come from the form; two are calculated by one shared function
(`compute_derived_features` in `ml_model.py`) that training imports too,
so the two sides can never drift apart.

| # | Feature | Meaning | Source |
|---|---|---|---|
| 0–10 | age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active | the 11 form inputs | form |
| 11 | `bmi` | weight ÷ height(m)², clipped to 10–70 | derived |
| 12 | `pulse_pressure` | systolic − diastolic | derived |

The **order is a contract**: the saved scaler matches values by position,
not by name, so `models/features.json` records the order and `prepare_input`
builds the vector in exactly that order.

## Preprocessing (six steps, in `train_model.py`)

1. Load the CSV (70,000 rows).
2. Basic cleaning: drop the `id` column, convert age days → years, shift
   cholesterol/glucose to the 1–3 scale.
3. Remove duplicate rows (the same patient must not be in train and test).
4. Remove physiologically impossible values — e.g. diastolic 10,000 mmHg.
   1,355 rows removed, leaving **68,645 usable records** (balance ≈ 1.02 : 1).
5. Median imputation for missing values (this file has none, but the step
   keeps the pipeline safe).
6. Feature engineering: BMI and pulse pressure.

## Training and model comparison

The clean data is split 80/20 with `stratify=y` and `random_state=42`, a
`StandardScaler` is fitted **on the training split only** (no leakage),
and three classifiers are trained and compared on the held-out 20%
(13,729 patients):

| Model | Accuracy | Precision | Recall | F1 | ROC AUC | CV AUC (5-fold) |
|---|---|---|---|---|---|---|
| Logistic Regression | 72.7% | 0.748 | 0.675 | 0.710 | 0.792 | 0.791 ± 0.004 |
| **Random Forest** | 73.0% | 0.752 | 0.678 | 0.713 | **0.800** | **0.800 ± 0.004** |
| XGBoost | 73.2% | 0.749 | 0.690 | 0.719 | 0.800 | 0.799 ± 0.004 |

**Random Forest is selected** and recorded in `models/results.json`. It
ties with XGBoost on test ROC AUC and has the best cross-validation AUC,
so it is the defensible choice on the evidence. Every random component
uses `random_state=42`, so re-running `python train_model.py` reproduces
the same models. The web application predicts with the average of all
three model probabilities (an ensemble); the per-model comparison stays
on the Model Performance page.

5-fold cross-validation runs on the training split inside a `Pipeline`,
which re-fits the scaler inside every fold — that prevents information
leaking between folds. The small standard deviations show the models are
stable, not lucky on one split.

Honest reading: the spread between the simplest and the most complex
model is about half a percentage point. On this dataset the features
carry most of the signal, and no model is dramatically better — which is
exactly why comparing them (instead of assuming) is part of the project.

## How a prediction works (the app side)

Handled by `/diagnosis` in `app.py`:

1. **Validate** — `ml_model.validate_input()` checks ranges on the server
   (age 18–120, systolic > diastolic, plausible BMI, …). The browser
   checks the same things first, but a form can be posted directly, so
   the server is the real gate.
2. **Prepare** — `ml_model.prepare_input()` builds the 13-feature vector
   in `FEATURE_ORDER`.
3. **Scale** — the saved scaler transforms the vector. It is *reused*,
   never refitted: one row has no spread, and the model must see the
   patient measured against the training population.
4. **Predict** — every saved model gives P(disease) with `predict_proba()[:, 1]`; the app averages the three probabilities (an ensemble).
5. **Decide** — probability ≥ 0.50 → "HIGH RISK", otherwise "LOW RISK".
   The same 0.50 threshold is used in training, so the app behaves
   exactly like the reported metrics.
6. **Save + show** — the inputs and result go to SQLite; the result page
   shows the risk level, the probability, and general advice.

## Route map

| Method | Route | Purpose |
|---|---|---|
| GET | `/` | Redirect to dashboard or login |
| GET, POST | `/login`, `/register` | Sign in / create a Patient account |
| GET | `/logout` | Sign out |
| GET | `/dashboard` | Totals, last result, risk-trend chart |
| GET, POST | `/diagnosis` | The assessment form / run a prediction |
| GET | `/history` | Own assessment history with filters |
| GET | `/report/<id>` | Download one assessment as a text report |
| GET, POST | `/profile` | Update name, email, password |
| GET | `/admin/users` | User list (admin only) |
| POST | `/admin/users/<id>/role`, `/ban`, `/delete` | Manage an account |
| GET | `/admin/predictions` | Every saved assessment (admin only) |
| POST | `/admin/predictions/<id>/delete` | Remove one record |
| GET | `/admin/models` | Model comparison, confusion matrix, feature importance |

---

## Security (plain and explainable)

| Measure | How |
|---|---|
| Passwords | PBKDF2-HMAC-SHA256, 260,000 rounds, per-user random salt (`database.py`) |
| Password check | `secrets.compare_digest` (constant-time) |
| Sessions | Flask signed cookies; `session.clear()` on every sign-in prevents session fixation |
| CSRF | One random token per session, embedded in every form, checked on every POST |
| SQL injection | Every query uses `?` placeholders |
| Access control | `@login_required` / `@admin_required` on every protected page; wrong role gets a 403 page |
| Privilege escalation | Registration always creates a Patient — the posted role is never read |
| Report access | Ownership checked before a report is downloaded |
| Last-admin guard | The only active admin cannot be suspended, demoted or deleted |

## Testing

```bash
python test_project.py
```

60 plain assert-style checks (no test framework) covering the shared
feature function, validation, a real end-to-end prediction with the saved
model, the database layer, and the web pages — including register → login
→ predict → history → report, the patient/admin access rules, and CSRF
rejection. The suite cleans up after itself and ends with a pass/fail
count.

## Retraining

```bash
python train_model.py
```

Rewrites everything in `models/` in about one minute and prints the
comparison table. Restart the app afterwards so it loads the fresh files.

## Dataset citation

Svetlana Ulianova, *Cardiovascular Disease dataset*, Kaggle.
<https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset>
