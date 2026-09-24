# File Guide — Heart Disease Prediction

Every file and folder in the simplified project, what it does, and what
depends on it.

---

## 1. Root files

| File | What it does |
|---|---|
| `app.py` | The whole Flask application: every route, login sessions, CSRF, role checks, admin pages, error pages. Entry point — `python app.py` |
| `database.py` | The only file that talks to SQLite. Two tables, PBKDF2 password hashing, `init_db()` runs at startup |
| `ml_model.py` | Loads the scaler and all three models once. Holds `validate_input()`, `prepare_input()`, `predict()` (ensemble average), and `compute_derived_features()` — the feature function training imports too |
| `train_model.py` | The training pipeline: clean → features → split → scale → train 3 models → evaluate → save into models/ |
| `test_project.py` | 59 plain-assert checks. Run with `python test_project.py` |
| `requirements.txt` | Flask, pandas, numpy, scikit-learn, xgboost |
| `README.md` | Setup, usage, metrics, security |
| `CHANGES.md` | The log of every change made to the project |
| `.gitignore` | Standard Python ignores; `heart_disease.db` is never committed |

## 2. `data/`

| File | What it is |
|---|---|
| `heart.csv` | The Kaggle Cardiovascular Disease dataset, 70,000 rows. Two quirks handled by training: age stored in days, cholesterol/glucose stored 0-2 |

## 3. `models/` — written by `train_model.py`

| File | What it is |
|---|---|
| `scaler.pkl` | The StandardScaler, fitted on the training split only. Reused at prediction time, never refitted |
| `logistic_regression.pkl` | Trained Logistic Regression |
| `random_forest.pkl` | Trained Random Forest (the best single model) |
| `xgboost.pkl` | Trained XGBoost |
| `features.json` | The 13 feature names, in the exact order the scaler expects |
| `results.json` | Every model's metrics, CV results, and `best_model` |

## 4. `templates/`

| File | Page |
|---|---|
| `base.html` | Layout: sidebar, flash messages, disclaimer |
| `login.html`, `register.html` | Sign-in and sign-up (Patient only) |
| `dashboard.html` | KPI cards + risk-trend chart |
| `predict.html` | The assessment form: 11 inputs, live BMI |
| `result.html` | Risk card + per-model probabilities table + advice |
| `history.html` | Saved assessments with filters |
| `profile.html` | Update name, email, password |
| `users.html` | Admin: manage accounts |
| `predictions.html` | Admin: every saved assessment |
| `models.html` | Admin: model comparison, confusion matrix, feature importance |
| `errors/400, 403, 404, 500` | Small error pages |

## 5. `static/`

| File | What it does |
|---|---|
| `css/style.css` | The whole theme |
| `js/main.js` | Sidebar, flash dismissal, confirm dialogs, applies `data-` attributes (bar widths, background images) |
| `js/predict.js` | Live BMI display + blood-pressure cross-checks on the form |
| `js/register.js` | Instant sign-up feedback |
| `js/charts.js` + `js/chart.min.js` | The one dashboard risk-trend chart (Chart.js, bundled — works offline) |
| `fonts/`, `img/` | Self-hosted Inter font and the heart illustration (works offline) |

## 6. `docs/`

`architecture.md`, `file_guide.md` (this file), `ml_pipeline.md`, and
`VIVA_QUESTIONS.md` — documentation for the project itself.

## 7. Dependency map

```
app.py ─────────► database.py ──────► heart_disease.db
   │
   └────────────► ml_model.py ────► models/*.pkl

train_model.py ─► ml_model.py ───► data/heart.csv ──► models/
test_project.py → all of the above
```

`app.py` is the only file that imports Flask. `database.py` and
`ml_model.py` are pure Python, which is why scripts and tests can use
them without a web server.
