# Architecture — Heart Disease Prediction

How the simplified system is put together. Everything described here is
visible in the four root files: `app.py`, `database.py`, `ml_model.py`
and `train_model.py`.

---

## 1. The shape of the system

```
┌──────────────────────────────────────────────┐
│ PRESENTATION   templates/  static/           │
│                Jinja pages, CSS, vanilla JS  │
├──────────────────────────────────────────────┤
│ ROUTES         app.py                        │
│                every route, session, CSRF    │
├──────────────────────────────────────────────┤
│ DOMAIN         database.py     ml_model.py   │
│                SQLite +        scaler +      │
│                passwords       3 models      │
├──────────────────────────────────────────────┤
│ DATA           heart_disease.db   models/    │
└──────────────────────────────────────────────┘

data/heart.csv ──► train_model.py ──► models/   (run once)
```

The important property: **the domain layer knows nothing about Flask**.
`database.py` and `ml_model.py` never import `request` or `session`.
That is why the test suite can exercise them with no server running.

## 2. Startup (`python app.py`)

1. `database.init_db()` — creates the two tables if missing and seeds
   the admin account the first time.
2. `ml_model.init()` — loads `models/features.json`, the saved scaler,
   and all three model pickles, once (not per request).
3. Flask starts on http://127.0.0.1:5000.

## 3. The life of a request

```
Browser → Flask routing → @login_required / @admin_required
        → CSRF check (POST only) → view function
        → render_template → Browser
```

View functions call `database` and `ml_model` — they never write SQL
themselves. Errors get small templates (400 bad CSRF token, 403 wrong
role, 404, 500); the real traceback goes to the server log only.

## 4. Walkthrough: one prediction (`POST /diagnosis`)

1. CSRF token checked, before the view runs.
2. `ml_model.validate_input(form)` — ranges and cross-field rules,
   on the server. The browser check is only a convenience.
3. `ml_model.prepare_input(form)` — the 11 answers become a
   13-feature vector in `FEATURE_ORDER`.
4. `ml_model.predict(vector)` — the saved scaler transforms the vector
   (reused, never refitted), every saved model gives
   `predict_proba()[:, 1]`, and the **average** is the final
   probability (the ensemble). Individual probabilities are returned
   as "votes".
5. `database.add_prediction(...)` — inputs + result saved.
6. `result.html` — risk card + per-model table + advice.

## 5. The role model

| | Patient | Admin |
|---|:---:|:---:|
| Run assessments, own history, own reports, own profile | ✅ | ✅ |
| Users page (role, suspend, delete) | — | ✅ |
| All Predictions page (everyone's records) | — | ✅ |
| Model Performance page | — | ✅ |

One scoping rule serves every page: `_scope_user_id()` in app.py
returns the user's id for a patient and `None` for an admin, and
`database.get_predictions()` filters by it.

Two safety rules on the admin pages: nobody may act on their own
account, and the last active administrator cannot be suspended,
demoted or deleted.

## 6. The database

Two tables, one SQLite file.

- `users`: id, username (UNIQUE), password_hash, role, fullname,
  email, is_banned, created_at.
- `predictions`: the 11 raw inputs, predicted_class, probability,
  model_used, timestamp, and a foreign key to users with
  ON DELETE CASCADE.

`PRAGMA foreign_keys = ON` is set on every connection, and every query
uses `?` placeholders.

## 7. Security summary

| Measure | Where |
|---|---|
| PBKDF2-HMAC-SHA256, 260,000 rounds, per-user salt | database.py |
| Constant-time password comparison | database.py |
| `session.clear()` on every sign-in (anti-fixation) | app.py |
| CSRF token checked on every POST | app.py |
| Server-side input validation | ml_model.py |
| `@login_required` / `@admin_required` | app.py |
| Registration hard-coded to Patient | app.py |
| Ownership check before report download | app.py |
| Last-admin protection | app.py |

## 8. Extending the system

- **A new page:** add a function in app.py with `@login_required` or
  `@admin_required`, a template extending base.html, and — for POST
  forms — the hidden `csrf_token` field.
- **A change of features:** `FEATURE_ORDER` in ml_model.py is a
  positional contract with `scaler.pkl`. Change the list and retrain
  (`python train_model.py`) together, never separately.
