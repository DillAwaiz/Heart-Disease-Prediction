"""
Heart Disease Prediction - database layer
==========================================
Every read and write to heart_disease.db (SQLite) happens in this file.

Two tables:
    users        accounts (patients and admins), with hashed passwords
    predictions  one row per saved risk assessment

Two things worth knowing:
  * Passwords are hashed with PBKDF2-HMAC-SHA256 and a per-user random
    salt. The plain password is never stored.
  * Every query is parameterised with ? placeholders, so SQL injection
    is not possible.

init_db() creates the tables and seeds one admin account the first time
the application runs.
"""

import os
import json
import sqlite3
import hashlib
import secrets

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "heart_disease.db")

# 260,000 rounds makes brute-force password guessing very slow
PBKDF2_ITERATIONS = 260000

ROLE_PATIENT = "Patient"
ROLE_ADMIN = "Admin"


# ─────────────────────────────────────────────
# Password hashing
# ─────────────────────────────────────────────
def hash_password(password):
    """Hash a password with a fresh random salt."""
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt,
                                 PBKDF2_ITERATIONS)
    return f"pbkdf2${PBKDF2_ITERATIONS}${salt.hex()}${digest.hex()}"


def verify_password(password, stored):
    """Check a typed password against the stored hash."""
    if not stored:
        return False
    try:
        _, iterations, salt_hex, hash_hex = stored.split("$")
        digest = hashlib.pbkdf2_hmac("sha256", password.encode(),
                                     bytes.fromhex(salt_hex), int(iterations))
        return secrets.compare_digest(digest.hex(), hash_hex)
    except (ValueError, TypeError):
        return False


def _connect():
    """Open the database with foreign keys switched ON."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    """Create any missing tables and seed one admin account if empty."""
    conn = _connect()
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL,
            fullname TEXT,
            email TEXT,
            is_banned INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""")

    c.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            age REAL, gender INTEGER, height REAL, weight REAL,
            ap_hi REAL, ap_lo REAL, cholesterol INTEGER, gluc INTEGER,
            smoke INTEGER, alco INTEGER, active INTEGER,
            predicted_class INTEGER, probability REAL, model_used TEXT,
            votes_json TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
        )""")

    # Small migration: databases created before votes_json existed get the
    # new column added, so each prediction can also store every single
    # model's probability. Safe to run repeatedly.
    columns = [row[1] for row in c.execute("PRAGMA table_info(predictions)")]
    if "votes_json" not in columns:
        c.execute("ALTER TABLE predictions ADD COLUMN votes_json TEXT")

    # Seed one admin account the very first time, so there is always a
    # way to sign in and manage the system.
    c.execute("SELECT COUNT(*) FROM users")
    if c.fetchone()[0] == 0:
        c.execute("""
            INSERT INTO users (username, password_hash, role, fullname, email)
            VALUES (?, ?, ?, ?, ?)
        """, ("admin", hash_password("admin123"), ROLE_ADMIN,
              "Administrator", "admin@example.com"))

    conn.commit()
    conn.close()


# ─────────────────────────────────────────────
# Authentication
# ─────────────────────────────────────────────
def validate_login(username, password):
    """
    Check a username and password.

    Returns (user_dict, "ok") on success, or (None, reason) where
    reason is "invalid" or "banned". The returned dict never contains
    the password hash.
    """
    conn = _connect()
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username=?", (username,))
    row = c.fetchone()
    conn.close()

    if not row:
        return None, "invalid"
    if not verify_password(password, row["password_hash"]):
        return None, "invalid"
    if row["is_banned"]:
        return None, "banned"

    user = dict(row)
    del user["password_hash"]
    return user, "ok"


def register_user(username, password, role, fullname, email):
    """
    Create a new account.

    Returns (new_id, None) on success, or (None, "taken") if the
    username already exists.
    """
    conn = _connect()
    c = conn.cursor()
    try:
        c.execute("""
            INSERT INTO users (username, password_hash, role, fullname, email)
            VALUES (?, ?, ?, ?, ?)
        """, (username, hash_password(password), role, fullname, email))
        conn.commit()
        return c.lastrowid, None
    except sqlite3.IntegrityError:
        return None, "taken"
    finally:
        conn.close()


def username_exists(username):
    """True if this username is already taken."""
    conn = _connect()
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM users WHERE LOWER(username)=LOWER(?)",
              (username.strip(),))
    found = c.fetchone()[0] > 0
    conn.close()
    return found


def email_exists(email):
    """True if any account already uses this email address."""
    if not email:
        return False
    conn = _connect()
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM users WHERE LOWER(email)=LOWER(?)",
              (email.strip(),))
    found = c.fetchone()[0] > 0
    conn.close()
    return found


# ─────────────────────────────────────────────
# Users
# ─────────────────────────────────────────────
def get_all_users():
    """Return every account as a list of dicts, ordered by id."""
    conn = _connect()
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("""SELECT id, username, role, fullname, email, is_banned, created_at
                 FROM users ORDER BY id""")
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows


def get_user_by_id(user_id):
    """Fetch one user row as a dict, or None."""
    conn = _connect()
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("""SELECT id, username, role, fullname, email, is_banned, created_at
                 FROM users WHERE id=?""", (user_id,))
    row = c.fetchone()
    conn.close()
    return dict(row) if row else None


def update_user_profile(user_id, fullname, email, new_password=None):
    """Save a user's name and email, and optionally set a new password."""
    conn = _connect()
    c = conn.cursor()
    if new_password:
        c.execute("UPDATE users SET fullname=?, email=?, password_hash=? WHERE id=?",
                  (fullname, email, hash_password(new_password), user_id))
    else:
        c.execute("UPDATE users SET fullname=?, email=? WHERE id=?",
                  (fullname, email, user_id))
    conn.commit()
    conn.close()


def update_user_role(user_id, new_role):
    """Change a user's role."""
    conn = _connect()
    conn.execute("UPDATE users SET role=? WHERE id=?", (new_role, user_id))
    conn.commit()
    conn.close()


def ban_user(user_id):
    """Suspend an account so it can no longer sign in."""
    conn = _connect()
    conn.execute("UPDATE users SET is_banned=1 WHERE id=?", (user_id,))
    conn.commit()
    conn.close()


def unban_user(user_id):
    """Reinstate a suspended account."""
    conn = _connect()
    conn.execute("UPDATE users SET is_banned=0 WHERE id=?", (user_id,))
    conn.commit()
    conn.close()


def delete_user(user_id):
    """
    Permanently remove an account.

    Foreign keys are ON (see _connect), so this also removes the user's
    saved predictions instead of leaving them behind.
    """
    conn = _connect()
    conn.execute("DELETE FROM users WHERE id=?", (user_id,))
    conn.commit()
    conn.close()


def count_active_admins():
    """How many Admin accounts can currently sign in."""
    conn = _connect()
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM users WHERE role=? AND is_banned=0",
              (ROLE_ADMIN,))
    total = c.fetchone()[0]
    conn.close()
    return total


def count_predictions_by_user():
    """Return {user_id: number_of_predictions} for each account."""
    conn = _connect()
    c = conn.cursor()
    c.execute("SELECT user_id, COUNT(*) FROM predictions GROUP BY user_id")
    counts = {row[0]: row[1] for row in c.fetchall()}
    conn.close()
    return counts


# ─────────────────────────────────────────────
# Predictions
# ─────────────────────────────────────────────
def add_prediction(user_id, values, predicted_class, probability, model_used,
                   votes=None):
    """
    Save one completed risk assessment and return its new id.

    `values` is the readable dict from ml_model.prepare_input(), holding
    the 11 raw measurements. `votes` holds each single model's
    probability, stored as JSON so a report can show single + ensemble
    results later without re-running the models.
    """
    conn = _connect()
    c = conn.cursor()
    c.execute("""
        INSERT INTO predictions
        (user_id, age, gender, height, weight, ap_hi, ap_lo,
         cholesterol, gluc, smoke, alco, active,
         predicted_class, probability, model_used, votes_json)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (user_id, values["age"], values["gender"], values["height"],
          values["weight"], values["ap_hi"], values["ap_lo"],
          values["cholesterol"], values["gluc"], values["smoke"],
          values["alco"], values["active"],
          predicted_class, probability, model_used,
          json.dumps(votes) if votes else None))
    conn.commit()
    new_id = c.lastrowid
    conn.close()
    return new_id


def get_predictions(user_id=None):
    """
    Return saved assessments, newest first.

    Passing a user_id returns only that person's records - this one rule
    stops a patient seeing anybody else's data. Passing None returns
    every record (with the account username), which is what the admin
    pages show.
    """
    conn = _connect()
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    if user_id:
        c.execute("SELECT * FROM predictions WHERE user_id=? ORDER BY timestamp DESC",
                  (user_id,))
    else:
        c.execute("""SELECT p.*, u.username AS account
                     FROM predictions p JOIN users u ON p.user_id=u.id
                     ORDER BY p.timestamp DESC""")
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows


def get_prediction_by_id(pred_id):
    """Fetch one prediction, joined with the account that created it."""
    conn = _connect()
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("""SELECT p.*, u.username AS account, u.fullname AS account_name
                 FROM predictions p JOIN users u ON p.user_id=u.id
                 WHERE p.id=?""", (pred_id,))
    row = c.fetchone()
    conn.close()
    return dict(row) if row else None


def delete_prediction(pred_id):
    """Remove a single assessment."""
    conn = _connect()
    conn.execute("DELETE FROM predictions WHERE id=?", (pred_id,))
    conn.commit()
    conn.close()
