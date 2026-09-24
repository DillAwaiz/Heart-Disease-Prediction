# Viva Questions and Answers — Heart Disease Prediction

Questions an examiner is likely to ask about THIS code, with short answers
grounded in the actual files. Read each answer, then practise saying it in
your own words. Where a weakness exists, say it yourself — examiners
respect that far more than a claimed perfection.

Files you should be able to open and point at while answering:
`app.py`, `database.py`, `ml_model.py`, `train_model.py`, `models/results.json`.

---

## A. The project overall

**Q1. Explain your project in one minute.**
It is a Flask web application that estimates cardiovascular disease risk
from eleven health measurements — age, gender, height, weight, both blood
pressures, cholesterol, glucose, smoking, alcohol and activity. The Kaggle
cardio dataset (70,000 records) is cleaned, two features are derived (BMI
and pulse pressure), and three models are trained and compared: Logistic
Regression, Random Forest and XGBoost. Random Forest was the best single
model (ROC AUC 0.800), and the web app predicts with the average of the
three models' probabilities (an ensemble). The probability is shown as
LOW or HIGH RISK, and each assessment is stored in SQLite so the user
keeps a personal history.

**Q2. Why Flask and not Django or Streamlit?**
Flask is small and explicit: every route, check and query is code I wrote
and can explain. Django would bring an ORM and an admin site that answer
the interesting questions for me. Streamlit re-runs the whole script on
every interaction, which makes real login sessions and per-role pages
awkward. This project needs only a handful of routes, so Flask fits.

**Q3. Walk me through what happens when a user submits the form.**
`POST /diagnosis` in `app.py` does five steps in order: (1) the CSRF token
is checked before the view runs; (2) `ml_model.validate_input()` checks
every value server-side; (3) `ml_model.prepare_input()` turns the 11
answers into a 13-feature vector; (4) `ml_model.predict()` scales the
vector with the saved scaler, calls every saved model's
`predict_proba` and averages the results (an ensemble); (5) the result is saved through `database.py` and the
result page is rendered.

**Q4. What are the two roles and what can each do?**
Patient: register, sign in, run assessments, see own history, download own
reports, edit own profile. Admin: everything a patient can do, plus the
Users page (change role, suspend, delete), the All Predictions page
(everyone's records), and the Model Performance page. An admin also sees
everyone's history because `_scope_user_id()` returns None for admins.

## B. Dataset and preprocessing

**Q5. What dataset did you use?**
The Cardiovascular Disease dataset by Svetlana Ulianova on Kaggle: 70,000
patient records with 11 measurements and a binary `cardio` label. After
cleaning, 68,645 records are used, almost perfectly balanced
(34,684 without disease, 33,961 with).

**Q6. What cleaning did you do?** (`train_model.py`, `load_and_clean`)
Six steps: drop the `id` column; convert age from days to years; shift
cholesterol and glucose from the 0–2 scale they are stored on to the
clinical 1–3 scale; remove duplicate rows; remove physiologically
impossible values (1,355 rows — for example diastolic readings of 10,000);
and median imputation for missing values.

**Q7. Why does the cholesterol shift matter so much?**
The raw file stores cholesterol and glucose as 0, 1, 2, but every filter
and the web form use 1, 2, 3. If you filter for values 1–3 *before*
shifting, every "normal cholesterol" patient (value 0) is silently
discarded — most of the dataset — and the remaining data is biased towards
sick patients while looking perfectly fine. It is a good example of a
silent data bug being worse than a crash.

**Q8. How did you decide which rows to delete?**
Only rows that describe a body that cannot exist: age outside 18–120,
height outside 100–250 cm, weight outside 20–300 kg, systolic outside
60–250, diastolic outside 40–200, systolic not above diastolic, or
cholesterol/glucose codes outside 1–3. These are recording errors, not
statistical outliers, so deleting them is safe.

**Q9. Why 80/20 train/test split, and what does stratify do?**
The split keeps 80% for training and holds out 20% (13,729 patients) that
the models never see until evaluation — that is what makes the reported
accuracy honest. `stratify=y` keeps the class ratio the same in both
halves, so the test set is representative instead of lucky.
`random_state=42` makes the split reproducible.

**Q10. What is data leakage and where could it happen in your project?**
Leakage is test-set information sneaking into training, making scores
optimistic. Two places it could happen here, both prevented: the scaler is
fitted on the training split only (never on all data), and inside
cross-validation the scaler is re-fitted within each fold because the
pipeline refits it from that fold's training rows.

## C. Features and scaling

**Q11. What are your 13 features?**
Eleven raw inputs from the form, plus two derived ones: BMI
(weight ÷ height in metres, squared, clipped to 10–70) and pulse pressure
(systolic minus diastolic).

**Q12. Why derive BMI and pulse pressure at all?**
BMI turns height and weight into one clinically meaningful number, and it
is non-linear (a ratio with a square), which a linear model cannot build
from the raw columns on its own. Pulse pressure is an established
cardiovascular marker in its own right — arterial stiffness shows up in
the *gap* between the two readings, not in either one alone.

**Q13. Where is the feature-engineering code, and why only in one place?**
In `compute_derived_features()` in `ml_model.py`. `train_model.py` imports
that exact function and applies it to every dataset row, so training and
prediction compute BMI and pulse pressure identically — the two
implementations physically cannot drift apart. My test suite checks this
(identity check in `test_project.py`).

**Q14. Why do you scale the data, and why fit the scaler only on training data?**
Age is around 55, height around 165, smoke is 0 or 1 — Logistic Regression
would let the bigger numbers dominate purely because of their size.
StandardScaler rewrites every column as "how many standard deviations from
the mean". The scaler must be fitted on training data only: if it saw the
test rows, their means and spreads would leak into training and the score
would be optimistic.

**Q15. What happens to the scaler at prediction time?**
It is unpickled once at startup and *reused*, never refitted
(`ml_model.init()` and `predict()`). Refitting on one patient is
meaningless — a single row has no spread — and the model must see the
patient measured against the same population it learned from.

**Q16. Why is feature ORDER so important in your app?**
The scaler matches values by position, not by name — it stores one mean
and one standard deviation per slot. If the vector order changed without
retraining, every value would be scaled with the wrong statistics: the app
would keep working and produce confident, wrong numbers. That is why
`models/features.json` records the order and my test compares it against
the constant in `ml_model.py`.

## D. Models and evaluation

**Q17. Which models did you train and why those three?**
Logistic Regression — the interpretable linear baseline; Random Forest —
bagging: 200 trees on random samples of the data, averaged; XGBoost —
boosting: each tree corrects the previous ones' errors. Together they
cover the three main approaches for tabular data, which makes the
comparison meaningful rather than decorative.

**Q18. What were the results?** (see `models/results.json`)
On the held-out test set: Logistic Regression 72.7% accuracy / 0.792 ROC
AUC; Random Forest 73.0% / 0.800; XGBoost 73.2% / 0.800. Random Forest
was selected: it ties with XGBoost on test AUC and has the best 5-fold
cross-validation AUC (0.7995 vs 0.7986). The honest observation is that
all three are within about half a percentage point — the features carry
the signal, not the model choice.

**Q19. Why does the app predict with an ensemble rather than one model?**
The three models use the same 13 features and land within 0.7 points of
each other, so averaging them buys stability, not a big accuracy jump —
the honest answer is that the ensemble scores roughly the same ROC AUC
(about 0.80) as the best single model. It still adds two things: the
result does not depend on one model's quirks, and the result page can
show each model's probability, demonstrating that three different
approaches — linear, bagging and boosting — broadly agree. Training still
records the best single model (Random Forest, ROC AUC 0.800) for the
comparison on the Model Performance page.

**Q20. Explain precision and recall in your project's words.**
Precision: of the people the model calls HIGH RISK, how many really have
disease — it measures false alarms. Recall: of the people who really have
disease, how many the model catches — it measures missed cases. My Random
Forest gets precision 0.752 and recall 0.678: it is more reliable when it
raises an alarm than when it says nothing, which is the normal trade-off
at the standard 0.5 threshold.

**Q21. What is a confusion matrix? Show me your selected model's one.**
Rows are the true outcomes, columns the predictions: TN, FP / FN, TP. It
is on the Model Performance page and in `results.json`. From it you can
compute everything else: recall = TP/(TP+FN), precision = TP/(TP+FP).

**Q22. Why did you use cross-validation, and how does it work here?**
A single train/test score can be lucky. 5-fold cross-validation splits
the training data into five parts, trains on four and validates on one,
five times, and reports the mean and standard deviation. My standard
deviation is about 0.004, so the models are stable. The scaler is inside
the pipeline, so it is re-fitted inside each fold — no leakage.

**Q23. Why is 73% accuracy acceptable?**
The dataset is deliberately hard: it is a general screening population,
not a hospital cohort, and about 1 in 4 predictions being wrong is
documented on every page via the disclaimer. The value of the project is
the complete, correct pipeline — cleaning, honest evaluation, a deployed
model — and stating the limitation is part of the evaluation.

## E. The Flask application

**Q24. Why is there no app factory or blueprints?**
Those patterns solve problems this app does not have — many route modules,
per-test app configurations, reusable sub-apps. With one route file, a
plain `app = Flask(__name__)` in `app.py` is easier to read, debug and
defend. I removed the factory deliberately during simplification.

**Q25. How do login sessions work?**
On successful login, `database.validate_login()` checks the PBKDF2 hash
and returns the user row without the password hash; the row goes into the
Flask `session`, a cookie signed with `SECRET_KEY` so it cannot be
tampered with. `session.clear()` before storing the user gives a fresh
session on every sign-in (prevents session fixation). `@login_required`
sends visitors to login; `@admin_required` returns 403 for non-admins.

**Q26. How is CSRF handled?**
Every form embeds a hidden `csrf_token` — one random hex value per
session, created by `csrf_token()` in `app.py`. A `before_request` hook
compares the posted token with the session's using
`secrets.compare_digest` on every POST and returns 400 on mismatch. A
page on another website can make your browser POST, but it cannot *read*
your session cookie, so it cannot know the token.

**Q27. How do you stop SQL injection?**
Every query in `database.py` is parameterised with `?` placeholders — user
input is never formatted into an SQL string.

**Q28. Where does the model load, and why there?**
In `ml_model.init()`, called once at startup in `app.py`. Loading 30 MB of
pickles per request would make the site unusably slow; loading once means
a prediction costs no disk access.

**Q29. What happens if the models folder is missing?**
`init()` prints a warning instead of crashing, `is_ready()` returns False,
and the app shows a "no trained model" message telling you to run
`python train_model.py`. The rest of the site (login, history) keeps
working.

## F. Database

**Q30. Describe your database.**
SQLite, one file, two tables. `users`: id, username, unique password hash,
role, fullname, email, is_banned, created_at. `predictions`: the 11 raw
inputs, predicted_class, probability, model_used, timestamp, and a
foreign key to users with ON DELETE CASCADE — deleting an account removes
its predictions too. Storing the raw inputs alongside the result means a
report can be regenerated later without re-running the model.

**Q31. How are passwords stored?**
PBKDF2-HMAC-SHA256 with 260,000 iterations and a 16-byte random salt per
user, stored as `pbkdf2$iterations$salt$hash` (`database.py`). The plain
password is never stored, the salt defeats rainbow tables, and the high
iteration count makes brute-force guessing slow. Comparison uses
`secrets.compare_digest`, which does not leak timing information.

**Q32. How does a patient get only their own history?**
One rule in one place: `_scope_user_id()` in `app.py` returns the user's
id for patients and None for admins, and `database.get_predictions()`
filters by it. There is no second query to forget to protect.

## G. Honest limitations (raise these yourself)

**Q33. What are the weaknesses of your project?**
About 27% of predictions are wrong — this is a screening aid, not a
diagnosis. Recall (0.678) is the weaker side at the 0.5 threshold, and a
real deployment might lower the threshold to catch more cases at the cost
of more false alarms. Three inputs are self-reported and under-reported.
The dataset is undocumented, so generalisation to other populations is
unknown. The model finds correlation, not causation.

**Q34. What would you do next to improve it?**
Threshold tuning with a clinician, probability calibration, testing on an
external dataset, and possibly more engineered features (e.g. MAP — mean
arterial pressure). On the web side: password reset by email and
pagination for large history tables.

---

## One-breath summary to memorise

"Flask app, SQLite storage, and three saved models averaged into one
ensemble prediction. Kaggle cardio data of 70,000 rows, cleaned to
68,645, eleven inputs plus BMI and pulse pressure become thirteen
features, scaled with the scaler fitted on training data only, three
models compared with accuracy, precision, recall, F1, ROC AUC and 5-fold
cross-validation, and their average serves every prediction at a 0.5
threshold. Login, roles and CSRF are implemented by hand, and a 60-check
test suite proves the pipeline end to end."
