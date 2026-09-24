# The Machine Learning Pipeline — Heart Disease Prediction

From `data/heart.csv` to a web prediction, step by step. Implemented in
`train_model.py` (training) and `ml_model.py` (prediction).

---

## 1. The dataset

**Cardiovascular Disease dataset** — Svetlana Ulianova, Kaggle:
<https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset>

70,000 patient records: 11 measurements plus a binary `cardio` label
(1 = disease present). Two quirks of the raw file, both handled by
training:

1. **`age` is stored in days** — converted by dividing by 365.25.
2. **`cholesterol` and `gluc` are stored 0-2**, while the clinical scale
   (and the web form) use 1-3. The pipeline shifts them +1 **before**
   filtering — otherwise every normal-cholesterol patient would be
   silently deleted, which would destroy the dataset.

## 2. Preprocessing — six steps

| Step | What happens | Rows |
|---|---|---|
| 1. Load | read `data/heart.csv` | 70,000 |
| 2. Basic cleaning | drop `id`; age days → years; cholesterol/glucose shifted to 1-3 | — |
| 3. Duplicates | exact duplicate rows removed | 0 found |
| 4. Physiological filter | impossible values removed: age 18-120, height 100-250 cm, weight 20-300 kg, systolic 60-250, diastolic 40-200, systolic > diastolic, codes in 1-3 | −1,355 |
| 5. Imputation | gaps filled with the column median | 0 found |
| 6. Feature engineering | BMI and pulse pressure added | **68,645** |

Why the filter is safe: a value a human body cannot take is a recording
error, not a real patient, so the row is deleted rather than trained on.

## 3. The 13 features

Eleven come from the form; two are derived by **one shared function**,
`compute_derived_features()` in `ml_model.py`, which `train_model.py`
imports — so training and prediction compute them identically and
cannot drift apart.

| # | Feature | Source |
|---|---|---|
| 0-10 | age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active | form |
| 11 | `bmi` = weight ÷ height(m)², clipped to 10-70 | derived |
| 12 | `pulse_pressure` = systolic − diastolic | derived |

The **order is a contract**: the scaler matches values by position, so
`models/features.json` records the order and both sides use the same
constant. Reordering without retraining would feed every value into the
wrong slot — the app would still run and produce confident, wrong
numbers.

## 4. Split, then scale — in that order

```python
train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler().fit(X_train)     # training rows ONLY
```

- 54,916 training rows, 13,729 test rows; `stratify=y` keeps the class
  ratio identical in both halves.
- Fitting the scaler on training data only prevents **data leakage** —
  the test rows' statistics never touch training.
- Cross-validation (5-fold) wraps the scaler in a `Pipeline`, so it is
  re-fitted inside every fold — no leakage between folds either.
- At prediction time the saved scaler is **reused, never refitted**.

## 5. The three models

| Model | Type | Settings |
|---|---|---|
| Logistic Regression | linear | max_iter=2000, class_weight="balanced" |
| Random Forest | bagging | 200 trees, max_depth=12, min_samples_split=10 |
| XGBoost | boosting | 300 rounds, depth 6, learning_rate 0.05 |

All use `random_state=42`, so retraining reproduces the same models.

## 6. Results — held-out test set (13,729 patients)

| Model | Accuracy | Precision | Recall | F1 | ROC AUC | CV AUC (5-fold) |
|---|---|---|---|---|---|---|
| Logistic Regression | 72.7% | 0.748 | 0.675 | 0.710 | 0.7924 | 0.7909 ± 0.0040 |
| **Random Forest** | 73.0% | 0.752 | 0.678 | 0.713 | **0.8000** | **0.7995 ± 0.0036** |
| XGBoost | 73.2% | 0.749 | 0.690 | 0.719 | 0.8000 | 0.7986 ± 0.0038 |

Honest reading: the spread from the simplest to the most complex model
is about half a percentage point — the features carry the signal.
Random Forest is the best single model: it ties XGBoost on test ROC AUC
and wins cross-validation.

## 7. Prediction — what the web app does

```
validate_input()  ranges, systolic > diastolic, plausible BMI
      ↓
prepare_input()   11 answers → 13 features, in FEATURE_ORDER
      ↓
scaler.transform  the saved scaler, reused
      ↓
predict_proba()   every saved model gives P(disease)
      ↓
AVERAGE           the ensemble probability
      ↓
≥ 0.50            HIGH RISK, otherwise LOW RISK
```

The per-model probabilities are shown on the result page, and the
stored record names the model as "Ensemble (3 models)".

## 8. Reproducibility and honest limitations

Every random component uses `random_state=42` — re-running
`python train_model.py` reproduces the same models. Limitations worth
stating in a viva: about 27% of predictions are wrong; recall is the
weaker side at the 0.50 threshold (0.678); three inputs are
self-reported; the dataset's population is undocumented; and the models
find correlation, not causation.
