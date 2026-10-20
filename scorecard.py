"""
Credit Risk Scorecard - Traditional Bank Style
==========================================
Interpretable logistic regression PD model with:
  - leakage-safe preprocessing (split first, fit on train only)
  - median imputation + outlier capping + standardization
  - AUC / Gini / KS validation
  - Risk Weight Matrix (ranked standardized coefficients)
  - PD -> 300-850 score conversion (points-to-double-odds)
  - Expected Loss = PD x LGD x EAD (illustrative LGD / EAD assumptions)

Data: Kaggle "Give Me Some Credit" (cs-training.csv), saved as data/credit_data.csv
Run:  python scorecard.py
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "credit_data.csv"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)
MODEL_FILE = OUT_DIR / "scorecard_model.pkl"
MATRIX_FILE = OUT_DIR / "risk_weight_matrix.csv"
TARGET = "SeriousDlqin2yrs"  # 1 = 90+ days past due within 2 years
SEED = 42

# Score scaling: 650 points = 20:1 good:bad odds, every 50 points doubles the odds
# (wide PDO spreads borrowers across the familiar 300-850 range)
BASE_SCORE, BASE_ODDS, PDO = 650, 20, 50
SCORE_MIN, SCORE_MAX = 300, 850

# Expected Loss = PD x LGD x EAD
# The dataset has no balances or recoveries, so LGD and EAD use illustrative,
# industry-typical assumptions for unsecured credit cards (clearly labelled).
CREDIT_LIMIT = 10_000   # assumed limit per borrower ($)
CCF = 0.50              # credit conversion factor: share of unused limit drawn before default
LGD = 0.80              # share of exposure lost after default (unsecured cards ~75-85%)

PAST_DUE_COLS = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
]


def line(char="=", n=72):
    print(char * n)


# ---------------------------------------------------------------------------
# Step 1 - Data ingestion and rule-based cleaning
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_FILE)
df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])

# 96 / 98 in the past-due counts are data-entry codes, not real counts.
# Treat them as missing so median imputation handles them.
for col in PAST_DUE_COLS:
    if col in df.columns:
        df.loc[df[col] >= 96, col] = np.nan

X = df.drop(columns=[TARGET])
y = df[TARGET]

line()
print("CREDIT RISK SCORECARD - LOGISTIC REGRESSION")
line()
print(f"Borrowers: {len(df):,} | Features: {X.shape[1]} | Default rate: {y.mean():.2%}")
missing = X.isna().mean()
missing = missing[missing > 0].sort_values(ascending=False)
if len(missing):
    print("Missing values (filled with training-set median):")
    for col, pct in missing.items():
        print(f"  {col:<42} {pct:6.1%}")

# ---------------------------------------------------------------------------
# Step 2 - Stratified 80/20 split BEFORE any fitting (prevents leakage)
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=SEED
)

# Outlier capping at the training set's 1st / 99th percentiles
caps = {c: (X_train[c].quantile(0.01), X_train[c].quantile(0.99)) for c in X.columns}


def cap(data):
    data = data.copy()
    for c, (lo, hi) in caps.items():
        data[c] = data[c].clip(lo, hi)
    return data


X_train, X_test = cap(X_train), cap(X_test)

# Quick multicollinearity check - correlated inputs make coefficients unstable
corr = X_train.corr().abs()
pairs = [
    (a, b, corr.loc[a, b])
    for i, a in enumerate(corr.columns)
    for b in corr.columns[i + 1:]
    if corr.loc[a, b] > 0.7
]
print(f"\nHighly correlated feature pairs (|r| > 0.7): {len(pairs)}")
for a, b, r in pairs:
    print(f"  {a} <-> {b}: {r:.2f}")

# ---------------------------------------------------------------------------
# Steps 3 & 4 - Imputation + scaling + logistic regression in one pipeline
# (imputer and scaler learn only from training data)
# ---------------------------------------------------------------------------
model = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
    ("logit", LogisticRegression(max_iter=1000)),
])
model.fit(X_train, y_train)

# ---------------------------------------------------------------------------
# Validation - the metrics credit teams use (accuracy is meaningless at ~7% defaults)
# ---------------------------------------------------------------------------
pd_test = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, pd_test)
fpr, tpr, _ = roc_curve(y_test, pd_test)
ks = np.max(tpr - fpr)

print()
line()
print("MODEL VALIDATION (20% hold-out)")
line()
print(f"  AUC  : {auc:.3f}")
print(f"  Gini : {2 * auc - 1:.3f}   (2 x AUC - 1)")
print(f"  KS   : {ks:.3f}   (max separation of good vs bad)")
print(f"  Calibration: predicted PD {pd_test.mean():.2%} vs actual default rate {y_test.mean():.2%}")

# ---------------------------------------------------------------------------
# Step 5 - Risk Weight Matrix
# ---------------------------------------------------------------------------
coefs = model.named_steps["logit"].coef_[0]
matrix = pd.DataFrame({"Feature": X.columns, "Coefficient": coefs})
matrix["Odds_x_per_1SD"] = np.exp(matrix["Coefficient"])
matrix["Direction"] = np.where(matrix["Coefficient"] > 0, "RAISES risk", "lowers risk")
matrix = matrix.reindex(matrix["Coefficient"].abs().sort_values(ascending=False).index)
matrix.insert(0, "Rank", range(1, len(matrix) + 1))

print()
line()
print("RISK WEIGHT MATRIX (standardized coefficients, ranked by impact)")
line()
print(f"{'Rank':<5}{'Feature':<42}{'Coef':>8}{'Odds x':>9}  Direction")
line("-")
for _, r in matrix.iterrows():
    print(f"{r['Rank']:<5}{r['Feature']:<42}{r['Coefficient']:>+8.3f}"
          f"{r['Odds_x_per_1SD']:>9.2f}  {r['Direction']}")
line("-")
print("Odds x = how default odds multiply for a 1 standard-deviation increase.")

# ---------------------------------------------------------------------------
# Score conversion: PD -> 300-850 (points to double the odds)
# ---------------------------------------------------------------------------
factor = PDO / np.log(2)
offset = BASE_SCORE - factor * np.log(BASE_ODDS)


def pd_to_score(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    good_bad_odds = (1 - p) / p
    return np.clip(np.round(offset + factor * np.log(good_bad_odds)), SCORE_MIN, SCORE_MAX)


scores = pd_to_score(pd_test)
bands = pd.cut(scores, bins=[299, 579, 669, 739, 799, 850],
               labels=["Poor 300-579", "Fair 580-669", "Good 670-739",
                       "V.Good 740-799", "Excep 800-850"])
band_table = (
    pd.DataFrame({"Band": bands, "Default": y_test.values})
    .groupby("Band", observed=False)["Default"]
    .agg(Borrowers="count", Default_Rate="mean")
)

print()
line()
print(f"CREDIT SCORE BANDS (hold-out) | {BASE_SCORE} pts = {BASE_ODDS}:1 odds, +{PDO} pts doubles odds")
line()
print(f"{'Score band':<16}{'Borrowers':>11}{'Share':>9}{'Default rate':>15}")
line("-")
for band, r in band_table.iterrows():
    share = r["Borrowers"] / len(scores)
    rate = f"{r['Default_Rate']:.2%}" if r["Borrowers"] else "-"
    print(f"{band:<16}{int(r['Borrowers']):>11,}{share:>9.1%}{rate:>15}")
line("-")
print("A working scorecard shows default rates falling as scores rise.")

# ---------------------------------------------------------------------------
# Expected Loss: EL = PD x LGD x EAD  (2-year horizon, matching the target)
# EAD = drawn balance + CCF x unused limit
# ---------------------------------------------------------------------------
utilization = X_test["RevolvingUtilizationOfUnsecuredLines"].clip(0, 1).values
drawn = utilization * CREDIT_LIMIT
ead = drawn + CCF * (CREDIT_LIMIT - drawn)
expected_loss = pd_test * LGD * ead

el_table = (
    pd.DataFrame({"Band": bands, "EAD": ead, "EL": expected_loss})
    .groupby("Band", observed=False)
    .agg(Borrowers=("EL", "count"), Total_EAD=("EAD", "sum"),
         Total_EL=("EL", "sum"), Avg_EL=("EL", "mean"))
)
total_ead, total_el = ead.sum(), expected_loss.sum()

print()
line()
print("EXPECTED LOSS (hold-out portfolio)  |  EL = PD x LGD x EAD")
line()
print(f"Assumptions: ${CREDIT_LIMIT:,} limit per borrower, CCF {CCF:.0%}, LGD {LGD:.0%}")
print(f"{'Score band':<16}{'Exposure':>14}{'Expected loss':>16}{'Avg EL':>10}{'Loss rate':>11}")
line("-")
for band, r in el_table.iterrows():
    if not r["Borrowers"]:
        continue
    print(f"{band:<16}{r['Total_EAD']:>14,.0f}{r['Total_EL']:>16,.0f}"
          f"{r['Avg_EL']:>10,.0f}{r['Total_EL'] / r['Total_EAD']:>11.2%}")
line("-")
print(f"{'Portfolio':<16}{total_ead:>14,.0f}{total_el:>16,.0f}"
      f"{total_el / len(ead):>10,.0f}{total_el / total_ead:>11.2%}")
poor_share = el_table.loc["Poor 300-579", "Total_EL"] / total_el
print(f"\nThe Poor band holds {(bands == 'Poor 300-579').mean():.0%} of borrowers "
      f"but {poor_share:.0%} of expected loss.")

# Worked example: one high-risk and one low-risk borrower from the hold-out set
hi, lo = int(np.argmax(pd_test)), int(np.argmin(pd_test))
print("\nWorked examples:")
for label, i in [("Highest-risk borrower", hi), ("Lowest-risk borrower", lo)]:
    print(f"  {label}: score {scores[i]:.0f} | PD {pd_test[i]:.1%} x LGD {LGD:.0%} x "
          f"EAD ${ead[i]:,.0f} = EL ${expected_loss[i]:,.0f}")

# ---------------------------------------------------------------------------
# Save everything needed to score a new applicant
# ---------------------------------------------------------------------------
with open(MODEL_FILE, "wb") as f:
    pickle.dump({
        "pipeline": model,
        "caps": caps,
        "features": list(X.columns),
        "score_params": {"base_score": BASE_SCORE, "base_odds": BASE_ODDS, "pdo": PDO},
        "el_assumptions": {"credit_limit": CREDIT_LIMIT, "ccf": CCF, "lgd": LGD},
        "metrics": {"auc": auc, "gini": 2 * auc - 1, "ks": ks},
    }, f)
matrix.to_csv(MATRIX_FILE, index=False)

print(f"\nSaved: outputs/{MODEL_FILE.name} (pipeline + caps + score settings), outputs/{MATRIX_FILE.name}")
