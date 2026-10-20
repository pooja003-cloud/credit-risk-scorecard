# Credit Risk Scorecard — Interpretable Logistic Regression — Full report

*This is the full technical report. For a short summary, see [README.md](README.md).*

A traditional bank-style credit scorecard that predicts the probability of serious delinquency (90+ days past due within two years), converts it to a 300–850 credit score with adverse action reason codes, and estimates the portfolio's expected credit loss. The model is built for **transparency**: every feature's effect on risk is visible, ranked and explainable, the way regulated lenders need it to be.

## Why logistic regression?

Banks must be able to explain their credit decisions — to model validators under model risk guidance (the Fed's SR 11-7, replaced in April 2026 by [SR 26-2](https://www.federalreserve.gov/supervisionreg/srletters/SR2602.htm)), and to customers through adverse action notices (ECOA / Regulation B). A logistic regression gives one clear, signed weight per feature, so you can say exactly what drives a borrower's risk up or down. That makes it the standard baseline for retail credit scoring.

The design rule throughout: **every feature in the model must be one a lender would be willing to show an applicant as a reason.**

## Results (20% stratified hold-out, 30,000 borrowers)

| Metric | Value | What it means |
|---|---|---|
| AUC | **0.861** | 86% chance a random defaulter is scored riskier than a random non-defaulter |
| Gini | **0.721** | 2 × AUC − 1, the standard industry ranking measure |
| KS | **0.557** | Maximum separation between good and bad borrowers |

<p align="center">
  <img src="outputs/roc_curve.png" width="48%" alt="ROC curve with AUC 0.861 and KS 0.557">
  <img src="outputs/calibration.png" width="48%" alt="Predicted vs actual default rate by decile">
</p>

### Calibration: are the predicted PDs right?

Ranking borrowers well is not enough for loss estimates; the predicted probabilities also need to match reality. The hold-out set is split into ten equal groups from safest to riskiest:

| Decile | Predicted PD | Actual default rate | Predicted ÷ actual |
|---|---|---|---|
| 1 (safest) | 0.93% | 0.37% | 2.54 |
| 2 | 1.29% | 0.53% | 2.42 |
| 3 | 1.61% | 0.80% | 2.01 |
| 4 | 1.96% | 1.50% | 1.31 |
| 5 | 2.38% | 1.83% | 1.30 |
| 6 | 3.04% | 2.73% | 1.11 |
| 7 | 4.24% | 4.63% | 0.92 |
| 8 | 6.43% | 7.03% | 0.91 |
| 9 | 10.36% | 11.00% | 0.94 |
| 10 (riskiest) | 34.26% | 36.40% | 0.94 |

**What this shows:** across the riskiest four deciles, predictions are within 6–9% of actual default rates (slightly understating risk). Deciles 4–6 overstate risk by 1.1–1.3×, and the **safest three deciles overstate it by 2–2.5×**. A straight-line model cannot bend enough to capture how very-low-risk borrowers behave. The error is conservative (it errs toward caution, not optimism), and it matters little in dollar terms because the safest deciles carry little expected loss. The fixes are WoE binning or a recalibration step (see *Limitations*).

### Risk Weight Matrix

Standardized coefficients, so features are directly comparable. *Odds ×* is how much default odds multiply for a one-standard-deviation increase.

| Rank | Feature | Coef | Odds × | Effect |
|---|---|---|---|---|
| 1 | Revolving utilization of unsecured lines | +0.686 | 1.99 | Raises risk |
| 2 | Times 90+ days late | +0.333 | 1.40 | Raises risk |
| 3 | Times 30–59 days past due | +0.307 | 1.36 | Raises risk |
| 4 | Age | −0.247 | 0.78 | Lowers risk |
| 5 | Times 60–89 days past due | +0.203 | 1.22 | Raises risk |
| 6 | Open credit lines and loans | +0.170 | 1.19 | Raises risk |
| 7 | Monthly income | −0.132 | 0.88 | Lowers risk |
| 8 | Real estate loans or lines | +0.104 | 1.11 | Raises risk |
| 9 | Past-due history recorded as special code (flag) | +0.103 | 1.11 | Raises risk |
| 10 | Number of dependents | +0.042 | 1.04 | Raises risk |
| 11 | Income not reported (flag) | −0.012 | 0.99 | Negligible |

**Key takeaway:** credit card utilization and past delinquency history dominate default risk. Utilization alone roughly doubles the odds of default per standard deviation, while monthly income has a much smaller effect. Every weight points in the direction a credit analyst would expect. The income-not-reported flag is close to zero, which fits its 5.6% default rate being near the 6.7% average.

### Credit score bands

Scores use the industry "points to double the odds" method: 650 points = 20:1 good-to-bad odds, and every 50 points doubles the odds.

| Score band | Borrowers | Share | Predicted PD | Actual default rate |
|---|---|---|---|---|
| Poor (300–579) | 3,679 | 12.3% | 30.25% | 32.16% |
| Fair (580–669) | 7,936 | 26.5% | 6.71% | 7.32% |
| Good (670–739) | 12,575 | 41.9% | 2.28% | 1.71% |
| Very Good (740–799) | 5,767 | 19.2% | 1.10% | 0.45% |
| Exceptional (800–850) | 43 | 0.1% | 0.56% | 0.00% |

Default rates fall steadily as scores rise, from about 32% in the lowest band to under 1% in the top bands. The gap between predicted and actual in the top bands is the same overstatement seen in the calibration table.

### Expected loss

Banks measure credit losses with **Expected Loss = PD × LGD × EAD**: the probability of default, the share of the exposure lost if default happens, and the amount owed at default. This model supplies the PD. The dataset has no balances or recovery data, so LGD and EAD use illustrative, industry-typical assumptions for unsecured credit cards:

- **Credit limit:** $10,000 per borrower (assumed)
- **EAD** = drawn balance + 50% of the unused limit (a 50% credit conversion factor, since borrowers tend to draw down their lines before defaulting)
- **LGD:** 80% (unsecured cards typically lose 75–85% after default)

| Score band | Exposure ($) | Expected loss ($) | Avg EL per borrower ($) | Loss rate |
|---|---|---|---|---|
| Poor (300–579) | 34,113,709 | 8,363,879 | 2,273 | 24.52% |
| Fair (580–669) | 63,098,569 | 3,508,309 | 442 | 5.56% |
| Good (670–739) | 71,016,906 | 1,315,415 | 105 | 1.85% |
| Very Good (740–799) | 29,856,977 | 264,247 | 46 | 0.89% |
| Exceptional (800–850) | 219,013 | 988 | 23 | 0.45% |
| **Portfolio** | **198,305,174** | **13,452,838** | **448** | **6.78%** |

**Key takeaway:** the Poor band holds 12% of borrowers but 62% of expected loss. That is the kind of concentration a bank acts on through approval cut-offs, credit limits and pricing.

The loss figures cover the same two-year horizon as the default target. They show how a PD model plugs into loss estimation; the dollar amounts depend on the assumptions above.

### Scoring a new applicant

`score_applicant()` takes an applicant's raw details and returns their score, PD and the top three **risk factors**: the features pushing their risk above the average borrower. If the applicant is declined, these are the reasons a lender would list on an adverse action notice. For a logistic model, each feature's contribution is simply its weight × the applicant's standardized value.

```python
from scorecard import score_applicant

score_applicant({
    "RevolvingUtilizationOfUnsecuredLines": 0.95, "age": 28,
    "NumberOfTime30-59DaysPastDueNotWorse": 3, "MonthlyIncome": 6000,
    "NumberOfOpenCreditLinesAndLoans": 8, "NumberOfTimes90DaysLate": 0,
    "NumberRealEstateLoansOrLines": 0, "NumberOfTime60-89DaysPastDueNotWorse": 0,
    "NumberOfDependents": 0,
})
```

| Applicant | Profile | Score | PD | Top risk factors |
|---|---|---|---|---|
| Maria | 28, uses 95% of her credit, 3 payments 30–59 days late, earns $6,000/month | **470** (Poor) | 37.8% | Payments 30–59 days late · High revolving utilization · Younger age |
| James | 52, uses 10% of his credit, never late, earns $4,500/month | **717** (Good) | 1.9% | Low risk; minor factors only |

Maria earns more than James, but her utilization and late payments make her about 20 times riskier. The model sets no approval cut-off; choosing one is a business decision about how much risk a lender will accept.

## Method

1. **Clean** — the 96/98 values in the past-due counts are data-entry codes, not real counts. Those borrowers default at 54.6% (about 8× average), so a flag keeps that signal before the codes are set to missing. A second flag marks the 19.8% of borrowers with no reported income.
2. **Select features** — debt ratio is left out (see *Limitations*), leaving nine original features plus the two flags.
3. **Split first** — stratified 80/20 train/test split *before* any fitting, so nothing from the test set leaks into training.
4. **Cap outliers** — clip each feature at the training set's 1st and 99th percentiles.
5. **Impute + scale + model** — median imputation, standardization and logistic regression, all in one scikit-learn `Pipeline` fitted on training data only.
6. **Validate** — AUC, Gini and KS on the hold-out set (accuracy is misleading when only ~7% of borrowers default), plus a 10-decile calibration check.
7. **Explain** — rank standardized coefficients into the Risk Weight Matrix.
8. **Score** — convert probability of default to a 300–850 score and check default rates by band.
9. **Estimate losses** — combine PD with assumed LGD and EAD to calculate expected loss by score band and for the whole portfolio.
10. **Risk factors** — score individual applicants and list the factors driving their risk.

## Project structure

```
credit-risk-scorecard/
├── scorecard.py            # training pipeline + score_applicant()
├── requirements.txt        # pinned versions (the saved model needs the same scikit-learn)
├── data/                   # place credit_data.csv here (not tracked)
└── outputs/
    ├── roc_curve.png
    ├── calibration.png
    ├── risk_weight_matrix.csv
    ├── scorecard_model.pkl  # pipeline + outlier caps + score and loss settings
    └── run_output.txt       # full console output of the latest run
```

## How to run

1. Download `cs-training.csv` from the [Give Me Some Credit competition](https://www.kaggle.com/c/GiveMeSomeCredit/data) (free Kaggle account required).
2. Save it as `data/credit_data.csv`.
3. Install and run (Python 3.11 or newer):

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scorecard.py
```

The dataset is not included in this repository, in line with Kaggle's competition rules.

## Limitations and next steps

- **Calibration for the safest borrowers.** The model overstates PD by about 2–2.5× in the safest three deciles. WoE binning (below) or an isotonic / Platt recalibration step on held-out data would correct it.
- **Debt ratio removed.** On its own, a higher debt ratio does go with more defaults (about 5% in the lowest decile vs 11% in the highest). But once utilization, open credit lines and real-estate loans are in the model, its weight turns slightly negative. Blanking it when income is missing and log-transforming it did not change that. Removing it costs no accuracy (AUC 0.860 with it, 0.861 without), so it is left out rather than kept and hidden from reason codes: "debt ratio too low" is not a reason anyone could be declined for.
- **Age and dependents.** Under ECOA / Regulation B, a statistically sound scoring system may use age, provided applicants aged 62 or older are not penalized for it. In this model, age has one negative weight, so being older only ever lowers risk; the 28% of borrowers aged 62+ are never scored down for age. Age and number of dependents are still sensitive, and a lender's fair-lending review would check both before use.
- **Linear effects only.** Features enter the model as straight lines. A production scorecard would use Weight of Evidence (WoE) binning with monotonic constraints and a points-per-attribute table, which captures non-linear patterns while staying interpretable.
- **Real LGD and EAD models.** With account-level balance, limit and recovery data, LGD and EAD would be modelled per borrower instead of assumed.
- **Monitoring.** Population Stability Index (PSI) tracking would detect when incoming applicants drift from the training population.

## Data

[Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) (Kaggle, 2011): 150,000 borrowers, 10 numeric features, target `SeriousDlqin2yrs`.
