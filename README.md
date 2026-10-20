# Credit Risk Scorecard — Interpretable Logistic Regression

A traditional bank-style credit scorecard that predicts the probability of serious delinquency (90+ days past due within two years) and converts it to a 300–850 credit score. The model is built for **transparency**: every feature's effect on risk is visible, ranked and explainable, the way regulated lenders need it to be.

## Why logistic regression?

Banks must be able to explain their credit decisions — to model validators under model-risk guidance (such as the Fed's SR 11-7), and to customers through adverse action notices (ECOA / Regulation B). A logistic regression gives one clear, signed weight per feature, so you can say exactly what drives a borrower's risk up or down. That makes it the standard baseline for retail credit scoring.

## Results (20% stratified hold-out, 30,000 borrowers)

| Metric | Value | What it means |
|---|---|---|
| AUC | **0.858** | 86% chance a random defaulter is scored riskier than a random non-defaulter |
| Gini | **0.717** | 2 × AUC − 1, the standard industry ranking measure |
| KS | **0.557** | Maximum separation between good and bad borrowers |

### Risk Weight Matrix

Standardized coefficients, so features are directly comparable. *Odds ×* is how much default odds multiply for a one-standard-deviation increase.

| Rank | Feature | Coef | Odds × | Effect |
|---|---|---|---|---|
| 1 | Revolving utilization of unsecured lines | +0.701 | 2.02 | Raises risk |
| 2 | Times 90+ days late | +0.325 | 1.38 | Raises risk |
| 3 | Times 30–59 days past due | +0.303 | 1.35 | Raises risk |
| 4 | Age | −0.252 | 0.78 | Lowers risk |
| 5 | Times 60–89 days past due | +0.200 | 1.22 | Raises risk |
| 6 | Open credit lines and loans | +0.153 | 1.17 | Raises risk |
| 7 | Monthly income | −0.139 | 0.87 | Lowers risk |
| 8 | Real estate loans or lines | +0.117 | 1.12 | Raises risk |
| 9 | Debt ratio | −0.078 | 0.93 | Lowers risk* |
| 10 | Number of dependents | +0.030 | 1.03 | Raises risk |

**Key takeaway:** credit card utilization and past delinquency history dominate default risk. Utilization alone roughly doubles the odds of default per standard deviation, while monthly income has a much smaller effect.

\*See *Limitations* for why debt ratio shows a counter-intuitive sign.

### Credit score bands

Scores use the industry "points to double the odds" method: 650 points = 20:1 good-to-bad odds, and every 50 points doubles the odds.

| Score band | Borrowers | Share | Default rate |
|---|---|---|---|
| Poor (300–579) | 3,868 | 12.9% | 30.53% |
| Fair (580–669) | 7,771 | 25.9% | 7.46% |
| Good (670–739) | 12,422 | 41.4% | 1.71% |
| Very Good (740–799) | 5,911 | 19.7% | 0.52% |
| Exceptional (800–850) | 28 | 0.1% | 0.00% |

Default rates fall steadily as scores rise, from about 31% in the lowest band to under 1% in the top bands, which is what a working scorecard should show.

## Method

1. **Clean** — load the data; treat the 96/98 codes in the past-due counts as missing (they are data-entry codes, not real counts).
2. **Split first** — stratified 80/20 train/test split *before* any fitting, so nothing from the test set leaks into training.
3. **Cap outliers** — clip each feature at the training set's 1st and 99th percentiles.
4. **Impute + scale + model** — median imputation, standardization and logistic regression, all in one scikit-learn `Pipeline` fitted on training data only.
5. **Validate** — AUC, Gini and KS on the hold-out set (accuracy is misleading when only ~7% of borrowers default).
6. **Explain** — rank standardized coefficients into the Risk Weight Matrix.
7. **Score** — convert probability of default to a 300–850 score and check default rates by band.

## Project structure

```
credit-risk-scorecard/
├── scorecard.py           # full pipeline: clean → train → validate → explain → score
├── requirements.txt
├── data/                   # place credit_data.csv here (not tracked)
└── outputs/
    ├── risk_weight_matrix.csv
    ├── scorecard_model.pkl  # pipeline + outlier caps + score settings
    └── run_output.txt      # full console output of the latest run
```

## How to run

1. Download `cs-training.csv` from the [Give Me Some Credit competition](https://www.kaggle.com/c/GiveMeSomeCredit/data) (free Kaggle account required).
2. Save it as `data/credit_data.csv`.
3. Install and run:

```bash
pip install -r requirements.txt
python scorecard.py
```

The dataset is not included in this repository, in line with Kaggle's competition rules.

## Limitations and next steps

- **Debt ratio sign.** In this dataset, debt ratio becomes extremely large when monthly income is missing, which distorts its relationship with default and produces a slightly negative weight. A production model would treat that case separately.
- **Linear effects only.** Features enter the model as straight lines. A production scorecard would use Weight of Evidence (WoE) binning with monotonic constraints to capture non-linear risk patterns while staying interpretable.
- **Reason codes.** Adding per-applicant reason codes (the top features pushing an individual's score down) would support adverse action notices.
- **Monitoring.** Population Stability Index (PSI) tracking would detect when incoming applicants drift from the training population.

## Data

[Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) (Kaggle, 2011): 150,000 borrowers, 10 numeric features, target `SeriousDlqin2yrs`.
