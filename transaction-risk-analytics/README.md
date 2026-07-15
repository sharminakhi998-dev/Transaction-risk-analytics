# Transaction Risk Analytics: Fraud Detection & Loss Quantification

An end-to-end transactional risk analytics project combining unsupervised anomaly detection, supervised fraud classification, and Monte Carlo-based risk quantification (VaR/CVaR).

## Motivation

Fraud/anomaly detection notebooks online commonly stop at "here are some outliers we found via clustering." This project goes further in two ways:

1. **Validates unsupervised findings against a labeled dataset** — anomaly detection alone can't tell you if it's actually catching fraud, since there's no ground truth. A second, labeled dataset is used to build and evaluate a genuine classifier.
2. **Translates model performance into risk terms** — precision/recall are model-evaluation metrics, not business metrics. This project converts classifier output into dollar-denominated expected loss and simulates the distribution of future losses to derive Value at Risk (VaR) and Conditional VaR (CVaR), the standard risk-quantification tools used in financial risk management.

## Datasets

| Dataset | Rows | Labeled? | Used for |
|---|---|---|---|
| [Bank Transaction Dataset for Fraud Detection](https://www.kaggle.com/datasets/valakhorasani/bank-transaction-dataset-for-fraud-detection) (V. Khorasani) | 2,512 | No | EDA, unsupervised anomaly detection |
| [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (ULB Machine Learning Group) | 284,807 | Yes | Supervised classification, risk quantification |

`bank_transactions_data_2.csv` is included in `data/raw/`. **`creditcard.csv` (144MB) is not included** — it exceeds GitHub's file size limit. Download it from the link above and place it in `data/raw/` before running notebooks 03–04.

## Project structure

```
transaction-risk-analytics/
├── data/
│   ├── raw/                # source CSVs
│   └── processed/          # intermediate outputs, trained model
├── notebooks/
│   ├── 01_eda.ipynb                     # structure, distributions, data quality
│   ├── 02_anomaly_detection.ipynb       # rule-based flagging, K-Means, DBSCAN
│   ├── 03_supervised_fraud_model.ipynb  # Logistic Regression, Random Forest, evaluation
│   └── 04_risk_quantification.ipynb     # expected loss, Monte Carlo, VaR/CVaR
├── reports/figures/        # exported charts
├── requirements.txt
└── README.md
```

## Methodology summary

**01 — EDA:** Profiles the bank transactions dataset. Identifies a data-quality issue (`AccountBalance` doesn't behave as a coherent running balance across an account's transaction history) and documents the decision to exclude it from feature engineering rather than silently using an unreliable column.

**02 — Anomaly detection:** Flags suspicious transactions three independent ways — a simple rule-based score, K-Means (distance-to-centroid), and DBSCAN (density-based outlier labeling) — and checks agreement across methods as a basic validation step, in the absence of ground-truth labels.

**03 — Supervised classification:** Uses the labeled credit card fraud dataset to train and compare Logistic Regression (baseline and class-balanced) against Random Forest. Evaluates using precision/recall/confusion matrix rather than accuracy, since accuracy is misleading on a dataset with 0.17% fraud prevalence (a model that always predicts "not fraud" scores 99.83% accuracy while catching zero fraud).

**04 — Risk quantification:** Converts the classifier's test-set performance into dollar terms (realized vs. prevented loss), then runs a 10,000-iteration Monte Carlo simulation of daily fraud loss (drawing fraud counts from a Binomial distribution and fraud amounts from the empirical historical distribution) to estimate VaR (95%/99%) and CVaR (95%) — the loss thresholds a bank would use to size capital reserves against fraud risk.

## Key results

- Random Forest (class-balanced) achieved **0.96 precision / 0.74 recall** on held-out fraud detection, a substantially better operational trade-off than an unweighted or naively-balanced Logistic Regression (the latter reached 0.92 recall but only 0.06 precision — over 1,300 false alarms, an unworkable false-positive rate).
- On the test set, the model reduced realized fraud loss by ~58% versus no detection.
- Monte Carlo simulation estimates **VaR (95%) ≈ $11,900/day** and **CVaR (95%) ≈ $13,100/day** in fraud loss exposure under current model performance.

## Limitations

- Unsupervised anomaly flags on the bank transactions dataset are not validated against true fraud labels (none exist for that dataset) — they identify statistical outliers, a related but distinct concept from confirmed fraud.
- The supervised model is evaluated on a single train/test split rather than cross-validation.
- The Monte Carlo simulation assumes historical fraud-amount distribution and model recall remain stable, which would need periodic recalibration in a production setting.
- No cost is assigned to false positives (manual review cost, customer friction) in the loss figures.

## Setup

```bash
pip install -r requirements.txt
jupyter notebook notebooks/
```

Run notebooks in order (01 → 04); 03 saves a trained model that 04 depends on.
