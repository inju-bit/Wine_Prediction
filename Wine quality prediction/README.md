# 🍷 Wine Quality Prediction with XGBoost

Predicting wine quality from 11 physicochemical lab measurements using gradient
boosted trees (XGBoost), with hyperparameter tuning through GridSearchCV and
5-fold cross-validation. Built on the classic UCI / Kaggle Wine Quality dataset.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-2.x-brightgreen)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange)

---

## Overview

Can a wine's quality be predicted from objective lab chemistry alone, without a
human taster? This project trains an XGBoost classifier to answer that, then
uses GridSearchCV to tune it and reports the gain honestly. On the white-wine
set the tuned model reaches **83.7% accuracy and 0.89 ROC-AUC**.

---

## Dataset

UCI Wine Quality dataset (Cortez et al., 2009), also hosted on Kaggle. Each row
is a Portuguese "Vinho Verde" wine, described by 11 physicochemical properties
and a sensory quality score from 0 to 10.

| File | Samples | Features |
|------|--------:|---------:|
| `winequality-red.csv`   | 1,599 | 11 |
| `winequality-white.csv` | 4,898 | 11 |

Features: fixed acidity, volatile acidity, citric acid, residual sugar,
chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates,
alcohol.

---

## Problem framing

The raw `quality` score is heavily imbalanced (most wines score 5 or 6), so the
target is reframed as binary classification:

* **good (1):** quality >= 6
* **not good (0):** quality < 6

A stratified 80/20 train/test split with a fixed seed (42) keeps the results
fully reproducible.

---

## Method

1. **Baseline:** an untuned XGBoost classifier with library defaults.
2. **Tuning:** `GridSearchCV` with 5-fold stratified cross-validation over
   `n_estimators`, `max_depth`, `learning_rate`, `subsample`,
   `colsample_bytree` and `min_child_weight`, optimised for accuracy.
3. **Evaluation:** accuracy, ROC-AUC, precision, recall, F1, confusion matrix
   and feature importance on the held-out test set.

---

## Results

Reproducible numbers from `src/train.py` (seed 42):

| Wine  | Baseline accuracy | Tuned accuracy | ROC-AUC | Change |
|-------|------------------:|---------------:|--------:|-------:|
| White | 0.829 | **0.837** | 0.894 | +0.8 pp |
| Red   | 0.825 | 0.816 | 0.889 | -0.9 pp |

### Feature importance

Alcohol and volatile acidity are clearly the strongest predictors on the white
set, followed by density and free sulfur dioxide. This lines up with the
original Cortez et al. findings that alcohol content dominates quality.

![Feature importance](outputs/feature_importance.png)

### ROC curve

The model separates good from not-good wines well above chance, with an area
under the curve of about 0.89.

![ROC curve](outputs/roc_curve.png)

### Confusion matrix

![Confusion matrix](outputs/confusion_matrix.png)

---

## Key findings

* On the larger **white** set, GridSearchCV gives a small but genuine gain by
  regularising a slightly overfit default model (lower learning rate, more
  trees, subsampling).
* On the smaller **red** set, the default model is already strong and tuning
  does not beat it on this split. This is a normal outcome for small datasets:
  XGBoost defaults are competitive and accuracy estimates carry real variance.
  Cross-validation is what stops a single lucky split from being over-read.
* Because the classes are imbalanced (about 67% good on white), accuracy alone
  is a weak metric, so ROC-AUC and per-class precision and recall are reported
  alongside it.

On the red set, sulphates rises into the top predictors, which is a known
difference between the two wine types.

Best white-wine parameters:
`n_estimators=800, max_depth=7, learning_rate=0.03, subsample=0.8,
colsample_bytree=0.8, min_child_weight=1`.

---

## How to run

```bash
pip install -r requirements.txt
python src/train.py --wine white   # or: --wine red
```

All figures, plus `results.json`, are written to `outputs/`.

---

## Project structure

```
wine-quality-xgboost/
├── data/
│   ├── winequality-red.csv
│   └── winequality-white.csv
├── src/
│   └── train.py
├── outputs/
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   ├── roc_curve.png
│   └── results.json
├── requirements.txt
└── README.md
```

---

## Reference

P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
*Modeling wine preferences by data mining from physicochemical properties.*
Decision Support Systems, 47(4):547-553, 2009.
