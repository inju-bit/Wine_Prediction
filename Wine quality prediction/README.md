# Wine Quality Prediction with XGBoost

Predicting wine quality from physicochemical lab measurements using gradient
boosted trees (XGBoost), with hyperparameter tuning via GridSearchCV.

## Dataset

UCI Wine Quality dataset (Cortez et al., 2009), also available on Kaggle.
Two files are included:

| File | Samples | Features |
|------|---------|----------|
| `winequality-red.csv`   | 1,599 | 11 |
| `winequality-white.csv` | 4,898 | 11 |

Each row is a Portuguese "Vinho Verde" wine described by 11 physicochemical
properties (fixed acidity, volatile acidity, citric acid, residual sugar,
chlorides, free and total sulfur dioxide, density, pH, sulphates, alcohol) and
a sensory quality score from 0 to 10.

> Reference: P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
> *Modeling wine preferences by data mining from physicochemical properties.*
> Decision Support Systems, 47(4):547-553, 2009.

## Problem framing

The raw `quality` score is highly imbalanced (most wines score 5 or 6), so it
is reframed as a binary classification task:

* **good (1):** quality >= 6
* **not good (0):** quality < 6

A stratified 80/20 train/test split is used with a fixed random seed (42) so
results are fully reproducible.

## Method

1. **Baseline:** an untuned XGBoost classifier with library defaults.
2. **Tuning:** `GridSearchCV` with 5-fold stratified cross validation over
   `n_estimators`, `max_depth`, `learning_rate`, `subsample`,
   `colsample_bytree` and `min_child_weight`, optimised for accuracy.
3. **Evaluation:** accuracy, ROC-AUC, precision, recall, F1, confusion matrix
   and feature importance on the held-out test set.

## Results

Reproducible numbers from `src/train.py` (seed 42):

| Wine  | Baseline accuracy | Tuned accuracy | ROC-AUC | Change |
|-------|------------------|----------------|---------|--------|
| White | 0.829 | **0.837** | 0.894 | +0.8 pp |
| Red   | 0.825 | 0.816 | 0.889 | -0.9 pp |

Notes on what the experiment actually shows:

* On the larger **white** set, GridSearchCV gives a small but genuine gain by
  regularising a slightly overfit default model (lower learning rate, more
  trees, subsampling).
* On the smaller **red** set, the default model is already strong and tuning
  does not beat it on this split. This is a normal outcome for small datasets:
  XGBoost defaults are competitive and accuracy estimates carry meaningful
  variance. The honest takeaway is that tuning is not guaranteed to help, and
  cross-validation prevents reading too much into a single lucky split.
* Alcohol, volatile acidity and sulphates are consistently the most predictive
  features, which matches the conclusions in the original Cortez et al. paper.

Best white-wine parameters found:
`n_estimators=800, max_depth=7, learning_rate=0.03, subsample=0.8,
colsample_bytree=0.8, min_child_weight=1`.

## How to run

```bash
pip install -r requirements.txt
python src/train.py --wine white   # or: --wine red
```

Outputs (confusion matrix, ROC curve, feature importance, `results.json`) are
written to `outputs/`.

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
