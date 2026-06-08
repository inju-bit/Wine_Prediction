"""
Wine Quality Prediction with XGBoost
====================================
Dataset : UCI Wine Quality (Cortez et al., 2009) -- also on Kaggle.
Task    : Binary classification, "good" wine (quality >= 6) vs "not good".
Pipeline: default XGBoost baseline -> GridSearchCV hyperparameter tuning
          -> evaluation (accuracy, ROC-AUC, precision/recall/F1).

Reference:
P. Cortez, A. Cerdeira, F. Almeida, T. Matos, J. Reis (2009).
"Modeling wine preferences by data mining from physicochemical properties."
Decision Support Systems, 47(4), 547-553.

Usage:
    python src/train.py --wine red
    python src/train.py --wine white
"""
import argparse, json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (accuracy_score, roc_auc_score, classification_report,
                             confusion_matrix, RocCurveDisplay)
from xgboost import XGBClassifier

RANDOM_STATE = 42


def load_data(wine: str) -> pd.DataFrame:
    path = f"data/winequality-{wine}.csv"
    df = pd.read_csv(path, sep=";")
    return df


def main(wine: str):
    os.makedirs("outputs", exist_ok=True)
    df = load_data(wine)
    print(f"Dataset: {wine} wine  |  rows={len(df)}  features={df.shape[1]-1}")

    # ---- Target: binary good/not-good at quality >= 6 ----
    X = df.drop(columns=["quality"])
    y = (df["quality"] >= 6).astype(int)
    print(f"Class balance (good=1): {y.mean():.3f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)

    # ---- 1) Baseline: default XGBoost (no tuning) ----
    baseline = XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss")
    baseline.fit(X_train, y_train)
    base_pred = baseline.predict(X_test)
    base_acc = accuracy_score(y_test, base_pred)
    print(f"\n[Baseline] default XGBoost accuracy = {base_acc:.4f}")

    # ---- 2) GridSearchCV hyperparameter optimisation ----
    param_grid = {
        "n_estimators":     [400, 800],
        "max_depth":        [5, 7, 9],
        "learning_rate":    [0.03, 0.05],
        "subsample":        [0.8],
        "colsample_bytree": [0.8],
        "min_child_weight": [1, 3],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    grid = GridSearchCV(
        XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss", n_jobs=1),
        param_grid, scoring="accuracy", cv=cv, n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)

    best = grid.best_estimator_
    tuned_pred = best.predict(X_test)
    tuned_proba = best.predict_proba(X_test)[:, 1]
    tuned_acc = accuracy_score(y_test, tuned_pred)
    tuned_auc = roc_auc_score(y_test, tuned_proba)

    print(f"\n[Tuned] best params: {grid.best_params_}")
    print(f"[Tuned] accuracy = {tuned_acc:.4f}  |  ROC-AUC = {tuned_auc:.4f}")
    print(f"[Tuned] improvement over baseline = {tuned_acc - base_acc:+.4f}")
    print("\nClassification report (tuned):")
    print(classification_report(y_test, tuned_pred, target_names=["not good", "good"]))

    # ---- Figures ----
    # Confusion matrix
    cm = confusion_matrix(y_test, tuned_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["not good", "good"], yticklabels=["not good", "good"])
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {wine} wine (tuned XGBoost)")
    plt.tight_layout(); plt.savefig("outputs/confusion_matrix.png", dpi=120); plt.close()

    # Feature importance
    imp = pd.Series(best.feature_importances_, index=X.columns).sort_values()
    plt.figure(figsize=(7, 5))
    imp.plot(kind="barh", color="#7a1f2b")
    plt.title(f"Feature Importance - {wine} wine")
    plt.xlabel("Importance (gain-weighted)")
    plt.tight_layout(); plt.savefig("outputs/feature_importance.png", dpi=120); plt.close()

    # ROC curve
    plt.figure(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_test, tuned_proba)
    plt.title(f"ROC Curve - {wine} wine (AUC={tuned_auc:.3f})")
    plt.tight_layout(); plt.savefig("outputs/roc_curve.png", dpi=120); plt.close()

    results = {
        "wine_type": wine,
        "n_samples": int(len(df)),
        "class_balance_good": round(float(y.mean()), 4),
        "baseline_accuracy": round(float(base_acc), 4),
        "tuned_accuracy": round(float(tuned_acc), 4),
        "tuned_roc_auc": round(float(tuned_auc), 4),
        "accuracy_improvement": round(float(tuned_acc - base_acc), 4),
        "best_params": grid.best_params_,
    }
    with open("outputs/results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved figures + results.json to outputs/")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wine", choices=["red", "white"], default="white")
    main(ap.parse_args().wine)
