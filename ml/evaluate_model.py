"""
evaluate_model.py
-----------------
Loads saved models and produces publication-quality graphs:
  1. Confusion matrices (RF and XGBoost side by side)
  2. Feature importance bar chart
  3. Per-class precision/recall/F1 table
  4. False positive rate per class

Works entirely offline — no hardware needed.

Usage:
    python -m ml.evaluate_model
    python -m ml.evaluate_model --data data/processed/simulated_traffic.csv
"""

import argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (confusion_matrix, classification_report,
                              ConfusionMatrixDisplay)
from sklearn.model_selection import train_test_split
from features.feature_definitions import FEATURE_COLUMNS, LABEL_COLUMN, LABEL_MAP, LABEL_NAMES
from utils.logger import get_logger

log = get_logger("evaluator")

LABEL_LIST = list(LABEL_MAP.keys())


def load_xy(data_path: str):
    df  = pd.read_csv(data_path)
    y   = df[LABEL_COLUMN].map(LABEL_MAP).values
    X   = df[FEATURE_COLUMNS].values
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    return X_test, y_test


def plot_confusion_matrices(rf, xgb, X_test, y_test, out_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("AdaptiGuard — Confusion Matrices", fontsize=14, fontweight="bold")

    for ax, model, name in zip(axes, [rf, xgb], ["Random Forest", "XGBoost"]):
        preds = model.predict(X_test)
        cm    = confusion_matrix(y_test, preds)
        disp  = ConfusionMatrixDisplay(cm, display_labels=LABEL_LIST)
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(name)

    plt.tight_layout()
    path = f"{out_dir}/confusion_matrices.png"
    plt.savefig(path, dpi=150)
    plt.close()
    log.info(f"Saved {path}")


def plot_feature_importance(rf, out_dir: str):
    importances = pd.Series(rf.feature_importances_, index=FEATURE_COLUMNS)
    importances = importances.sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, 7))
    importances.plot(kind="barh", ax=ax, color="#534AB7", edgecolor="none")
    ax.set_title("Feature Importance — Random Forest", fontweight="bold")
    ax.set_xlabel("Importance score")
    ax.axvline(x=importances.mean(), color="gray", linestyle="--", linewidth=0.8,
               label="Mean importance")
    ax.legend()
    plt.tight_layout()
    path = f"{out_dir}/feature_importance.png"
    plt.savefig(path, dpi=150)
    plt.close()
    log.info(f"Saved {path}")


def print_fpr(model, X_test, y_test):
    """False positive rate per class."""
    preds = model.predict(X_test)
    cm    = confusion_matrix(y_test, preds)
    log.info("False Positive Rate per class:")
    for i, label in LABEL_NAMES.items():
        fp = cm[:, i].sum() - cm[i, i]
        tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
        fpr = fp / max(fp + tn, 1)
        log.info(f"  {label:<12} FPR = {fpr:.4f} ({fpr*100:.2f}%)")


def evaluate(data_path: str, model_dir: str = "data/models",
             out_dir: str = "data/models"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    log.info("Loading models ...")
    rf  = joblib.load(f"{model_dir}/random_forest.pkl")
    xgb = joblib.load(f"{model_dir}/xgboost.pkl")

    log.info("Loading test data ...")
    X_test, y_test = load_xy(data_path)

    log.info("=== Random Forest ===")
    rf_preds = rf.predict(X_test)
    log.info(f"\n{classification_report(y_test, rf_preds, target_names=LABEL_LIST)}")
    print_fpr(rf, X_test, y_test)

    log.info("=== XGBoost ===")
    xgb_preds = xgb.predict(X_test)
    log.info(f"\n{classification_report(y_test, xgb_preds, target_names=LABEL_LIST)}")
    print_fpr(xgb, X_test, y_test)

    plot_confusion_matrices(rf, xgb, X_test, y_test, out_dir)
    plot_feature_importance(rf, out_dir)
    log.info("Evaluation complete.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",      default="data/processed/simulated_traffic.csv")
    parser.add_argument("--model-dir", default="data/models")
    parser.add_argument("--out-dir",   default="data/models")
    args = parser.parse_args()
    evaluate(args.data, args.model_dir, args.out_dir)


if __name__ == "__main__":
    main()
