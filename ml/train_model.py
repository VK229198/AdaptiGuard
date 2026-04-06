"""
train_model.py
--------------
Loads processed CSV(s), trains Random Forest and XGBoost classifiers,
evaluates both, saves the better model to data/models/best_model.pkl.

Works entirely offline — no hardware needed.

Usage:
    python -m ml.train_model
    python -m ml.train_model --data data/processed/simulated_traffic.csv
    python -m ml.train_model --data data/processed/  (merges all CSVs in folder)
"""

import argparse
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, f1_score)
from xgboost import XGBClassifier
from features.feature_definitions import FEATURE_COLUMNS, LABEL_COLUMN, LABEL_MAP
from utils.logger import get_logger

log = get_logger("trainer")


def load_data(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.is_dir():
        csvs = list(p.glob("*.csv"))
        log.info(f"Merging {len(csvs)} CSV files from {path}")
        return pd.concat([pd.read_csv(f) for f in csvs], ignore_index=True)
    return pd.read_csv(path)


def train(data_path: str, out_dir: str = "data/models"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    df = load_data(data_path)
    log.info(f"Loaded {len(df)} rows")
    log.info(f"Label distribution:\n{df[LABEL_COLUMN].value_counts()}")

    # Encode labels
    le = LabelEncoder()
    le.classes_ = np.array(list(LABEL_MAP.keys()))
    y = df[LABEL_COLUMN].map(LABEL_MAP).values
    X = df[FEATURE_COLUMNS].values

    # Train/test split — stratified to preserve class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    log.info(f"Train: {len(X_train)}  Test: {len(X_test)}")

    results = {}

    # ── Random Forest ─────────────────────────────────────────────────────────
    log.info("Training Random Forest ...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_acc   = accuracy_score(y_test, rf_preds)
    rf_f1    = f1_score(y_test, rf_preds, average="weighted")

    # k-fold cross validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rf_cv = cross_val_score(rf, X, y, cv=cv, scoring="accuracy")

    log.info(f"RF  Accuracy : {rf_acc:.4f}")
    log.info(f"RF  F1 Score : {rf_f1:.4f}")
    log.info(f"RF  CV Scores: {rf_cv.round(4)}  mean={rf_cv.mean():.4f}")
    log.info(f"\n{classification_report(y_test, rf_preds, target_names=list(LABEL_MAP.keys()))}")
    results["random_forest"] = {"model": rf, "acc": rf_acc, "f1": rf_f1, "cv_mean": rf_cv.mean()}

    # ── XGBoost ───────────────────────────────────────────────────────────────
    log.info("Training XGBoost ...")
    xgb = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )
    xgb.fit(X_train, y_train)
    xgb_preds = xgb.predict(X_test)
    xgb_acc   = accuracy_score(y_test, xgb_preds)
    xgb_f1    = f1_score(y_test, xgb_preds, average="weighted")
    xgb_cv    = cross_val_score(xgb, X, y, cv=cv, scoring="accuracy")

    log.info(f"XGB Accuracy : {xgb_acc:.4f}")
    log.info(f"XGB F1 Score : {xgb_f1:.4f}")
    log.info(f"XGB CV Scores: {xgb_cv.round(4)}  mean={xgb_cv.mean():.4f}")
    log.info(f"\n{classification_report(y_test, xgb_preds, target_names=list(LABEL_MAP.keys()))}")
    results["xgboost"] = {"model": xgb, "acc": xgb_acc, "f1": xgb_f1, "cv_mean": xgb_cv.mean()}

    # ── Save both, mark best ───────────────────────────────────────────────────
    best_name = max(results, key=lambda k: results[k]["acc"])
    best      = results[best_name]

    joblib.dump(rf,  f"{out_dir}/random_forest.pkl")
    joblib.dump(xgb, f"{out_dir}/xgboost.pkl")
    joblib.dump(best["model"], f"{out_dir}/best_model.pkl")
    joblib.dump(FEATURE_COLUMNS, f"{out_dir}/feature_columns.pkl")

    log.info(f"Best model: {best_name}  (acc={best['acc']:.4f})")
    log.info(f"Saved to {out_dir}/")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/simulated_traffic.csv")
    parser.add_argument("--out",  default="data/models")
    args = parser.parse_args()
    train(args.data, args.out)


if __name__ == "__main__":
    main()
