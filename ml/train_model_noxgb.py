"""Fallback trainer — Random Forest only, no XGBoost dependency."""
import joblib, pandas as pd, numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from features.feature_definitions import FEATURE_COLUMNS, LABEL_COLUMN, LABEL_MAP
from utils.logger import get_logger
log = get_logger("trainer")

def train(data_path="data/processed/simulated_traffic.csv", out_dir="data/models"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(data_path)
    y  = df[LABEL_COLUMN].map(LABEL_MAP).values
    X  = df[FEATURE_COLUMNS].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)
    acc   = accuracy_score(y_test, preds)
    cv    = cross_val_score(rf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42))
    log.info(f"Accuracy: {acc:.4f}  CV mean: {cv.mean():.4f}")
    log.info(f"\n{classification_report(y_test, preds, target_names=list(LABEL_MAP.keys()))}")
    joblib.dump(rf, f"{out_dir}/random_forest.pkl")
    joblib.dump(rf, f"{out_dir}/best_model.pkl")
    joblib.dump(FEATURE_COLUMNS, f"{out_dir}/feature_columns.pkl")
    log.info(f"Saved to {out_dir}/")

if __name__ == "__main__":
    train()
