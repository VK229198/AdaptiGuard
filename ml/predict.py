"""
predict.py
----------
Loads the saved best_model.pkl and classifies a single flow dict.
Used internally by main.py. Also callable standalone for testing.

Usage:
    python -m ml.predict
"""

import joblib
import numpy as np
from pathlib import Path
from features.feature_definitions import FEATURE_COLUMNS, LABEL_NAMES
from utils.logger import get_logger

log = get_logger("predictor")

_model = None
_model_path = "data/models/best_model.pkl"


def load_model(path: str = _model_path):
    global _model
    if _model is None:
        if not Path(path).exists():
            raise FileNotFoundError(
                f"No model found at {path}. Run: python -m ml.train_model"
            )
        _model = joblib.load(path)
        log.info(f"Model loaded from {path}")
    return _model


def predict_flow(flow: dict) -> tuple[str, float]:
    """
    Classify a single flow.

    Parameters:
        flow: dict with keys matching FEATURE_COLUMNS

    Returns:
        (label_str, confidence)  e.g. ("ddos", 0.97)
    """
    model = load_model()
    row   = np.array([[flow.get(f, 0) for f in FEATURE_COLUMNS]])
    pred  = model.predict(row)[0]
    proba = model.predict_proba(row)[0]
    label = LABEL_NAMES[pred]
    conf  = float(proba[pred])
    return label, conf


def main():
    # Quick smoke test with a dummy DDoS-like flow
    test_flow = {
        "duration": 0.05, "packet_count": 2000, "byte_count": 100000,
        "avg_packet_size": 50.0, "packet_rate": 40000, "byte_rate": 2000000,
        "iat_mean": 0.00002, "iat_std": 0.000005, "iat_min": 0.000001,
        "iat_max": 0.0001, "syn_count": 1800, "rst_count": 2, "ack_count": 5,
        "fin_count": 0, "urg_count": 0, "unique_dst_ports": 1,
        "src_port": 54321, "dst_port": 80,
    }
    label, conf = predict_flow(test_flow)
    log.info(f"Prediction: {label}  (confidence: {conf:.2%})")


if __name__ == "__main__":
    main()
