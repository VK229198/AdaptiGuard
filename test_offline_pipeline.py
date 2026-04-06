"""
test_offline_pipeline.py
------------------------
Run this BEFORE lab day to confirm every software component is working.
Zero hardware needed. If all tests pass, you are ready for lab.

Usage:
    python test_offline_pipeline.py
"""

import sys
import traceback
import pandas as pd
import joblib
from pathlib import Path
from utils.logger import get_logger

log = get_logger("preflight")

PASS = "  [PASS]"
FAIL = "  [FAIL]"
results = []


def check(name, fn):
    try:
        fn()
        log.info(f"{PASS} {name}")
        results.append((name, True))
    except Exception as e:
        log.info(f"{FAIL} {name}")
        log.info(f"         {e}")
        results.append((name, False))


# ── Test 1: Simulate traffic ──────────────────────────────────────────────────
def t_simulate():
    from capture.simulate_traffic import generate
    df = generate(n_samples=500)
    assert len(df) == 500
    assert "label" in df.columns
    assert set(df["label"].unique()) == {"normal", "ddos", "intrusion"}
    df.to_csv("data/processed/test_sim.csv", index=False)

check("Traffic simulation (3 classes, 500 flows)", t_simulate)


# ── Test 2: Feature columns consistent ───────────────────────────────────────
def t_features():
    from features.feature_definitions import FEATURE_COLUMNS, LABEL_COLUMN
    assert len(FEATURE_COLUMNS) == 18
    df = pd.read_csv("data/processed/test_sim.csv")
    missing = [f for f in FEATURE_COLUMNS if f not in df.columns]
    assert not missing, f"Missing features: {missing}"

check("Feature definitions match simulated CSV (18 features)", t_features)


# ── Test 3: Model training ────────────────────────────────────────────────────
def t_train():
    from ml.train_model_noxgb import train
    results_dict = train("data/processed/test_sim.csv", "data/models")
    # With real data this will be ~95%+ — 100% here is fine for synthetic
    assert Path("data/models/best_model.pkl").exists()
    assert Path("data/models/random_forest.pkl").exists()

check("Model training (Random Forest, saves .pkl)", t_train)


# ── Test 4: Prediction — each class ──────────────────────────────────────────
def t_predict_normal():
    from ml.predict import predict_flow, _model
    import importlib, ml.predict as mp
    mp._model = None  # reset cache
    flow = {
        "duration": 2.5, "packet_count": 30, "byte_count": 20000,
        "avg_packet_size": 666, "packet_rate": 12, "byte_rate": 8000,
        "iat_mean": 0.08, "iat_std": 0.02, "iat_min": 0.01, "iat_max": 0.3,
        "syn_count": 1, "rst_count": 0, "ack_count": 18, "fin_count": 1,
        "urg_count": 0, "unique_dst_ports": 1, "src_port": 54123, "dst_port": 80,
    }
    label, conf = predict_flow(flow)
    assert label == "normal", f"Expected normal, got {label}"
    assert conf > 0.5

check("Prediction: normal flow classified correctly", t_predict_normal)


def t_predict_ddos():
    from ml.predict import predict_flow
    import ml.predict as mp; mp._model = None
    flow = {
        "duration": 0.05, "packet_count": 3000, "byte_count": 150000,
        "avg_packet_size": 50, "packet_rate": 60000, "byte_rate": 3000000,
        "iat_mean": 0.00001, "iat_std": 0.000002, "iat_min": 0.000001, "iat_max": 0.0001,
        "syn_count": 2800, "rst_count": 1, "ack_count": 2, "fin_count": 0,
        "urg_count": 0, "unique_dst_ports": 1, "src_port": 54321, "dst_port": 80,
    }
    label, conf = predict_flow(flow)
    assert label == "ddos", f"Expected ddos, got {label}"

check("Prediction: DDoS flow classified correctly", t_predict_ddos)


def t_predict_intrusion():
    from ml.predict import predict_flow
    import ml.predict as mp; mp._model = None
    flow = {
        "duration": 8.0, "packet_count": 200, "byte_count": 12000,
        "avg_packet_size": 60, "packet_rate": 25, "byte_rate": 1500,
        "iat_mean": 0.04, "iat_std": 0.01, "iat_min": 0.005, "iat_max": 0.1,
        "syn_count": 120, "rst_count": 90, "ack_count": 5, "fin_count": 2,
        "urg_count": 1, "unique_dst_ports": 850, "src_port": 44444, "dst_port": 22,
    }
    label, conf = predict_flow(flow)
    assert label == "intrusion", f"Expected intrusion, got {label}"

check("Prediction: intrusion flow classified correctly", t_predict_intrusion)


# ── Test 5: ACL dry run ───────────────────────────────────────────────────────
def t_acl():
    from response.push_acl import push_block, build_commands
    cmds = build_commands("192.168.20.10")
    assert any("deny ip host 192.168.20.10" in c for c in cmds)
    assert any("write memory" in c for c in cmds)
    result = push_block("192.168.20.10", dry_run=True)
    assert result is True

check("ACL response: correct commands, dry-run returns True", t_acl)


# ── Test 6: Offline main pipeline ─────────────────────────────────────────────
def t_main_offline():
    import subprocess, sys
    r = subprocess.run(
        [sys.executable, "main.py", "--offline", "--data", "data/processed/test_sim.csv"],
        capture_output=True, text=True, timeout=30
    )
    assert r.returncode == 0, f"main.py exited with {r.returncode}\n{r.stderr}"
    assert "flows processed" in r.stdout

check("Full offline pipeline (main.py --offline) runs end-to-end", t_main_offline)


# ── Test 7: Paramiko installed (needed for live SSH) ─────────────────────────
def t_paramiko():
    import paramiko
    assert paramiko.__version__

check("paramiko installed (needed for live SSH to switch)", t_paramiko)


# ── Test 8: Wireshark CLI available (needed for live capture) ─────────────────
def t_wireshark():
    import shutil
    assert shutil.which("wireshark") or shutil.which("tshark"), \
        "Neither wireshark nor tshark found in PATH — install Wireshark"

check("Wireshark/tshark available in PATH", t_wireshark)


# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*52)
passed = sum(1 for _, ok in results if ok)
total  = len(results)
print(f"  {passed}/{total} checks passed")
print("="*52)

critical = ["Traffic simulation", "Feature definitions", "Model training",
            "Prediction: normal", "Prediction: DDoS", "Prediction: intrusion",
            "ACL response", "Full offline pipeline"]

critical_failed = [n for n, ok in results if not ok and any(c in n for c in critical)]
if critical_failed:
    print("\n  Fix these before lab day:")
    for n in critical_failed:
        print(f"    - {n}")
    sys.exit(1)
else:
    print("\n  All critical checks passed. Software is lab-ready.")
    warn = [n for n, ok in results if not ok]
    if warn:
        print("  Non-critical warnings (won't block lab):")
        for n in warn:
            print(f"    - {n}")
