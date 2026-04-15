"""
main.py
-------
AdaptiGuard full pipeline entry point.

Offline mode (no hardware):
    python main.py --offline

Live mode (PC4 with SPAN port):
    sudo python main.py --iface eth1

What it does:
    1. Captures traffic (or loads from CSV in offline mode)
    2. Extracts flow features in a sliding window
    3. Classifies each flow with the trained ML model
    4. If attack detected: logs it + optionally pushes ACL to switch
"""

import argparse
import time
import pandas as pd
from pathlib import Path
from ml.predict import load_model, predict_flow
from response.push_acl import push_block
from features.feature_definitions import FEATURE_COLUMNS
from utils.logger import get_logger

log = get_logger("main")

CONFIDENCE_THRESHOLD = 0.85   # Only act if model is >85% confident
DRY_RUN = True                # Set to False only after hardware is set up


def run_offline(csv_path: str):
    """Simulate live classification on a pre-labeled CSV (for dev/demo)."""
    log.info(f"OFFLINE MODE — classifying flows from {'D:\Taufeeq Riyaz\Projects\AdaptiGuard\data\processed\simulated_traffic.csv'}")
    load_model()

    df = pd.read_csv("D:\Taufeeq Riyaz\Projects\AdaptiGuard\data\processed\simulated_traffic.csv")
    total = len(df)
    attacks = 0

    for i, row in df.iterrows():
        flow = row[FEATURE_COLUMNS].to_dict()
        label, conf = predict_flow(flow)

        if label != "normal" and conf >= CONFIDENCE_THRESHOLD:
            attacks += 1
            src_ip = f"192.168.20.{(i % 254) + 1}"  # simulated src IP
            log.info(
                f"[ALERT] Flow {i:>5} | {label.upper():<10} "
                f"conf={conf:.2%} | src={src_ip}"
            )
            push_block(src_ip, dry_run=DRY_RUN)

    log.info(f"Done. {total} flows processed, {attacks} attacks detected.")


def run_live(iface: str):
    """
    Live mode — requires Phase 1 hardware to be set up.
    Captures packets in windows, extracts features, classifies, responds.
    """
    log.info(f"LIVE MODE — listening on {iface}")
    log.info("Phase 1 hardware must be complete before using this mode.")

    try:
        from scapy.all import sniff, IP, TCP, UDP
    except ImportError:
        log.error("scapy not installed.")
        return

    load_model()

    WINDOW = 2.0  # seconds per classification window
    buffer = []

    def process_window(pkts):
        if not pkts:
            return
        # Minimal feature extraction from window
        n = len(pkts)
        if n < 5:
            return
        sizes  = [len(p) for p in pkts]
        ts     = [float(p.time) for p in pkts]
        dur    = ts[-1] - ts[0] if n > 1 else 1e-6
        iats   = [ts[i+1] - ts[i] for i in range(len(ts)-1)] or [0]

        flow = {
            "duration":         dur,
            "packet_count":     n,
            "byte_count":       sum(sizes),
            "avg_packet_size":  sum(sizes) / n,
            "packet_rate":      n / max(dur, 1e-6),
            "byte_rate":        sum(sizes) / max(dur, 1e-6),
            "iat_mean":         sum(iats) / len(iats),
            "iat_std":          pd.Series(iats).std() or 0,
            "iat_min":          min(iats),
            "iat_max":          max(iats),
            "syn_count":        sum(1 for p in pkts if p.haslayer(TCP) and p[TCP].flags & 0x02),
            "rst_count":        sum(1 for p in pkts if p.haslayer(TCP) and p[TCP].flags & 0x04),
            "ack_count":        sum(1 for p in pkts if p.haslayer(TCP) and p[TCP].flags & 0x10),
            "fin_count":        sum(1 for p in pkts if p.haslayer(TCP) and p[TCP].flags & 0x01),
            "urg_count":        sum(1 for p in pkts if p.haslayer(TCP) and p[TCP].flags & 0x20),
            "unique_dst_ports": len(set(p[TCP].dport for p in pkts if p.haslayer(TCP))),
            "src_port":         pkts[0][TCP].sport if pkts[0].haslayer(TCP) else 0,
            "dst_port":         pkts[0][TCP].dport if pkts[0].haslayer(TCP) else 0,
        }

        label, conf = predict_flow(flow)
        if label != "normal" and conf >= CONFIDENCE_THRESHOLD:
            src_ip = pkts[0][IP].src if pkts[0].haslayer(IP) else "unknown"
            log.info(f"[ALERT] {label.upper()} conf={conf:.2%} src={src_ip}")
            push_block(src_ip, dry_run=DRY_RUN)

    def collect(pkt):
        buffer.append(pkt)

    log.info(f"Starting live capture (window={WINDOW}s) ...")
    while True:
        buffer.clear()
        sniff(iface=iface, prn=collect, timeout=WINDOW, store=False)
        process_window(list(buffer))


def main():
    parser = argparse.ArgumentParser(description="AdaptiGuard")
    parser.add_argument("--offline", action="store_true",
                        help="Run on simulated/pre-captured CSV (no hardware needed)")
    parser.add_argument("--data",    default="data/processed/simulated_traffic.csv")
    parser.add_argument("--iface",   default="eth1",
                        help="Network interface for live capture (PC4 SPAN port)")
    parser.add_argument("--live",    action="store_true",
                        help="Disable dry-run and push real ACL rules to switch")
    args = parser.parse_args()

    global DRY_RUN
    DRY_RUN = not args.live

    if args.offline:
        run_offline(args.data)
    else:
        run_live(args.iface)


if __name__ == "__main__":
    main()
