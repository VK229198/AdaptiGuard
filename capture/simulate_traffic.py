"""
simulate_traffic.py
-------------------
Generates a synthetic labeled traffic CSV that mirrors what real
pcap extraction will produce. Run this while hardware isn't ready.
Once real .pcap files are captured, switch to extract_features.py
— the output CSV format is identical, so no ML code changes needed.

Usage:
    python -m capture.simulate_traffic
    python -m capture.simulate_traffic --samples 5000 --out data/processed/sim.csv
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from features.feature_definitions import FEATURE_COLUMNS, LABEL_COLUMN
from utils.logger import get_logger

log = get_logger("simulator")


def _normal_flow(rng: np.random.Generator) -> dict:
    """HTTP/DNS/FTP-like benign traffic."""
    duration   = rng.uniform(0.1, 5.0)
    pkt_count  = rng.integers(5, 80)
    byte_count = pkt_count * rng.integers(64, 1400)
    iats       = rng.exponential(0.05, max(pkt_count - 1, 1))
    return {
        "duration":         duration,
        "packet_count":     int(pkt_count),
        "byte_count":       int(byte_count),
        "avg_packet_size":  byte_count / pkt_count,
        "packet_rate":      pkt_count / max(duration, 1e-6),
        "byte_rate":        byte_count / max(duration, 1e-6),
        "iat_mean":         float(iats.mean()),
        "iat_std":          float(iats.std()),
        "iat_min":          float(iats.min()),
        "iat_max":          float(iats.max()),
        "syn_count":        rng.integers(1, 3),
        "rst_count":        rng.integers(0, 2),
        "ack_count":        int(pkt_count * rng.uniform(0.4, 0.7)),
        "fin_count":        rng.integers(0, 2),
        "urg_count":        0,
        "unique_dst_ports": rng.integers(1, 3),
        "src_port":         int(rng.integers(1024, 65535)),
        "dst_port":         int(rng.choice([80, 443, 53, 21, 22, 25])),
        LABEL_COLUMN:       "normal",
    }


def _ddos_flow(rng: np.random.Generator) -> dict:
    """UDP/SYN flood — high packet rate, tiny IAT, mostly SYN flags."""
    duration   = rng.uniform(0.01, 0.5)
    pkt_count  = rng.integers(500, 5000)
    byte_count = pkt_count * rng.integers(40, 64)
    iats       = rng.exponential(0.0001, max(pkt_count - 1, 1))
    return {
        "duration":         duration,
        "packet_count":     int(pkt_count),
        "byte_count":       int(byte_count),
        "avg_packet_size":  byte_count / pkt_count,
        "packet_rate":      pkt_count / max(duration, 1e-6),
        "byte_rate":        byte_count / max(duration, 1e-6),
        "iat_mean":         float(iats.mean()),
        "iat_std":          float(iats.std()),
        "iat_min":          float(iats.min()),
        "iat_max":          float(iats.max()),
        "syn_count":        int(pkt_count * rng.uniform(0.7, 1.0)),
        "rst_count":        rng.integers(0, 5),
        "ack_count":        rng.integers(0, 10),
        "fin_count":        0,
        "urg_count":        0,
        "unique_dst_ports": rng.integers(1, 3),
        "src_port":         int(rng.integers(1024, 65535)),
        "dst_port":         int(rng.choice([80, 443, 53])),
        LABEL_COLUMN:       "ddos",
    }


def _intrusion_flow(rng: np.random.Generator) -> dict:
    """Port scan / brute-force — many unique dst ports, RST responses."""
    duration   = rng.uniform(1.0, 10.0)
    pkt_count  = rng.integers(50, 300)
    byte_count = pkt_count * rng.integers(40, 100)
    iats       = rng.uniform(0.01, 0.1, max(pkt_count - 1, 1))
    return {
        "duration":         duration,
        "packet_count":     int(pkt_count),
        "byte_count":       int(byte_count),
        "avg_packet_size":  byte_count / pkt_count,
        "packet_rate":      pkt_count / max(duration, 1e-6),
        "byte_rate":        byte_count / max(duration, 1e-6),
        "iat_mean":         float(iats.mean()),
        "iat_std":          float(iats.std()),
        "iat_min":          float(iats.min()),
        "iat_max":          float(iats.max()),
        "syn_count":        int(pkt_count * rng.uniform(0.4, 0.7)),
        "rst_count":        int(pkt_count * rng.uniform(0.2, 0.5)),
        "ack_count":        rng.integers(0, 20),
        "fin_count":        rng.integers(0, 5),
        "urg_count":        rng.integers(0, 3),
        "unique_dst_ports": rng.integers(20, 1024),
        "src_port":         int(rng.integers(1024, 65535)),
        "dst_port":         int(rng.integers(1, 1024)),
        LABEL_COLUMN:       "intrusion",
    }


def generate(n_samples: int = 3000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    generators = [_normal_flow, _ddos_flow, _intrusion_flow]
    # 60% normal, 25% ddos, 15% intrusion — realistic imbalance
    weights = [0.60, 0.25, 0.15]
    choices = rng.choice(len(generators), size=n_samples, p=weights)
    rows = [generators[c](rng) for c in choices]
    df = pd.DataFrame(rows)
    log.info(f"Generated {n_samples} flows — label distribution:")
    log.info(str(df[LABEL_COLUMN].value_counts()))
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=3000)
    parser.add_argument("--out", default="data/processed/simulated_traffic.csv")
    args = parser.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df = generate(args.samples)
    df.to_csv(args.out, index=False)
    log.info(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
