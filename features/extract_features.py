"""
extract_features.py
-------------------
Reads a labeled .pcap file and produces a CSV of flow-level features.
Each "flow" = all packets sharing the same (src_ip, dst_ip, src_port,
dst_port, protocol) 5-tuple within a 60-second timeout window.

HARDWARE REQUIRED for real .pcap input.
For offline development, use simulate_traffic.py instead.

Usage:
    python -m features.extract_features data/raw/capture.pcap
    python -m features.extract_features data/raw/capture.pcap --label ddos
    python -m features.extract_features data/raw/capture.pcap --out data/processed/ddos_flows.csv
"""

import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path
from features.feature_definitions import FEATURE_COLUMNS, LABEL_COLUMN
from utils.logger import get_logger

log = get_logger("extractor")

FLOW_TIMEOUT = 60  # seconds — flows inactive longer than this are closed


def _empty_flow():
    return {
        "packets": [],
        "timestamps": [],
        "bytes": [],
        "flags": {"SYN": 0, "RST": 0, "ACK": 0, "FIN": 0, "URG": 0},
        "dst_ports": set(),
        "src_port": 0,
        "dst_port": 0,
    }


def _flow_to_features(flow: dict, label: str) -> dict:
    pkts = flow["packets"]
    ts   = sorted(flow["timestamps"])
    n    = len(pkts)

    if n == 0:
        return None

    duration   = ts[-1] - ts[0] if n > 1 else 1e-6
    byte_total = sum(flow["bytes"])
    iats       = np.diff(ts) if n > 1 else np.array([0.0])

    return {
        "duration":         round(duration, 6),
        "packet_count":     n,
        "byte_count":       byte_total,
        "avg_packet_size":  round(byte_total / n, 2),
        "packet_rate":      round(n / max(duration, 1e-6), 4),
        "byte_rate":        round(byte_total / max(duration, 1e-6), 4),
        "iat_mean":         round(float(iats.mean()), 6),
        "iat_std":          round(float(iats.std()), 6),
        "iat_min":          round(float(iats.min()), 6),
        "iat_max":          round(float(iats.max()), 6),
        "syn_count":        flow["flags"]["SYN"],
        "rst_count":        flow["flags"]["RST"],
        "ack_count":        flow["flags"]["ACK"],
        "fin_count":        flow["flags"]["FIN"],
        "urg_count":        flow["flags"]["URG"],
        "unique_dst_ports": len(flow["dst_ports"]),
        "src_port":         flow["src_port"],
        "dst_port":         flow["dst_port"],
        LABEL_COLUMN:       label,
    }


def extract_from_pcap(pcap_path: str, label: str) -> pd.DataFrame:
    try:
        from scapy.all import rdpcap, TCP, UDP, IP
    except ImportError:
        log.error("scapy not installed. Run: pip install scapy")
        return pd.DataFrame()

    log.info(f"Reading {pcap_path} ...")
    packets = rdpcap(pcap_path)
    log.info(f"Loaded {len(packets)} packets")

    flows = defaultdict(_empty_flow)
    last_seen = {}

    for pkt in packets:
        if not pkt.haslayer(IP):
            continue
        ip = pkt[IP]
        proto = "TCP" if pkt.haslayer(TCP) else ("UDP" if pkt.haslayer(UDP) else None)
        if proto is None:
            continue

        layer = pkt[TCP] if proto == "TCP" else pkt[UDP]
        key = (ip.src, ip.dst, layer.sport, layer.dport, proto)
        ts  = float(pkt.time)

        # Timeout check — close old flow and start new one
        if key in last_seen and (ts - last_seen[key]) > FLOW_TIMEOUT:
            del flows[key]

        last_seen[key] = ts
        f = flows[key]
        f["packets"].append(pkt)
        f["timestamps"].append(ts)
        f["bytes"].append(len(pkt))
        f["src_port"] = layer.sport
        f["dst_port"] = layer.dport
        f["dst_ports"].add(layer.dport)

        if proto == "TCP":
            flags = layer.flags
            if flags & 0x02: f["flags"]["SYN"] += 1
            if flags & 0x04: f["flags"]["RST"] += 1
            if flags & 0x10: f["flags"]["ACK"] += 1
            if flags & 0x01: f["flags"]["FIN"] += 1
            if flags & 0x20: f["flags"]["URG"] += 1

    rows = [_flow_to_features(f, label) for f in flows.values()]
    rows = [r for r in rows if r is not None]
    df = pd.DataFrame(rows)
    log.info(f"Extracted {len(df)} flows from {pcap_path}")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pcap", help="Path to input .pcap file")
    parser.add_argument("--label", default="normal",
                        choices=["normal", "ddos", "intrusion"],
                        help="Traffic label for this capture session")
    parser.add_argument("--out", default=None,
                        help="Output CSV path (default: data/processed/<pcap_stem>.csv)")
    args = parser.parse_args()

    out = args.out or f"data/processed/{Path(args.pcap).stem}.csv"
    Path(out).parent.mkdir(parents=True, exist_ok=True)

    df = extract_from_pcap(args.pcap, args.label)
    df.to_csv(out, index=False)
    log.info(f"Saved to {out}")


if __name__ == "__main__":
    main()
