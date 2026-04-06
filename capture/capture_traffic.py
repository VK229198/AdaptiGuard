"""
capture_traffic.py
------------------
Runs on PC4. Listens on the SPAN/mirror port interface and writes
packets to a timestamped .pcap file in data/raw/.

HARDWARE REQUIRED — this script will not work without Phase 1 done.
Until then, use simulate_traffic.py.

Usage (on PC4, must be root/sudo):
    sudo python -m capture.capture_traffic --iface eth1 --duration 300
    sudo python -m capture.capture_traffic --iface eth1 --count 100000

Arguments:
    --iface     Network interface connected to SPAN port (default: eth1)
    --duration  Capture duration in seconds (0 = run until Ctrl+C)
    --count     Max packet count (0 = unlimited)
    --out       Output .pcap path (default: data/raw/<timestamp>.pcap)
"""

import argparse
from datetime import datetime
from pathlib import Path
from utils.logger import get_logger

log = get_logger("capture")


def capture(iface: str, duration: int, count: int, out: str):
    try:
        from scapy.all import sniff, wrpcap
    except ImportError:
        log.error("scapy not installed. Run: pip install scapy")
        return

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    log.info(f"Starting capture on interface '{iface}'")
    log.info(f"Output: {out}")

    kwargs = {"iface": iface, "store": True}
    if duration > 0:
        kwargs["timeout"] = duration
        log.info(f"Duration: {duration}s")
    if count > 0:
        kwargs["count"] = count
        log.info(f"Max packets: {count}")

    packets = sniff(**kwargs)
    wrpcap(out, packets)
    log.info(f"Captured {len(packets)} packets → {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iface",    default="eth1")
    parser.add_argument("--duration", type=int, default=0)
    parser.add_argument("--count",    type=int, default=0)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser.add_argument("--out", default=f"data/raw/capture_{timestamp}.pcap")
    args = parser.parse_args()
    capture(args.iface, args.duration, args.count, args.out)


if __name__ == "__main__":
    main()
