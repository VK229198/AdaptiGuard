# All 18 flow-level features used by AdaptiGuard.
# This file is the single source of truth — both the extractor and
# the ML model import FEATURE_COLUMNS from here so they never go out of sync.

FEATURE_COLUMNS = [
    "duration",           # Flow duration in seconds
    "packet_count",       # Total packets in flow
    "byte_count",         # Total bytes in flow
    "avg_packet_size",    # Mean bytes per packet
    "packet_rate",        # Packets per second
    "byte_rate",          # Bytes per second
    "iat_mean",           # Mean inter-arrival time (seconds)
    "iat_std",            # Std dev of inter-arrival time
    "iat_min",            # Min inter-arrival time
    "iat_max",            # Max inter-arrival time
    "syn_count",          # Number of SYN packets
    "rst_count",          # Number of RST packets
    "ack_count",          # Number of ACK packets
    "fin_count",          # Number of FIN packets
    "urg_count",          # Number of URG packets
    "unique_dst_ports",   # Distinct destination ports (high = port scan)
    "src_port",           # Source port number
    "dst_port",           # Destination port number
]

# Label column
LABEL_COLUMN = "label"

# Label encoding
LABEL_MAP = {
    "normal":    0,
    "ddos":      1,
    "intrusion": 2,
}

LABEL_NAMES = {v: k for k, v in LABEL_MAP.items()}
