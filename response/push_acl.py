"""
push_acl.py
-----------
SSHes into the 48-port core switch and pushes an ACL rule to block
a malicious source IP.

HARDWARE REQUIRED — needs Phase 1 done and SSH enabled on the switch.
Until hardware is ready, runs in DRY_RUN mode by default and just
prints the commands it would send.

Usage:
    python -m response.push_acl --src-ip 192.168.20.10 --dry-run
    python -m response.push_acl --src-ip 192.168.20.10  (real switch)
"""

import argparse
import time
from utils.logger import get_logger

log = get_logger("response")

# ── Switch connection settings — edit these for your switch ──────────────────
SWITCH_HOST = "192.168.30.1"    # Management IP of core switch (VLAN 30 SVI)
SWITCH_USER = "admin"
SWITCH_PASS = "admin"           # Change to your switch password
SSH_PORT    = 22
ACL_NAME    = "ADAPTIBLOCK"     # ACL name applied on ingress of VLAN 20 int
# ─────────────────────────────────────────────────────────────────────────────


def build_commands(src_ip: str, acl_name: str = ACL_NAME) -> list[str]:
    """
    Returns Cisco IOS-style config commands to block a source IP.
    Adjust syntax if your switch uses a different OS (e.g. Comware).
    """
    return [
        "configure terminal",
        f"ip access-list extended {acl_name}",
        f"deny ip host {src_ip} any",
        "permit ip any any",
        "exit",
        "interface vlan 20",                    # Apply on attacker-facing VLAN
        f"ip access-group {acl_name} in",
        "exit",
        "end",
        "write memory",
    ]


def push_block(src_ip: str, dry_run: bool = True):
    commands = build_commands(src_ip)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    if dry_run:
        log.info(f"[DRY RUN] Would block {src_ip} at {timestamp}")
        log.info("Commands that would be sent:")
        for cmd in commands:
            log.info(f"  {cmd}")
        return True

    # ── Real SSH execution (needs hardware) ───────────────────────────────────
    try:
        import paramiko
    except ImportError:
        log.error("paramiko not installed. Run: pip install paramiko")
        return False

    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(SWITCH_HOST, port=SSH_PORT,
                    username=SWITCH_USER, password=SWITCH_PASS,
                    timeout=10)

        shell = ssh.invoke_shell()
        time.sleep(0.5)

        for cmd in commands:
            shell.send(cmd + "\n")
            time.sleep(0.3)

        output = shell.recv(4096).decode("utf-8", errors="ignore")
        ssh.close()

        log.info(f"Blocked {src_ip} at {timestamp}")
        log.info(f"Switch response:\n{output}")
        return True

    except Exception as e:
        log.error(f"SSH failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-ip",  required=True, help="IP to block")
    parser.add_argument("--dry-run", action="store_true", default=True,
                        help="Print commands without sending (default: True)")
    parser.add_argument("--live",    action="store_true",
                        help="Actually push to switch (overrides --dry-run)")
    args = parser.parse_args()
    dry = not args.live
    push_block(args.src_ip, dry_run=dry)


if __name__ == "__main__":
    main()
