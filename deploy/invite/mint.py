#!/usr/bin/env python3
"""Mint a time-limited pilot invite code / signed URL.

Usage (repo root, INVITE_SECRET in env or .env):
  python deploy/invite/mint.py --ttl 72h --label client-acme
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Allow running as `python deploy/invite/mint.py`
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tokens import mint, parse_ttl  # noqa: E402


def _load_dotenv() -> None:
    root = Path(__file__).resolve().parents[2]
    env_path = root / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


def main() -> int:
    parser = argparse.ArgumentParser(description="Mint a time-limited pilot invite")
    parser.add_argument("--ttl", default="72h", help="TTL like 24h, 72h, 7d (default 72h)")
    parser.add_argument("--label", default="", help="Optional label stored in the token")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("INVITE_BASE_URL", "https://ai-doc-pilot.roxanatapia.dev"),
        help="Public pilot host for the redeem URL",
    )
    args = parser.parse_args()
    _load_dotenv()

    try:
        ttl = parse_ttl(args.ttl)
        token = mint(ttl, label=args.label)
    except (RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    base = args.base_url.rstrip("/")
    print(f"ttl_seconds={ttl}")
    if args.label:
        print(f"label={args.label}")
    print(f"code={token}")
    print(f"url={base}/invite/redeem?token={token}")
    print("Deliver out of band. Do not commit this output.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
