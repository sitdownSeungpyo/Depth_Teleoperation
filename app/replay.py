"""Replay a JSONL skeleton recording through the full Phase 1 chain.

Usage (PowerShell):
    python -m app.replay --config .\\config\\default.yaml --replay .\\tests\\fixtures\\arm_circle.jsonl
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from app.main import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a skeleton JSONL through the controller")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--publisher", choices=("mock", "udp"), default="mock")
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run(
        config_path=args.config,
        tracker_kind="mock",
        publisher_kind=args.publisher,
        replay=args.replay,
        duration_s=args.duration,
    )


if __name__ == "__main__":
    main()
