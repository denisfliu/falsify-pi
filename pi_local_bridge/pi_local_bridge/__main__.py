"""CLI entry point: ``python -m pi_local_bridge --config bridge.yaml``."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from pi_local_bridge.server import run_server


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="pi-local-bridge")
    ap.add_argument("--config", "-c", required=True, type=Path,
                    help="Bridge YAML (listen, auth, policy, spec_overrides)")
    ap.add_argument("--log-level", default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        run_server(args.config)
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
