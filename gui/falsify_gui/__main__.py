from __future__ import annotations

import argparse
import os

import uvicorn

from .paths import REPO_ROOT


def load_secrets_env() -> None:
    """Read tools/secrets.env (gitignored `export KEY=value` lines) into the
    server environment so spawned jobs and the bridge proxy inherit the Pi
    keys. Variables already set in the environment win."""
    p = REPO_ROOT / "tools" / "secrets.env"
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, sep, val = line.partition("=")
        key, val = key.strip(), val.strip().strip("'").strip('"')
        if sep and key and key not in os.environ:
            os.environ[key] = val


def main() -> None:
    ap = argparse.ArgumentParser(description="falsify management GUI")
    ap.add_argument("--host", default="127.0.0.1",
                    help="bind host (0.0.0.0 exposes job launching to the LAN "
                         "— see README before doing that)")
    ap.add_argument("--port", type=int, default=9000)
    args = ap.parse_args()

    load_secrets_env()
    uvicorn.run("falsify_gui.app:create_app", factory=True,
                host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
