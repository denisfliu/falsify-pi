"""Ops CLI for the bridge's admin endpoints.

Talks to the bridge's HTTP admin port (same as the WS port) over plain
stdlib urllib — no extra deps. The same `Authorization: Api-Key` header
used by WS connections also gates admin calls.

Usage::

    # List registered policies + which is active
    python -m pi_local_bridge.switch \\
        --admin-url http://moraband.stanford.edu:8765 \\
        --api-key-env PI_BRIDGE_API_KEYS --list

    # Switch the active policy (blocks while the new checkpoint loads)
    python -m pi_local_bridge.switch \\
        --admin-url http://moraband.stanford.edu:8765 \\
        --api-key-env PI_BRIDGE_API_KEYS \\
        --policy-id v7-nonhistory-center

`PiGatewayPolicy` on the falsify side already calls /admin/switch_policy
during its handshake when `bridge_admin_url` is set; this CLI exists for
manual ops / debugging.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Optional


def _resolve_api_key(args: argparse.Namespace) -> str:
    if args.api_key:
        return args.api_key
    if args.api_key_env:
        v = os.environ.get(args.api_key_env)
        if not v:
            raise SystemExit(f"env var {args.api_key_env!r} is unset")
        # If the env var holds a comma-separated allow-list (as the bridge
        # expects), pick the first key.
        return v.split(",", 1)[0].strip()
    raise SystemExit("either --api-key or --api-key-env must be supplied")


def _request(
    method: str,
    url: str,
    api_key: str,
    timeout: float,
) -> tuple[int, bytes]:
    req = urllib.request.Request(url, method=method)
    req.add_header("Authorization", f"Api-Key {api_key}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def _cmd_list(args: argparse.Namespace) -> int:
    api_key = _resolve_api_key(args)
    url = args.admin_url.rstrip("/") + "/admin/policies"
    code, body = _request("GET", url, api_key, args.timeout)
    if code != 200:
        sys.stderr.write(f"GET {url} → {code}\n{body.decode('utf-8', errors='replace')}\n")
        return 1
    parsed = json.loads(body)
    active = parsed["active_policy_id"]
    print(f"active: {active}")
    print(f"loaded_at: {parsed.get('active_loaded_at')}")
    print("registered:")
    for entry in parsed["policies"]:
        marker = "*" if entry["is_active"] else " "
        print(f"  {marker} {entry['policy_id']:32s}  {entry['ws_path']}")
        if args.verbose:
            print(f"       ckpt: {entry.get('ckpt_path')}")
    return 0


def _cmd_switch(args: argparse.Namespace) -> int:
    api_key = _resolve_api_key(args)
    qs = urllib.parse.urlencode({"policy_id": args.policy_id})
    url = args.admin_url.rstrip("/") + "/admin/switch_policy?" + qs
    # Admin uses GET (the `websockets` sync server's HTTP parser rejects
    # any non-GET method before our process_request hook runs). The
    # mutation is idempotent and the endpoint is internal-only.
    code, body = _request("GET", url, api_key, args.timeout)
    text = body.decode("utf-8", errors="replace")
    if code != 200:
        sys.stderr.write(f"POST {url} → {code}\n{text}\n")
        return 1
    parsed = json.loads(text)
    print(f"active: {parsed['active_policy_id']}")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="pi-local-bridge.switch")
    ap.add_argument("--admin-url", required=True,
                    help="Base URL of the bridge (e.g. http://moraband.stanford.edu:8765)")
    auth = ap.add_mutually_exclusive_group()
    auth.add_argument("--api-key", help="Inline API key (avoid in shell history)")
    auth.add_argument("--api-key-env", help="Env var holding the API key (or comma-list; first is used)")
    ap.add_argument("--timeout", type=float, default=300.0,
                    help="HTTP timeout in seconds (default 300; swaps can take ~60s on cold load)")
    ap.add_argument("--list", dest="list_only", action="store_true",
                    help="List registered policies and which is active")
    ap.add_argument("--policy-id", help="Target policy id to switch to")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="Show ckpt_path under each entry when listing")
    args = ap.parse_args(argv)

    if args.list_only:
        return _cmd_list(args)
    if not args.policy_id:
        ap.error("either --list or --policy-id is required")
    return _cmd_switch(args)


if __name__ == "__main__":
    sys.exit(main())
