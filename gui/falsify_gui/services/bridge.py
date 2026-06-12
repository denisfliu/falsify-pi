"""Proxy for the pi_local_bridge admin API (mirrors pi_local_bridge/switch.py:
stdlib urllib + `Authorization: Api-Key` header).

Admin URL resolution: explicit param > most-common bridge_admin_url across the
pi_gateway policy YAMLs > $FALSIFY_GUI_BRIDGE_URL.
The key comes from $PI_BRIDGE_API_KEYS (first entry) or $PI_API_KEY.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter

from . import configs_enum

SWITCH_TIMEOUT_S = 420.0   # cold JAX load on the bridge can take minutes


def _api_key() -> str | None:
    keys = os.environ.get("PI_BRIDGE_API_KEYS", "")
    if keys.strip():
        return keys.split(",")[0].strip()
    return os.environ.get("PI_API_KEY") or None


def default_admin_url() -> str | None:
    env = os.environ.get("FALSIFY_GUI_BRIDGE_URL")
    urls = [p.get("bridge_admin_url") for p in configs_enum.get_configs()["policies"]
            if p.get("bridge_admin_url")]
    if urls:
        return Counter(urls).most_common(1)[0][0]
    return env


def _get(url: str, timeout: float) -> dict:
    req = urllib.request.Request(url)
    key = _api_key()
    if key:
        req.add_header("Authorization", f"Api-Key {key}")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def list_policies(admin_url: str | None = None) -> dict:
    base = (admin_url or default_admin_url() or "").rstrip("/")
    if not base:
        return {"reachable": False, "error": "no bridge admin URL configured"}
    try:
        doc = _get(base + "/admin/policies", timeout=5.0)
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as e:
        body = ""
        if isinstance(e, urllib.error.HTTPError):
            try:
                body = e.read().decode(errors="replace")[:200]
            except OSError:
                pass
        return {"reachable": False, "admin_url": base,
                "error": f"{e}{(' — ' + body) if body else ''}",
                "key_present": _api_key() is not None}
    # decorate with local YAML metadata keyed by bridge_policy_id
    by_bridge_id = {p["bridge_policy_id"]: p
                    for p in configs_enum.get_configs()["policies"]
                    if p.get("bridge_policy_id")}
    for pol in doc.get("policies", []):
        local = by_bridge_id.get(pol.get("policy_id"))
        if local:
            pol["yaml_name"] = local["name"]
            pol["yaml_path"] = local["path"]
            pol["traceability"] = local.get("traceability")
    return {"reachable": True, "admin_url": base, **doc}


def switch_policy(policy_id: str, admin_url: str | None = None) -> dict:
    base = (admin_url or default_admin_url() or "").rstrip("/")
    q = urllib.parse.urlencode({"policy_id": policy_id})
    try:
        return {"ok": True,
                **_get(f"{base}/admin/switch_policy?{q}", timeout=SWITCH_TIMEOUT_S)}
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode(errors="replace")[:300]
        except OSError:
            pass
        return {"ok": False, "error": f"HTTP {e.code}: {body or e.reason}"}
    except (urllib.error.URLError, OSError) as e:
        return {"ok": False, "error": str(e)}
