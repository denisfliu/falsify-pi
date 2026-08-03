"""Re-run posthoc classification on an already-captured campaign.

For every trial under a campaign dir, loads ``rollout_states.npz`` +
``trial_card.json`` + ``episode_summary.json``, re-converts the positions
into MOCAP, and re-invokes ``classify_trajectory_posthoc`` with the
campaign's current settings (notably ``expected_dy_sign`` derived from
the trial's scene_key suffix).

Rewrites the trial's ``episode_summary.json`` in place (preserving
everything else) and re-aggregates ``campaign_summary.json``. Backs up
the originals into a ``<file>.bak`` sibling on first run so a second
invocation doesn't destroy the pre-reclassify history.

Usage:

    PYTHONPATH=src python scripts/eval/reclassify_campaign.py \\
        --campaign runs/eval_campaigns/<campaign>

Add ``--dry-run`` to print the new vs old outcomes without writing.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _expected_dy_sign(scene_key: str) -> Optional[int]:
    if scene_key.endswith("_from_left"):
        return -1
    if scene_key.endswith("_from_right"):
        return +1
    return None


def _ned_to_mocap(scene_yaml: Path, positions_ned: np.ndarray) -> np.ndarray:
    from falsify.geometry import Point
    from falsify.io import build_frame_graph, load_yaml
    scene_cfg = load_yaml(scene_yaml)
    fg = build_frame_graph(scene_cfg, base_path=scene_yaml.parent)
    ned_frame = fg.frame("ned")
    out = [
        fg.convert(Point(np.asarray(p, dtype=np.float64), frame=ned_frame), to="mocap").xyz
        for p in positions_ned
    ]
    return np.asarray(out, dtype=np.float64)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--campaign", required=True, type=Path)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    from falsify.io import load_yaml
    from falsify.safety.posthoc import classify_trajectory_posthoc
    from falsify.safety.records import FailureType

    summaries_path = list(sorted(args.campaign.glob("*/trial_*/episode_summary.json")))
    if not summaries_path:
        raise SystemExit(f"No trials under {args.campaign}")

    by_outcome_before: dict[str, int] = defaultdict(int)
    by_outcome_after: dict[str, int] = defaultdict(int)
    changes: list[tuple[str, str, str]] = []

    for sp in summaries_path:
        s = json.loads(sp.read_text())
        scene_key = s["scene_key"]
        trial_dir = sp.parent
        roll = trial_dir / "rollout_states.npz"
        if not roll.is_file():
            continue
        scene_yaml = REPO_ROOT / s["scene"]
        scene_cfg = load_yaml(scene_yaml)

        data = np.load(roll, allow_pickle=True)
        positions_ned = np.asarray(data["positions_ned"])
        positions_mocap = _ned_to_mocap(scene_yaml, positions_ned)
        failure_type_str = (s.get("failure") or {}).get("failure_type") if isinstance(s.get("failure"), dict) else None
        runtime_failure_type: Optional[FailureType] = None
        if failure_type_str:
            # FailureType members in records.py: "GOAL_REACHED" et al.
            runtime_failure_type = getattr(FailureType, failure_type_str, None)
        # The runtime failure_type also lives directly in
        # `rollout_states.npz` as `failure_type` (a numpy object scalar).
        if runtime_failure_type is None and "failure_type" in data.files:
            raw = data["failure_type"]
            try:
                v = raw.item()
            except AttributeError:
                v = raw
            if v is not None and isinstance(v, str):
                runtime_failure_type = getattr(FailureType, v, None)
            elif isinstance(v, FailureType):
                runtime_failure_type = v

        gate_deltas = (s.get("perturbations_manifest") or {}).get("gate_deltas") \
            if isinstance(s.get("perturbations_manifest"), dict) else None

        # True gate-aperture corners (gate surface) for the directional
        # transit check. Prefer the safety path the campaign recorded;
        # fall back to the scenes↔safety naming convention
        # (configs/scenes/<x>.yaml → configs/safety/<x>.yaml) for older
        # summaries that predate `safety_yaml` being persisted.
        from falsify.safety.posthoc import apertures_from_safety_cfg
        safety_yaml = (REPO_ROOT / s["safety_yaml"]) if s.get("safety_yaml") \
            else (REPO_ROOT / "configs" / "safety" / scene_yaml.name)
        aperture_corners = None
        if safety_yaml.is_file():
            _ap = apertures_from_safety_cfg(load_yaml(safety_yaml))
            aperture_corners = _ap[0] if _ap else None

        expected = _expected_dy_sign(scene_key)
        result = classify_trajectory_posthoc(
            positions_mocap=positions_mocap,
            scene_cfg=scene_cfg,
            runtime_failure_type=runtime_failure_type,
            horizon_steps=int(s.get("horizon_s", 25.0) * s.get("hz", 30)),
            n_states=int(s.get("n_states", positions_mocap.shape[0])),
            gate_deltas_mocap=gate_deltas,
            expected_dy_sign=expected,
            aperture_corners=aperture_corners,
        )

        old_outcome = s.get("posthoc_outcome", "UNKNOWN")
        new_outcome = result["outcome"]
        by_outcome_before[old_outcome] += 1
        by_outcome_after[new_outcome] += 1

        if old_outcome != new_outcome:
            changes.append((f"{scene_key}/trial_{s['trial_index']:03d}",
                            old_outcome, new_outcome))

        if args.dry_run:
            continue

        # Backup + write the updated summary in place.
        if not (sp.with_suffix(".json.bak")).exists():
            shutil.copy2(sp, sp.with_suffix(".json.bak"))
        s["posthoc_outcome"] = new_outcome
        s["transited"] = result["transited"]
        s["transit_first_step"] = result["first_inside_step"]
        s["transit_last_step"]  = result["last_inside_step"]
        if expected is not None:
            s["expected_dy_sign"]   = expected
            s["correct_crossings"]  = result.get("correct_crossings")
            s["wrong_crossings"]    = result.get("wrong_crossings")
            s["gate_plane_y_mocap"] = result.get("gate_plane_y_mocap")
        sp.write_text(json.dumps(s, indent=2, default=str))

    print("[reclassify] before:", dict(by_outcome_before))
    print("[reclassify] after: ", dict(by_outcome_after))
    if changes:
        print(f"[reclassify] {len(changes)} trials changed outcome:")
        for trial, old, new in changes:
            print(f"  {trial:<40} {old:<20} -> {new}")
    else:
        print("[reclassify] no outcome changes")

    # Update campaign_summary.json's by_outcome / n_succeeded.
    cs_path = args.campaign / "campaign_summary.json"
    if cs_path.is_file() and not args.dry_run:
        cs = json.loads(cs_path.read_text())
        if not (cs_path.with_suffix(".json.bak")).exists():
            shutil.copy2(cs_path, cs_path.with_suffix(".json.bak"))
        cs["by_outcome"] = dict(by_outcome_after)
        cs["n_succeeded"] = by_outcome_after.get("SUCCESS", 0)
        # Recount per-scene successes.
        per_scene: dict[str, dict] = defaultdict(lambda: {"n": 0, "succeeded": 0})
        for sp in summaries_path:
            s2 = json.loads(sp.read_text())
            sk = s2["scene_key"]
            per_scene[sk]["n"] += 1
            if s2.get("posthoc_outcome") == "SUCCESS":
                per_scene[sk]["succeeded"] += 1
        cs["by_scene"] = dict(per_scene)
        cs_path.write_text(json.dumps(cs, indent=2, default=str))
        print(f"[reclassify] rewrote {cs_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
