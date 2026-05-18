"""Stage + combine the 4 derived datasets via combine_lerobot.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/build_combined_datasets.py

Builds:
    data/gate_scenes_real_synth   (real + synthetic)        — 200 ep
    data/gate_scenes_real_center  (real + central)          — 200 ep
    data/gate_scenes_center       (central only)            — 100 ep
    data/gate_scenes_all          (real + synthetic + central) — 300 ep
"""
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"

LEFT = "go through the gate on the left and hover over the stuffed animal"
RIGHT = "go through the gate on the right and hover over the stuffed animal"
CENTER = "go through the center gate and hover over the stuffed animal"

# (output_name, list of source dataset names under data/, list of --task specs)
PLANS = [
    ("gate_scenes_real_synth",
     ["gate_scenes_real_combined", "synth_left_gate", "synth_right_gate"],
     [f"50:{LEFT}", f"50:{RIGHT}", f"50:{LEFT}", f"rest:{RIGHT}"]),
    ("gate_scenes_real_center",
     ["gate_scenes_real_combined", "synth_center_from_left", "synth_center_from_right"],
     [f"50:{LEFT}", f"50:{RIGHT}", f"50:{CENTER}", f"rest:{CENTER}"]),
    ("gate_scenes_center",
     ["synth_center_from_left", "synth_center_from_right"],
     [f"rest:{CENTER}"]),
    ("gate_scenes_all",
     ["gate_scenes_real_combined", "synth_center_from_left", "synth_center_from_right",
      "synth_left_gate", "synth_right_gate"],
     [f"50:{LEFT}", f"50:{RIGHT}", f"50:{CENTER}", f"50:{CENTER}",
      f"50:{LEFT}", f"rest:{RIGHT}"]),
]


def main():
    for name, sources, task_specs in PLANS:
        stage = REPO / "runs" / f"combine_stage_{name}"
        out = DATA / name

        print()
        print(f"=== {name} → {out} ===")

        # Wipe + re-stage.
        if stage.exists():
            # Remove old symlinks/directory.
            for child in stage.iterdir():
                if child.is_symlink() or child.is_file():
                    child.unlink()
                elif child.is_dir():
                    shutil.rmtree(child)
            stage.rmdir()
        stage.mkdir(parents=True)
        if out.exists():
            shutil.rmtree(out)

        for s in sources:
            src_path = (DATA / s).resolve()
            (stage / s).symlink_to(src_path, target_is_directory=True)
        print(f"  staged: {sources}")

        cmd = [
            str(REPO / ".venv/bin/python"),
            "-m", "falsify.cli.combine_lerobot",
            "--src", str(stage),
            "--out", str(out),
            "--overwrite",
        ]
        for spec in task_specs:
            cmd += ["--task", spec]
        result = subprocess.run(
            cmd,
            cwd=str(REPO),
            env={**__import__("os").environ, "PYTHONPATH": str(REPO / "src")},
            capture_output=True,
            text=True,
        )
        # Print last few lines of stdout
        out_lines = result.stdout.strip().splitlines()
        for line in out_lines[-6:]:
            print(f"  {line}")
        if result.returncode != 0:
            print(f"  STDERR: {result.stderr[:500]}")
            sys.exit(result.returncode)

    print()
    print("=== summary ===")
    for name, _, _ in PLANS:
        out = DATA / name
        n = len(list((out / "data" / "chunk-000").glob("*.parquet")))
        print(f"  {out}  {n} episodes")


if __name__ == "__main__":
    main()
