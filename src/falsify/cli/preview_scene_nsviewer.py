"""Launch nerfstudio's ``ns-viewer`` against a falsify scene, applying any
``scene_edits`` declared in the scene YAML before the viewer starts.

Mirrors ``nerfstudio.scripts.viewer.run_viewer.RunViewer.main`` step-by-step
so the only behavioural difference is the in-place edit between
``eval_setup`` and ``_start_viewer`` — the live gsplat the viewer renders
is exactly what ``GSplatRenderer`` would render in a falsify rollout.

Example::

    PYTHONPATH=src:external/FiGS/src:external/splatnav \\
    .venv/bin/python -m falsify.cli.preview_scene_nsviewer \\
        --scene configs/scenes/center_gate.yaml \\
        --port 7007
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--scene", required=True, type=Path,
                   help="Path to a falsify scene YAML (e.g. configs/scenes/center_gate.yaml).")
    p.add_argument("--host", default="0.0.0.0",
                   help="Bind host for viser (default 0.0.0.0).")
    p.add_argument("--port", type=int, default=7007,
                   help="viser websocket port (default 7007).")
    args = p.parse_args(argv)

    # Heavy imports — keep them lazy so --help is snappy.
    from falsify.io import build_frame_graph, load_yaml
    from falsify.sim.scene_edits import apply_edits_to_pipeline, load_scene_edits

    from nerfstudio.utils.eval_utils import eval_setup
    from nerfstudio.scripts.viewer.run_viewer import _start_viewer

    scene_cfg = load_yaml(args.scene)
    scene_dir = args.scene.parent

    def _resolve(p: str) -> Path:
        pp = Path(p)
        return pp if pp.is_absolute() else (scene_dir / pp).resolve()

    gsplat_yml = _resolve(scene_cfg["gsplat_config_yml"])
    data_cwd = _resolve(scene_cfg["gsplat_data_cwd"]) if "gsplat_data_cwd" in scene_cfg else None
    fg = build_frame_graph(scene_cfg, base_path=scene_dir)
    edits = load_scene_edits(scene_cfg)

    print(f"[scene]   {args.scene}")
    print(f"[gsplat]  {gsplat_yml}")
    print(f"[cwd]     {data_cwd}")
    print(f"[edits]   {len(edits)} declared: {[e.name for e in edits]}")

    # Nerfstudio resolves the training config's `data:` field relative to
    # the current working directory; chdir into the dataset root the same
    # way ``GSplatRenderer`` does.
    prev_cwd = os.getcwd()
    if data_cwd is not None:
        os.chdir(data_cwd)
    try:
        config, pipeline, _, step = eval_setup(
            gsplat_yml, eval_num_rays_per_chunk=None, test_mode="test",
        )
    finally:
        os.chdir(prev_cwd)

    if edits:
        n = apply_edits_to_pipeline(pipeline, edits, fg)
        print(f"[edits]   applied {len(edits)} edit(s); {n} Gaussians modified")

    # Mirror RunViewer.main without re-instantiating its dataclass.
    config.vis = "viewer"
    config.viewer.websocket_host = args.host
    config.viewer.websocket_port = args.port
    # The viewer's data manager probes the training dataset; that needs the
    # scene cwd too because the dataparser config's `data:` is relative.
    if data_cwd is not None:
        os.chdir(data_cwd)
    try:
        print(f"[viser]   http://{args.host}:{args.port}")
        _start_viewer(config, pipeline, step)
    finally:
        os.chdir(prev_cwd)
    return 0


if __name__ == "__main__":
    sys.exit(main())
