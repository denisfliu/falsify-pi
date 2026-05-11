"""SplatNav recovery planner — frame-aware wrapper.

Single translation site: NED in, NED out. Everything inside this module
operates in the NS (Nerfstudio-internal) frame because that's what
`splatplan.splatplan.SplatPlan` expects; the frame contract for callers is
strictly NED.

Heavy imports (torch, splatnav, nerfstudio) are deferred to construction
time so importing this module is cheap on machines without CUDA.

The default `_SplatNavBackend` actually calls SplatPlan; tests pass a stub
backend to exercise the wrapper without splatnav installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Protocol, Sequence

import numpy as np

from falsify.geometry import FrameGraph, Point, Trajectory, assert_frame


# ---------------------------------------------------------------------------
# Backend protocol
# ---------------------------------------------------------------------------


class PlannerBackend(Protocol):
    """A minimal callable surface that the wrapper depends on.

    The real backend wraps SplatPlan; the test backend returns a hand-rolled
    waypoint list. Frame is **always NS** at this layer.
    """

    def generate_path(self, x0_ns: np.ndarray, xf_ns: np.ndarray) -> np.ndarray:
        ...


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class RecoveryConfig:
    """Planner config in **NED** (the public-facing frame).

    All bounds and the goal are declared in NED for convenience. The
    wrapper converts them to NS internally via the active `FrameGraph`.
    """
    bounds_lower_ned: Sequence[float]
    bounds_upper_ned: Sequence[float]
    radius_m: float = 0.05
    vmax: float = 2.0
    amax: float = 3.0
    voxel_resolution: int = 100


@dataclass
class RecoveryResult:
    trajectory: Trajectory                  # frame == "ned"
    feasible: bool = True
    info: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------


class SplatNavPlanner:
    """Frame-aware recovery planner.

    Parameters
    ----------
    cfg, frame_graph
        Configuration and active frame registry.
    backend
        Implementation of `PlannerBackend`. If None, the real splatnav
        backend is constructed lazily from `gsplat_config_path`.
    gsplat_config_path
        Path to a Nerfstudio ``config.yml`` (used only when `backend` is None).
    horizon_s, hz
        Used to attach times to the returned `Trajectory` — splatnav itself
        returns positions only, so we stamp them at the controller rate.
    """

    def __init__(
        self,
        cfg: RecoveryConfig,
        frame_graph: FrameGraph,
        *,
        backend: Optional[PlannerBackend] = None,
        gsplat_config_path: Optional[str | Path] = None,
        horizon_s: float = 5.0,
        hz: int = 10,
    ) -> None:
        self.cfg = cfg
        self.frame_graph = frame_graph
        self._backend = backend
        self._gsplat_config_path = gsplat_config_path
        self._horizon_s = float(horizon_s)
        self._hz = int(hz)

    # ---- public API ----------------------------------------------------

    def plan(self, start: Point, goal: Point) -> RecoveryResult:
        assert_frame(start, "ned")
        assert_frame(goal, "ned")

        start_ns = self.frame_graph.convert(start, to="ns").xyz
        goal_ns = self.frame_graph.convert(goal, to="ns").xyz

        backend = self._get_backend()
        traj_ns_xyz = np.asarray(backend.generate_path(start_ns, goal_ns), dtype=np.float64)
        if traj_ns_xyz.ndim != 2 or traj_ns_xyz.shape[1] < 3:
            raise ValueError(
                f"backend returned shape {traj_ns_xyz.shape}; expected (N, ≥3)"
            )
        # Trajectory points are in NS — convert to NED in one place.
        ns_frame = self.frame_graph.frame("ns")
        n = traj_ns_xyz.shape[0]
        times = np.linspace(0.0, self._horizon_s, n)
        traj_ns = Trajectory(
            times=times, positions=traj_ns_xyz[:, :3].copy(), frame=ns_frame,
        )
        traj_ned = self.frame_graph.convert(traj_ns, to="ned")
        assert_frame(traj_ned, "ned")
        return RecoveryResult(trajectory=traj_ned)

    # ---- internals -----------------------------------------------------

    def _get_backend(self) -> PlannerBackend:
        if self._backend is None:
            if self._gsplat_config_path is None:
                raise ValueError(
                    "no backend supplied and no gsplat_config_path set — "
                    "either inject a PlannerBackend or provide a config.yml"
                )
            self._backend = _make_splatnav_backend(
                config_path=self._gsplat_config_path,
                cfg=self.cfg,
                frame_graph=self.frame_graph,
            )
        return self._backend


# ---------------------------------------------------------------------------
# Real splatnav backend (lazy-loaded)
# ---------------------------------------------------------------------------


def _make_splatnav_backend(
    *,
    config_path: str | Path,
    cfg: RecoveryConfig,
    frame_graph: FrameGraph,
) -> PlannerBackend:
    """Construct a backend that actually calls SplatPlan.

    Lazy-imports torch + splatnav inside, so importing this module costs
    nothing on machines without CUDA.
    """
    import torch  # type: ignore
    from splat.splat_utils import GSplatLoader  # type: ignore
    from splatplan.splatplan import SplatPlan  # type: ignore
    from splatplan.spline_utils import SplinePlanner  # type: ignore

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gsplat = GSplatLoader(Path(config_path), device)

    # Convert NED bounds to NS once. We do it via the frame graph so future
    # scenes with different transforms work without code changes.
    lower_ned = Point(np.asarray(cfg.bounds_lower_ned, dtype=np.float64), frame=frame_graph.frame("ned"))
    upper_ned = Point(np.asarray(cfg.bounds_upper_ned, dtype=np.float64), frame=frame_graph.frame("ned"))
    lower_ns = frame_graph.convert(lower_ned, to="ns").xyz
    upper_ns = frame_graph.convert(upper_ned, to="ns").xyz
    # Element-wise min/max to handle frames that flip signs.
    lo = torch.tensor(np.minimum(lower_ns, upper_ns), device=device, dtype=torch.float32)
    hi = torch.tensor(np.maximum(lower_ns, upper_ns), device=device, dtype=torch.float32)

    spline_planner = SplinePlanner(spline_deg=6, N_sec=10, device=device)
    planner = SplatPlan(
        gsplat,
        {"radius": cfg.radius_m, "vmax": cfg.vmax, "amax": cfg.amax},
        {"lower_bound": lo, "upper_bound": hi, "resolution": cfg.voxel_resolution},
        spline_planner,
        device,
    )

    class _RealBackend:
        def __init__(self, _planner, _device):
            self._planner = _planner
            self._device = _device

        def generate_path(self, x0_ns: np.ndarray, xf_ns: np.ndarray) -> np.ndarray:
            x0 = torch.tensor(x0_ns, dtype=torch.float32, device=self._device)
            xf = torch.tensor(xf_ns, dtype=torch.float32, device=self._device)
            out = self._planner.generate_path(x0, xf)
            traj = out["traj"] if isinstance(out, dict) else out
            return np.asarray(traj)

    return _RealBackend(planner, device)
