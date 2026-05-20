"""`GaussianBoxDistribution` — 6-d truncated-by-clipping multivariate Gaussian.

Models the CEM search distribution over the perturbation vector
``[start_dx, start_dy, start_dz, gate_dx, gate_dy, gate_dyaw]``. The
distribution is parameterised by (mean, cov) of a multivariate Gaussian
and a per-dimension half-width vector ``bounds``; samples are clipped
into ``[-bounds, +bounds]`` element-wise.

Initial state matches the uniform recipe: mean=0, cov=diag(h^2 / 3) (the
variance of ``Uniform(-h, +h)``). With ``cov_shrink=1.0`` the
distribution stays uniform-equivalent across refits; with
``cov_shrink=0.0`` the refit uses the empirical elite covariance
verbatim. The default 0.1 blends a small prior weight in to stop the
covariance from collapsing when the elite set is tiny.

JSON schema (what ``save_json`` writes and ``load_json`` consumes)::

    {
      "target_failure_type": "COLLISION_GATE",
      "param_names": ["start_dx", "start_dy", "start_dz",
                      "gate_dx", "gate_dy", "gate_dyaw"],
      "mean": [6 floats],
      "cov":  [[6x6 floats]],
      "bounds": [6 floats],                # half-widths; gate_dz pinned 0
      "bounds_layout": {
        "start_half_widths_mocap":  [hx, hy, hz],
        "gate_offset_half_widths":  [hx, hy, 0.0],
        "gate_yaw_half_width_rad":  theta
      },
      "provenance": {...}                  # free-form; producer fills it
    }
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


# 6-d ordering used everywhere. Don't change without updating
# generate_eval_bundles.py + run_eval_campaign.py + the scorer.
PARAM_NAMES: tuple[str, ...] = (
    "start_dx", "start_dy", "start_dz",
    "gate_dx",  "gate_dy",  "gate_dyaw",
)
PARAM_DIM = len(PARAM_NAMES)
# Indices into the 6-d vector — keep callers from hard-coding 0..5.
IDX_START = slice(0, 3)
IDX_GATE_XY = slice(3, 5)
IDX_GATE_YAW = 5


def _box_to_bounds_vec(box: dict) -> np.ndarray:
    """Flatten the layout dict into the 6-d half-width vector.

    Pins ``gate_dz`` to 0 unconditionally — the gate doesn't levitate.
    """
    start = np.asarray(box["start_half_widths_mocap"], dtype=np.float64).reshape(-1)
    if start.shape != (3,):
        raise ValueError(
            f"start_half_widths_mocap must be (3,); got {start.shape}"
        )
    gate_xyz = np.asarray(box["gate_offset_half_widths"], dtype=np.float64).reshape(-1)
    if gate_xyz.shape != (3,):
        raise ValueError(
            f"gate_offset_half_widths must be (3,); got {gate_xyz.shape}"
        )
    if gate_xyz[2] != 0.0:
        raise ValueError(
            "gate_offset_half_widths[2] (gate_dz) must be 0 — gates don't "
            f"levitate. Got {gate_xyz[2]}."
        )
    yaw_hw = float(box["gate_yaw_half_width_rad"])
    if (start < 0).any() or (gate_xyz < 0).any() or yaw_hw < 0:
        raise ValueError(
            "All bounds must be non-negative half-widths."
        )
    # Order: start_xyz, gate_xy, gate_yaw. gate_dz is *not* a free dim —
    # it's pinned to 0 by zeroing its bound and the sampler always emits 0
    # for it via PARAM_NAMES.index('gate_dz')... oh wait, it's not in
    # PARAM_NAMES. The bounds vector is 6-d to match PARAM_NAMES exactly:
    #   [start_dx, start_dy, start_dz, gate_dx, gate_dy, gate_dyaw]
    return np.array([
        start[0], start[1], start[2],
        gate_xyz[0], gate_xyz[1],
        yaw_hw,
    ], dtype=np.float64)


def _bounds_vec_to_layout(bounds: np.ndarray) -> dict:
    """Inverse of ``_box_to_bounds_vec`` — gate_dz always 0 on round-trip."""
    return {
        "start_half_widths_mocap":  [float(bounds[0]), float(bounds[1]), float(bounds[2])],
        "gate_offset_half_widths":  [float(bounds[3]), float(bounds[4]), 0.0],
        "gate_yaw_half_width_rad":  float(bounds[5]),
    }


@dataclass
class GaussianBoxDistribution:
    """6-d truncated-by-clipping multivariate Gaussian over the perturbation
    vector. Truncation is implemented by element-wise clipping of samples
    into ``[-bounds, +bounds]`` — it's a degenerate but simple choice that
    matches the existing uniform sampler's box exactly. Switch to true
    rejection sampling later if the clipping bias becomes a problem.

    All shapes are (PARAM_DIM,) and (PARAM_DIM, PARAM_DIM).
    """

    mean: np.ndarray
    cov: np.ndarray
    bounds: np.ndarray
    target_failure_type: Optional[str] = None
    provenance: dict = field(default_factory=dict)

    # ---- construction helpers --------------------------------------------

    def __post_init__(self) -> None:
        self.mean = np.asarray(self.mean, dtype=np.float64).reshape(-1)
        self.cov = np.asarray(self.cov, dtype=np.float64)
        self.bounds = np.asarray(self.bounds, dtype=np.float64).reshape(-1)
        if self.mean.shape != (PARAM_DIM,):
            raise ValueError(f"mean must be ({PARAM_DIM},); got {self.mean.shape}")
        if self.cov.shape != (PARAM_DIM, PARAM_DIM):
            raise ValueError(f"cov must be ({PARAM_DIM},{PARAM_DIM}); got {self.cov.shape}")
        if self.bounds.shape != (PARAM_DIM,):
            raise ValueError(f"bounds must be ({PARAM_DIM},); got {self.bounds.shape}")
        if (self.bounds < 0).any():
            raise ValueError(f"bounds must be non-negative; got {self.bounds}")

    @classmethod
    def uniform_prior(
        cls,
        bounds_layout: dict,
        target_failure_type: Optional[str] = None,
        provenance: Optional[dict] = None,
    ) -> "GaussianBoxDistribution":
        """Build the iter-0 distribution: mean=0, cov = diag(h^2 / 3).

        The (h^2 / 3) variance matches ``Uniform(-h, +h)``, so iter-0 samples
        are statistically equivalent to the existing uniform recipe (modulo
        the clipping at the box boundary, which is rare for a Gaussian with
        this variance).
        """
        bounds = _box_to_bounds_vec(bounds_layout)
        var = (bounds ** 2) / 3.0
        # Pin gate_dz: bounds[2] is start-z (free); the gate-z slot doesn't
        # exist in the 6-d vector at all. So no extra pinning is needed
        # here — the parameterization itself excludes gate_dz.
        return cls(
            mean=np.zeros(PARAM_DIM),
            cov=np.diag(var),
            bounds=bounds,
            target_failure_type=target_failure_type,
            provenance=dict(provenance or {}),
        )

    # ---- sampling --------------------------------------------------------

    def sample(self, rng: np.random.Generator, n: int = 1) -> np.ndarray:
        """Draw ``n`` samples, shape (n, PARAM_DIM), each clipped into the box."""
        if n < 1:
            raise ValueError(f"n must be >= 1; got {n}")
        # multivariate_normal needs a PSD cov; add a tiny ridge for safety.
        cov_psd = self.cov + 1e-12 * np.eye(PARAM_DIM)
        raw = rng.multivariate_normal(self.mean, cov_psd, size=n)
        return np.clip(raw, -self.bounds, +self.bounds)

    # ---- refitting -------------------------------------------------------

    def refit(
        self,
        elites: np.ndarray,
        cov_shrink: float = 0.1,
        min_eigenvalue: float = 1e-6,
    ) -> "GaussianBoxDistribution":
        """Return a *new* distribution fit to the elite set.

        ``elites`` is (k, PARAM_DIM). Mean is the elite mean; covariance is
        a shrinkage blend ``(1 - α) · Σ_elites + α · Σ_prior`` where
        ``Σ_prior = diag(bounds^2 / 3)`` (the uniform prior's covariance).
        The blend prevents covariance collapse with small elite sets and
        keeps the distribution from getting stuck if the elite set is
        rank-deficient.

        Provenance, bounds, and target_failure_type are carried over.
        """
        elites = np.asarray(elites, dtype=np.float64)
        if elites.ndim != 2 or elites.shape[1] != PARAM_DIM:
            raise ValueError(
                f"elites must be (k, {PARAM_DIM}); got {elites.shape}"
            )
        if elites.shape[0] < 1:
            raise ValueError("refit needs at least one elite sample")
        if not (0.0 <= cov_shrink <= 1.0):
            raise ValueError(f"cov_shrink must be in [0, 1]; got {cov_shrink}")

        mean_new = elites.mean(axis=0)
        if elites.shape[0] >= 2:
            cov_elites = np.cov(elites, rowvar=False, ddof=0)
        else:
            # With one elite, empirical cov is zero — rely fully on prior.
            cov_elites = np.zeros((PARAM_DIM, PARAM_DIM))
        cov_prior = np.diag((self.bounds ** 2) / 3.0)
        cov_new = (1.0 - cov_shrink) * cov_elites + cov_shrink * cov_prior

        # Floor eigenvalues to keep cov_new strictly PD — avoids
        # multivariate_normal warnings on near-degenerate covs.
        eigvals, eigvecs = np.linalg.eigh(cov_new)
        eigvals_floor = np.clip(eigvals, min_eigenvalue, None)
        cov_new = (eigvecs * eigvals_floor) @ eigvecs.T
        # Symmetrise to kill any numerical asymmetry.
        cov_new = 0.5 * (cov_new + cov_new.T)

        return GaussianBoxDistribution(
            mean=mean_new,
            cov=cov_new,
            bounds=self.bounds.copy(),
            target_failure_type=self.target_failure_type,
            provenance=dict(self.provenance),
        )

    # ---- (de)serialization ----------------------------------------------

    def to_dict(self) -> dict:
        return {
            "target_failure_type": self.target_failure_type,
            "param_names": list(PARAM_NAMES),
            "mean": self.mean.tolist(),
            "cov":  self.cov.tolist(),
            "bounds": self.bounds.tolist(),
            "bounds_layout": _bounds_vec_to_layout(self.bounds),
            "provenance": dict(self.provenance),
        }

    def save_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path

    @classmethod
    def from_dict(cls, data: dict) -> "GaussianBoxDistribution":
        param_names = tuple(data.get("param_names") or PARAM_NAMES)
        if param_names != PARAM_NAMES:
            raise ValueError(
                "GaussianBoxDistribution param_names mismatch:\n"
                f"  expected: {PARAM_NAMES}\n"
                f"  got:      {param_names}\n"
                "If the parameter space has changed, bump the schema or "
                "regenerate the distribution."
            )
        return cls(
            mean=np.asarray(data["mean"], dtype=np.float64),
            cov=np.asarray(data["cov"], dtype=np.float64),
            bounds=np.asarray(data["bounds"], dtype=np.float64),
            target_failure_type=data.get("target_failure_type"),
            provenance=dict(data.get("provenance") or {}),
        )

    @classmethod
    def load_json(cls, path: str | Path) -> "GaussianBoxDistribution":
        path = Path(path)
        return cls.from_dict(json.loads(path.read_text()))

    # ---- unpacking helpers ----------------------------------------------

    @staticmethod
    def unpack(theta: np.ndarray) -> dict:
        """Turn one 6-d sample into the trial-card-shaped payload.

        Returns ``{"start_delta_mocap": [dx,dy,dz], "gate_delta_xyz":
        [gx,gy,0.0], "gate_delta_yaw_rad": yaw}``. ``gate_delta_xyz``'s
        z-component is always 0 by construction.
        """
        theta = np.asarray(theta, dtype=np.float64).reshape(-1)
        if theta.shape != (PARAM_DIM,):
            raise ValueError(f"theta must be ({PARAM_DIM},); got {theta.shape}")
        return {
            "start_delta_mocap":   [float(theta[0]), float(theta[1]), float(theta[2])],
            "gate_delta_xyz":      [float(theta[3]), float(theta[4]), 0.0],
            "gate_delta_yaw_rad":  float(theta[5]),
        }
