"""Observation-side perturbations.

Each perturbation mutates one keyed modality in an `Observation`. Because
`Observation` is frozen, perturbations return a new `Observation` whose
`data` dict has the updated key.

Frame-tagged values inside `obs.data` (e.g. `state.pos: Point[ned]`) are not
mutated unless a perturbation explicitly targets them; the canonical
`StateNoise` perturbation builds a new `Point` in the same frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from falsify.geometry import Point
from falsify.policy.observation import Observation
from .base import ObservationPerturbation, _jsonable


@dataclass
class ImageGaussianNoise(ObservationPerturbation):
    """Add Gaussian noise to one camera's image."""
    camera: str = "forward"
    std: float = 5.0   # in 0–255 pixel intensity units
    name: str = ""

    def __post_init__(self):
        if not self.name:
            self.name = f"image_noise[{self.camera}]"
        self._key = f"images.{self.camera}"

    def apply(self, obs: Observation, *, rng: np.random.Generator) -> Observation:
        img = obs.get(self._key)
        if img is None:
            return obs   # camera not present this episode; no-op
        noise = rng.normal(scale=self.std, size=img.shape)
        new_img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(img.dtype)
        new_data = dict(obs.data)
        new_data[self._key] = new_img
        return Observation(state=obs.state, data=new_data, prompt=obs.prompt)

    def manifest(self) -> dict:
        return {"name": self.name, "type": "ImageGaussianNoise",
                "camera": self.camera, "std": _jsonable(self.std)}


@dataclass
class ImageBlur(ObservationPerturbation):
    """Box-blur one camera's image (uniform kernel)."""
    camera: str = "forward"
    kernel: int = 5
    name: str = ""

    def __post_init__(self):
        if self.kernel < 1 or self.kernel % 2 == 0:
            raise ValueError(f"kernel must be a positive odd int; got {self.kernel}")
        if not self.name:
            self.name = f"image_blur[{self.camera}]"
        self._key = f"images.{self.camera}"

    def apply(self, obs: Observation, *, rng: np.random.Generator) -> Observation:
        img = obs.get(self._key)
        if img is None:
            return obs
        new_img = _box_blur(img, self.kernel)
        new_data = dict(obs.data)
        new_data[self._key] = new_img
        return Observation(state=obs.state, data=new_data, prompt=obs.prompt)

    def manifest(self) -> dict:
        return {"name": self.name, "type": "ImageBlur",
                "camera": self.camera, "kernel": self.kernel}


@dataclass
class StateNoise(ObservationPerturbation):
    """Add Gaussian noise to the ``state.pos`` Point (frame preserved)."""
    std: float = 0.02
    name: str = "state_noise"

    def apply(self, obs: Observation, *, rng: np.random.Generator) -> Observation:
        pos: Point = obs.require("state.pos")
        noise = rng.normal(scale=self.std, size=(3,))
        new_pos = Point(pos.xyz + noise, frame=pos.frame)
        new_data = dict(obs.data)
        new_data["state.pos"] = new_pos
        return Observation(state=obs.state, data=new_data, prompt=obs.prompt)

    def manifest(self) -> dict:
        return {"name": self.name, "type": "StateNoise", "std": _jsonable(self.std)}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _box_blur(img: np.ndarray, k: int) -> np.ndarray:
    """Vectorized box blur using a separable uniform kernel.

    Avoids opencv dependency. For full sensor-pipeline realism (motion blur,
    bayer noise, defocus) we'll switch to a richer backend; this is enough
    for smoke testing.
    """
    if img.ndim == 2:
        return _box_blur_2d(img, k)
    out = np.empty_like(img, dtype=img.dtype)
    for c in range(img.shape[2]):
        out[..., c] = _box_blur_2d(img[..., c], k)
    return out


def _box_blur_2d(plane: np.ndarray, k: int) -> np.ndarray:
    arr = plane.astype(np.float32)
    half = k // 2
    padded = np.pad(arr, half, mode="edge")
    cs = padded.cumsum(axis=0).cumsum(axis=1)
    cs = np.pad(cs, ((1, 0), (1, 0)), mode="constant")
    h, w = arr.shape
    total = (
        cs[k:k+h, k:k+w] - cs[0:h, k:k+w] - cs[k:k+h, 0:w] + cs[0:h, 0:w]
    )
    return (total / (k * k)).astype(plane.dtype)
