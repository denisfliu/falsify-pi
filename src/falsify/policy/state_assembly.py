"""Embodiment-driven state vector assembly.

Before this module, three sites built the policy-facing state vector:
``training.exporter._build_state`` consulted a ``_STATE_GETTERS`` table
keyed by the embodiment YAML's state schema, but both
``policy.pi_gateway.observe`` and ``policy.vla.observe`` hardcoded the
v7 layout ``[px, py, pz, -yaw, 0, 0, 0]`` directly. If an embodiment
ever declared a different layout the policies would silently send the
wrong shape.

This module exposes the **same** schema-driven builder for everyone:

    state_vec = build_state_vector(embodiment, pos_mocap, yaw_mocap,
                                   pos_ned=..., yaw_ned=...)

so train and eval share both the per-camera postprocess
(``CameraPostprocess``) and the state-vector assembly.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


# Per-state-field getter: takes the runtime context as kwargs, returns a float.
_GetterFn = Callable[..., float]


def _g_x_mocap(**kw) -> float: return float(kw["pos_mocap"][0])
def _g_y_mocap(**kw) -> float: return float(kw["pos_mocap"][1])
def _g_z_mocap(**kw) -> float: return float(kw["pos_mocap"][2])
def _g_yaw_mocap(**kw) -> float: return float(kw["yaw_mocap"])
def _g_x_ned(**kw) -> float: return float(kw["pos_ned"][0])
def _g_y_ned(**kw) -> float: return float(kw["pos_ned"][1])
def _g_z_ned(**kw) -> float: return float(kw["pos_ned"][2])
def _g_yaw_ned(**kw) -> float: return float(kw["yaw_ned"])
def _g_gripper(**kw) -> float: return 0.0   # currently always 0.0
def _g_zero(**kw) -> float: return 0.0


STATE_GETTERS: dict[str, _GetterFn] = {
    "x_mocap":   _g_x_mocap,
    "y_mocap":   _g_y_mocap,
    "z_mocap":   _g_z_mocap,
    "yaw_mocap": _g_yaw_mocap,
    "x_ned":     _g_x_ned,
    "y_ned":     _g_y_ned,
    "z_ned":     _g_z_ned,
    "yaw_ned":   _g_yaw_ned,
    "gripper":   _g_gripper,
    "zero":      _g_zero,
}


def build_state_vector(
    embodiment,
    *,
    pos_mocap: np.ndarray,
    yaw_mocap: float,
    pos_ned: np.ndarray | None = None,
    yaw_ned: float | None = None,
) -> np.ndarray:
    """Assemble the policy-facing state vector by walking ``embodiment.state``.

    Parameters
    ----------
    embodiment
        An ``EmbodimentSpec`` (or any object exposing ``.state`` as an
        iterable of fields with a ``.name`` attribute and ``.state_dim()``).
    pos_mocap, yaw_mocap
        Drone pose in MOCAP frame — typical server-frame for v7+
        finetunes (yaw already sign-flipped from NED).
    pos_ned, yaw_ned
        Optional NED-frame values. Required iff the embodiment declares
        any ``*_ned`` state field.

    Returns
    -------
    np.ndarray, shape ``(embodiment.state_dim(),)``, dtype ``float32``.

    Raises
    ------
    ValueError
        If the embodiment names a state field with no registered getter,
        or if NED values are required but not provided.
    """
    out = np.zeros(embodiment.state_dim(), dtype=np.float32)
    ctx = {
        "pos_mocap": pos_mocap,
        "yaw_mocap": yaw_mocap,
        "pos_ned": pos_ned,
        "yaw_ned": yaw_ned,
    }
    for i, field in enumerate(embodiment.state):
        getter = STATE_GETTERS.get(field.name)
        if getter is None:
            raise ValueError(
                f"build_state_vector: no getter registered for state field "
                f"{field.name!r} (declared by embodiment {embodiment.name!r}). "
                f"Add one to STATE_GETTERS in falsify.policy.state_assembly."
            )
        if field.name.endswith("_ned") and ctx["pos_ned"] is None:
            raise ValueError(
                f"build_state_vector: embodiment {embodiment.name!r} requests "
                f"{field.name!r} but pos_ned was not provided"
            )
        out[i] = getter(**ctx)
    return out
