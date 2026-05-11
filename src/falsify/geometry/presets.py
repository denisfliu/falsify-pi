"""Named axis-permutation presets for NED ↔ Z-up style conversions.

FiGS uses a small integer ``permutation`` field (0–7) to pick which axis-and-
sign pattern relates NED to the MOCAP Z-up world. The integer scheme was
opaque, so falsify exposes the same patterns by **name**.

Each preset returns a 3×3 ``np.ndarray`` that you can wrap in an `SE3` or
`Sim3` with the appropriate src/dst frame names.

The presets here mirror the bit-encoding used in SousVide:
- bit 0: flip y
- bit 1: flip z
- bit 2: also swap x ↔ y

These are kept verbatim from SousVide so existing alignment files continue to
work. New scenes are encouraged to declare a custom ``se3_inline`` transform
instead of using these.
"""

from __future__ import annotations

import numpy as np


def _from_bits(bits: int) -> np.ndarray:
    """Recover the 3×3 axis-permutation/sign matrix for the SousVide-style
    permutation integer.

    A permutation integer ``bits`` decomposes as:
      bit 0 → flip y
      bit 1 → flip z
      bit 2 → swap x,y axes (applied first)
    """
    swap_xy = bool(bits & 0b100)
    flip_y = bool(bits & 0b001)
    flip_z = bool(bits & 0b010)

    if swap_xy:
        # [[0,1,0],[1,0,0],[0,0,1]]
        M = np.array([[0.0, 1.0, 0.0],
                      [1.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0]])
    else:
        M = np.eye(3)

    if flip_y:
        M = np.diag([1.0, -1.0, 1.0]) @ M
    if flip_z:
        M = np.diag([1.0, 1.0, -1.0]) @ M
    return M


# Named entries — at minimum, ship the perm5 SousVide presets used. Extend as
# new scenes need them; users may also pass a literal preset name not in this
# table by writing ``perm<int>`` (handled by `from_name` below).

_NAMED: dict[str, np.ndarray] = {
    "identity": np.eye(3),
    "perm0": _from_bits(0),
    "perm1": _from_bits(1),
    "perm2": _from_bits(2),
    "perm3": _from_bits(3),
    "perm4": _from_bits(4),
    "perm5": _from_bits(5),   # the SousVide default for both left_gate / right_gate
    "perm6": _from_bits(6),
    "perm7": _from_bits(7),
    # Friendly aliases.
    "ned_to_zup_xyflip": _from_bits(5),  # alias for perm5 (most common in SousVide)
}


def axis_permutation(name: str) -> np.ndarray:
    """Return the 3×3 matrix for a named preset.

    Accepts entries from the table above plus any string of the form
    ``perm<int>`` for direct access to the SousVide bit encoding.
    """
    if name in _NAMED:
        return _NAMED[name].copy()
    if name.startswith("perm"):
        try:
            bits = int(name[4:])
        except ValueError as exc:
            raise KeyError(f"unrecognized preset {name!r}") from exc
        if 0 <= bits <= 7:
            return _from_bits(bits)
    raise KeyError(f"unknown axis-permutation preset {name!r}")


def available_presets() -> tuple[str, ...]:
    return tuple(sorted(_NAMED.keys()))
