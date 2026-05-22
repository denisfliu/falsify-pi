"""Evaluation-pipeline support: per-trial card sampling, seed derivation."""

from .sampling import (
    sample_gate_perturbation,
    sample_start_mocap,
    seed_for,
)

__all__ = ["sample_gate_perturbation", "sample_start_mocap", "seed_for"]
