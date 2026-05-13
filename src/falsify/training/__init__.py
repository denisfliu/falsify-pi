"""Training-data export pipeline.

See ``CLAUDE.md`` in this directory for the three-layer contract
(Trajectory → Scene + Embodiment → Parquet) and how to author embodiments
without touching code.
"""

from .trajectory import (
    Trajectory,
    load_trajectory,
    save_trajectory,
    from_episode_trace,
    from_vla_run_dir,
    resample,
)
from .embodiment import EmbodimentSpec, load_embodiment
from .parquet_writer import ParquetWriter
from .exporter import TrainingDataExporter, ExportResult

__all__ = [
    "Trajectory",
    "load_trajectory",
    "save_trajectory",
    "from_episode_trace",
    "from_vla_run_dir",
    "resample",
    "EmbodimentSpec",
    "load_embodiment",
    "ParquetWriter",
    "TrainingDataExporter",
    "ExportResult",
]
