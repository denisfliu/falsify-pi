"""Parquet writer that matches the LeRobot v2-style schema used by
DroneVLA2.0 (reference: ``~/Downloads/episode_000008.parquet``).

Per-row columns
---------------
- one ``struct<bytes: binary, path: string>`` per image column
  (PNG bytes embedded; ``path`` is a relative ``frame_NNNNNN.png`` for HF tooling)
- ``state``   : ``fixed_size_list<float32>[state_dim]``
- ``actions`` : ``fixed_size_list<float32>[action_dim]``
- ``timestamp``     float32   seconds since episode start
- ``frame_index``   int64     per-episode 0..N-1
- ``episode_index`` int64     episode number (caller supplies)
- ``index``         int64     global frame index (caller supplies via offset)
- ``task_index``    int64     0 for single-task episodes

Schema metadata
---------------
Schema-level metadata carries a ``huggingface`` JSON blob describing each
feature. This lets a LeRobot/HF loader open the parquet directly.

The writer accepts in-memory PNG bytes per image column — the caller
(``TrainingDataExporter``) handles channel-order conversion and encoding.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


@dataclass
class SampleRow:
    """One frame's worth of data, pre-encoded."""
    images: dict[str, bytes]      # column_name -> PNG bytes
    state: np.ndarray             # (state_dim,) float32
    actions: np.ndarray           # (action_dim,) float32
    timestamp_s: float
    frame_index: int


def _build_hf_metadata(
    image_columns: Sequence[str],
    state_dim: int,
    action_dim: int,
) -> dict:
    features: dict = {}
    for col in image_columns:
        features[col] = {"_type": "Image"}
    features["state"] = {
        "feature": {"dtype": "float32", "_type": "Value"},
        "length": state_dim,
        "_type": "Sequence",
    }
    features["actions"] = {
        "feature": {"dtype": "float32", "_type": "Value"},
        "length": action_dim,
        "_type": "Sequence",
    }
    for col, dtype in (
        ("timestamp", "float32"),
        ("frame_index", "int64"),
        ("episode_index", "int64"),
        ("index", "int64"),
        ("task_index", "int64"),
    ):
        features[col] = {"dtype": dtype, "_type": "Value"}
    return {"info": {"features": features}}


def _image_struct_type() -> pa.DataType:
    return pa.struct([
        pa.field("bytes", pa.binary()),
        pa.field("path", pa.string()),
    ])


class ParquetWriter:
    """Buffer rows, flush to a single parquet matching the reference schema."""

    def __init__(
        self,
        image_columns: Sequence[str],
        state_dim: int,
        action_dim: int,
        *,
        episode_index: int = 0,
        task_index: int = 0,
        index_offset: int = 0,
    ) -> None:
        self.image_columns = tuple(image_columns)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.episode_index = int(episode_index)
        self.task_index = int(task_index)
        self.index_offset = int(index_offset)
        self._rows: list[SampleRow] = []

    def add(self, row: SampleRow) -> None:
        if set(row.images) != set(self.image_columns):
            raise ValueError(
                f"row.images keys {sorted(row.images)} != writer columns "
                f"{sorted(self.image_columns)}"
            )
        if row.state.shape != (self.state_dim,):
            raise ValueError(f"state shape {row.state.shape} != ({self.state_dim},)")
        if row.actions.shape != (self.action_dim,):
            raise ValueError(f"actions shape {row.actions.shape} != ({self.action_dim},)")
        self._rows.append(row)

    def __len__(self) -> int:
        return len(self._rows)

    def flush(self, out_path: str | Path) -> Path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if not self._rows:
            raise ValueError("no rows to write")

        n = len(self._rows)
        image_struct_t = _image_struct_type()
        image_arrays: dict[str, pa.Array] = {}
        for col in self.image_columns:
            structs = [
                {"bytes": r.images[col], "path": f"frame_{r.frame_index:06d}.png"}
                for r in self._rows
            ]
            image_arrays[col] = pa.array(structs, type=image_struct_t)

        state_arr = pa.array(
            [r.state.astype(np.float32).tolist() for r in self._rows],
            type=pa.list_(pa.float32(), self.state_dim),
        )
        actions_arr = pa.array(
            [r.actions.astype(np.float32).tolist() for r in self._rows],
            type=pa.list_(pa.float32(), self.action_dim),
        )

        timestamps = pa.array(
            [float(r.timestamp_s) for r in self._rows], type=pa.float32(),
        )
        frame_indices = pa.array(
            [int(r.frame_index) for r in self._rows], type=pa.int64(),
        )
        episode_indices = pa.array([self.episode_index] * n, type=pa.int64())
        global_indices = pa.array(
            [self.index_offset + r.frame_index for r in self._rows], type=pa.int64(),
        )
        task_indices = pa.array([self.task_index] * n, type=pa.int64())

        columns = {}
        for col in self.image_columns:
            columns[col] = image_arrays[col]
        columns["state"] = state_arr
        columns["actions"] = actions_arr
        columns["timestamp"] = timestamps
        columns["frame_index"] = frame_indices
        columns["episode_index"] = episode_indices
        columns["index"] = global_indices
        columns["task_index"] = task_indices

        table = pa.table(columns)
        hf_meta = _build_hf_metadata(self.image_columns, self.state_dim, self.action_dim)
        meta = {b"huggingface": json.dumps(hf_meta).encode("utf-8")}
        table = table.replace_schema_metadata(meta)

        pq.write_table(table, out_path)
        return out_path
