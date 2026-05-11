"""Pluggable observation pipeline.

Sensors produce named modalities into an `ObservationBuilder` each timestep.
Policies declare which modalities they need; the `SensorRig` asserts coverage
at construction. Adding a new sensor (lidar, IMU, event camera) is a matter
of subclassing `Sensor` — no churn to `Observation` or `Policy`.

See ``CLAUDE.md`` in this directory for the contract.
"""

from .base import Sensor, SensorRig
from .state import StateSensor
from .prompt import PromptSensor
from .camera import CameraSensor, CameraSpec, make_camera_sensor_from_yaml
from .factory import build_sensor_rig

__all__ = [
    "Sensor", "SensorRig",
    "StateSensor", "PromptSensor",
    "CameraSensor", "CameraSpec", "make_camera_sensor_from_yaml",
    "build_sensor_rig",
]
