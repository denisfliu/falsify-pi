"""Policy layer.

`Policy` declares its modality requirements (`required_modalities`) and
produces a `Trajectory` in the NED frame from an `Observation`. The
`sensors` package decouples *what* the policy needs from *how* it's filled.
"""

from .base import Policy
from .observation import Observation, ObservationBuilder
from .mock import MockStraightLine, MockStraightLineConfig, MockNoisy, MockNoisyConfig
from .vla import VLAPolicy, VLAPolicyConfig, register_policy_host
from .pi_gateway import PiGatewayPolicy, PiGatewayConfig

__all__ = [
    "Policy", "Observation", "ObservationBuilder",
    "MockStraightLine", "MockStraightLineConfig",
    "MockNoisy", "MockNoisyConfig",
    "VLAPolicy", "VLAPolicyConfig", "register_policy_host",
    "PiGatewayPolicy", "PiGatewayConfig",
]
