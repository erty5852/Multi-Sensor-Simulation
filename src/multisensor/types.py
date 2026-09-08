from dataclasses import dataclass


@dataclass
class GroundTruthState:
    """Ground-truth state of the target vehicle at a given timestamp."""

    timestamp: float
    x: float
    y: float
    vx: float
    vy: float