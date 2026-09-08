from dataclasses import dataclass


@dataclass
class GroundTruthState:
    """Ground-truth state of the target vehicle at a given timestamp."""

    timestamp: float
    x: float
    y: float
    vx: float
    vy: float


@dataclass
class Measurement:
    """A sensor measurement at a given timestamp."""

    timestamp: float
    sensor_name: str
    measurement_type: str
    values: tuple[float, ...]
    is_valid: bool = True
    confidence: float | None = None