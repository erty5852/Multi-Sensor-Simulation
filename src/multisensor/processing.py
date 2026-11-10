from dataclasses import dataclass
from math import cos, sin

from multisensor.types import Measurement


@dataclass
class PositionMeasurement:
    """An observed position in the common Cartesian frame."""

    timestamp: float
    sensor_name: str
    x: float
    y: float
    confidence: float | None = None


def to_cartesian(measurement: Measurement) -> PositionMeasurement | None:
    """Convert a valid measurement to Cartesian coordinates."""
    if not measurement.is_valid:
        return None

    match measurement.measurement_type:
        case "cartesian_position":
            x, y = measurement.values

        case "polar_position":
            distance, bearing = measurement.values
            x = distance * cos(bearing)
            y = distance * sin(bearing)

        case "bearing_and_range":
            bearing, distance = measurement.values
            x = distance * cos(bearing)
            y = distance * sin(bearing)

        case _:
            raise ValueError(
                f"Unsupported measurement type: {measurement.measurement_type}"
            )

    return PositionMeasurement(
        timestamp=measurement.timestamp,
        sensor_name=measurement.sensor_name,
        x=x,
        y=y,
        confidence=measurement.confidence,
    )


def group_positions_by_timestamp(
    measurements: list[Measurement],
) -> dict[float, list[PositionMeasurement]]:
    """Group valid Cartesian positions by simulation timestamp."""
    groups: dict[float, list[PositionMeasurement]] = {}

    for measurement in measurements:
        position = to_cartesian(measurement)
        if position is None:
            continue

        groups.setdefault(position.timestamp, []).append(position)

    return dict(sorted(groups.items()))