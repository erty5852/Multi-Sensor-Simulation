from dataclasses import dataclass
from math import isfinite

from multisensor.processing import PositionMeasurement


@dataclass
class FusedPosition:
    """A fused position and the sensors that contributed to it."""

    timestamp: float
    x: float
    y: float
    sensor_names: tuple[str, ...]


class WeightedPositionFusion:
    """Fuse positions using fixed per-sensor uncertainty settings."""

    def __init__(
        self,
        position_std_by_sensor: dict[str, float],
    ) -> None:
        if not position_std_by_sensor:
            raise ValueError("At least one sensor uncertainty is required")

        for name, std in position_std_by_sensor.items():
            if not isfinite(std) or std <= 0.0:
                raise ValueError(
                    f"Position standard deviation for {name} "
                    "must be finite and positive"
                )

        self.position_std_by_sensor = dict(position_std_by_sensor)

    def fuse(
        self,
        positions: list[PositionMeasurement],
    ) -> FusedPosition | None:
        if not positions:
            return None

        timestamp = positions[0].timestamp
        sensor_names: list[str] = []
        weighted_x = 0.0
        weighted_y = 0.0
        total_weight = 0.0

        for position in positions:
            if position.timestamp != timestamp:
                raise ValueError(
                    "All positions must have the same timestamp"
                )

            name = position.sensor_name

            if name in sensor_names:
                raise ValueError(
                    f"Duplicate sensor in fusion group: {name}"
                )

            if name not in self.position_std_by_sensor:
                raise ValueError(
                    f"No position uncertainty configured for {name}"
                )

            std = self.position_std_by_sensor[name]
            weight = 1.0 / std**2

            weighted_x += weight * position.x
            weighted_y += weight * position.y
            total_weight += weight
            sensor_names.append(name)

        return FusedPosition(
            timestamp=timestamp,
            x=weighted_x / total_weight,
            y=weighted_y / total_weight,
            sensor_names=tuple(sensor_names),
        )