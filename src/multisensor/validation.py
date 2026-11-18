from dataclasses import dataclass
from math import hypot, sqrt

from multisensor.processing import to_cartesian
from multisensor.types import GroundTruthState, Measurement


@dataclass
class SensorMetrics:
    """Detection and position metrics for one sensor."""

    attempts: int
    detections: int
    detection_rate: float
    position_mae: float | None
    position_rmse: float | None
    bias_x: float | None
    bias_y: float | None


def evaluate_sensors(
    ground_truth: list[GroundTruthState],
    measurements: list[Measurement],
) -> dict[str, SensorMetrics]:
    """Evaluate each sensor using the shared simulation timestamps."""
    truth_by_time = {state.timestamp: state for state in ground_truth}
    attempts_by_sensor: dict[str, int] = {}
    errors_by_sensor: dict[str, list[tuple[float, float]]] = {}

    for measurement in measurements:
        name = measurement.sensor_name
        attempts_by_sensor[name] = attempts_by_sensor.get(name, 0) + 1

        position = to_cartesian(measurement)
        if position is None:
            continue

        truth = truth_by_time.get(position.timestamp)
        if truth is None:
            raise ValueError(
                f"No ground truth for {name} at t={position.timestamp}"
            )

        dx = position.x - truth.x
        dy = position.y - truth.y
        errors_by_sensor.setdefault(name, []).append((dx, dy))

    results: dict[str, SensorMetrics] = {}

    for name, attempts in attempts_by_sensor.items():
        errors = errors_by_sensor.get(name, [])
        detections = len(errors)

        mae = rmse = bias_x = bias_y = None
        if detections:
            mae = sum(hypot(dx, dy) for dx, dy in errors) / detections
            rmse = sqrt(
                sum(dx**2 + dy**2 for dx, dy in errors) / detections
            )
            bias_x = sum(dx for dx, _ in errors) / detections
            bias_y = sum(dy for _, dy in errors) / detections

        results[name] = SensorMetrics(
            attempts=attempts,
            detections=detections,
            detection_rate=detections / attempts,
            position_mae=mae,
            position_rmse=rmse,
            bias_x=bias_x,
            bias_y=bias_y,
        )

    return results