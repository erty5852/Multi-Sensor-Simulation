from dataclasses import dataclass
from math import sqrt

from multisensor.fusion import FusedPosition, WeightedPositionFusion
from multisensor.processing import group_positions_by_timestamp
from multisensor.simulation.runner import SimulationResult


@dataclass
class FusionEvaluation:
    """Fusion outputs and RMSE on shared comparison timestamps."""

    fused_positions: list[FusedPosition]
    output_coverage: float | None
    common_sample_count: int
    sensor_rmse: dict[str, float | None]
    fused_rmse: float | None


def evaluate_fusion(
    simulation: SimulationResult,
    position_std_by_sensor: dict[str, float],
) -> FusionEvaluation:
    fusion = WeightedPositionFusion(position_std_by_sensor)
    groups = group_positions_by_timestamp(simulation.measurements)
    truth_by_time = {state.timestamp: state for state in simulation.ground_truth}
    expected_sensors = set(position_std_by_sensor)

    fused_positions: list[FusedPosition] = []
    squared_error_by_sensor = dict.fromkeys(position_std_by_sensor, 0.0)
    fused_squared_error = 0.0
    common_sample_count = 0

    for timestamp, positions in groups.items():
        truth = truth_by_time.get(timestamp)
        if truth is None:
            raise ValueError(f"No ground truth at t={timestamp}")

        fused = fusion.fuse(positions)
        if fused is None:
            continue

        fused_positions.append(fused)

        observed_sensors = set(fused.sensor_names)
        if observed_sensors != expected_sensors:
            continue

        common_sample_count += 1
        for position in positions:
            dx = position.x - truth.x
            dy = position.y - truth.y
            squared_error_by_sensor[position.sensor_name] += dx**2 + dy**2

        dx = fused.x - truth.x
        dy = fused.y - truth.y
        fused_squared_error += dx**2 + dy**2

    def rmse(squared_error: float) -> float | None:
        if common_sample_count == 0:
            return None
        return sqrt(squared_error / common_sample_count)

    return FusionEvaluation(
        fused_positions=fused_positions,
        output_coverage=(
            len(fused_positions) / len(truth_by_time) if truth_by_time else None
        ),
        common_sample_count=common_sample_count,
        sensor_rmse={
            name: rmse(error) for name, error in squared_error_by_sensor.items()
        },
        fused_rmse=rmse(fused_squared_error),
    )