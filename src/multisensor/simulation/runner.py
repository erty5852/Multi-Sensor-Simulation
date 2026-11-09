from dataclasses import dataclass

from multisensor.simulation.sensors import SensorModel
from multisensor.types import GroundTruthState, Measurement


@dataclass
class SimulationResult:
    """Store ground truth and sensor measurements from one simulation."""

    ground_truth: list[GroundTruthState]
    measurements: list[Measurement]


class SimulationRunner:
    """Run sensor models against a sequence of ground-truth states."""

    def __init__(self, sensors: list[SensorModel]) -> None:
        if not sensors:
            raise ValueError("At least one sensor is required")

        sensor_names = [sensor.name for sensor in sensors]

        if len(sensor_names) != len(set(sensor_names)):
            raise ValueError("Sensor names must be unique")

        self.sensors = list(sensors)

    def run(
        self, 
        ground_truth: list[GroundTruthState],
    ) -> SimulationResult:
        measurements: list[Measurement] = []

        for state in ground_truth:
            for sensor in self.sensors:
                if sensor.is_update_time(state.timestamp):
                    measurement = sensor.measure(state)
                    measurements.append(measurement)

        return SimulationResult(
            ground_truth=list(ground_truth),
            measurements=measurements,
        )