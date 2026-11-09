import pytest

from multisensor.simulation.runner import SimulationRunner
from multisensor.simulation.sensors import CameraSensor, LidarSensor, RadarSensor
from multisensor.types import GroundTruthState


@pytest.fixture
def short_trajectory():
    return [
        GroundTruthState(timestamp=0.00, x=10.0, y=0.0, vx=2.0, vy=0.0),
        GroundTruthState(timestamp=0.05, x=10.1, y=0.0, vx=2.0, vy=0.0),
        GroundTruthState(timestamp=0.10, x=10.2, y=0.0, vx=2.0, vy=0.0),
    ]


def test_runner_samples_each_sensor_at_expected_times(short_trajectory):
    runner = SimulationRunner([
        LidarSensor(detection_probability=1.0),
        RadarSensor(detection_probability=1.0),
        CameraSensor(detection_probability=1.0),
    ])

    result = runner.run(short_trajectory)

    assert result.ground_truth == short_trajectory
    assert len(result.measurements) == 7

    expected_times = {
        "lidar": [0.00, 0.10],
        "radar": [0.00, 0.05, 0.10],
        "camera": [0.00, 0.10],
    }

    for sensor_name, timestamps in expected_times.items():
        actual_times = [
            measurement.timestamp
            for measurement in result.measurements
            if measurement.sensor_name == sensor_name
        ]
        assert actual_times == timestamps, sensor_name


def test_runner_keeps_missed_detections(short_trajectory):
    runner = SimulationRunner([
        CameraSensor(detection_probability=0.0),
    ])

    result = runner.run(short_trajectory)

    assert [m.timestamp for m in result.measurements] == [0.00, 0.10]

    for measurement in result.measurements:
        assert measurement.is_valid is False
        assert measurement.values == ()
        assert measurement.confidence is None


def test_runner_is_reproducible_with_fresh_seeded_sensors(short_trajectory):
    runner_a = SimulationRunner([
        LidarSensor(seed=1),
        RadarSensor(seed=2),
        CameraSensor(seed=3),
    ])
    runner_b = SimulationRunner([
        LidarSensor(seed=1),
        RadarSensor(seed=2),
        CameraSensor(seed=3),
    ])

    result_a = runner_a.run(short_trajectory)
    result_b = runner_b.run(short_trajectory)

    assert len(result_a.measurements) == 7
    assert result_a.measurements == result_b.measurements