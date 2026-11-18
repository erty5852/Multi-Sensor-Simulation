from math import sqrt

import pytest

from multisensor.types import GroundTruthState, Measurement
from multisensor.validation import evaluate_sensors


def test_evaluates_each_sensor_separately():
    ground_truth = [
        GroundTruthState(t, 10.0, 0.0, 0.0, 0.0)
        for t in (0.00, 0.10, 0.20)
    ]
    measurements = [
        Measurement(0.00, "lidar", "cartesian_position", (13.0, -4.0)),
        Measurement(0.10, "lidar", "cartesian_position", (10.0, 0.0)),
        Measurement(0.20, "lidar", "cartesian_position", (), is_valid=False),
        Measurement(0.10, "radar", "polar_position", (10.0, 0.0)),
    ]

    results = evaluate_sensors(ground_truth, measurements)

    lidar = results["lidar"]
    assert lidar.attempts == 3
    assert lidar.detections == 2
    assert lidar.detection_rate == pytest.approx(2 / 3)
    assert lidar.position_mae == pytest.approx(2.5)
    assert lidar.position_rmse == pytest.approx(5 / sqrt(2))
    assert lidar.bias_x == pytest.approx(1.5)
    assert lidar.bias_y == pytest.approx(-2.0)

    radar = results["radar"]
    assert radar.attempts == radar.detections == 1
    assert radar.detection_rate == 1.0
    assert radar.position_rmse == pytest.approx(0.0)


def test_all_misses_have_zero_detection_rate_but_no_position_errors():
    ground_truth = [GroundTruthState(0.0, 10.0, 0.0, 0.0, 0.0)]
    measurements = [
        Measurement(0.0, "camera", "bearing_and_range", (), is_valid=False),
    ]

    camera = evaluate_sensors(ground_truth, measurements)["camera"]

    assert camera.attempts == 1
    assert camera.detections == 0
    assert camera.detection_rate == 0.0
    assert camera.position_mae is None
    assert camera.position_rmse is None
    assert camera.bias_x is None
    assert camera.bias_y is None


def test_rejects_valid_measurement_without_matching_ground_truth():
    measurements = [
        Measurement(0.10, "lidar", "cartesian_position", (10.0, 0.0)),
    ]

    with pytest.raises(ValueError, match="No ground truth"):
        evaluate_sensors([], measurements)