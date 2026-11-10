from math import atan2

import pytest

from multisensor.processing import group_positions_by_timestamp, to_cartesian
from multisensor.types import Measurement


@pytest.mark.parametrize(
    ("sensor_name", "measurement_type", "values", "confidence"),
    [
        ("lidar", "cartesian_position", (3.0, 4.0), None),
        ("radar", "polar_position", (5.0, atan2(4.0, 3.0)), None),
        ("camera", "bearing_and_range", (atan2(4.0, 3.0), 5.0), 0.8),
    ],
)
def test_to_cartesian_converts_each_sensor(
    sensor_name, measurement_type, values, confidence
):
    measurement = Measurement(
        timestamp=0.10,
        sensor_name=sensor_name,
        measurement_type=measurement_type,
        values=values,
        is_valid=True,
        confidence=confidence,
    )

    position = to_cartesian(measurement)

    assert position is not None
    assert (position.x, position.y) == pytest.approx((3.0, 4.0))
    assert position.timestamp == 0.10
    assert position.sensor_name == sensor_name
    assert position.confidence == confidence


def test_to_cartesian_returns_none_for_missed_detection():
    measurement = Measurement(
        timestamp=0.10,
        sensor_name="camera",
        measurement_type="bearing_and_range",
        values=(),
        is_valid=False,
    )

    assert to_cartesian(measurement) is None


def test_to_cartesian_rejects_unsupported_measurement_type():
    measurement = Measurement(
        timestamp=0.10,
        sensor_name="unknown",
        measurement_type="unsupported",
        values=(3.0, 4.0),
        is_valid=True,
    )

    with pytest.raises(ValueError, match="Unsupported measurement type"):
        to_cartesian(measurement)


def test_groups_valid_positions_by_timestamp():
    measurements = [
        Measurement(0.05, "radar", "polar_position", (5.0, 0.0)),
        Measurement(0.00, "lidar", "cartesian_position", (4.0, 0.0)),
        Measurement(0.00, "camera", "bearing_and_range", (0.0, 4.0)),
        Measurement(0.00, "radar", "polar_position", (), is_valid=False),
    ]

    groups = group_positions_by_timestamp(measurements)

    assert list(groups) == [0.00, 0.05]
    assert [p.sensor_name for p in groups[0.00]] == ["lidar", "camera"]
    assert [p.sensor_name for p in groups[0.05]] == ["radar"]


def test_all_missed_detections_produce_no_position_group():
    measurements = [
        Measurement(0.10, "camera", "bearing_and_range", (), is_valid=False),
    ]

    assert group_positions_by_timestamp(measurements) == {}
    assert len(measurements) == 1
    assert measurements[0].is_valid is False