from math import atan2

import pytest

from multisensor.simulation.sensors import (
    LidarSensor,
    RadarSensor,
    CameraSensor,
    SensorModel,
)
from multisensor.types import GroundTruthState, Measurement


class DummySensor(SensorModel):
    """Minimal concrete sensor used only for testing SensorModel."""

    def measure(self, state: GroundTruthState) -> Measurement:
        if not self._detection_succeeds():
            return Measurement(
                timestamp=state.timestamp,
                sensor_name=self.name,
                measurement_type="cartesian_position",
                values=(),
                is_valid=False,
            )

        return Measurement(
            timestamp=state.timestamp,
            sensor_name=self.name,
            measurement_type="cartesian_position",
            values=(state.x, state.y),
            is_valid=True,
        )


def create_ground_truth_state() -> GroundTruthState:
    return GroundTruthState(
        timestamp=0.1,
        x=31.0,
        y=0.2,
        vx=10.0,
        vy=0.4,
    )


def test_abstract_sensor_model_cannot_be_instantiated():
    with pytest.raises(TypeError):
        SensorModel(
            name="test_sensor",
            update_rate_hz=10.0,
            detection_probability=1.0,
            seed=42,
        )


def test_sensor_stores_configuration():
    sensor = DummySensor(
        name="dummy",
        update_rate_hz=10.0,
        detection_probability=0.9,
        seed=42,
    )

    assert sensor.name == "dummy"
    assert sensor.update_rate_hz == pytest.approx(10.0)
    assert sensor.detection_probability == pytest.approx(0.9)


def test_sampling_interval_is_calculated_from_update_rate():
    sensor = DummySensor(
        name="dummy",
        update_rate_hz=10.0,
        detection_probability=1.0,
    )

    assert sensor.sampling_interval == pytest.approx(0.1)


def test_sensor_identifies_update_timestamps():
    sensor = DummySensor(
        name="dummy",
        update_rate_hz=10.0,
        detection_probability=1.0,
    )

    assert sensor.is_update_time(0.0) is True
    assert sensor.is_update_time(0.05) is False
    assert sensor.is_update_time(0.1) is True
    assert sensor.is_update_time(0.2) is True


def test_sensor_rejects_invalid_configuration():
    with pytest.raises(ValueError):
        DummySensor(
            name="dummy",
            update_rate_hz=0.0,
            detection_probability=1.0,
        )

    with pytest.raises(ValueError):
        DummySensor(
            name="dummy",
            update_rate_hz=10.0,
            detection_probability=1.1,
        )


def test_sensor_with_full_detection_probability_returns_valid_measurement():
    sensor = DummySensor(
        name="dummy",
        update_rate_hz=10.0,
        detection_probability=1.0,
        seed=42,
    )

    measurement = sensor.measure(create_ground_truth_state())

    assert measurement.is_valid is True
    assert measurement.values == (31.0, 0.2)


def test_sensor_with_zero_detection_probability_returns_invalid_measurement():
    sensor = DummySensor(
        name="dummy",
        update_rate_hz=10.0,
        detection_probability=0.0,
        seed=42,
    )

    measurement = sensor.measure(create_ground_truth_state())

    assert measurement.is_valid is False
    assert measurement.values == ()


def test_same_seed_produces_same_detection_sequence():
    first_sensor = DummySensor(
        name="first",
        update_rate_hz=10.0,
        detection_probability=0.5,
        seed=42,
    )

    second_sensor = DummySensor(
        name="second",
        update_rate_hz=10.0,
        detection_probability=0.5,
        seed=42,
    )

    ground_truth = create_ground_truth_state()

    first_results = [
        first_sensor.measure(ground_truth).is_valid
        for _ in range(10)
    ]

    second_results = [
        second_sensor.measure(ground_truth).is_valid
        for _ in range(10)
    ]

    assert first_results == second_results


def test_lidar_uses_expected_default_configuration():
    lidar = LidarSensor(seed=42)

    assert lidar.name == "lidar"
    assert lidar.update_rate_hz == pytest.approx(10.0)
    assert lidar.detection_probability == pytest.approx(0.95)
    assert lidar.noise_std_x == pytest.approx(0.10)
    assert lidar.noise_std_y == pytest.approx(0.10)


def test_lidar_updates_at_ten_hertz():
    lidar = LidarSensor(seed=42)

    assert lidar.is_update_time(0.0) is True
    assert lidar.is_update_time(0.05) is False
    assert lidar.is_update_time(0.10) is True
    assert lidar.is_update_time(0.20) is True


def test_lidar_with_zero_noise_matches_ground_truth():
    lidar = LidarSensor(
        noise_std_x=0.0,
        noise_std_y=0.0,
        detection_probability=1.0,
        seed=42,
    )

    state = create_ground_truth_state()
    measurement = lidar.measure(state)

    assert measurement.timestamp == pytest.approx(state.timestamp)
    assert measurement.sensor_name == "lidar"
    assert measurement.measurement_type == "cartesian_position"
    assert measurement.values[0] == pytest.approx(state.x)
    assert measurement.values[1] == pytest.approx(state.y)
    assert measurement.is_valid is True
    assert measurement.confidence is None


def test_lidar_adds_position_noise():
    lidar = LidarSensor(
        noise_std_x=0.10,
        noise_std_y=0.10,
        detection_probability=1.0,
        seed=42,
    )

    state = create_ground_truth_state()
    measurement = lidar.measure(state)

    assert measurement.is_valid is True
    assert len(measurement.values) == 2
    assert measurement.values != (state.x, state.y)


def test_lidar_returns_invalid_measurement_after_missed_detection():
    lidar = LidarSensor(
        detection_probability=0.0,
        seed=42,
    )

    state = create_ground_truth_state()
    measurement = lidar.measure(state)

    assert measurement.timestamp == pytest.approx(state.timestamp)
    assert measurement.sensor_name == "lidar"
    assert measurement.measurement_type == "cartesian_position"
    assert measurement.values == ()
    assert measurement.is_valid is False


def test_lidar_measurements_are_reproducible_with_same_seed():
    first_lidar = LidarSensor(
        detection_probability=1.0,
        seed=42,
    )

    second_lidar = LidarSensor(
        detection_probability=1.0,
        seed=42,
    )

    state = create_ground_truth_state()

    first_measurements = [
        first_lidar.measure(state).values
        for _ in range(5)
    ]

    second_measurements = [
        second_lidar.measure(state).values
        for _ in range(5)
    ]

    assert first_measurements == second_measurements


def test_lidar_rejects_negative_noise_standard_deviation():
    with pytest.raises(ValueError):
        LidarSensor(noise_std_x=-0.10)

    with pytest.raises(ValueError):
        LidarSensor(noise_std_y=-0.10)


def test_radar_uses_expected_default_configuration():
    radar = RadarSensor(seed=42)

    assert radar.name == "radar"
    assert radar.update_rate_hz == pytest.approx(20.0)
    assert radar.detection_probability == pytest.approx(0.90)
    assert radar.range_noise_std == pytest.approx(0.30)
    assert radar.bearing_noise_std == pytest.approx(0.01)


def test_radar_updates_at_twenty_hertz():
    radar = RadarSensor(seed=42)

    assert radar.is_update_time(0.00) is True
    assert radar.is_update_time(0.025) is False
    assert radar.is_update_time(0.05) is True
    assert radar.is_update_time(0.10) is True


def test_radar_with_zero_noise_converts_cartesian_to_polar():
    radar = RadarSensor(
        range_noise_std=0.0,
        bearing_noise_std=0.0,
        detection_probability=1.0,
        seed=42,
    )

    state = GroundTruthState(
        timestamp=0.0,
        x=3.0,
        y=4.0,
        vx=0.0,
        vy=0.0,
    )

    measurement = radar.measure(state)

    measured_range, measured_bearing = measurement.values

    assert measurement.timestamp == pytest.approx(0.0)
    assert measurement.sensor_name == "radar"
    assert measurement.measurement_type == "polar_position"
    assert measurement.is_valid is True
    assert measurement.confidence is None

    assert measured_range == pytest.approx(5.0)
    assert measured_bearing == pytest.approx(atan2(4.0, 3.0))


def test_radar_adds_range_and_bearing_noise():
    radar = RadarSensor(
        range_noise_std=0.30,
        bearing_noise_std=0.01,
        detection_probability=1.0,
        seed=42,
    )

    state = GroundTruthState(
        timestamp=0.0,
        x=3.0,
        y=4.0,
        vx=0.0,
        vy=0.0,
    )

    measurement = radar.measure(state)

    true_values = (
        5.0,
        atan2(4.0, 3.0),
    )

    assert measurement.is_valid is True
    assert len(measurement.values) == 2
    assert measurement.values != true_values


def test_radar_returns_invalid_measurement_after_missed_detection():
    radar = RadarSensor(
        detection_probability=0.0,
        seed=42,
    )

    state = create_ground_truth_state()
    measurement = radar.measure(state)

    assert measurement.timestamp == pytest.approx(state.timestamp)
    assert measurement.sensor_name == "radar"
    assert measurement.measurement_type == "polar_position"
    assert measurement.values == ()
    assert measurement.is_valid is False


def test_radar_measurements_are_reproducible_with_same_seed():
    first_radar = RadarSensor(
        detection_probability=1.0,
        seed=42,
    )

    second_radar = RadarSensor(
        detection_probability=1.0,
        seed=42,
    )

    state = create_ground_truth_state()

    first_measurements = [
        first_radar.measure(state).values
        for _ in range(5)
    ]

    second_measurements = [
        second_radar.measure(state).values
        for _ in range(5)
    ]

    assert first_measurements == second_measurements


def test_radar_rejects_negative_noise_standard_deviation():
    with pytest.raises(ValueError):
        RadarSensor(range_noise_std=-0.30)

    with pytest.raises(ValueError):
        RadarSensor(bearing_noise_std=-0.01)


def test_camera_default_configuration():
    sensor = CameraSensor()

    assert sensor.name == "camera"
    assert sensor.update_rate_hz == 10.0
    assert sensor.detection_probability == 0.85
    assert sensor.bearing_noise_std == 0.02
    assert sensor.estimated_range_noise_std == 1.50
    assert sensor.confidence_min == 0.60
    assert sensor.confidence_max == 0.95
    assert sensor.sampling_interval == pytest.approx(0.1)


def test_camera_update_times():
    sensor = CameraSensor()

    assert sensor.is_update_time(0.0)
    assert sensor.is_update_time(0.1)
    assert sensor.is_update_time(0.2)

    assert not sensor.is_update_time(0.05)
    assert not sensor.is_update_time(0.15)


def test_camera_zero_noise_measurement():
    sensor = CameraSensor(
        bearing_noise_std=0.0,
        estimated_range_noise_std=0.0,
        confidence_min=0.80,
        confidence_max=0.80,
        detection_probability=1.0,
        seed=42,
    )

    state = GroundTruthState(
        timestamp=0.1,
        x=3.0,
        y=4.0,
        vx=0.0,
        vy=0.0,
    )

    measurement = sensor.measure(state)

    assert measurement.timestamp == state.timestamp
    assert measurement.sensor_name == "camera"
    assert measurement.measurement_type == "bearing_and_range"
    assert measurement.is_valid is True

    assert measurement.values == pytest.approx(
        (atan2(4.0, 3.0), 5.0)
    )
    assert measurement.confidence == pytest.approx(0.80)


def test_camera_missed_detection():
    sensor = CameraSensor(
        detection_probability=0.0,
        seed=42,
    )

    state = GroundTruthState(
        timestamp=0.1,
        x=3.0,
        y=4.0,
        vx=0.0,
        vy=0.0,
    )

    measurement = sensor.measure(state)

    assert measurement.timestamp == state.timestamp
    assert measurement.sensor_name == "camera"
    assert measurement.measurement_type == "bearing_and_range"
    assert measurement.is_valid is False
    assert measurement.values == ()
    assert measurement.confidence is None


def test_camera_confidence_within_configured_bounds():
    sensor = CameraSensor(
        confidence_min=0.65,
        confidence_max=0.90,
        detection_probability=1.0,
        seed=42,
    )

    state = GroundTruthState(
        timestamp=0.1,
        x=3.0,
        y=4.0,
        vx=0.0,
        vy=0.0,
    )

    for _ in range(100):
        measurement = sensor.measure(state)

        assert measurement.confidence is not None
        assert 0.65 <= measurement.confidence <= 0.90


def test_camera_estimated_range_is_non_negative():
    sensor = CameraSensor(
        estimated_range_noise_std=10.0,
        detection_probability=1.0,
        seed=42,
    )

    state = GroundTruthState(
        timestamp=0.0,
        x=0.0,
        y=0.0,
        vx=0.0,
        vy=0.0,
    )

    for _ in range(100):
        measurement = sensor.measure(state)

        estimated_range = measurement.values[1]

        assert estimated_range >= 0.0


def test_camera_same_seed_produces_same_measurements():
    sensor_a = CameraSensor(seed=42)
    sensor_b = CameraSensor(seed=42)

    state = GroundTruthState(
        timestamp=0.1,
        x=30.0,
        y=5.0,
        vx=2.0,
        vy=0.0,
    )

    measurements_a = [
        sensor_a.measure(state)
        for _ in range(20)
    ]

    measurements_b = [
        sensor_b.measure(state)
        for _ in range(20)
    ]

    assert measurements_a == measurements_b


@pytest.mark.parametrize(
    "kwargs",
    [
        {"bearing_noise_std": -0.01},
        {"estimated_range_noise_std": -1.0},
        {"confidence_min": -0.1},
        {"confidence_max": 1.1},
        {"confidence_min": 0.9, "confidence_max": 0.6},
    ],
)
def test_camera_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        CameraSensor(**kwargs)

