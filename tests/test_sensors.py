import pytest

from multisensor.simulation.sensors import LidarSensor, SensorModel
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