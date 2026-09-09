import pytest

from multisensor.simulation.sensors import SensorModel
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