from multisensor.types import GroundTruthState, Measurement


def test_ground_truth_state_stores_values():
    state = GroundTruthState(
        timestamp=0.0,
        x=30.0,
        y=0.0,
        vx=10.0,
        vy=0.0,
    )

    assert state.timestamp == 0.0
    assert state.x == 30.0
    assert state.y == 0.0
    assert state.vx == 10.0
    assert state.vy == 0.0


def test_lidar_measurement_stores_cartesian_position():
    measurement = Measurement(
        timestamp=0.0,
        sensor_name="lidar",
        measurement_type="cartesian_position",
        values=(30.0, 0.0),
    )

    assert measurement.sensor_name == "lidar"
    assert measurement.measurement_type == "cartesian_position"
    assert measurement.values == (30.0, 0.0)
    assert measurement.is_valid is True
    assert measurement.confidence is None


def test_radar_measurement_stores_polar_position():
    measurement = Measurement(
        timestamp=0.05,
        sensor_name="radar",
        measurement_type="polar_position",
        values=(30.5, 0.02),
    )

    assert measurement.sensor_name == "radar"
    assert measurement.values == (30.5, 0.02)


def test_camera_measurement_stores_confidence():
    measurement = Measurement(
        timestamp=0.1,
        sensor_name="camera",
        measurement_type="bearing_and_range",
        values=(0.03, 31.0),
        confidence=0.82,
    )

    assert measurement.sensor_name == "camera"
    assert measurement.values == (0.03, 31.0)
    assert measurement.confidence == 0.82