import pytest

from multisensor.fusion import WeightedPositionFusion
from multisensor.processing import PositionMeasurement


def test_fusion_uses_inverse_variance_weights():
    fusion = WeightedPositionFusion({
        "lidar": 1.0,
        "radar": 2.0,
    })

    positions = [
        PositionMeasurement(0.1, "lidar", 10.0, 4.0),
        PositionMeasurement(0.1, "radar", 12.0, 8.0),
    ]

    result = fusion.fuse(positions)

    assert result is not None
    assert result.timestamp == 0.1
    assert result.x == pytest.approx(10.4)
    assert result.y == pytest.approx(4.8)
    assert result.sensor_names == ("lidar", "radar")


def test_single_position_is_preserved():
    fusion = WeightedPositionFusion({"radar": 2.0})

    result = fusion.fuse([
        PositionMeasurement(0.05, "radar", 12.0, 8.0),
    ])

    assert result is not None
    assert result.timestamp == 0.05
    assert result.x == pytest.approx(12.0)
    assert result.y == pytest.approx(8.0)
    assert result.sensor_names == ("radar",)


def test_empty_input_returns_none():
    fusion = WeightedPositionFusion({"lidar": 1.0})

    assert fusion.fuse([]) is None


def test_rejects_different_timestamps():
    fusion = WeightedPositionFusion({"lidar": 1.0, "radar": 2.0})

    positions = [
        PositionMeasurement(0.0, "lidar", 10.0, 0.0),
        PositionMeasurement(0.1, "radar", 12.0, 0.0),
    ]

    with pytest.raises(ValueError, match="same timestamp"):
        fusion.fuse(positions)


def test_rejects_duplicate_sensor():
    fusion = WeightedPositionFusion({"lidar": 1.0})

    positions = [
        PositionMeasurement(0.1, "lidar", 10.0, 0.0),
        PositionMeasurement(0.1, "lidar", 11.0, 0.0),
    ]

    with pytest.raises(ValueError, match="Duplicate sensor"):
        fusion.fuse(positions)


def test_rejects_missing_uncertainty():
    fusion = WeightedPositionFusion({"lidar": 1.0})

    positions = [
        PositionMeasurement(0.1, "camera", 10.0, 0.0),
    ]

    with pytest.raises(ValueError, match="No position uncertainty"):
        fusion.fuse(positions)


@pytest.mark.parametrize("std", [0.0, -1.0, float("inf"), float("nan")])
def test_rejects_invalid_uncertainty(std):
    with pytest.raises(ValueError, match="finite and positive"):
        WeightedPositionFusion({"lidar": std})


def test_rejects_empty_configuration():
    with pytest.raises(ValueError, match="At least one sensor"):
        WeightedPositionFusion({})