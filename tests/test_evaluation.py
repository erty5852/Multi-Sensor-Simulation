import pytest

from multisensor.evaluation import evaluate_fusion
from multisensor.simulation.runner import SimulationResult
from multisensor.types import GroundTruthState, Measurement


def test_fusion_rmse_uses_only_shared_timestamps():
    simulation = SimulationResult(
        ground_truth=[
            GroundTruthState(t, 10.0, 0.0, 0.0, 0.0)
            for t in (0.00, 0.05, 0.10)
        ],
        measurements=[
            Measurement(0.00, "lidar", "cartesian_position", (10.0, 0.0)),
            Measurement(0.00, "radar", "polar_position", (12.0, 0.0)),
            Measurement(0.00, "camera", "bearing_and_range", (0.0, 14.0)),
            Measurement(0.05, "radar", "polar_position", (100.0, 0.0)),
            Measurement(0.10, "radar", "polar_position", (), is_valid=False),
        ],
    )

    report = evaluate_fusion(
        simulation,
        {"lidar": 1.0, "radar": 1.0, "camera": 1.0},
    )

    assert [p.timestamp for p in report.fused_positions] == [0.00, 0.05]
    assert report.fused_positions[1].x == pytest.approx(100.0)
    assert report.output_coverage == pytest.approx(2 / 3)
    assert report.common_sample_count == 1
    assert report.sensor_rmse == pytest.approx({
        "lidar": 0.0,
        "radar": 2.0,
        "camera": 4.0,
    })
    assert report.fused_rmse == pytest.approx(2.0)


def test_fusion_has_zero_coverage_when_all_sensors_miss():
    simulation = SimulationResult(
        ground_truth=[
            GroundTruthState(0.0, 30.0, 0.0, 10.0, 0.4),
        ],
        measurements=[
            Measurement(0.0, "lidar", "cartesian_position", (), is_valid=False),
            Measurement(0.0, "radar", "polar_position", (), is_valid=False),
            Measurement(0.0, "camera", "bearing_and_range", (), is_valid=False),
        ],
    )

    report = evaluate_fusion(
        simulation,
        {"lidar": 1.0, "radar": 1.0, "camera": 1.0},
    )

    assert report.fused_positions == []
    assert report.output_coverage == 0.0
    assert report.common_sample_count == 0
    assert report.fused_rmse is None
    assert report.sensor_rmse == {
        "lidar": None,
        "radar": None,
        "camera": None,
    }