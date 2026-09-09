import pytest

from multisensor.simulation.trajectory import TrajectoryGenerator


def create_default_trajectory():
    generator = TrajectoryGenerator(
        duration=20.0,
        dt=0.05,
    )

    return generator.generate()


def test_trajectory_contains_expected_number_of_states():
    trajectory = create_default_trajectory()

    assert len(trajectory) == 401


def test_first_state_has_expected_values():
    trajectory = create_default_trajectory()
    first_state = trajectory[0]

    assert first_state.timestamp == pytest.approx(0.0)
    assert first_state.x == pytest.approx(30.0)
    assert first_state.y == pytest.approx(0.0)
    assert first_state.vx == pytest.approx(10.0)
    assert first_state.vy == pytest.approx(0.4)


def test_longitudinal_velocity_changes_as_expected():
    trajectory = create_default_trajectory()

    state_at_10_seconds = trajectory[200]
    final_state = trajectory[-1]

    assert state_at_10_seconds.vx == pytest.approx(15.0)
    assert final_state.vx == pytest.approx(10.0)


def test_target_moves_forward():
    trajectory = create_default_trajectory()

    first_state = trajectory[0]
    final_state = trajectory[-1]

    assert final_state.x > first_state.x


def test_lateral_position_stays_within_expected_range():
    trajectory = create_default_trajectory()

    lateral_positions = [state.y for state in trajectory]

    assert min(lateral_positions) >= -2.0
    assert max(lateral_positions) <= 2.0