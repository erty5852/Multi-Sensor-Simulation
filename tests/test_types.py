from multisensor.types import GroundTruthState


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