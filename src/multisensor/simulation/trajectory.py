from math import cos, sin

from multisensor.types import GroundTruthState


class TrajectoryGenerator:
    """Generate a synthetic ground-truth trajectory for one target vehicle."""

    def __init__(
        self,
        duration: float,
        dt: float,
        initial_x: float = 30.0,
        initial_vx: float = 10.0,
    ) -> None:
        if duration <= 0:
            raise ValueError("duration must be greater than zero")

        if dt <= 0:
            raise ValueError("dt must be greater than zero")

        if duration / dt != round(duration / dt):
            raise ValueError("duration must be divisible by dt")

        self.duration = duration
        self.dt = dt
        self.initial_x = initial_x
        self.initial_vx = initial_vx

    def generate(self) -> list[GroundTruthState]:
        """Generate and return the complete ground-truth trajectory."""

        trajectory: list[GroundTruthState] = []

        x = self.initial_x
        vx = self.initial_vx
        number_of_steps = int(round(self.duration / self.dt))

        for step in range(number_of_steps + 1):
            timestamp = step * self.dt

            y, vy = self._get_lateral_state(timestamp)

            state = GroundTruthState(
                timestamp=timestamp,
                x=x,
                y=y,
                vx=vx,
                vy=vy,
            )

            trajectory.append(state)

            if step == number_of_steps:
                break

            acceleration = self._get_longitudinal_acceleration(timestamp)

            x = x + vx * self.dt + 0.5 * acceleration * self.dt**2
            vx = vx + acceleration * self.dt

        return trajectory

    def _get_longitudinal_acceleration(self, timestamp: float) -> float:
        """Return the longitudinal acceleration for a given timestamp."""

        if timestamp < 5.0:
            return 0.0

        if timestamp < 10.0:
            return 1.0

        if timestamp < 15.0:
            return 0.0

        return -1.0

    def _get_lateral_state(self, timestamp: float) -> tuple[float, float]:
        """Return lateral position and velocity for a given timestamp."""

        y = 2.0 * sin(0.2 * timestamp)
        vy = 0.4 * cos(0.2 * timestamp)

        return y, vy