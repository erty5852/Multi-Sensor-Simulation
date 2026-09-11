from abc import ABC, abstractmethod
from math import isclose
from random import Random

from multisensor.types import GroundTruthState, Measurement


class SensorModel(ABC):
    """Abstract base class for all synthetic sensor models."""

    def __init__(
        self,
        name: str,
        update_rate_hz: float,
        detection_probability: float,
        seed: int | None = None,
    ) -> None:
        if not name.strip():
            raise ValueError("sensor name must not be empty")

        if update_rate_hz <= 0:
            raise ValueError("update_rate_hz must be greater than zero")

        if not 0.0 <= detection_probability <= 1.0:
            raise ValueError(
                "detection_probability must be between 0.0 and 1.0"
            )

        self.name = name
        self.update_rate_hz = update_rate_hz
        self.detection_probability = detection_probability
        self._random_generator = Random(seed)

    @property
    def sampling_interval(self) -> float:
        """Return the time interval between two sensor updates."""

        return 1.0 / self.update_rate_hz

    def is_update_time(self, timestamp: float) -> bool:
        """Return whether the sensor should produce data at this timestamp."""

        if timestamp < 0:
            return False

        number_of_cycles = timestamp * self.update_rate_hz

        return isclose(
            number_of_cycles,
            round(number_of_cycles),
            rel_tol=0.0,
            abs_tol=1e-9,
        )

    def _detection_succeeds(self) -> bool:
        """Randomly determine whether the sensor detects the target."""

        random_value = self._random_generator.random()

        return random_value < self.detection_probability

    @abstractmethod
    def measure(self, state: GroundTruthState) -> Measurement:
        """Generate a sensor measurement from a ground-truth state."""

        raise NotImplementedError


class LidarSensor(SensorModel):
    """Synthetic object-level LiDAR position sensor."""

    def __init__(
        self,
        noise_std_x: float = 0.10,
        noise_std_y: float = 0.10,
        update_rate_hz: float = 10.0,
        detection_probability: float = 0.95,
        seed: int | None = None,
    ) -> None:
        if noise_std_x < 0:
            raise ValueError("noise_std_x must not be negative")

        if noise_std_y < 0:
            raise ValueError("noise_std_y must not be negative")

        super().__init__(
            name="lidar",
            update_rate_hz=update_rate_hz,
            detection_probability=detection_probability,
            seed=seed,
        )

        self.noise_std_x = noise_std_x
        self.noise_std_y = noise_std_y

    def measure(self, state: GroundTruthState) -> Measurement:
        """Generate a noisy Cartesian position measurement."""

        if not self._detection_succeeds():
            return Measurement(
                timestamp=state.timestamp,
                sensor_name=self.name,
                measurement_type="cartesian_position",
                values=(),
                is_valid=False,
            )

        measured_x = state.x + self._random_generator.gauss(
            0.0,
            self.noise_std_x,
        )

        measured_y = state.y + self._random_generator.gauss(
            0.0,
            self.noise_std_y,
        )

        return Measurement(
            timestamp=state.timestamp,
            sensor_name=self.name,
            measurement_type="cartesian_position",
            values=(measured_x, measured_y),
            is_valid=True,
        )