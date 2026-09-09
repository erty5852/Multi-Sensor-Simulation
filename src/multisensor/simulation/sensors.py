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