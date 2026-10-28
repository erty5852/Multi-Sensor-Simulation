from abc import ABC, abstractmethod
from math import atan2, hypot, isclose
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


class RadarSensor(SensorModel):
    """Synthetic object-level radar range and bearing sensor."""

    def __init__(
        self,
        range_noise_std: float = 0.30,
        bearing_noise_std: float = 0.01,
        update_rate_hz: float = 20.0,
        detection_probability: float = 0.90,
        seed: int | None = None,
    ) -> None:
        if range_noise_std < 0:
            raise ValueError("range_noise_std must not be negative")

        if bearing_noise_std < 0:
            raise ValueError("bearing_noise_std must not be negative")

        super().__init__(
            name="radar",
            update_rate_hz=update_rate_hz,
            detection_probability=detection_probability,
            seed=seed,
        )

        self.range_noise_std = range_noise_std
        self.bearing_noise_std = bearing_noise_std

    def measure(self, state: GroundTruthState) -> Measurement:
        """Generate a noisy polar range and bearing measurement."""

        if not self._detection_succeeds():
            return Measurement(
                timestamp=state.timestamp,
                sensor_name=self.name,
                measurement_type="polar_position",
                values=(),
                is_valid=False,
            )

        true_range = hypot(state.x, state.y)
        true_bearing = atan2(state.y, state.x)

        measured_range = true_range + self._random_generator.gauss(
            0.0,
            self.range_noise_std,
        )

        measured_bearing = true_bearing + self._random_generator.gauss(
            0.0,
            self.bearing_noise_std,
        )

        measured_range = max(0.0, measured_range)

        return Measurement(
            timestamp=state.timestamp,
            sensor_name=self.name,
            measurement_type="polar_position",
            values=(measured_range, measured_bearing),
            is_valid=True,
        )


class CameraSensor(SensorModel):
    """Simulate a camera that measures bearing and estimated range."""

    def __init__(
        self,
        bearing_noise_std: float = 0.02,
        estimated_range_noise_std: float = 1.50,
        confidence_min: float = 0.60,
        confidence_max: float = 0.95,
        update_rate_hz: float = 10.0,
        detection_probability: float = 0.85,
        seed: int | None = None,
    ) -> None:
        if bearing_noise_std < 0:
            raise ValueError("bearing_noise_std must be non-negative")

        if estimated_range_noise_std < 0:
            raise ValueError(
                "estimated_range_noise_std must be non-negative"
            )

        if not 0.0 <= confidence_min <= confidence_max <= 1.0:
            raise ValueError(
                "confidence values must satisfy "
                "0 <= confidence_min <= confidence_max <= 1"
            )

        super().__init__(
            name="camera",
            update_rate_hz=update_rate_hz,
            detection_probability=detection_probability,
            seed=seed,
        )

        self.bearing_noise_std = bearing_noise_std
        self.estimated_range_noise_std = estimated_range_noise_std
        self.confidence_min = confidence_min
        self.confidence_max = confidence_max

    def measure(self, state: GroundTruthState) -> Measurement:
        """Generate a camera measurement from a ground-truth state."""

        if not self._detection_succeeds():
            return Measurement(
                timestamp=state.timestamp,
                sensor_name=self.name,
                measurement_type="bearing_and_range",
                values=(),
                is_valid=False,
                confidence=None,
            )

        true_range = hypot(state.x, state.y)
        true_bearing = atan2(state.y, state.x)

        measured_bearing = true_bearing + self._random_generator.gauss(
            0.0,
            self.bearing_noise_std,
        )

        estimated_range = true_range + self._random_generator.gauss(
            0.0,
            self.estimated_range_noise_std,
        )

        estimated_range = max(0.0, estimated_range)

        confidence = self._random_generator.uniform(
            self.confidence_min,
            self.confidence_max,
        )

        return Measurement(
            timestamp=state.timestamp,
            sensor_name=self.name,
            measurement_type="bearing_and_range",
            values=(measured_bearing, estimated_range),
            is_valid=True,
            confidence=confidence,
        )