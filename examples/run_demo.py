from multisensor.evaluation import FusionEvaluation, evaluate_fusion
from multisensor.simulation.runner import SimulationResult, SimulationRunner
from multisensor.simulation.sensors import CameraSensor, LidarSensor, RadarSensor
from multisensor.types import GroundTruthState
from multisensor.validation import evaluate_sensors
from multisensor.simulation.trajectory import TrajectoryGenerator

from math import hypot
from pathlib import Path

import matplotlib.pyplot as plt

from multisensor.processing import group_positions_by_timestamp


# Assumed Cartesian position standard deviations in meters.
POSITION_STD_M = {"lidar": 0.10, "radar": 0.50, "camera": 1.50}


def build_ground_truth() -> list[GroundTruthState]:
    """Generate the standard 20-second simulation trajectory."""
    generator = TrajectoryGenerator(duration=20.0, dt=0.05)
    return generator.generate()


def format_rmse(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.3f} m"


def save_shared_error_plot(
    simulation: SimulationResult,
    report: FusionEvaluation,
    output_path: Path,
) -> None:
    """Plot position errors on timestamps shared by all configured sensors."""
    groups = group_positions_by_timestamp(simulation.measurements)
    truth_by_time = {state.timestamp: state for state in simulation.ground_truth}
    expected_sensors = set(report.sensor_rmse)

    timestamps: list[float] = []
    errors: dict[str, list[float]] = {
        name: [] for name in (*report.sensor_rmse, "fused")
    }

    for fused in report.fused_positions:
        if set(fused.sensor_names) != expected_sensors:
            continue

        t = fused.timestamp
        truth = truth_by_time[t]
        timestamps.append(t)

        for position in groups[t]:
            error = hypot(position.x - truth.x, position.y - truth.y)
            errors[position.sensor_name].append(error)

        errors["fused"].append(hypot(fused.x - truth.x, fused.y - truth.y))

    if not timestamps:
        print("\nNo shared timestamps; skipping the error plot.")
        return

    fig, (ax_all, ax_detail) = plt.subplots(
        2, 1, sharex=True, figsize=(10, 6), layout="constrained"
    )
    colors = {
        "lidar": "tab:blue",
        "radar": "tab:orange",
        "camera": "tab:green",
        "fused": "tab:red",
    }

    for name, values in errors.items():
        ax_all.scatter(timestamps, values, s=10, label=name, color=colors[name])

    for name in ("lidar", "fused"):
        ax_detail.scatter(
            timestamps, errors[name], s=10, label=name, color=colors[name]
        )

    ax_all.set_title(f"2D position error on shared timestamps (n={len(timestamps)})")
    ax_detail.set_title("Detail: LiDAR vs. fusion (different y-axis scale)")
    ax_detail.set_xlabel("Simulation time [s]")

    for ax in (ax_all, ax_detail):
        ax.set_ylabel("Position error [m]")
        ax.grid(alpha=0.3)
        ax.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved plot to {output_path}")


def main() -> None:
    runner = SimulationRunner([
        LidarSensor(seed=1),
        RadarSensor(seed=2),
        CameraSensor(seed=3),
    ])

    simulation = runner.run(build_ground_truth())
    sensor_metrics = evaluate_sensors(simulation.ground_truth, simulation.measurements)
    fusion_report = evaluate_fusion(simulation, POSITION_STD_M)

    print(f"Simulation ticks: {len(simulation.ground_truth)}")
    print("\nDetection rates (per sensor's scheduled attempts):")
    for name, metrics in sensor_metrics.items():
        print(
            f"  {name:6s} {metrics.detection_rate:6.1%}"
            f" ({metrics.detections}/{metrics.attempts})"
        )

    coverage = fusion_report.output_coverage
    print("\nFusion output coverage (all simulation ticks):")
    print("  N/A" if coverage is None else f"  {coverage:.1%}")

    print(f"\n2D RMSE on shared timestamps (n={fusion_report.common_sample_count}):")
    for name, rmse in fusion_report.sensor_rmse.items():
        print(f"  {name:6s} {format_rmse(rmse)}")
    print(f"  {'fused':6s} {format_rmse(fusion_report.fused_rmse)}")

    save_shared_error_plot(
        simulation,
        fusion_report,
        Path("docs/images/shared_position_error.png"),
    )



if __name__ == "__main__":
    main()