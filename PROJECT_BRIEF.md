# Multi-Sensor Simulation, Validation & Fusion Framework

## 1. Project Objective

This project aims to develop an extensible Python framework for synthetic multi-sensor simulation, validation, and uncertainty-aware fusion in an ADAS-oriented scenario.

The framework will simulate heterogeneous LiDAR, radar, and camera measurements of a moving target vehicle. It will then transform, synchronize, validate, and fuse the measurements to estimate the target's two-dimensional position.

The project is designed to demonstrate transferable engineering skills for:

- Simulation Engineer roles
- Sensor Fusion Engineer roles
- ADAS Application Engineer roles
- Robotics and other simulation-related roles

The project emphasizes:

- 40% simulation
- 30% validation
- 30% sensor fusion

All data used by the project will be synthetically generated.

## 2. Scenario

### 2.1 Scenario Description

The V1 scenario contains:

- One stationary ego vehicle reference frame
- One moving target vehicle
- A two-dimensional environment
- Synthetic LiDAR, radar, and camera measurements
- A predefined target trajectory

The ego vehicle itself will remain stationary in V1. The target vehicle will move relative to the fixed ego coordinate frame.

The target trajectory will include:

- Longitudinal motion in the positive x direction
- A period of acceleration and/or deceleration
- A small lateral displacement in the y direction

This simplified scenario represents a moving target vehicle with changing longitudinal speed and mild lateral motion. It is intended to demonstrate sensor integration concepts without requiring a complete road, vehicle-dynamics, or traffic simulation model.

### 2.2 Coordinate System

The project will use a two-dimensional ego coordinate system:

- `x`: longitudinal direction, pointing forward from the ego vehicle
- `y`: lateral direction
- The origin: the ego vehicle reference point

All measurements must eventually be represented in the common ego/fusion coordinate frame before fusion.

### 2.3 Simulation Configuration

The baseline experiment will use:

- Simulation duration: 20 seconds
- Base simulation time step: `0.05 seconds`
- Base simulation frequency: 20 Hz
- Reproducible random seed
- One target vehicle

The trajectory model will be intentionally simple and interpretable. It will not represent a complete physical vehicle model.

## 3. Sensor Models

The framework will provide a common sensor interface and separate implementations for each sensor type.

### 3.1 LiDAR

LiDAR will generate a noisy Cartesian position measurement:

```text
(x, y)
```

Expected characteristics:

- Relatively small position uncertainty
- 10 Hz update rate
- Approximately 95% detection probability
- Occasional missed detections

### 3.2 Radar

Radar will generate a noisy polar measurement:

```text
(range, bearing)
```

The radar measurement must be converted to Cartesian coordinates before fusion.

Expected characteristics:

- Medium range and bearing uncertainty
- 20 Hz update rate
- Approximately 90% detection probability
- Occasional missed detections

### 3.3 Camera

The camera will generate a simplified synthetic measurement containing:

```text
(bearing, estimated_range, confidence)
```

The estimated range is a simplified synthetic depth estimate. It does not represent a complete monocular depth-estimation or image-processing system.

The camera measurement must be converted to Cartesian coordinates before fusion.

Expected characteristics:

- Larger bearing uncertainty than LiDAR
- Larger range uncertainty than LiDAR and radar
- 10 Hz update rate
- Approximately 85% detection probability
- A confidence score associated with each valid detection

### 3.4 Sensor Reliability Assumption

The baseline experiment will use the following reliability ordering:

```text
LiDAR: most reliable
Radar: medium reliability
Camera: lowest reliability
```

Noise parameters, detection probabilities, update rates, and random seeds should be configurable rather than hard-coded inside the sensor implementations.

## 4. Processing Pipeline

The V1 processing pipeline will contain the following stages:

1. Generate the target ground-truth trajectory.
2. Generate sensor-specific measurements.
3. Represent measurements with timestamps and metadata.
4. Convert measurements into the common ego/fusion frame.
5. Synchronize measurements using a configurable time tolerance.
6. Reject invalid or unavailable measurements.
7. Validate individual sensor measurements against ground truth.
8. Fuse the available sensor measurements.
9. Validate the fused estimate against ground truth.
10. Generate plots and an experiment summary.

## 5. Validation Goals and KPIs

The framework will evaluate both individual sensors and the fused output.

### 5.1 Position Accuracy

The following metrics will be calculated:

- Position RMSE
- Mean Absolute Error (MAE)
- Bias

Metrics should be available for:

- The x direction
- The y direction
- Overall two-dimensional position error

### 5.2 Sensor Availability

The framework will report:

- Detection rate
- Number of missed detections
- Number of valid measurements
- Number of rejected measurements, where applicable

### 5.3 Fusion Performance

The framework will compare:

- LiDAR RMSE
- Radar RMSE
- Camera RMSE
- Fused RMSE

The main fusion KPI is whether the fused estimate improves upon at least one individual sensor under the baseline configuration.

### 5.4 Robustness Experiments

V1 should include at least two robustness experiments:

1. Increase the missed-detection rate of one sensor and observe whether the fusion pipeline continues to operate.
2. Introduce a measurement delay or timestamp offset and measure its effect before and after synchronization.

## 6. Fusion Approach

V1 will use uncertainty-weighted position fusion.

The fusion process will:

1. Receive valid, synchronized Cartesian measurements.
2. Select the measurements available at the current fusion timestamp.
3. Assign each measurement a weight based on its uncertainty.
4. Calculate a fused two-dimensional position.
5. Return the fused position and relevant metadata.

The V1 fusion model will use the following simplifying assumptions:

- All measurements belong to the same target.
- Data association is known and is not part of V1.
- x and y measurement errors are treated as independent.
- Sensor uncertainty is represented using diagonal covariance.
- Lower uncertainty results in a higher fusion weight.
- The system must continue operating when one or more sensor measurements are missing.

The fusion design should use an abstraction that can later support alternative implementations, such as a Kalman Filter.

## 7. V1 Scope

V1 will include:

- Python-based object-oriented software structure
- Synthetic ground-truth trajectory generation
- A common sensor interface
- LiDAR, radar, and camera sensor models
- Configurable sensor noise
- Configurable detection probability
- Different sensor update frequencies
- Timestamped measurements
- Polar-to-Cartesian conversion
- Basic two-dimensional frame transformation
- Measurement synchronization
- Invalid and missing measurement handling
- Individual sensor validation
- Uncertainty-weighted position fusion
- Fused-result validation
- Reproducible experiments
- Plots for trajectories, errors, sensor availability, and RMSE comparison
- Unit tests for important mathematical and processing functions

## 8. Out of Scope for V1

The following features are intentionally excluded from V1:

- Multiple targets
- Complex data association
- Multi-object tracking
- A complete ego-vehicle motion model
- Full vehicle dynamics
- Camera image generation
- Deep-learning-based perception
- Real sensor datasets
- 3D point clouds
- ROS2 or Gazebo integration
- HIL or SIL execution
- Real-time deployment
- Graphical user interface
- Extended Kalman Filter or Unscented Kalman Filter implementation

Kalman Filtering, multi-target tracking, real-data integration, and robotics middleware integration may be considered future extensions after V1 is complete.

## 9. Software Design Principles

The implementation should follow these principles:

- Separate simulation, processing, validation, fusion, and visualization responsibilities.
- Use classes for sensor behavior and core system components.
- Use dataclasses for structured state and measurement data.
- Use type hints for public methods.
- Keep configuration separate from implementation logic.
- Avoid hidden global state.
- Make experiments reproducible through explicit configuration and random seeds.
- Ensure that a new sensor can be added without rewriting the complete experiment runner.
- Keep visualization code independent from simulation and fusion logic.

## 10. Success Criteria

V1 will be considered complete when:

1. A single documented command runs a complete synthetic experiment.
2. The system generates a reproducible ground-truth trajectory.
3. LiDAR, radar, and camera produce timestamped measurements.
4. At least two sensors use different measurement representations.
5. Measurements are converted into a common coordinate frame.
6. Measurements are synchronized before fusion.
7. The system handles missing sensor measurements without crashing.
8. Individual sensor RMSE, MAE, bias, and detection rate are reported.
9. A fused position estimate is generated and evaluated.
10. The fused result is compared with individual sensor results.
11. At least one robustness experiment demonstrates the effect of dropout or timing misalignment.
12. Unit tests cover coordinate conversion, metric calculation, and fusion behavior.
13. The README explains the architecture, assumptions, limitations, and example results.
14. All project data is clearly identified as synthetic.
15. Kalman Filtering and other advanced features are documented as future work rather than presented as completed functionality.

## 11. Portfolio Positioning

This project demonstrates a complete but focused engineering workflow for heterogeneous sensor systems:

```text
Synthetic Scenario
        -> Sensor Simulation
        -> Coordinate and Time Alignment
        -> Quantitative Validation
        -> Uncertainty-Aware Fusion
```

The framework is primarily ADAS-oriented, but its architecture is transferable to robotics, autonomous systems, mobility systems, and other simulation-related applications.
