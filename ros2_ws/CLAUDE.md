# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a ROS 2 (Jazzy) workspace for the CSE276A (Introduction to Robotics, UC San Diego) course, targeting the
Qualcomm **RubikPi** single-board computer (aarch64). The code under `src/rubikpi_ros2` originates from
`AutonomousVehicleLaboratory/rubikpi_ros2`; it was imported into this repo as plain files (no submodule/subtree
link back to that remote). The workspace root (`~/ros2_ws`) follows the standard colcon layout
(`src/`, `build/`, `install/`, `log/`) — only `src/` is version controlled.

## Build & test commands

```bash
# Build everything (run from the workspace root)
cd ~/ros2_ws && colcon build --symlink-install

# Build a single package
colcon build --packages-select robot_control
colcon build --packages-select robot_vision_camera   # note: package name, not the "robot_vision" directory
colcon build --packages-select apriltag_ros

# Source the workspace after every build, in every new shell
source ~/ros2_ws/install/setup.bash

# Run tests for a package and show results
colcon test --packages-select robot_control
colcon test-result --verbose
```

Running a node/launch file always requires `source /opt/ros/jazzy/setup.bash` and
`source ~/ros2_ws/install/setup.bash` first.

### Linting

- `robot_control` (ament_python) has `ament_copyright`, `ament_flake8`, `ament_pep257` tests under
  `robot_control/test/`, run via `colcon test`.
- `apriltag_ros` and `robot_vision` (ament_cmake) use `ament_lint_auto` and, for `apriltag_ros`,
  `clang-format` (config in `apriltag_ros/.clang-format`) and `cppcheck`.

## Architecture

Three packages under `src/rubikpi_ros2/`:

### `robot_control` (ament_python)

Drives the physical robot through a separate, non-ROS motor-controller board over USB serial
(`/dev/ttyUSB0` @ 115200), using newline-delimited JSON with a `"T"` type field:
- `T:1` — drive command, `{"L", "R"}` wheel speeds clipped to `[-0.5, 0.5]`
- `T:126` — IMU data request (sent by this node)
- `T:1002` — IMU reply from the board (`r/p/y` in degrees, `gx/gy/gz`, `ax/ay/az` in mg)

- `motor_control.py` owns the serial connection. It subscribes to `motor_commands`
  (`std_msgs/Float32MultiArray`, `[L, R]`), publishes `imu_data` (`sensor_msgs/Imu`, converting the
  board's Euler degrees to quaternion), and runs a **0.15s command-timeout watchdog** that force-stops
  the motors if no command arrives in time — this is a safety feature, don't remove it when touching
  the threading/command-handling logic.
- `keyboard_control.py` is a separate node: WASD/ESC teleop, publishing to `motor_commands` at 20Hz.
  `launch/robot_teleop_launch.py` starts both nodes together and shuts the whole launch down if
  *either* one exits.
- `teleop_example.py` is a plain Python reference script (no `rclpy`) kept for comparison — it is not
  registered in `setup.py` and is not a ROS node.
- `velocity_control.py` is the waypoint follower: it reads `x,y,theta` waypoints (metres, radians,
  relative to the start pose) and publishes `motor_commands`. There are no wheel encoders, so position
  is dead-reckoned from the commanded wheel speeds (heading optionally from `imu_data`); the
  `speed_per_unit`/`wheel_base` calibration in `config/velocity_control.yaml` determines accuracy.
  The turn-drive-turn control logic lives in the ROS-free `waypoint_controller.py` so it can be unit
  tested in simulation. `launch/waypoint_launch.py` starts it together with `motor_control`.
- `config/waypoints.txt` is a symlink to the repo-root `waypoints.txt`.
- **HW4 path planning** (ROS-free modules, unit-tested in `test/test_hw4_planning.py`):
  `config/hw4_world.yaml` is the single source of measured inputs (arena, obstacle boxes, robot
  radius, start/goal, landmark poses, planner settings) in the HW3 world frame (SW corner origin,
  x east, y north). `world_model.py` (clearance, C-obstacles) → `visibility_planner.py` (fast:
  visibility graph + time-cost A* + corner merge) / `gvd_planner.py` (safe: brushfire clearance
  grid + Voronoi + Dijkstra + RDP; also a clearance-weighted comparison planner) →
  `path_tools.py` (waypoints, metrics, time model). Waypoints with theta `nan` are pass-through
  (`WaypointController`, `pass_radius`). `velocity_control` takes `world_file` (landmarks + start
  pose for the EKF, optional `clearance_guard`) and `initial_pose`. `ros2 run robot_control hw4
  plan|sim|eval|list` (`hw4_tool.py`); `launch/hw4_launch.py mode:=fast|safe` plans at launch
  into `<workspace>/hw4_logs/<mode>_<date>/` with `config/hw4.yaml` node params.
- **HW4 extra credit, SLAM-mapped obstacle** (`test/test_hw4_slam.py`): `launch/hw4_slam_launch.py
  mode:=fast|safe [prior_landmarks:=none|walls]` with `config/hw4_world_slam.yaml` (its
  `obstacles` and box tags are ground truth for sim/eval only; the robot gets `slam.prior`) and
  `config/hw4_slam.yaml`. `velocity_control` with `slam_plan_mode` runs EKF-SLAM from the world
  start pose; `slam_obstacle.py` estimates the box from mapped box tags (intersection of per-face
  bounds and the prior, widened by k·sigma + margin) and `SlamPlanner` replans on it.
  `ros2 run robot_control hw4_slam sim|eval|prior|list` (`hw4_slam_tool.py`). Box tags must be
  36h11 (one detector): a second apriltag_node got almost no frames on the RubikPi.

### `robot_vision` (ament_cmake)

The package name is `robot_vision_camera` (it differs from the `robot_vision` directory name — use
`robot_vision_camera` in `ros2 run`/`ros2 launch`/`colcon build --packages-select` commands).

Single-node camera driver for the RubikPi's onboard camera, built on GStreamer using the Qualcomm
`qtiqmmfsrc` source element (`pipeline: qtiqmmfsrc ! capsfilter ! videoconvert ! videobalance !
videoconvert ! capsfilter ! appsink`) — this only works on the RubikPi hardware itself, not a generic
USB webcam. It publishes `sensor_msgs/Image` (raw), a JPEG-compressed variant, and `CameraInfo`
(loaded from `config/camera_parameter.yaml`, OpenCV calibration YAML format), with optional
undistortion via `image_rectify`. `launch/robot_vision_camera.launch.py` also starts `foxglove_bridge`
(port 8765) for remote visualization.

### `apriltag_ros` (ament_cmake)

Largely a vendored copy of the upstream `christianrauch/apriltag_ros` node (see its own
`apriltag_ros/README.md` for parameters and topic details). Provides AprilTag detection both as a
standalone `apriltag_node` executable and as an `rclcpp_components` composable node, consuming
rectified images + `CameraInfo` and publishing `/tf` and `apriltag_msgs/AprilTagDetectionArray`.
`launch/camera_36h11.launch.yml` wires it into one component container together with the *upstream*
`camera_ros` and `image_proc` nodes — this is a different, independent camera pipeline from this
project's own `robot_vision_camera` node; pick whichever matches the camera/hardware actually in use.

## Development environment

Developed via the `.devcontainer/` at the repo root (ROS2 Jazzy desktop image, noVNC on port 6080).
`robot_control`'s motor-controller connection and `robot_vision`'s camera driver only work on the
RubikPi hardware itself; from this devcontainer (e.g. on macOS), the expectation is to reach the
robot's own ROS2 nodes over the network via DDS discovery, not to run those hardware-bound nodes
locally.
