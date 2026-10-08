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
- **Known gap**: `setup.py` declares a `velocity_control` console script entry point, but
  `robot_control/velocity_control.py` does not exist — building/running that entry point will fail
  until it's added.

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
