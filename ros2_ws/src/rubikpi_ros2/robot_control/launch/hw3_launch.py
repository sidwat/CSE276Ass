"""
HW3 EKF-SLAM run: camera + apriltag_node + motor_control + velocity_control (use_slam).

  ros2 launch robot_control hw3_launch.py path:=square            # 3 laps of the 4 ft square
  ros2 launch robot_control hw3_launch.py path:=octagon record_bag:=true
  ros2 launch robot_control hw3_launch.py path:=square landmark_dof:=2

Every run writes ~/hw3_logs/run_<time>/ (see robot_control/slam_logger.py); record_bag also
records the raw topics to ~/hw3_logs/bags/ so a run can be replayed later, and record_images adds
the compressed camera stream to that bag (large: ~1 MB/s).
"""

import os
import time

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, OpaqueFunction,
    RegisterEventHandler, Shutdown)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

# path name -> (waypoint file, waypoints per lap)
PATHS = {
    'square': ('waypoints_hw3_square.txt', 4),
    'octagon': ('waypoints_hw3_octagon.txt', 8),
}

BAG_TOPICS = ['/tf', '/detections', '/motor_commands', '/imu_data', '/camera/camera_info',
              '/slam/pose', '/slam/path', '/slam/landmarks', '/rosout']


def launch_setup(context):
    def arg(name):
        return context.launch_configurations[name]

    from ament_index_python.packages import get_package_share_directory
    config_dir = os.path.join(get_package_share_directory('robot_control'), 'config')
    path = arg('path')
    if path not in PATHS and not arg('waypoints_file'):
        raise RuntimeError(f'path must be one of {sorted(PATHS)} (or give waypoints_file)')
    wp_name, per_lap = PATHS.get(path, ('', 0))
    waypoints_file = arg('waypoints_file') or os.path.join(config_dir, wp_name)
    per_lap = int(arg('waypoints_per_lap') or per_lap)
    params_file = arg('params_file') or os.path.join(config_dir, 'hw3.yaml')

    overrides = {
        'waypoints_file': waypoints_file,
        'waypoints_per_lap': per_lap,
        'path_name': path,
    }
    if arg('landmark_dof'):
        overrides['landmark_dof'] = int(arg('landmark_dof'))
    if arg('use_imu'):
        overrides['use_imu'] = arg('use_imu').lower() == 'true'

    velocity_node = Node(
        package='robot_control',
        executable='velocity_control',
        name='velocity_control_node',
        output='screen',
        emulate_tty=True,
        parameters=[params_file, overrides],
    )
    actions = [
        velocity_node,
        # Once the waypoint follower finishes (or dies), bring the whole system down.
        RegisterEventHandler(OnProcessExit(
            target_action=velocity_node,
            on_exit=[Shutdown(reason='Waypoint follower exited')])),
    ]

    if arg('record_bag').lower() == 'true':
        topics = list(BAG_TOPICS)
        if arg('record_images').lower() == 'true':
            topics.append('/camera/image_raw/compressed')
        bag_dir = os.path.join(os.path.expanduser(arg('bag_dir')),
                               time.strftime(f'hw3_{path}_%Y%m%d_%H%M%S'))
        os.makedirs(os.path.dirname(bag_dir), exist_ok=True)
        actions.append(ExecuteProcess(
            cmd=['ros2', 'bag', 'record', '-o', bag_dir] + topics, output='screen'))
    return actions


def generate_launch_description():
    args = [
        DeclareLaunchArgument('path', default_value='square',
                              description='square (n = 4 ft) or octagon (m = 2 ft), 3 laps each'),
        DeclareLaunchArgument('waypoints_file', default_value='',
                              description='override the waypoint file of the path'),
        DeclareLaunchArgument('waypoints_per_lap', default_value='',
                              description='override the lap length (map snapshot per lap)'),
        DeclareLaunchArgument('params_file', default_value='',
                              description='parameters (default: config/hw3.yaml)'),
        DeclareLaunchArgument('landmark_dof', default_value='',
                              description='2 or 3 (default: from the params file)'),
        DeclareLaunchArgument('use_imu', default_value='',
                              description='true/false (default: from the params file)'),
        DeclareLaunchArgument('record_bag', default_value='false',
                              description='also record the raw topics with ros2 bag'),
        DeclareLaunchArgument('record_images', default_value='false',
                              description='add the compressed camera stream to the bag'),
        DeclareLaunchArgument('bag_dir', default_value='~/hw3_logs/bags'),
    ]

    # Camera (rectified images + camera_info, and foxglove_bridge) and the tag detector
    camera = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution(
            [FindPackageShare('robot_vision_camera'), 'launch', 'robot_vision_camera.launch.py'])),
        launch_arguments={'image_rectify': 'true'}.items(),
    )
    apriltag = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution(
            [FindPackageShare('apriltag_ros'), 'launch', 'apriltag_launch.py'])),
    )
    motor_node = Node(
        package='robot_control',
        executable='motor_control',
        name='motor_controller_node',
        output='screen',
        emulate_tty=True,
    )
    # If the motor controller exits for any reason, also bring everything down.
    shutdown_when_motor_exits = RegisterEventHandler(OnProcessExit(
        target_action=motor_node, on_exit=[Shutdown(reason='Motor controller exited')]))

    return LaunchDescription(args + [
        camera,
        apriltag,
        motor_node,
        shutdown_when_motor_exits,
        OpaqueFunction(function=launch_setup),
    ])
