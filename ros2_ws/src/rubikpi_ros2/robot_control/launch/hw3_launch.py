"""
HW3 EKF-SLAM run: camera + apriltag_node + motor_control + velocity_control (use_slam).

  ros2 launch robot_control hw3_launch.py path:=square            # 3 laps of the 4 ft square
  ros2 launch robot_control hw3_launch.py path:=octagon record_bag:=true
  ros2 launch robot_control hw3_launch.py path:=square landmark_dof:=2 start:=0.92,0.90,0.0

Every run is one folder in <workspace>/hw3_logs (robot_control/hw3_paths.py), named
<path>_<dof>dof_<date>_<time>, e.g. square_3dof_2026-11-12_15-04-31. It holds the SLAM logs
(robot_control/slam_logger.py), the bag in bag/ with record_bag:=true (record_images adds the
compressed camera stream: ~1 MB/s), and eval/ with the maps: when the run ends a post-run hook
runs `hw3 eval` on it against gt_file, using start (default: the path's nominal start pose).
"""

import os

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

    from robot_control import hw3_paths
    log_dir = os.path.expanduser(arg('log_dir')) if arg('log_dir') else hw3_paths.logs_dir()
    dof = int(arg('landmark_dof')) if arg('landmark_dof') else _yaml_dof(params_file)
    name = hw3_paths.run_name(path, dof, arg('run_tag'))
    run_dir = os.path.join(log_dir, name)
    os.makedirs(run_dir, exist_ok=True)
    start = arg('start') or ','.join(f'{v:.4f}' for v in hw3_paths.NOMINAL_START.get(path, ()))
    gt_file = arg('gt_file') or hw3_paths.default_gt_file()

    overrides = {
        'waypoints_file': waypoints_file,
        'waypoints_per_lap': per_lap,
        'path_name': path,
        'log_dir': log_dir,
        'run_name': name,
        'start_pose': start,
        'post_run_eval': arg('auto_eval').lower() == 'true',
        'gt_file': gt_file,
        'landmark_dof': dof,
    }
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
        actions.append(ExecuteProcess(
            cmd=['ros2', 'bag', 'record', '-o', os.path.join(run_dir, 'bag')] + topics,
            output='screen'))
    print(f'[hw3] run folder: {run_dir}\n[hw3] start pose (world): {start}  ground truth: '
          f'{gt_file}' + ('  (STILL THE TEMPLATE: fill in measured tag positions)'
                          if hw3_paths.gt_is_template(gt_file) else ''), flush=True)
    return actions


def _yaml_dof(params_file):
    import yaml
    with open(params_file) as f:
        data = yaml.safe_load(f)
    return int(next(iter(data.values()))['ros__parameters'].get('landmark_dof', 3))


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
        DeclareLaunchArgument('log_dir', default_value='',
                              description='runs folder (default: <workspace>/hw3_logs)'),
        DeclareLaunchArgument('run_tag', default_value='',
                              description='optional suffix for the run folder name, e.g. try2'),
        DeclareLaunchArgument('start', default_value='',
                              description='measured world start pose "x,y,theta" '
                                          '(default: the path\'s nominal start)'),
        DeclareLaunchArgument('gt_file', default_value='',
                              description='ground-truth tags (default: config/'
                                          'landmarks_hw3_gt.txt)'),
        DeclareLaunchArgument('auto_eval', default_value='true',
                              description='post-run hook: evaluate the run when it ends'),
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
