import os
import time

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, OpaqueFunction,
    RegisterEventHandler, Shutdown)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


# Every topic needed to replay or re-analyse a run; images only with record_images:=true
RECORDED_TOPICS = ['/tf', '/tf_static', '/detections', '/motor_commands', '/imu_data',
                   '/camera/camera_info', '/rosout']


def start_recording(context, run_name):
    """ros2 bag record into <log_dir>/<run_name>/bag, next to velocity_control's logs."""
    if LaunchConfiguration('record').perform(context).lower() != 'true':
        return []
    log_dir = os.path.expanduser(LaunchConfiguration('log_dir').perform(context))
    topics = list(RECORDED_TOPICS)
    if LaunchConfiguration('record_images').perform(context).lower() == 'true':
        topics.append('/camera/image_raw/compressed')
    output = os.path.join(log_dir, run_name, 'bag')
    return [ExecuteProcess(cmd=['ros2', 'bag', 'record', '-s', 'mcap', '-o', output, *topics],
                           output='log')]


def generate_launch_description():
    config_dir = PathJoinSubstitution([FindPackageShare('robot_control'), 'config'])
    run_name = time.strftime('run_%Y%m%d_%H%M%S')

    waypoints_file = DeclareLaunchArgument(
        'waypoints_file',
        default_value=PathJoinSubstitution([config_dir, 'waypoints_hw2.txt']),
        description='File of "x,y,theta" waypoints (metres, radians), relative to the start pose',
    )
    landmarks_file = DeclareLaunchArgument(
        'landmarks_file',
        default_value=PathJoinSubstitution([config_dir, 'landmarks.txt']),
        description='File of "id, x, y, yaw" AprilTag positions in the start frame',
    )
    log_dir = DeclareLaunchArgument(
        'log_dir', default_value='~/hw2_logs',
        description='Each run writes <log_dir>/run_<time>/ (CSVs, plots, config, bag)')
    record = DeclareLaunchArgument(
        'record', default_value='true', description='Record a ros2 bag of the run')
    record_images = DeclareLaunchArgument(
        'record_images', default_value='false',
        description='Also record the compressed camera images (about 1.5 MB/s)')
    params_file = DeclareLaunchArgument(
        'params_file',
        default_value=PathJoinSubstitution([config_dir, 'hw2.yaml']),
        description='Calibration, controller and EKF parameters for velocity_control',
    )

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

    velocity_node = Node(
        package='robot_control',
        executable='velocity_control',
        name='velocity_control_node',
        output='screen',
        emulate_tty=True,
        parameters=[
            LaunchConfiguration('params_file'),
            {
                'waypoints_file': LaunchConfiguration('waypoints_file'),
                'landmarks_file': LaunchConfiguration('landmarks_file'),
                'log_dir': LaunchConfiguration('log_dir'),
                'run_name': run_name,
            },
        ],
    )

    # Once the waypoint follower finishes (or dies), bring the whole system down.
    shutdown_when_velocity_exits = RegisterEventHandler(
        OnProcessExit(
            target_action=velocity_node,
            on_exit=[Shutdown(reason='Waypoint follower exited')]
        )
    )

    # If the motor controller exits for any reason, also bring everything down.
    shutdown_when_motor_exits = RegisterEventHandler(
        OnProcessExit(
            target_action=motor_node,
            on_exit=[Shutdown(reason='Motor controller exited')]
        )
    )

    return LaunchDescription([
        waypoints_file,
        landmarks_file,
        params_file,
        log_dir,
        record,
        record_images,
        OpaqueFunction(function=start_recording, args=[run_name]),
        camera,
        apriltag,
        motor_node,
        velocity_node,
        shutdown_when_velocity_exits,
        shutdown_when_motor_exits,
    ])
