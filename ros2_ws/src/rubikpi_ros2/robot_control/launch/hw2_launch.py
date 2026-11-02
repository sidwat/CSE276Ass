from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, IncludeLaunchDescription, RegisterEventHandler, Shutdown)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    config_dir = PathJoinSubstitution([FindPackageShare('robot_control'), 'config'])

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
        camera,
        apriltag,
        motor_node,
        velocity_node,
        shutdown_when_velocity_exits,
        shutdown_when_motor_exits,
    ])
