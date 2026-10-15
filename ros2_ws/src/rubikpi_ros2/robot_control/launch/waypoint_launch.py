from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler, Shutdown
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    config_dir = PathJoinSubstitution([FindPackageShare('robot_control'), 'config'])

    waypoints_file = DeclareLaunchArgument(
        'waypoints_file',
        default_value=PathJoinSubstitution([config_dir, 'waypoints.txt']),
        description='File of "x,y,theta" waypoints (metres, radians), relative to the start pose',
    )
    params_file = DeclareLaunchArgument(
        'params_file',
        default_value=PathJoinSubstitution([config_dir, 'velocity_control.yaml']),
        description='Calibration and controller parameters for velocity_control',
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
            {'waypoints_file': LaunchConfiguration('waypoints_file')},
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
        params_file,
        motor_node,
        velocity_node,
        shutdown_when_velocity_exits,
        shutdown_when_motor_exits,
    ])
