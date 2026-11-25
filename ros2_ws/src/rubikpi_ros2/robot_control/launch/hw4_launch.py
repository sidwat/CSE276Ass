"""
HW4 path-planning run: camera + apriltag_node + motor_control + velocity_control (known map).

  ros2 launch robot_control hw4_launch.py mode:=fast
  ros2 launch robot_control hw4_launch.py mode:=safe record_bag:=true run_tag:=try2
  ros2 launch robot_control hw4_launch.py mode:=fast world:=/path/to/hw4_world.yaml

At launch the plan is computed from the world file (deterministic, under a second) and written
into a new run folder <workspace>/hw4_logs/<mode>_<date>_<time>[_<tag>]/ together with the tag
map, so the plan, the map and the logs of a run always belong together. velocity_control then
starts at the world start pose and logs into the same folder; when it finishes a post-run hook
runs `hw4 eval` on it (eval/ in the run folder). Afterwards, tape-measure where the robot
stopped and run:  ros2 run robot_control hw4 eval latest --end X Y [THETA]
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

BAG_TOPICS = ['/tf', '/detections', '/motor_commands', '/imu_data', '/camera/camera_info',
              '/rosout']


def launch_setup(context):
    def arg(name):
        return context.launch_configurations[name]

    import json
    import time

    from robot_control import hw4_plan
    from robot_control.hw4_world import load_world

    mode = arg('mode')
    if mode not in hw4_plan.MODES:
        raise RuntimeError(f'mode must be one of {hw4_plan.MODES}')
    world_file = os.path.expanduser(arg('world') or hw4_plan.default_world_file())
    params_file = os.path.expanduser(arg('params_file') or hw4_plan.default_params_file())
    world = load_world(world_file)
    for problem in world.check():
        print(f'[hw4] WARNING: {problem}', flush=True)

    log_dir = os.path.expanduser(arg('log_dir')) if arg('log_dir') else hw4_plan.logs_dir()
    name = hw4_plan.run_name(mode, arg('run_tag'))
    run_dir = os.path.join(log_dir, name)
    plan = hw4_plan.make_plan(world, mode)
    files = hw4_plan.write_plan(plan, world, run_dir)
    waypoints_file = files['waypoints']
    if arg('waypoints_file'):  # hand-made override (the plan is still saved for reference)
        waypoints_file = os.path.expanduser(arg('waypoints_file'))
    with open(os.path.join(run_dir, 'run_info.json'), 'w') as f:
        json.dump({'mode': mode, 'simulated': False, 'started': time.strftime('%Y-%m-%d %H:%M:%S'),
                   'world_file': world_file, 'params_file': params_file,
                   'waypoints_file': waypoints_file}, f, indent=2)
    with open(os.path.join(run_dir, 'end_pose.txt'), 'w') as f:
        f.write('# Tape-measured final pose of the robot centre (world frame): x, y[, theta]\n'
                '# Fill in after the run, then: ros2 run robot_control hw4 eval '
                f'{name}\n')

    start = ','.join(f'{v:.4f}' for v in world.start)
    overrides = {
        'waypoints_file': waypoints_file,
        'landmarks_file': files['landmarks'],
        'start_pose': start,
        'world_file': files.get('world', world_file),
        'log_dir': log_dir,
        'run_name': name,
        'path_name': mode,
        'post_run_eval': arg('auto_eval').lower() == 'true',
        'post_run_tool': 'hw4_tool',
        'use_slam': False,
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

    m = plan.metrics
    print(f'[hw4] mode {mode}: {hw4_plan.DESCRIPTIONS[mode]}\n'
          f'[hw4] run folder: {run_dir}\n'
          f'[hw4] start {start}  goal {world.goal}\n'
          f'[hw4] plan: {m["length_m"]:.2f} m, {m["waypoints"]} waypoints ({m["stops"]} stops), '
          f'min body gap {m["min_body_gap_m"]:.3f} m, estimated {m["est_time_s"]:.1f} s',
          flush=True)
    return actions


def generate_launch_description():
    args = [
        DeclareLaunchArgument('mode', default_value='fast',
                              description='fast (minimum time) | safe (maximum clearance) | '
                                          'weighted (trade-off)'),
        DeclareLaunchArgument('world', default_value='',
                              description='world file (default: config/hw4_world.yaml)'),
        DeclareLaunchArgument('params_file', default_value='',
                              description='node parameters (default: config/hw4.yaml)'),
        DeclareLaunchArgument('waypoints_file', default_value='',
                              description='override the planned waypoints (world frame)'),
        DeclareLaunchArgument('use_imu', default_value='',
                              description='true/false (default: from the params file)'),
        DeclareLaunchArgument('record_bag', default_value='false',
                              description='also record the raw topics with ros2 bag'),
        DeclareLaunchArgument('record_images', default_value='false',
                              description='add the compressed camera stream to the bag'),
        DeclareLaunchArgument('log_dir', default_value='',
                              description='runs folder (default: <workspace>/hw4_logs)'),
        DeclareLaunchArgument('run_tag', default_value='',
                              description='optional suffix for the run folder name'),
        DeclareLaunchArgument('auto_eval', default_value='true',
                              description='post-run hook: evaluate the run when it ends'),
    ]

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
    shutdown_when_motor_exits = RegisterEventHandler(OnProcessExit(
        target_action=motor_node, on_exit=[Shutdown(reason='Motor controller exited')]))

    return LaunchDescription(args + [
        camera,
        apriltag,
        motor_node,
        shutdown_when_motor_exits,
        OpaqueFunction(function=launch_setup),
    ])
