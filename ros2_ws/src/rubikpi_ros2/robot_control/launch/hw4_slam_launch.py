"""
HW4 extra credit: plan with the obstacle mapped by EKF-SLAM (camera + apriltag + motors + node).

  ros2 launch robot_control hw4_slam_launch.py mode:=fast
  ros2 launch robot_control hw4_slam_launch.py mode:=safe record_bag:=true run_tag:=try2
  ros2 launch robot_control hw4_slam_launch.py mode:=fast prior_landmarks:=walls

The robot knows the arena walls, its start pose, the goal and only a prior region for the box
(config/hw4_world_slam.yaml, slam.prior). velocity_control runs EKF-SLAM from the world start
pose, estimates the box from the mapped box tags and (re)plans on that estimate
(robot_control/slam_obstacle.py). The run folder <workspace>/hw4_logs/slam_<mode>_<date>_<time>
holds the world file used, the plan made on the prior alone (waypoints_prior.txt), the SLAM
logs (trajectory, measurements, map snapshots, map_final.csv), obstacle_estimates.csv and
replans.json; when the node exits `hw4_slam eval` scores it against the measured box and tags.
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
              '/slam/pose', '/slam/path', '/slam/landmarks', '/rosout']


def launch_setup(context):
    def arg(name):
        return context.launch_configurations[name]

    import yaml

    from robot_control import hw4_slam_tool, hw4_tool
    from robot_control.path_tools import write_waypoints
    from robot_control.slam_obstacle import SlamConfig, SlamPlanner
    from robot_control.world_model import world_from_dict

    mode = arg('mode')
    if mode not in ('fast', 'safe'):
        raise RuntimeError("mode must be 'fast' or 'safe'")
    src = os.path.realpath(os.path.expanduser(arg('world_file')
                                              or hw4_slam_tool.default_world_file()))
    params_file = arg('params_file') or hw4_slam_tool.default_params_file()
    log_dir = os.path.expanduser(arg('log_dir')) if arg('log_dir') else hw4_tool.logs_dir()
    name = hw4_tool.run_name(f'slam_{mode}', arg('run_tag'))
    run_dir = os.path.join(log_dir, name)
    os.makedirs(run_dir, exist_ok=True)

    # The run folder gets the exact world the node and the evaluation use
    with open(src) as f:
        data = yaml.safe_load(f)
    if arg('prior_landmarks'):
        data.setdefault('slam', {})['prior_landmarks'] = arg('prior_landmarks')
    world_file = os.path.join(run_dir, 'world.yaml')
    with open(world_file, 'w') as f:
        f.write(f'# copied from {src} by hw4_slam_launch.py\n')
        yaml.safe_dump(data, f, sort_keys=False)
    world = world_from_dict(data, source=world_file)
    cfg = SlamConfig.from_world(world)

    # What the robot would do knowing nothing about the box: plan around the prior region.
    # The node replaces it with a plan on the SLAM estimate once the start delay is over.
    prior = SlamPlanner(world, mode, cfg)
    wps = prior.update(0.0, world.start, {})
    if wps is None:
        raise RuntimeError(f'no path around the prior region: {prior.history[-1]["note"]}')
    wp_file = os.path.join(run_dir, 'waypoints_prior.txt')
    write_waypoints(wp_file, wps, [
        f'HW4 SLAM {mode}: plan on the PRIOR obstacle region only (before any mapping)',
        'world frame; theta nan = pass-through; the node replans on the SLAM estimate'])
    others = sorted(str(t) for t in world.landmarks if isinstance(t, str))
    print(f'[hw4_slam] {mode}: obstacle prior centre {cfg.prior_center} +- '
          f'{cfg.center_tolerance} m, sides <= {cfg.max_size} m; prior_landmarks '
          f'{cfg.prior_landmarks}\n[hw4_slam] plan on the prior: {len(wps)} waypoints, '
          f'{prior.history[-1]["length"]:.2f} m\n[hw4_slam] run folder: {run_dir}'
          + (f'\n[hw4_slam] WARNING: tags {others} are not 36h11; only the 36h11 detector '
             f'runs, so they will not be seen' if others else ''), flush=True)

    overrides = {
        'waypoints_file': wp_file,
        'world_file': world_file,
        'log_dir': log_dir,
        'run_name': name,
        'path_name': f'slam_{mode}',
        'slam_plan_mode': mode,
        'use_slam': True,
        'start_pose': ','.join(f'{v:.4f}' for v in world.start),
        'post_run_eval': arg('auto_eval').lower() == 'true',
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
    return actions


def generate_launch_description():
    args = [
        DeclareLaunchArgument('mode', default_value='fast',
                              description='fast (visibility graph + A*) or safe (Voronoi), '
                                          'replanned on the SLAM-mapped obstacle'),
        DeclareLaunchArgument('world_file', default_value='',
                              description='world yaml (default: config/hw4_world_slam.yaml)'),
        DeclareLaunchArgument('params_file', default_value='',
                              description='node parameters (default: config/hw4_slam.yaml)'),
        DeclareLaunchArgument('prior_landmarks', default_value='',
                              description="'none' (map every tag) or 'walls' (wall tags known); "
                                          'default: slam.prior_landmarks of the world file'),
        DeclareLaunchArgument('use_imu', default_value='',
                              description='true/false (default: from the params file)'),
        DeclareLaunchArgument('record_bag', default_value='false'),
        DeclareLaunchArgument('record_images', default_value='false'),
        DeclareLaunchArgument('log_dir', default_value='',
                              description='runs folder (default: <workspace>/hw4_logs)'),
        DeclareLaunchArgument('run_tag', default_value='',
                              description='optional suffix for the run folder name'),
        DeclareLaunchArgument('auto_eval', default_value='true',
                              description='run hw4_slam eval on the run folder when it ends'),
    ]
    camera = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution(
            [FindPackageShare('robot_vision_camera'), 'launch', 'robot_vision_camera.launch.py'])),
        launch_arguments={'image_rectify': 'true'}.items(),
    )
    # ONE detector (36h11) for wall and box tags: a second apriltag_node on the same image
    # stream received almost no frames on the RubikPi
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
