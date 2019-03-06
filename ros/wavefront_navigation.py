#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
import rospy
import sys
import yaml

# Robot motion commands:
# http://docs.ros.org/api/geometry_msgs/html/msg/Twist.html
from geometry_msgs.msg import Twist
# Position.
from tf import TransformListener
# Goal.
from geometry_msgs.msg import PoseStamped
# Path.
from nav_msgs.msg import Path
# For pose information.
from tf.transformations import euler_from_quaternion

from gazebo_msgs.msg import ModelStates

# Import the potential_field.py code rather than copy-pasting.
directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../wavefront')
sys.path.insert(0, directory)

directory_rrt = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../rrt')
sys.path.insert(0, directory_rrt)
try:
    import wavefront as wavefront
    import plot_trajectory
except ImportError:
    raise ImportError(
        'Unable to import wavefront.py. Make sure this file is in "{}"'.format(
            directory))

SPEED = .2
EPSILON = .1
GOAL_POSITION = np.array([-1., -2.5], dtype=np.float32)

X = 0
Y = 1
YAW = 2


def feedback_linearized(pose, velocity, epsilon):
    # Implementation of feedback-linearization to follow the velocity
    # vector given as argument. Epsilon corresponds to the distance of
    # linearized point in front of the robot.
    u = 0.  # [m/s]
    w = 0.  # [rad/s] going counter-clockwise.

    dx_p = velocity[0]
    dy_p = velocity[1]
    theta = pose[YAW]

    u = dx_p * np.cos(theta) + dy_p * np.sin(theta)
    w = 1 / epsilon * (-dx_p * np.sin(theta) + dy_p * np.cos(theta))
    return u, w


def get_area(a, b, c):
    return 1 / 2 * abs(
        (b[X] - a[X]) * (c[Y] - a[Y]) - (b[Y] - a[Y]) * (c[X] - a[X]))


def get_curvature(a, b, c):
    A = get_area(a, b, c)
    l1 = np.linalg.norm(b - a)
    l2 = np.linalg.norm(c - a)
    l3 = np.linalg.norm(c - b)
    return 4 * A / (l1 * l2 * l3)


def get_velocity(position, path_points):
    max_acc = 1.
    max_velocity = 1.
    v = np.zeros_like(position)
    if len(path_points) == 0:
        return v
    # Stop moving if the goal is reached.
    if np.linalg.norm(position - path_points[-1]) < .2:
        return v

    # Find the currently closest point in the path
    min_dist = np.linalg.norm(position - path_points[0])
    min_point = 0
    for i, p in enumerate(path_points):
        dist = np.linalg.norm(position - p)
        if dist < min_dist:
            min_dist = dist
            min_point = i

    # Move in the direction of the next point
    if len(path_points) <= min_point + 3:
        direction = path_points[-1]
        v = direction - position
    else:
        if min_point < 2:
            a_points = path_points[min_point: min_point + 11]
            b_points = path_points[min_point + 1: min_point + 12]
            c_points = path_points[min_point + 2: min_point + 13]
        else:
            a_points = path_points[min_point - 2: min_point + 11]
            b_points = path_points[min_point - 1: min_point + 12]
            c_points = path_points[min_point: min_point + 13]
        curvatures = [get_curvature(a, b, c) for a, b, c in
                      zip(a_points, b_points, c_points)]
        # curvature = max(
        #     sum(curvatures) / len(curvatures),
        #     sum(curvatures[:5]) / len(curvatures[:5]),
        #     # sum(curvatures[:3]) / len(curvatures[:3])
        # )

        curvature = sum(curvatures) / len(curvatures)

        factor = max_velocity

        if curvature > max_acc:
            factor *= np.sqrt(max_acc / curvature)

        direction = path_points[min_point + 4]
        v = factor * (direction - position) / np.linalg.norm(
            direction - position)

    if np.linalg.norm(v) > max_velocity:
        v *= max_velocity / np.linalg.norm(v)

    return v


class GroundtruthPose(object):
    def __init__(self, name='turtlebot3_burger'):
        rospy.Subscriber('/gazebo/model_states', ModelStates, self.callback)
        self._pose = np.array([np.nan, np.nan, np.nan], dtype=np.float32)
        self._name = name

    def callback(self, msg):
        idx = [i for i, n in enumerate(msg.name) if n == self._name]
        if not idx:
            raise ValueError(
                'Specified name "{}" does not exist.'.format(self._name))
        idx = idx[0]
        self._pose[X] = msg.pose[idx].position.x
        self._pose[Y] = msg.pose[idx].position.y
        _, _, yaw = euler_from_quaternion([
            msg.pose[idx].orientation.x,
            msg.pose[idx].orientation.y,
            msg.pose[idx].orientation.z,
            msg.pose[idx].orientation.w])
        self._pose[YAW] = yaw

    @property
    def ready(self):
        return not np.isnan(self._pose[0])

    @property
    def pose(self):
        return self._pose


class GoalPose(object):
    def __init__(self):
        # rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.callback)
        self._position = GOAL_POSITION

    def callback(self, msg):
        # The pose from RViz is with respect to the "map".
        self._position[X] = msg.pose.position.x
        self._position[Y] = msg.pose.position.y
        print('Received new goal position:', self._position)

    @property
    def ready(self):
        return not np.isnan(self._position[0])

    @property
    def position(self):
        return self._position


def run(args, occ_grid):
    rospy.init_node('wavefront_navigation')
    with open('/tmp/gazebo_race_path.txt', 'w'):
        pass
    with open('/tmp/gazebo_race_trajectory.txt', 'w'):
        pass

    # Update control every 100 ms.
    rate_limiter = rospy.Rate(100)
    publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=5)
    path_publisher = rospy.Publisher('/path', Path, queue_size=1)
    groundtruth = GroundtruthPose()
    goal = GoalPose()
    frame_id = 0
    current_path = []
    pose_history = []
    previous_time = rospy.Time.now().to_sec()

    # Stop moving message.
    stop_msg = Twist()
    stop_msg.linear.x = 0.
    stop_msg.angular.z = 0.

    # Make sure the robot is stopped.
    i = 0
    while i < 10 and not rospy.is_shutdown():
        publisher.publish(stop_msg)
        rate_limiter.sleep()
        i += 1

    while not rospy.is_shutdown():
        current_time = rospy.Time.now().to_sec()

        # Make sure all measurements are ready.
        # Get map and current position through SLAM:
        # > roslaunch exercises slam.launch
        # if not goal.ready or not slam.ready:
        #   rate_limiter.sleep()
        #   continue

        goal_reached = np.linalg.norm(groundtruth.pose[:2] - goal.position) < .4
        if goal_reached:
            plot_trajectory.plot_race(occ_grid)
            publisher.publish(stop_msg)
            rate_limiter.sleep()
            continue

        # Follow path using feedback linearization.
        position = np.array([
            groundtruth.pose[X] + EPSILON * np.cos(groundtruth.pose[YAW]),
            groundtruth.pose[Y] + EPSILON * np.sin(groundtruth.pose[YAW])],
            dtype=np.float32)
        v = get_velocity(position, np.array(current_path, dtype=np.float32))
        u, w = feedback_linearized(groundtruth.pose, v, epsilon=EPSILON)
        vel_msg = Twist()
        vel_msg.linear.x = u
        vel_msg.angular.z = w
        publisher.publish(vel_msg)

        # Log groundtruth positions in /tmp/gazebo_exercise.txt
        pose_history.append(
            np.concatenate([groundtruth.pose, position], axis=0))
        if len(pose_history) % 10:
            with open('/tmp/gazebo_race_trajectory.txt', 'a') as fp:
                fp.write('\n'.join(
                    ','.join(str(v) for v in p) for p in pose_history) + '\n')
                pose_history = []

        # Update plan every 1s.
        time_since = current_time - previous_time
        if current_path and time_since < 30.:
            rate_limiter.sleep()
            continue
        previous_time = current_time

        # Run Wavefront.
        current_path = wavefront.run_path_planning(occ_grid, groundtruth.pose[:2], goal.position)

        if len(current_path) == 0:
            print('Unable to reach goal position:', goal.position)

        # Log groundtruth positions in /tmp/gazebo_exercise.txt
        with open('/tmp/gazebo_race_path.txt', 'a') as fp:
            fp.write('\n'.join(
                ','.join(str(v) for v in p) for p in current_path) + '\n')
            pose_history = []

        rate_limiter.sleep()
        frame_id += 1


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Uses the Wavefront algorithm to reach the goal.')
    parser.add_argument('--map', action='store',
                        default='/home/roxana/catkin_ws/src/exercises/exercises/robot-racing/rrt/maps/circuit',
                        help='Which map to use.')
    args, unknown = parser.parse_known_args()

    # Load map.
    with open(args.map + '.yaml') as fp:
        data = yaml.load(fp)
    img = wavefront.read_pgm(os.path.join(os.path.dirname(args.map), data['image']))
    occupancy_grid = np.empty_like(img, dtype=np.int8)
    occupancy_grid[:] = wavefront.UNKNOWN
    occupancy_grid[img < .1] = wavefront.OCCUPIED
    occupancy_grid[img > .9] = wavefront.FREE
    print(len(occupancy_grid[img > .9]))
    # Transpose (undo ROS processing).
    occupancy_grid = occupancy_grid.T
    # Invert Y-axis.
    occupancy_grid = occupancy_grid[:, ::-1]

    # Invisible wall
    # if args.map == 'maps/circuit':
    occupancy_grid[170, 144:170] = wavefront.OCCUPIED
    GOAL_POSITION = np.array([-1., -2.],
                             dtype=np.float32)  # Any orientation is good.
    START_POSE = np.array([-2.5, -2.5, np.pi / 2], dtype=np.float32)
    if 'square' in args.map:
        occupancy_grid[177, 160:180] = wavefront.OCCUPIED
        GOAL_POSITION = np.array([-1., -1.5],
                                 dtype=np.float32)  # Any orientation is good.
        START_POSE = np.array([-1.5, -1.5, np.pi / 2], dtype=np.float32)

    occupancy_grid = wavefront.OccupancyGrid(occupancy_grid, data['origin'],
                                       data['resolution'])

    try:
        run(args, occupancy_grid)
    except rospy.ROSInterruptException:
        pass

