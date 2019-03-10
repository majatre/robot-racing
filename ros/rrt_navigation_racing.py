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
# Occupancy grid.
from nav_msgs.msg import OccupancyGrid
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
directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../rrt')
sys.path.insert(0, directory)
try:
  import rrt_star_lines_opt as rrt
  import plot_trajectory
except ImportError:
  raise ImportError('Unable to import potential_field.py. Make sure this file is in "{}"'.format(directory))

from steering_control import PID, get_velocity


EPSILON = .1
GOAL_POSITION = np.array([-1., -2.5], dtype=np.float32)

X = 0
Y = 1
YAW = 2

class GroundtruthPose(object):
  def __init__(self, name='turtlebot3_burger'):
    rospy.Subscriber('/gazebo/model_states', ModelStates, self.callback)
    self._pose = np.array([np.nan, np.nan, np.nan], dtype=np.float32)
    self._velocity = np.array([np.nan, np.nan, np.nan], dtype=np.float32)
    self._name = name

  def callback(self, msg):
    idx = [i for i, n in enumerate(msg.name) if n == self._name]
    if not idx:
      raise ValueError('Specified name "{}" does not exist.'.format(self._name))
    idx = idx[0]
    self._pose[X] = msg.pose[idx].position.x
    self._pose[Y] = msg.pose[idx].position.y
    _, _, yaw = euler_from_quaternion([
        msg.pose[idx].orientation.x,
        msg.pose[idx].orientation.y,
        msg.pose[idx].orientation.z,
        msg.pose[idx].orientation.w])
    self._pose[YAW] = yaw
    self._velocity[0] = msg.twist[idx].linear.x
    self._velocity[1] = msg.twist[idx].linear.y
    self._velocity[2] = msg.twist[idx].linear.z
   # print(msg.twist[idx])

  @property
  def ready(self):
    return not np.isnan(self._pose[0])

  @property
  def pose(self):
    return self._pose

  @property
  def velocity(self):
    return self._velocity


def run(args, occ_grid):
  sart_time = 0
  rospy.init_node('rrt_navigation')
  file_path = directory + '/../metrics/{}_rrt_gazebo_race_path.txt'.format(args.map.split('/')[1])
  file_trajectory = directory + '/../metrics/{}_rrt_gazebo_race_trajectory.txt'.format(args.map.split('/')[1])
  with open(file_path, 'w'):
   pass
  with open(file_trajectory, 'w'):
   pass

  # Update control every 100 ms.
  rate_limiter = rospy.Rate(50)
  publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=5)
  groundtruth = GroundtruthPose()
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
    #slam.update()
    current_time = rospy.Time.now().to_sec()

    # Make sure all measurements are ready.
     # Get map and current position through SLAM:
    # > roslaunch exercises slam.launch
    # if not goal.ready or not slam.ready:
    #   rate_limiter.sleep()
    #   continue

    goal_reached = np.linalg.norm(groundtruth.pose[:2] - GOAL_POSITION) < .4
    if goal_reached:
      finish_time = rospy.Time.now().to_sec()
      print('------- Time:', finish_time - start_time)
      plot_trajectory.plot_race(occ_grid, file_path, file_trajectory)
      plot_trajectory.plot_velocity(occ_grid, file_path, file_trajectory)
      publisher.publish(stop_msg)
      rate_limiter.sleep()
      continue

    # Follow path using feedback linearization.
    position = np.array([
        groundtruth.pose[X] + EPSILON * np.cos(groundtruth.pose[YAW]),
        groundtruth.pose[Y] + EPSILON * np.sin(groundtruth.pose[YAW])], dtype=np.float32)
    v = get_velocity(position, np.array(current_path, dtype=np.float32))
    u, w = PID(groundtruth.pose, np.array(current_path, dtype=np.float32), v, np.linalg.norm(groundtruth.velocity))
    #u, w = feedback_linearized(groundtruth.pose, v, epsilon=EPSILON)
    vel_msg = Twist()
    vel_msg.linear.x = u
    vel_msg.linear.y = .0
    vel_msg.linear.z = .0
    vel_msg.angular.x = .0
    vel_msg.angular.y = .0
    vel_msg.angular.z = w
    publisher.publish(vel_msg)

    #print(u, np.linalg.norm(groundtruth.velocity), groundtruth.velocity)
    # Log groundtruth positions and velocities
    pose_history.append([groundtruth.pose[X], groundtruth.pose[Y], np.linalg.norm(groundtruth.velocity)])
    if len(pose_history) % 10 == 0:
      with open(file_trajectory, 'a') as fp:
        fp.write('\n'.join(','.join(str(v) for v in p) for p in pose_history) + '\n')
        pose_history = []

    # Update plan every 1s.
    time_since = current_time - previous_time
    if current_path and time_since < 60.:
      rate_limiter.sleep()
      continue
    previous_time = current_time

    # Run RRT.
    #current_path = rrt.run_path_planning(groundtruth.pose, goal.position, occ_grid)
    xy_path = np.genfromtxt(directory + '/paths/rrt_path_sharp.txt', delimiter=',')
    current_path = [(xy[0], xy[1]) for xy in xy_path]

    for a,b in zip(current_path, current_path[1:]):
      print(np.linalg.norm(np.array(b) - np.array(a)))

    print('Path', current_path)
    if not current_path:
      print('Unable to reach goal position:', GOAL_POSITION)
    else:
      start_time = rospy.Time.now().to_sec()

     # Log path
    with open(file_path, 'a') as fp:
      fp.write('\n'.join(','.join(str(v) for v in p) for p in current_path) + '\n')
      pose_history = []
      
    rate_limiter.sleep()
    frame_id += 1


if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='Uses RRT to reach the goal.')
  parser.add_argument('--map', action='store', default='maps/sharp_turn', help='Which map to use.')
  args, unknown = parser.parse_known_args()

  # Load map.
  with open(directory + '/' + args.map + '.yaml') as fp:
      data = yaml.load(fp)
  img = rrt.read_pgm(os.path.join(os.path.dirname(directory + '/' + args.map), data['image']))
  occupancy_grid = np.empty_like(img, dtype=np.int8)
  occupancy_grid[:] = rrt.UNKNOWN
  occupancy_grid[img < .1] = rrt.OCCUPIED
  occupancy_grid[img > .9] = rrt.FREE
  print(len(occupancy_grid[img > .9]))
  # Transpose (undo ROS processing).
  occupancy_grid = occupancy_grid.T
  # Invert Y-axis.
  occupancy_grid = occupancy_grid[:, ::-1]

  #Invisible wall
  if args.map == 'maps/circuit':
    occupancy_grid[170, 144:170] = rrt.OCCUPIED
    GOAL_POSITION = np.array([-1., -2.5], dtype=np.float32)  # Any orientation is good.
    START_POSE = np.array([-2.5, -2.5, np.pi / 2], dtype=np.float32)
  if args.map == 'maps/square':
    occupancy_grid[176, 160:180] = rrt.OCCUPIED
    GOAL_POSITION = np.array([-1., -1.5], dtype=np.float32)  # Any orientation is good.
    START_POSE = np.array([-1.5, -1.5, np.pi / 2], dtype=np.float32)
  elif args.map == 'maps/sharp_turn':
    GOAL_POSITION = np.array([0.7, -1], dtype=np.float32)  # Any orientation is good.
    START_POSE = np.array([-0.3, -1, np.pi / 2], dtype=np.float32)
  elif args.map == 'maps/smooth_turn':
    occupancy_grid[175, 160:180] = rrt.OCCUPIED
    GOAL_POSITION = np.array([-1., -1.5], dtype=np.float32)  # Any orientation is good.
    START_POSE = np.array([-1.5, -1.5, np.pi / 2], dtype=np.float32)


  occupancy_grid = rrt.OccupancyGrid(occupancy_grid, data['origin'], data['resolution'])


  try:
    run(args, occupancy_grid)
  except rospy.ROSInterruptException:
    pass
