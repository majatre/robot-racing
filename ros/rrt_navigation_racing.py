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
    
  u = dx_p*np.cos(theta) + dy_p*np.sin(theta)
  w = 1/epsilon * (-dx_p*np.sin(theta) + dy_p*np.cos(theta))
  return u, w


def get_area(a,b,c):
  return 1/2 * abs((b[X] - a[X]) * (c[Y] - a[Y]) - (b[Y] - a[Y]) *(c[X] - a[X]))

def get_curvature(a,b,c):
  A = get_area(a,b,c)
  l1 = np.linalg.norm(b-a)
  l2 = np.linalg.norm(c-a)
  l3 = np.linalg.norm(c-b)
  return 4*A / (l1*l2*l3)

def get_velocity(position, path_points):
  max_acc = 3
  max_velocity = 1
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

  print('Min point', position, path_points[min_point])

  # Move in the direction of the next point
  if len(path_points) <= min_point + 3:
    direction = path_points[-1]
    v = direction - position
  else:
    direction = path_points[min_point+3]
    curvature = get_curvature(path_points[min_point+1],path_points[min_point+2],path_points[min_point+3])
    print('Curva', curvature)
    v = 0.5 * (direction - position) / np.linalg.norm(direction - position)
    if curvature > 0.1:
      v *= np.sqrt(max_acc / curvature)

  if np.linalg.norm(v) > max_velocity:
    v *= max_velocity / np.linalg.norm(v)
  print(v)

  # Scale the velocity to have a magnitude of 0.2.
  return  v


class GroundtruthPose(object):
  def __init__(self, name='turtlebot3_burger'):
    rospy.Subscriber('/gazebo/model_states', ModelStates, self.callback)
    self._pose = np.array([np.nan, np.nan, np.nan], dtype=np.float32)
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

  @property
  def ready(self):
    return not np.isnan(self._pose[0])

  @property
  def pose(self):
    return self._pose


class GoalPose(object):
  def __init__(self):
    #rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.callback)
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


def get_path(final_node):
  # Construct path from RRT solution.
  if final_node is None:
    return []
  path_reversed = []
  path_reversed.append(final_node)
  while path_reversed[-1].parent is not None:
    path_reversed.append(path_reversed[-1].parent)
  path = list(reversed(path_reversed))
  # Put a point every 5 cm.
  distance = 0.05
  offset = 0.
  points_x = []
  points_y = []
  for u, v in zip(path, path[1:]):
    center, radius = rrt.find_circle(u, v)
    du = u.position - center
    theta1 = np.arctan2(du[1], du[0])
    dv = v.position - center
    theta2 = np.arctan2(dv[1], dv[0])
    # Check if the arc goes clockwise.
    clockwise = np.cross(u.direction, du).item() > 0.
    # Generate a point every 5cm apart.
    da = distance / radius
    offset_a = offset / radius
    if clockwise:
      da = -da
      offset_a = -offset_a
      if theta2 > theta1:
        theta2 -= 2. * np.pi
    else:
      if theta2 < theta1:
        theta2 += 2. * np.pi
    angles = np.arange(theta1 + offset_a, theta2, da)
    offset = distance - (theta2 - angles[-1]) * radius
    points_x.extend(center[X] + np.cos(angles) * radius)
    points_y.extend(center[Y] + np.sin(angles) * radius)
  print(zip(points_x, points_y))
  return zip(points_x, points_y)
  

def run(args, occ_grid):
  rospy.init_node('rrt_navigation')
  with open('/tmp/gazebo_race_path.txt', 'w'):
   pass
  with open('/tmp/gazebo_race_trajectory.txt', 'w'):
   pass

  # Update control every 100 ms.
  rate_limiter = rospy.Rate(100)
  publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=5)
  path_publisher = rospy.Publisher('/path', Path, queue_size=1)
  groundtruth = GroundtruthPose()
  goal =  GoalPose()
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

    goal_reached = np.linalg.norm(groundtruth.pose[:2] - goal.position) < .4
    if goal_reached:
      plot_trajectory.plot_race(occ_grid)
      publisher.publish(stop_msg)
      rate_limiter.sleep()
      continue

    # Follow path using feedback linearization.
    position = np.array([
        groundtruth.pose[X] + EPSILON * np.cos(groundtruth.pose[YAW]),
        groundtruth.pose[Y] + EPSILON * np.sin(groundtruth.pose[YAW])], dtype=np.float32)
    v = get_velocity(position, np.array(current_path, dtype=np.float32))
    u, w = feedback_linearized(groundtruth.pose, v, epsilon=EPSILON)
    vel_msg = Twist()
    vel_msg.linear.x = u
    vel_msg.angular.z = w
    publisher.publish(vel_msg)

    # Log groundtruth positions in /tmp/gazebo_exercise.txt
    pose_history.append(np.concatenate([groundtruth.pose, position], axis=0))
    if len(pose_history) % 10:
      with open('/tmp/gazebo_race_trajectory.txt', 'a') as fp:
        fp.write('\n'.join(','.join(str(v) for v in p) for p in pose_history) + '\n')
        pose_history = []

    # Update plan every 1s.
    time_since = current_time - previous_time
    if current_path and time_since < 30.:
      rate_limiter.sleep()
      continue
    previous_time = current_time

    # Run RRT.
    print(groundtruth.pose, goal.position)
    current_path = rrt.run_path_planning(groundtruth.pose, goal.position, occ_grid)
    print('Path', current_path)
    if not current_path:
      print('Unable to reach goal position:', goal.position)

     # Log groundtruth positions in /tmp/gazebo_exercise.txt
    with open('/tmp/gazebo_race_path.txt', 'a') as fp:
      fp.write('\n'.join(','.join(str(v) for v in p) for p in current_path) + '\n')
      pose_history = []
      
    rate_limiter.sleep()
    frame_id += 1


if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='Uses RRT to reach the goal.')
  parser.add_argument('--map', action='store', default='/home/maja/catkin_ws/src/exercises/robot-racing/rrt/maps/circuit', help='Which map to use.')
  args, unknown = parser.parse_known_args()

  # Load map.
  with open(args.map + '.yaml') as fp:
      data = yaml.load(fp)
  img = rrt.read_pgm(os.path.join(os.path.dirname(args.map), data['image']))
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
 #if args.map == 'maps/circuit':
  occupancy_grid[170, 144:170] = rrt.OCCUPIED
  GOAL_POSITION = np.array([-1., -2.5], dtype=np.float32)  # Any orientation is good.
  START_POSE = np.array([-2.5, -2.5, np.pi / 2], dtype=np.float32)
  if args.map == 'maps/square':
      occupancy_grid[177, 160:180] = rrt.OCCUPIED
      GOAL_POSITION = np.array([-1., -1.5], dtype=np.float32)  # Any orientation is good.
      START_POSE = np.array([-1.5, -1.5, np.pi / 2], dtype=np.float32)

  occupancy_grid = rrt.OccupancyGrid(occupancy_grid, data['origin'], data['resolution'])


  try:
    run(args, occupancy_grid)
  except rospy.ROSInterruptException:
    pass