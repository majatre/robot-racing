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

prev_error = 0
error_sum = 0
tau_p = 2
tau_d = 20 # 20
tau_i = 0 #0.01
k = 1  # control gain


def PID(pose, path, velocity):
  global prev_error, error_sum, tau_p, tau_d, tau_i
  cte, angle_diff = calculate_error(pose, path)
  error = angle_diff + np.arctan2(k * cte, np.linalg.norm(velocity)) 
  error_change = error - prev_error
  prev_error = error
  
  print('Errors', cte, error_change)

  error_sum += error
  u = np.linalg.norm(velocity) * max(0.3, (1 - abs(error)))
  #u = 1.3 * np.linalg.norm(velocity) * max(0.3, (1 - abs(error)))
  #w = 1/epsilon * (-dx_p*np.sin(theta) + dy_p*np.cos(theta))
  w = tau_p*error + tau_d*error_change + tau_i*error_sum
 
  return u, w


def stanley_steering(pose, path, velocity):
  cte, angle_diff = calculate_error(pose, path)
  
  # theta_e corrects the heading error
  theta_e = angle_diff
  # theta_d corrects the cross track error
  theta_d = np.arctan2(k * cte, np.linalg.norm(velocity))
  # Steering control
  delta = theta_e + theta_d

  print('Errors', theta_e, theta_d)

  dx_p = np.cos(delta)
  dy_p = np.sin(delta)
  theta = pose[YAW] 

  u = np.linalg.norm(velocity) 
  w = 2 * delta * u #1/epsilon * (-dx_p*np.sin(theta) + dy_p*np.cos(theta))
  # w = tau_p*cte + tau_d*cte_change + tau_i*error_sum
  print(u, w)
  return u, w


def calculate_error(pose, path_points):
  if len(path_points) == 0:
    return 0, 0

  p = np.array([pose[X], pose[Y]], dtype=np.float32)

  # Find the currently closest point in the path
  min_dist = np.linalg.norm(p - path_points[0])
  min_point = 0
  for i, point in enumerate(path_points):
    dist = np.linalg.norm(p - point)
    if dist < min_dist:
      min_dist = dist
      min_point = i

  p1 = path_points[min_point]
  p2 = path_points[min_point+1]
  p3 = path_points[min_point+2]

  dist = -np.cross(p2-p1,p-p1)/np.linalg.norm(p2-p1)
  print('Distance', dist)
  angle = np.arctan2((p3-p2)[1], (p3-p2)[0])
  angle_diff = np.arctan2(np.sin(angle-pose[YAW]), np.cos(angle-pose[YAW]))
  print('Angle diff', angle_diff)


  return dist, angle_diff



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
  max_acc = 0.75
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

  # Move in the direction of the next point
  if len(path_points) <= min_point + 3:
    direction = path_points[-1]
    v = direction - position
  else:
    if min_point < 2:
      a_points = path_points[min_point : min_point+11]
      b_points = path_points[min_point+1 : min_point+12]
      c_points = path_points[min_point+2 : min_point+13]
    else:  
      a_points = path_points[min_point-2 : min_point+11]
      b_points = path_points[min_point-1 : min_point+12]
      c_points = path_points[min_point : min_point+13]
    curvatures = [get_curvature(a,b,c) for a,b,c in zip(a_points, b_points, c_points)]
    curvature = max(
      sum(curvatures) / len(curvatures), 
      sum(curvatures[:5]) / len(curvatures[:5]), 
      sum(curvatures[:3]) / len(curvatures[:3])
      )
    direction = sum(path_points[min_point+1:min_point+4])/len(path_points[min_point+1:min_point+4])
    factor = max_velocity
   
    if curvature > max_acc:
      #print('Curva', curvature)
      factor *= np.sqrt(max_acc / curvature)

    v = factor * (direction - position) / np.linalg.norm(direction - position)
    #print(v, np.linalg.norm(v))


  if np.linalg.norm(v) > max_velocity:
    v *= max_velocity / np.linalg.norm(v)
  # Scale the velocity to have a magnitude of 0.2.
  return  v


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
  #publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=5)
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
      plot_trajectory.plot_velocity(occ_grid)
      publisher.publish(stop_msg)
      rate_limiter.sleep()
      continue

    # Follow path using feedback linearization.
    position = np.array([
        groundtruth.pose[X] + EPSILON * np.cos(groundtruth.pose[YAW]),
        groundtruth.pose[Y] + EPSILON * np.sin(groundtruth.pose[YAW])], dtype=np.float32)
    v = get_velocity(position, np.array(current_path, dtype=np.float32))
    u, w = PID(groundtruth.pose, np.array(current_path, dtype=np.float32), v)
    #u, w = feedback_linearized(groundtruth.pose, v, epsilon=EPSILON)
    vel_msg = Twist()
    vel_msg.linear.x = u
    vel_msg.angular.z = w
    publisher.publish(vel_msg)


    print(u, np.linalg.norm(groundtruth.velocity), groundtruth.velocity)
    # Log groundtruth positions in /tmp/gazebo_exercise.txt
    pose_history.append([groundtruth.pose[X], groundtruth.pose[Y], np.linalg.norm(groundtruth.velocity)])
    if len(pose_history) % 10:
      with open('/tmp/gazebo_race_trajectory.txt', 'a') as fp:
        fp.write('\n'.join(','.join(str(v) for v in p) for p in pose_history) + '\n')
        pose_history = []

    # Update plan every 1s.
    time_since = current_time - previous_time
    if current_path and time_since <60.:
      rate_limiter.sleep()
      continue
    previous_time = current_time

    # Run RRT.
    print(groundtruth.pose, goal.position)
    current_path = rrt.run_path_planning(groundtruth.pose, goal.position, occ_grid)
    print('Path', current_path)
    if not current_path:
      print('Unable to reach goal position:', goal.position)

    prev_cte = 0
    error_sum = 0
    tau_p = 1 
    tau_d = 0 
    tau_i = 0

     # Log groundtruth positions in /tmp/gazebo_exercise.txt
    with open('/tmp/gazebo_race_path.txt', 'a') as fp:
      fp.write('\n'.join(','.join(str(v) for v in p) for p in current_path) + '\n')
      pose_history = []
      
    rate_limiter.sleep()
    frame_id += 1


if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='Uses RRT to reach the goal.')
  parser.add_argument('--map', action='store', default='maps/circuit', help='Which map to use.')
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

  occupancy_grid = rrt.OccupancyGrid(occupancy_grid, data['origin'], data['resolution'])


  try:
    run(args, occupancy_grid)
  except rospy.ROSInterruptException:
    pass
