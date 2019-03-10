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

MAX_SPEED = 1.5
EPSILON = .1
GOAL_POSITION = np.array([-1., -2.5], dtype=np.float32)

X = 0
Y = 1
YAW = 2

prev_error = 0
tau_p = 8
tau_d = 160 # 20
tau_i = 0 #0.01
k = 1  # control gain
k_v = 1.5 #speed proportional gain 

dt = 0.1

def PID(pose, path, velocity, current_speed):
  global prev_error
  cte, angle_diff = calculate_error(pose, path)
  error = angle_diff + np.arctan2(k * cte, np.linalg.norm(velocity)) 
  error_change = error - prev_error
  prev_error = error

  target_u = np.linalg.norm(velocity) * max(0.3, (1 - abs(error/3))) 
  if target_u - current_speed > dt*k_v:
    u = current_speed + dt*k_v # *(target_u - current_speed)
  else:
    u = target_u
  w = (tau_p*error + tau_d*error_change) * current_speed
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

  lookahead = int(MAX_SPEED * 3) 
  if len(path_points) <= min_point + lookahead:
    return 0, 0

  p1 = path_points[min_point]
  p2 = path_points[min_point+1]
  p3 = path_points[min_point+lookahead-1]
  p4 = sum(path_points[min_point+lookahead:min_point+lookahead+3]) / len(path_points[min_point+lookahead:min_point+lookahead+3])

  dist = -np.cross(p2-p1,p-p1)/np.linalg.norm(p2-p1)
  angle = np.arctan2((p4-p3)[1], (p4-p3)[0])
  angle_diff = np.arctan2(np.sin(angle-pose[YAW]), np.cos(angle-pose[YAW]))

  return dist, angle_diff


def get_area(a,b,c):
  return 1/2 * abs((b[X] - a[X]) * (c[Y] - a[Y]) - (b[Y] - a[Y]) *(c[X] - a[X]))

def get_curvature(a,b,c):
  A = get_area(a,b,c)
  l1 = np.linalg.norm(b-a)
  l2 = np.linalg.norm(c-a)
  l3 = np.linalg.norm(c-b)
  return 4*A / (l1*l2*l3)

def get_velocity(position, path_points):
  max_acc = MAX_SPEED / 1.8
  max_velocity = MAX_SPEED
  v = np.zeros_like(position)
  if len(path_points) == 0:
    return v
  # Stop moving if the goal is reached.
  if np.linalg.norm(position - path_points[-1]) < .2:
    return v

  path_points = path_points[::3]

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
    lookahead = int(10 * max_velocity)
    a_points = path_points[min_point : min_point+lookahead]
    b_points = path_points[min_point+1 : min_point+lookahead]
    c_points = path_points[min_point+2 : min_point+lookahead]
    curvatures = [get_curvature(a,b,c) for a,b,c in zip(a_points, b_points, c_points)]
    curvature = sum(
      [sum(curvatures) / len(curvatures), 
      sum(curvatures[:int(lookahead/2)]) / len(curvatures[:int(lookahead/2)]), 
      sum(curvatures[:int(lookahead/4)]) / len(curvatures[:int(lookahead/4)])]
      ) / 3

    print(curvature)

    direction = path_points[min_point+1] #sum(path_points[min_point+1:min_point+4])/len(path_points[min_point+1:min_point+4])
    factor = max_velocity
   
    if curvature > 0.3:
      factor = np.sqrt(max_acc / curvature)

    v = factor * (direction - position) / np.linalg.norm(direction - position)

  if np.linalg.norm(v) > max_velocity:
    v *= max_velocity / np.linalg.norm(v)
  # Scale the velocity to have a magnitude of 0.2.
  return  v
