from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import matplotlib.pylab as plt
import matplotlib.patches as patches
import numpy as np
import os
import re
import scipy.signal
import yaml
import math
import time

# Constants used for indexing.
X = 0
Y = 1
YAW = 2

# Constants for occupancy grid.
FREE = 0
UNKNOWN = 1
OCCUPIED = 2

ROBOT_RADIUS = 0.105 / 2.
GOAL_POSITION = np.array([1.5, 1.5], dtype=np.float32)  # Any orientation is good.
START_POSE = np.array([-1.5, -1.5, 0.], dtype=np.float32)
MAX_ITERATIONS = 500
SAMPLING = 'uniform'

def sample_random_position(occupancy_grid):
  # Sample a valid random position using Gaussian distribution.
  # The corresponding cell must be free in the occupancy grid.
  if SAMPLING=='uniform':
    i = np.floor(np.random.random() * occupancy_grid.values.shape[0])
    j = np.floor(np.random.random() * occupancy_grid.values.shape[1])
    position = occupancy_grid.get_position(i,j)
    while not occupancy_grid.is_free(position):
      i = np.floor(np.random.random() * occupancy_grid.values.shape[0])
      j = np.floor(np.random.random() * occupancy_grid.values.shape[1])
      position = occupancy_grid.get_position(i,j)
    
  elif SAMPLING=='gaussian':
    mean = GOAL_POSITION
    cov = [[2, 0], [0, 2]]
    position = np.random.multivariate_normal(mean, cov, 1)[0]
    while not occupancy_grid.is_free(position):
      position = np.random.multivariate_normal(mean, cov, 1)[0]
  
  elif SAMPLING=='ellipse':
    direction = GOAL_POSITION - START_POSE[0:2]
    r1 = np.linalg.norm(direction) * 0.75
    r2 = np.linalg.norm(direction) * 0.15
    centre = (GOAL_POSITION - START_POSE[0:2])/2
    angle = np.arctan2(-direction[1], direction[0])
    x,y = np.random.random(2) * 2.0 - 1.0
    x2 = x * r1 * np.cos(angle) + y * r2 * np.sin(angle)
    y2 = -x * r1 * np.sin(angle) + y * r2 * np.cos(angle)
    position = np.array([x2,y2] + centre)
    while not occupancy_grid.is_free(position):
      x,y = np.random.random(2) * 2.0 - 1.0
      x2 = x * r1 * np.cos(angle) + y * r2 * np.sin(angle)
      y2 = -x * r1 * np.sin(angle) + y * r2 * np.cos(angle)
      position = np.array([x2,y2] + centre)

  return position


def find_arc(node_a, node_b):
  def perpendicular(v):
    w = np.empty_like(v)
    w[X] = -v[Y]
    w[Y] = v[X]
    return w

  start = node_a.position
  perp_start = perpendicular(node_a.direction)
  middle = (node_b.position + node_a.position) / 2.
  perp_middle = perpendicular(node_b.position - node_a.position)

  #  perp_middle * t + middle = perp_start* s + start
  #  perp_middle * t - perp_start* s = start -middle
  t, s = np.linalg.solve(np.array([perp_middle, -perp_start]).T, start - middle)
  center = perp_start * s + start
  radius = np.linalg.norm(center - start)
  return center, radius



def adjust_pose(node, final_position, occupancy_grid):
  # Check whether there exists a simple path that links node.pose
  # to final_position. This function needs to return a new node that has
  # the same position as final_position and a valid yaw. The yaw is such that
  # there exists an arc of a circle that passes through node.pose and the
  # adjusted final pose. If no such arc exists (e.g., collision) return None.
  # Assume that the robot always goes forward.
  # Feel free to use the find_circle() function below.
  final_pose = node.pose.copy()
  final_pose[:2] = final_position
  final_node = Node(final_pose)

  center, radius = find_arc(node, final_node)
  t1 = np.arctan2(center[Y] - node.pose[Y], center[X] - node.pose[X])
  t2 = np.arctan2(center[Y] - final_pose[Y], center[X] - final_pose[X]) 

  # Get the final orientation by rotating the start orientation by the angle
  # difference of radii
  final_pose[2] = t2 - t1 + node.pose[2]
  final_node = Node(final_pose)

  if check_collisions(node, final_position, center, radius, occupancy_grid):
    return None
  else:
    return final_node

def check_collisions(node, final_position, center, radius, occupancy_grid):
  distance = occupancy_grid.resolution
  offset = 0.
  points_x = []
  points_y = []
  
  du = node.pose[0:2] - center
  theta1 = np.arctan2(du[1], du[0])
  dv = final_position - center
  theta2 = np.arctan2(dv[1], dv[0])
  # Check if the arc goes clockwise.
  clockwise = np.cross(node.direction, du).item() > 0.
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

  for x, y in zip(points_x, points_y):
    if occupancy_grid.is_occupied(np.array([x,y])):
      return True
  return False


def get_path(final_node):
  # Construct path from RRT solution.
  if final_node is None:
    return []
  path_reversed = []
  path_reversed.append(final_node)
  while path_reversed[-1].parent is not None:
    path_reversed.append(path_reversed[-1].parent)
  path = list(reversed(path_reversed))
  return path

def get_path_length(path):
  # Calculate length of the path madeout of arcs of circles.
  path_len = 0
  for u, v in zip(path, path[1:]):
    path_len += get_arc_length(u, v)
  return path_len

def get_arc_length(u, v):
  # Calculate length of an arc of circle.
  center, radius = find_circle(u, v)
  du = u.position - center
  theta1 = np.arctan2(du[1], du[0])
  dv = v.position - center
  theta2 = np.arctan2(dv[1], dv[0])
  # Check if the arc goes clockwise.
  clockwise = np.cross(u.direction, du).item() > 0.
  if clockwise:
    if theta2 > theta1:
      theta1 += 2. * np.pi
    arc_angle = theta1 - theta2
  else:
    if theta2 < theta1:
      theta2 += 2. * np.pi
    arc_angle = theta2 - theta1
  return arc_angle * radius


# Defines an occupancy grid.
class OccupancyGrid(object):
  def __init__(self, values, origin, resolution):
    self._original_values = values.copy()
    self._values = values.copy()
    # Inflate obstacles (using a convolution).
    inflated_grid = np.zeros_like(values)
    inflated_grid[values == OCCUPIED] = 1.
    w = 2 * int(ROBOT_RADIUS / resolution) + 1
    inflated_grid = scipy.signal.convolve2d(inflated_grid, np.ones((w, w)), mode='same')
    self._values[inflated_grid > 0.] = OCCUPIED
    self._origin = np.array(origin[:2], dtype=np.float32)
    self._origin -= resolution / 2.
    assert origin[YAW] == 0.
    self._resolution = resolution

  @property
  def values(self):
    return self._values

  @property
  def resolution(self):
    return self._resolution

  @property
  def origin(self):
    return self._origin

  def draw(self):
    plt.imshow(self._original_values.T, interpolation='none', origin='lower',
               extent=[self._origin[X],
                       self._origin[X] + self._values.shape[0] * self._resolution,
                       self._origin[Y],
                       self._origin[Y] + self._values.shape[1] * self._resolution])
    plt.set_cmap('gray_r')

  def get_index(self, position):
    idx = ((position - self._origin) / self._resolution).astype(np.int32)
    if len(idx.shape) == 2:
      idx[:, 0] = np.clip(idx[:, 0], 0, self._values.shape[0] - 1)
      idx[:, 1] = np.clip(idx[:, 1], 0, self._values.shape[1] - 1)
      return (idx[:, 0], idx[:, 1])
    idx[0] = np.clip(idx[0], 0, self._values.shape[0] - 1)
    idx[1] = np.clip(idx[1], 0, self._values.shape[1] - 1)
    return tuple(idx)

  def get_position(self, i, j):
    return np.array([i, j], dtype=np.float32) * self._resolution + self._origin

  def is_occupied(self, position):
    return self._values[self.get_index(position)] == OCCUPIED

  def is_free(self, position):
    return self._values[self.get_index(position)] == FREE


# Defines a node of the graph.
class Node(object):
  def __init__(self, pose):
    self._pose = pose.copy()
    self._neighbors = []
    self._parent = None
    self._cost = 0.

  @property
  def pose(self):
    return self._pose

  def add_neighbor(self, node):
    self._neighbors.append(node)

  @property
  def parent(self):
    return self._parent

  @parent.setter
  def parent(self, node):
    self._parent = node

  @property
  def neighbors(self):
    return self._neighbors

  @property
  def position(self):
    return self._pose[:2]

  @property
  def yaw(self):
    return self._pose[YAW]
  
  @property
  def direction(self):
    return np.array([np.cos(self._pose[YAW]), np.sin(self._pose[YAW])], dtype=np.float32)

  @property
  def cost(self):
      return self._cost

  @cost.setter
  def cost(self, c):
    self._cost = c

def calculate_cost(parent, position, occupancy_grid):
  if parent.cost == 0:
    parent.cost = get_path_length(get_path(parent))

  center, radius = find_arc(parent, Node(position))
  final_pose = parent.pose.copy()
  final_pose[:2] = position

  t1 = math.atan2(center[Y] - parent.pose[Y], center[X] - parent.pose[X])
  t2 = math.atan2(center[Y] - position[Y], center[X] - position[X]) 
  final_pose[2] = t2 - t1 + parent.pose[2]
  v = Node(final_pose)

  arc_len = get_arc_length(parent, v)

  if v is None:
    return float("inf")

  v.parent = parent
  v.cost = parent.cost + arc_len
  return v.cost

def filter_distance(n, position):
  # Pick a node at least some distance away but not too far.
  # We also verify that the angles are aligned (within pi / 4).
  d =  np.linalg.norm(position - n.position)
  return d > .2 and d < 1.5 and n.direction.dot(position - n.position) / d > 0.70710678118


def rrt(start_pose, goal_position, occupancy_grid):
  # RRT builds a graph one node at a time.
  graph = []
  start_node = Node(start_pose)
  final_node = None

  if not occupancy_grid.is_free(goal_position):
    print('Goal position is not in the free space.')
    return start_node, final_node
  graph.append(start_node)
  for _ in range(MAX_ITERATIONS): 
    position = sample_random_position(occupancy_grid)
    #print(position)
    # With a random chance, draw the goal position.
    if np.random.rand() < .05:
      position = goal_position
    # Find best parent node in the graph.
    parents_to_check = filter(lambda n: 
        np.linalg.norm(position - n.position) > .2 and 
        np.linalg.norm(position - n.position) < 1.5 and 
        n.direction.dot(position - n.position) / np.linalg.norm(position - n.position) > 0.70710678118, 
      graph)
    potential_parent = sorted(
      ((n, calculate_cost(n, position, occupancy_grid)) for n in parents_to_check), key=lambda x: x[1])
    
    if len(potential_parent) > 0:
      u = potential_parent[0][0]
    else:
      u = None
      continue
    v = adjust_pose(u, position, occupancy_grid)
    if v is None:
      continue
    u.add_neighbor(v)
    v.parent = u
    graph.append(v)
    if np.linalg.norm(v.position - goal_position) < .2:
      final_node = v
      break
  return start_node, final_node


def find_circle(node_a, node_b):
  def perpendicular(v):
    w = np.empty_like(v)
    w[X] = -v[Y]
    w[Y] = v[X]
    return w
  db = perpendicular(node_b.direction)
  dp = node_a.position - node_b.position
  t = np.dot(node_a.direction, db)
  if np.abs(t) < 1e-3:
    # By construction node_a and node_b should be far enough apart,
    # so they must be on opposite end of the circle.
    center = (node_b.position + node_a.position) / 2.
    radius = np.linalg.norm(center - node_b.position)
  else:
    radius = np.dot(node_a.direction, dp) / t
    center = radius * db + node_b.position
  return center, np.abs(radius)


def read_pgm(filename, byteorder='>'):
  """Read PGM file."""
  with open(filename, 'rb') as fp:
    buf = fp.read()
  try:
    header, width, height, maxval = re.search(
        b'(^P5\s(?:\s*#.*[\r\n])*'
        b'(\d+)\s(?:\s*#.*[\r\n])*'
        b'(\d+)\s(?:\s*#.*[\r\n])*'
        b'(\d+)\s(?:\s*#.*[\r\n]\s)*)', buf).groups()
  except AttributeError:
    raise ValueError('Invalid PGM file: "{}"'.format(filename))
  maxval = int(maxval)
  height = int(height)
  width = int(width)
  img = np.frombuffer(buf,
                      dtype='u1' if maxval < 256 else byteorder + 'u2',
                      count=width * height,
                      offset=len(header)).reshape((height, width))
  return img.astype(np.float32) / 255.


def draw_solution(start_node, final_node=None):
  ax = plt.gca()

  def draw_path(u, v, arrow_length=.1, color=(.8, .8, .8), lw=1):
    du = u.direction
    plt.arrow(u.pose[X], u.pose[Y], du[0] * arrow_length, du[1] * arrow_length,
              head_width=.05, head_length=.1, fc=color, ec=color)
    dv = v.direction
    plt.arrow(v.pose[X], v.pose[Y], dv[0] * arrow_length, dv[1] * arrow_length,
              head_width=.05, head_length=.1, fc=color, ec=color)
    center, radius = find_circle(u, v)
    du = u.position - center
    theta1 = np.arctan2(du[1], du[0])
    dv = v.position - center
    theta2 = np.arctan2(dv[1], dv[0])
    # Check if the arc goes clockwise.
    if np.cross(u.direction, du).item() > 0.:
      theta1, theta2 = theta2, theta1
    ax.add_patch(patches.Arc(center, radius * 2., radius * 2.,
                             theta1=theta1 / np.pi * 180., theta2=theta2 / np.pi * 180.,
                             color=color, lw=lw))

  points = []
  s = [(start_node, None)]  # (node, parent).
  while s:
    v, u = s.pop()
    if hasattr(v, 'visited'):
      continue
    v.visited = True
    # Draw path from u to v.
    if u is not None:
      draw_path(u, v)
    points.append(v.pose[:2])
    for w in v.neighbors:
      s.append((w, v))

  points = np.array(points)
  plt.scatter(points[:, 0], points[:, 1], s=10, marker='o', color=(.8, .8, .8))
  if final_node is not None:
    plt.scatter(final_node.position[0], final_node.position[1], s=10, marker='o', color='k')
    # Draw final path.
    v = final_node
    while v.parent is not None:
      draw_path(v.parent, v, color='k', lw=2)
      v = v.parent


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Uses RRT to reach the goal.')
  parser.add_argument('--map', action='store', default='map', help='Which map to use.')
  args, unknown = parser.parse_known_args()

  # Load map.
  with open(args.map + '.yaml') as fp:
    data = yaml.load(fp)
  img = read_pgm(os.path.join(os.path.dirname(args.map), data['image']))
  occupancy_grid = np.empty_like(img, dtype=np.int8)
  occupancy_grid[:] = UNKNOWN
  occupancy_grid[img < .1] = OCCUPIED
  occupancy_grid[img > .9] = FREE
  print(len(occupancy_grid[img > .9]))
  # Transpose (undo ROS processing).
  occupancy_grid = occupancy_grid.T
  # Invert Y-axis.
  occupancy_grid = occupancy_grid[:, ::-1]
  occupancy_grid = OccupancyGrid(occupancy_grid, data['origin'], data['resolution'])

  # Run RRT.
  start_node, final_node = rrt(START_POSE, GOAL_POSITION, occupancy_grid)
  path = get_path(final_node)
  print(get_path_length(path))

  # timing = []
  # path_lengths = []

  # for i in range(30):
  #   start = time.time()
  #   start_node, final_node = rrt(START_POSE, GOAL_POSITION, occupancy_grid)
  #   end = time.time()
  #   path_len = get_path_length(get_path(final_node))
  #   if path_len > 0:
  #     path_lengths.append(path_len)
  #     timing.append(end - start)

  # print(path_lengths)
  # print(timing)

  # print('Avg. length:', sum(path_lengths)/len(path_lengths))
  # print('Avg. time:', sum(timing)/len(timing))

  # Plot environment.
  fig, ax = plt.subplots()
  occupancy_grid.draw()
  plt.scatter(.3, .2, s=10, marker='o', color='green', zorder=1000)
  draw_solution(start_node, final_node)
  plt.scatter(START_POSE[0], START_POSE[1], s=10, marker='o', color='green', zorder=1000)
  plt.scatter(GOAL_POSITION[0], GOAL_POSITION[1], s=10, marker='o', color='red', zorder=1000)
  
  plt.axis('equal')
  plt.xlabel('x')
  plt.ylabel('y')
  plt.xlim([-.5 - 2., 2. + .5])
  plt.ylim([-.5 - 2., 2. + .5])
  plt.show()
  