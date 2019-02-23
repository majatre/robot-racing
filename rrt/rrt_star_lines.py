from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import copy

import matplotlib.pylab as plt
import matplotlib.patches as patches
import numpy as np
import os
import re
import scipy.signal
import yaml
import math
import time

from cubic_spline_planner import Spline2D

X = 0
Y = 1
YAW = 2

# Constants for occupancy grid.
FREE = 0
UNKNOWN = 1
OCCUPIED = 2

ROBOT_RADIUS = 0.105 / 2.
GOAL_POSITION = np.array([-1., -2.5], dtype=np.float32)  # Any orientation is good.
START_POSE = np.array([-2.5, -2.5, np.pi / 2], dtype=np.float32)
MAX_ITERATIONS = 1000
SAMPLING = 'uniform'


def sample_random_position(occupancy_grid):
    # Sample a valid random position using Gaussian distribution.
    # The corresponding cell must be free in the occupancy grid.
    if SAMPLING == 'uniform':
        i = np.floor(np.random.random() * occupancy_grid.values.shape[0])
        j = np.floor(np.random.random() * occupancy_grid.values.shape[1])
        position = occupancy_grid.get_position(i, j)
        while not occupancy_grid.is_free(position):
            i = np.floor(np.random.random() * occupancy_grid.values.shape[0])
            j = np.floor(np.random.random() * occupancy_grid.values.shape[1])
            position = occupancy_grid.get_position(i, j)

    elif SAMPLING == 'gaussian':
        mean = GOAL_POSITION
        cov = [[2, 0], [0, 2]]
        position = np.random.multivariate_normal(mean, cov, 1)[0]
        while not occupancy_grid.is_free(position):
            position = np.random.multivariate_normal(mean, cov, 1)[0]

    elif SAMPLING == 'ellipse':
        direction = GOAL_POSITION - START_POSE[0:2]
        r1 = np.linalg.norm(direction) * 0.75
        r2 = np.linalg.norm(direction) * 0.15
        centre = (GOAL_POSITION - START_POSE[0:2]) / 2
        angle = np.arctan2(-direction[1], direction[0])
        x, y = np.random.random(2) * 2.0 - 1.0
        x2 = x * r1 * np.cos(angle) + y * r2 * np.sin(angle)
        y2 = -x * r1 * np.sin(angle) + y * r2 * np.cos(angle)
        position = np.array([x2, y2] + centre)
        while not occupancy_grid.is_free(position):
            x, y = np.random.random(2) * 2.0 - 1.0
            x2 = x * r1 * np.cos(angle) + y * r2 * np.sin(angle)
            y2 = -x * r1 * np.sin(angle) + y * r2 * np.cos(angle)
            position = np.array([x2, y2] + centre)

    return position


def find_path(node, parent):
    theta = math.atan2(node.pose[Y] - parent.pose[Y], node.pose[X] - parent.pose[X])
    newNode = Node(node.pose)
    d = np.linalg.norm(node.position - parent.position)

    newNode.cost = parent.cost + d
    newNode.pose[YAW] = theta
    newNode.parent = parent

    return newNode


def adjust_pose(node, final_position, occupancy_grid):
    # Check whether there exists a simple path that links node.pose
    # to final_position.
    final_pose = node.pose.copy()
    final_pose[:2] = final_position
    final_node = Node(final_pose)

    theta = math.atan2(final_position[Y] - node.pose[Y], final_position[X] - node.pose[X])
    d = np.linalg.norm(final_position - node.position)

    final_node.pose[YAW] = theta
    final_node.cost = node.cost + d

    if check_collisions(node, final_position, occupancy_grid):
        return None
    else:
        return final_node


def check_collisions(node, final_position, occupancy_grid):
  d = np.linalg.norm(final_position - node.position)
  res = 0.05
  x = node.pose[X]
  y = node.pose[Y]

  theta = math.atan2(final_position[Y] - node.pose[Y], final_position[X] - node.pose[X])
  for i in range(int(d / res)):
    x += res * math.cos(theta)
    y += res * math.sin(theta)
    if occupancy_grid.is_occupied(np.array([x, y])):
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
        path_len += np.linalg.norm(u.position - v.position)
    return path_len


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
        self.path_x = []
        self.path_y = []
        self.path_yaw = []

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
    # if parent.cost == 0:
    #   parent.cost = get_path_length(get_path(parent))

    d = np.linalg.norm(position - parent.position)

    if check_collisions(parent, position, occupancy_grid):
        return float("inf")
    else:
      return parent.cost + d


def find_near_nodes(graph, newNode):
    nnode = len(graph)
    r = 4.0 * math.sqrt((math.log(nnode) / nnode))
    dlist = [(node.pose[X] - newNode.pose[X]) ** 2 +
             (node.pose[Y] - newNode.pose[Y]) ** 2 for node in graph]
    nearinds = [dlist.index(i) for i in dlist if i <= r ** 2]
    return nearinds


def rewire(newNode, nearinds, graph, occupancy_grid):
    for i in nearinds:
        nearNode = graph[i]
        tNode = find_path(nearNode, newNode)
        if nearNode.cost > tNode.cost:
            if not check_collisions(nearNode, newNode.position, occupancy_grid):
              nearNode.parent = tNode.parent
              nearNode.cost = tNode.cost


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
        # With a random chance, draw the goal position.
        if np.random.rand() < .05:
            position = goal_position
        # Find best parent node in the graph.
        parents_to_check = filter(lambda n:
                                  np.linalg.norm(position - n.position) > .1 and
                                  np.linalg.norm(position - n.position) < 1.5,
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
        nearinds = find_near_nodes(graph, v)
        graph.append(v)
        rewire(v, nearinds, graph, occupancy_grid)
        if np.linalg.norm(v.position - goal_position) < .2:
            final_node = v
            break
    print('Done')
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
        plt.plot([u.pose[X], v.pose[X]], [u.pose[Y], v.pose[Y]], color=color)

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


def get_target_point(path, targetL):
  le = 0
  ti = 0
  lastPairLen = 0
  for i in range(len(path) - 1):
    dx = path[i + 1].pose[0] - path[i].pose[0]
    dy = path[i + 1].pose[1] - path[i].pose[1]
    d = math.sqrt(dx * dx + dy * dy)
    le += d
    if le >= targetL:
      ti = i - 1
      lastPairLen = d
      break

  partRatio = (le - targetL) / lastPairLen
  #  print(partRatio)
  #  print((ti,len(path),path[ti],path[ti+1]))

  x = path[ti].pose[0] + (path[ti + 1].pose[0] - path[ti].pose[0]) * partRatio
  y = path[ti].pose[1] + (path[ti + 1].pose[1] - path[ti].pose[1]) * partRatio
  #  print((x,y))

  return [x, y, ti]


def path_smoothing(path, maxIter, occupancy_grid):
  #  print("PathSmoothing")

  le = get_path_length(path)

  for i in range(maxIter):
    # Sample two points
    pickPoints = [np.random.uniform(0, le), np.random.uniform(0, le)]
    pickPoints.sort()
    #  print(pickPoints)
    first = get_target_point(path, pickPoints[0])
    #  print(first)
    second = get_target_point(path, pickPoints[1])
    #  print(second)

    if first[2] <= 0 or second[2] <= 0:
      continue

    if (second[2] + 1) > len(path):
      continue

    if second[2] == first[2]:
      continue

    # collision check
    if check_collisions(Node(np.array(first[0:2])), np.array(second[0:2]), occupancy_grid):
      continue

    # Create New path
    newPath = []
    newPath.extend(path[:first[2] + 1])
    newPath.append(Node(np.array([first[0], first[1]])))
    newPath.append(Node(np.array([second[0], second[1]])))
    newPath.extend(path[second[2] + 1:])
    path = newPath
    le = get_path_length(path)

  return path


def plan_cubic_splines(x, y):
    print("Spline 2D test")
    import matplotlib.pyplot as plt
    ds = 0.1  # [m] distance of each intepolated points

    sp = Spline2D(x, y)
    s = np.arange(0, sp.s[-1], ds)

    rx, ry, ryaw, rk = [], [], [], []
    for i_s in s:
        ix, iy = sp.calc_position(i_s)
        rx.append(ix)
        ry.append(iy)
        ryaw.append(sp.calc_yaw(i_s))
        rk.append(sp.calc_curvature(i_s))

    # plt.subplots(1)
    # plt.plot(x, y, "xb", label="input")
    # plt.plot(rx, ry, "-r", label="spline")
    # plt.grid(True)
    # plt.axis("equal")
    # plt.xlabel("x[m]")
    # plt.ylabel("y[m]")
    # plt.legend()

    # plt.subplots(1)
    # plt.plot(s, [np.rad2deg(iyaw) for iyaw in ryaw], "-r", label="yaw")
    # plt.grid(True)
    # plt.legend()
    # plt.xlabel("line length[m]")
    # plt.ylabel("yaw angle[deg]")
    #
    plt.subplots(1)
    plt.plot(s, rk, "-r", label="curvature")
    plt.grid(True)
    plt.legend()
    plt.xlabel("line length[m]")
    plt.ylabel("curvature [1/m]")

    #plt.show()
    return x,y,rx,ry

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

    #Invisible wall
    if args.map == 'maps/circuit':
        occupancy_grid[170, 144:170] = OCCUPIED
        GOAL_POSITION = np.array([-1., -2.5], dtype=np.float32)  # Any orientation is good.
        START_POSE = np.array([-2.5, -2.5, np.pi / 2], dtype=np.float32)
    elif args.map == 'maps/square':
        occupancy_grid[177, 160:180] = OCCUPIED
        GOAL_POSITION = np.array([-1., -1.5], dtype=np.float32)  # Any orientation is good.
        START_POSE = np.array([-1.5, -1.5, np.pi / 2], dtype=np.float32)

    occupancy_grid = OccupancyGrid(occupancy_grid, data['origin'], data['resolution'])

    # x = -1.13
    # for y in np.arange(-2., -1., 0.05):
    #   print('grid', occupancy_grid.get_index(np.array([x, y])))

    # Run RRT.
    start_node, final_node = rrt(START_POSE, GOAL_POSITION, occupancy_grid)
    path = get_path(final_node)
    print(get_path_length(path))

    splines_x = [n.position[X] for n in path]
    splines_y = [n.position[Y] for n in path]

    x,y,rx,ry = plan_cubic_splines(splines_x, splines_y)


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
    # plt.scatter(.3, .2, s=10, marker='o', color='green', zorder=1000)
    draw_solution(start_node, final_node)
    plt.scatter(START_POSE[0], START_POSE[1], s=10, marker='o', color='green', zorder=1000)
    plt.scatter(GOAL_POSITION[0], GOAL_POSITION[1], s=10, marker='o', color='red', zorder=1000)

    plt.plot(x, y, "xb", label="input")
    plt.plot(rx, ry, "-r", label="spline")

    # new_path = path_smoothing(path, 500, occupancy_grid)
    # # for node in new_path:
    # #   plt.scatter(node.pose[0], node.pose[1], s=10, marker='o', color='blue', zorder=1000)
    #
    # splines_x = [n.position[X] for n in new_path]
    # splines_y = [n.position[Y] for n in new_path]
    #
    # x, y, rx, ry = plan_cubic_splines(splines_x, splines_y)
    # plt.plot(x, y, "xg", label="smoothed_input")
    # plt.plot(rx, ry, "-g", label="smoothed_spline")

    plt.axis('equal')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim([- 3., 3.])
    plt.ylim([- 3., 3.])
    plt.show()
