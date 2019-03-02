from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import re

import matplotlib.pylab as plt
import numpy as np
import scipy.signal
import yaml

# Constants used for indexing.
X = 0
Y = 1
YAW = 2

# Constants for occupancy grid.
FREE = 0
UNKNOWN = 1
OCCUPIED = 2

ROBOT_RADIUS = 0.105 / 2.
GOAL_POSITION = np.array([1, 1],
                         dtype=np.float32)  # Any orientation is good.
START_POSE = np.array([-1, -1, 0.], dtype=np.float32)
MAX_ITERATIONS = 500


def compute_weights(occupancy_grid_values, goal_index):
    weights = np.copy(occupancy_grid_values).astype(np.int64)
    grid_shape_x, grid_shape_y = occupancy_grid_values.shape

    neighbours = [[] for _ in range(grid_shape_x * grid_shape_y)]
    neighbours[0] = [goal_index]

    visited = np.zeros(occupancy_grid_values.shape)
    visited[goal_index[0], goal_index[1]] = 1

    i = np.int64(0)
    condition = True

    while condition:
        neighbours[i + 1] = []
        for n in neighbours[i]:
            x, y = n
            weights[x][y] = i + OCCUPIED + 1
            for a in [x - 1, x, x + 1]:
                for b in [y - 1, y, y + 1]:
                    if 0 <= a < grid_shape_x and 0 <= b < grid_shape_y:
                        if occupancy_grid_values[a][b] == 0 and visited[a][
                            b] == 0:
                            visited[a][b] = 1
                            neighbours[i + 1].append([a, b])
        if len(neighbours[i + 1]) == 0:
            condition = False
        else:
            i += 1

    return weights


def compute_path_second(occupancy_grid_values, start_index, goal_index):
    weights = compute_weights(occupancy_grid_values, goal_index)
    visited = np.zeros(occupancy_grid_values.shape)
    path = []

    current_node = start_index
    visited[current_node[0]][current_node[1]] = 1
    path.append(current_node)
    while current_node != goal_index:
        min_d = np.iinfo(np.int32).max
        min_neigh = current_node
        x, y = current_node
        for a in [x - 1, x, x + 1]:
            for b in [y - 1, y, y + 1]:
                # print("got here")
                # print(a, b, weights[a][b], visited[a][b])

                if 0 <= a < len(weights) and 0 <= b < len(weights[0]) and \
                        weights[a][b] > OCCUPIED:
                    # print(a, b, weights[a, b], visited[a, b])
                    visited[a][b] = 1

                    if weights[a][b] <= min_d:
                        min_d = weights[a][b]
                        min_neigh = [a, b]
        if current_node == min_neigh:
            break
        current_node = min_neigh
        path.append(current_node)
        # print(current_node)

    return path[0::5]


# og = np.array([[0, 0, 1, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 1, 0], [0, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0]])
# print(og)
# print(compute_weights(og, [6, 3]))
# print(compute_path_second(og, [0, 0], [6, 3]))


# Defines an occupancy grid.
class OccupancyGrid(object):
    def __init__(self, values, origin, resolution):
        self._original_values = values.copy()
        self._values = values.copy()
        # Inflate obstacles (using a convolution).
        inflated_grid = np.zeros_like(values)
        inflated_grid[values == OCCUPIED] = 1.
        w = 2 * int(ROBOT_RADIUS / resolution) + 1
        inflated_grid = scipy.signal.convolve2d(inflated_grid, np.ones((w, w)),
                                                mode='same')
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
        plt.imshow(self._original_values.T, interpolation='none',
                   origin='lower',
                   extent=[self._origin[X],
                           self._origin[X] + self._values.shape[
                               0] * self._resolution,
                           self._origin[Y],
                           self._origin[Y] + self._values.shape[
                               1] * self._resolution])
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
        return np.array([i, j],
                        dtype=np.float32) * self._resolution + self._origin

    def is_occupied(self, position):
        return self._values[self.get_index(position)] == OCCUPIED

    def is_free(self, position):
        return self._values[self.get_index(position)] == FREE


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Uses RRT to reach the goal.')
    parser.add_argument('--map', action='store', default='map',
                        help='Which map to use.')
    args, unknown = parser.parse_known_args()

    # Load map.
    with open(args.map + '.yaml') as fp:
        data = yaml.load(fp)
    img = read_pgm(os.path.join(os.path.dirname(args.map), data['image']))
    occupancy_grid = np.empty_like(img, dtype=np.int8)
    occupancy_grid[:] = UNKNOWN
    occupancy_grid[img < .1] = OCCUPIED
    occupancy_grid[img > .9] = FREE
    # Transpose (undo ROS processing).
    occupancy_grid = occupancy_grid.T
    # Invert Y-axis.
    occupancy_grid = occupancy_grid[:, ::-1]

    if 'maps/circuit' in args.map:
        occupancy_grid[170, 144:170] = OCCUPIED
        GOAL_POSITION = np.array([-1., -2.],
                                 dtype=np.float32)  # Any orientation is good.
        START_POSE = np.array([-2.5, -2.5, np.pi / 2], dtype=np.float32)
    elif 'maps/square' in args.map:
        occupancy_grid[177, 160:180] = OCCUPIED
        GOAL_POSITION = np.array([-1., -1.5],
                                 dtype=np.float32)  # Any orientation is good.
        START_POSE = np.array([-1.5, -1.5, np.pi / 2], dtype=np.float32)

    occupancy_grid = OccupancyGrid(occupancy_grid, data['origin'],
                                   data['resolution'])

    fig, ax = plt.subplots()
    occupancy_grid.draw()

    p = compute_path_second(occupancy_grid.values,
                            occupancy_grid.get_index(START_POSE[:2]),
                            occupancy_grid.get_index(GOAL_POSITION))
    for x, y in p:
        n = occupancy_grid.get_position(x, y)
        plt.scatter(n[0], n[1], color='red')

    plt.axis('equal')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim([-.5 - 2., 2. + .5])
    plt.ylim([-.5 - 2., 2. + .5])
    plt.show()