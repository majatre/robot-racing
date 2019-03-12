from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pylab as plt
# from rrt import OccupancyGrid
from matplotlib.collections import LineCollection

X = 0
Y = 1

def get_area(a, b, c):
    return 1 / 2 * abs(
        (b[X] - a[X]) * (c[Y] - a[Y]) - (b[Y] - a[Y]) * (c[X] - a[X]))


def get_curvature(a, b, c):
    A = get_area(a, b, c)
    l1 = np.linalg.norm(b - a)
    l2 = np.linalg.norm(c - a)
    l3 = np.linalg.norm(c - b)
    return 4 * A / (l1 * l2 * l3)


def plot_path_curvature(path):
    l = len(path)
    print(l)
    curvatures = []
    line_length = []

    for p1, p2, p3 in zip(path[:-2], path[1:-1], path[2:]):
        curvatures.append(get_curvature(p1, p2, p3))
        if len(line_length) == 0:
            line_length.append(np.linalg.norm(p2 - p1) + np.linalg.norm(p3 - p2))
        else:
            ll = line_length[-1]
            line_length.append(ll + np.linalg.norm(p3 - p2))
    print(line_length)

    curvatures2 = []
    for i in range(len(curvatures)):
        curvatures2.append(sum(curvatures[i:i+26])/ len(curvatures[i:i+26]))

    #line_length = [(c1+c2)/2 for c1, c2 in zip(line_length, line_length[1:])]

    return line_length, curvatures2


def plot_curvatures(file_paths):
  #data_path = np.genfromtxt(file_path, delimiter=',')
  plt.style.use('ggplot')
  for fp in file_paths:
    data_path = np.genfromtxt(fp, delimiter=',')
    path = [np.array(x[:2], dtype=np.float32) for x in data_path]
    line_length, curvatures = plot_path_curvature(path)
    plt.plot(line_length, curvatures, color = "r" if "rrt" in fp else "b")

  plt.ylabel('Curvature [1/m]')
  plt.xlabel('Distance from the start of the track [m]')
  plt.show()

plot_curvatures(['sharp_turn_rrt_gazebo_race_path.txt', 'sharp_turn_wavefront_gazebo_race_path.txt'])


