from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pylab as plt
from rrt import OccupancyGrid
from matplotlib.collections import LineCollection

def plot_race(occ_grid):
  data_path = np.genfromtxt('/tmp/gazebo_race_path.txt', delimiter=',')
  data_trajectory = np.genfromtxt('/tmp/gazebo_race_trajectory.txt', delimiter=',')

  plt.figure()
  plt.plot(data_path[:, 0], data_path[:, 1], 'b', label='path')
  plt.plot(data_trajectory[:, 0], data_trajectory[:, 1], 'g', label='true')

  occ_grid.draw()
  plt.show()


def plot_velocity(occ_grid):
  data_path = np.genfromtxt('/tmp/gazebo_race_path.txt', delimiter=',')
  data_trajectory = np.genfromtxt('/tmp/gazebo_race_trajectory.txt', delimiter=',')

  fig, axs = plt.subplots()
  velocities = data_trajectory[:, 2]
  # Create a continuous norm to map from data points to colors
  norm = plt.Normalize(velocities.min(), velocities.max())
  points = np.array([data_trajectory[:, 0], data_trajectory[:, 1]]).T.reshape(-1, 1, 2)
  segments = np.concatenate([points[:-1], points[1:]], axis=1)
  lc = LineCollection(segments, cmap='viridis', norm=norm)
  # Set the values used for colormapping
  lc.set_array(velocities)
  lc.set_linewidth(2)
  line = axs.add_collection(lc)
  plt.colorbar(line, ax=axs)

  occ_grid.draw()

  plt.show()