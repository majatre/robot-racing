from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pylab as plt
from rrt import OccupancyGrid

def plot_race(occ_grid):
  data_path = np.genfromtxt('/tmp/gazebo_race_path.txt', delimiter=',')
  data_trajectory = np.genfromtxt('/tmp/gazebo_race_trajectory.txt', delimiter=',')

  plt.figure()
  plt.plot(data_path[:, 0], data_path[:, 1], 'b', label='path')
  plt.plot(data_trajectory[:, 0], data_trajectory[:, 1], 'g', label='true')

  # if data.shape[1] == 6:
  #   plt.figure()
  #   error = np.linalg.norm(data[:, :2] - data[:, 3:5], axis=1)
  #   plt.plot(error, c='b', lw=2)
  #   plt.ylabel('Error [m]')
  #   plt.xlabel('Timestep')

  occ_grid.draw()

  plt.show()