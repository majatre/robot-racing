from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pylab as plt
# from rrt import OccupancyGrid
from matplotlib.collections import LineCollection

TIME_INTERVAL = 0.1
# def plot_race(occ_grid):
#   data_path = np.genfromtxt('/tmp/gazebo_race_path.txt', delimiter=',')
#   data_trajectory = np.genfromtxt('/tmp/gazebo_race_trajectory.txt', delimiter=',')

#   plt.figure()
#   plt.plot(data_path[:, 0], data_path[:, 1], 'b', label='path')
#   plt.plot(data_trajectory[:, 0], data_trajectory[:, 1], 'g', label='true')

#   occ_grid.draw()
#   plt.show()


# def plot_velocity(occ_grid):
#   data_path = np.genfromtxt('/tmp/gazebo_race_path.txt', delimiter=',')
#   data_trajectory = np.genfromtxt('/tmp/gazebo_race_trajectory.txt', delimiter=',')

#   fig, axs = plt.subplots()
#   velocities = data_trajectory[:, 2]
#   # Create a continuous norm to map from data points to colors
#   norm = plt.Normalize(velocities.min(), velocities.max())
#   points = np.array([data_trajectory[:, 0], data_trajectory[:, 1]]).T.reshape(-1, 1, 2)
#   segments = np.concatenate([points[:-1], points[1:]], axis=1)
#   lc = LineCollection(segments, cmap='Greens', norm=norm)
#   # Set the values used for colormapping
#   lc.set_array(velocities)
#   lc.set_linewidth(2)
#   line = axs.add_collection(lc)
#   plt.colorbar(line, ax=axs)

#   occ_grid.draw()

#   plt.show()


def velocity_histogram(file_paths):
    plt.style.use('ggplot')
    data = []
    labels = []
    for fp in file_paths:
        data_path = np.genfromtxt(fp, delimiter=',')
        velocities = [x[2] for x in data_path]
        data.append(velocities)
        labels.append(fp)
    plt.hist(data, 25)
    plt.xlabel('Velocity Buckets [m/s]')
    plt.ylabel('Frequency [100 ms]')
    plt.title('Histogram of velocity frequencies')
    plt.legend(labels)
    plt.show()


def velocity_over_time(file_paths):
    plt.style.use('ggplot')
    labels = []
    for fp in file_paths:
        data_path = np.genfromtxt(fp, delimiter=',')
        velocities = [x[2] for x in data_path]
        labels.append(fp)
        plt.plot(range(len(velocities)), velocities)
    plt.xlabel('Time [s]')
    plt.ylabel('Velocity [m/s]')
    plt.title('Plot of velocities over time')
    plt.legend(labels)
    plt.show()


def acceleration_over_time(file_paths):
    time_interv = TIME_INTERVAL * 5
    plt.style.use('ggplot')
    labels = []
    for fp in file_paths:
        data_path = np.genfromtxt(fp, delimiter=',')
        accelerations = []
        prev_x, prev_y, prev_v = data_path[0]
        data_path = data_path[0::20]
        for x, y, v in data_path:
            acc = (v - prev_v) / time_interv
            prev_v = v
            accelerations.append(acc)
        times = [x * time_interv for x in range(len(accelerations))]
        labels.append(fp)
        plt.plot(times, accelerations)
    plt.xlabel('Time [s]')
    plt.ylabel('Acceleration [m/s' + r'$^2$' + ']')
    plt.title('Plot of velocities over time')
    plt.legend(labels)
    plt.show()


velocity_histogram(['sharp_turn_rrt_gazebo_race_trajectory.txt','sharp_turn_wavefront_gazebo_race_trajectory.txt'])
velocity_over_time(['sharp_turn_rrt_gazebo_race_trajectory.txt','sharp_turn_wavefront_gazebo_race_trajectory.txt'])
acceleration_over_time(['sharp_turn_rrt_gazebo_race_trajectory.txt','sharp_turn_wavefront_gazebo_race_trajectory.txt'])