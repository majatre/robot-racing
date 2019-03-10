from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pylab as plt
# from rrt import OccupancyGrid
from matplotlib.collections import LineCollection

TIME_INTERVAL = 0.1


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


MAX_SPEEDS = [1.1, 1.3, 1.5, 1.7]
RESULTS_SHARP = {"wavefront_sharp_turn": [7.47, 7.1, 6.56, 6.43], "rrt_sharp_turn": [7.51, 6.94, 6.27, 6.23]}
RESULTS_SMOOTH = {"wavefront_smooth_turn": [11.22, 10.1, 9.44, 9.23], "rrt_smooth_turn": [11.2, 10.13, 9.45, 9.22]}
RESULTS_CIRCUIT = {"wavefront_circuit": [16.31, 15.4, 14.42, 13.93], "rrt_circuit": [16.89, 15.43, 13.87, 12.97]}


def max_speed_performance(max_speeds, results):
    plt.style.use('ggplot')
    labels = []
    for k, v in results.items():
        keywords = k.split("_")
        title = ""
        for kw in keywords[1:]:
            title += kw + " "
        plt.plot(max_speeds, v)
        labels.append(keywords[0])
    plt.xlabel('Maximum speed [m/s]')
    plt.ylabel('Time to finish the track [s]')
    plt.title('Time to finish the ' + title + 'track')
    plt.legend(labels)
    plt.show()



max_speed_performance(MAX_SPEEDS, RESULTS_SHARP)
max_speed_performance(MAX_SPEEDS, RESULTS_SMOOTH)
max_speed_performance(MAX_SPEEDS, RESULTS_CIRCUIT)


# velocity_histogram(['gazebo_race_trajectory.txt',
#                     'circuit_wavefront_gazebo_race_trajectory.txt'])
# velocity_over_time(
#     ['gazebo_race_trajectory.txt',
#      'circuit_wavefront_gazebo_race_trajectory.txt'])
# acceleration_over_time(
#     ['gazebo_race_trajectory.txt',
#      'circuit_wavefront_gazebo_race_trajectory.txt'])
