from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pylab as plt
import os
import sys
import yaml
import re
# from rrt import OccupancyGrid
from matplotlib.collections import LineCollection

directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../wavefront')
sys.path.insert(0, directory)

try:
    import wavefront as wavefront
except ImportError:
    raise ImportError(
        'Unable to import wavefront.py. Make sure this file is in "{}"'.format(
            directory))

TIME_INTERVAL = 0.1


def velocity_histogram(file_paths):
    plt.style.use('ggplot')
    data = []
    labels = []
    for fp in file_paths:
        data_path = np.genfromtxt(fp, delimiter=',')
        velocities = [x[2] for x in data_path]
        data.append(velocities)
        labels.append("RRT" if "rrt" in fp else "Wavefront")
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
        labels.append("RRT" if "rrt" in fp else "Wavefront")
        plt.plot(range(len(velocities)), velocities, color="r" if "rrt" in fp else "b")
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
        labels.append("RRT" if "rrt" in fp else "Wavefront")
        plt.plot(times, accelerations)
    plt.xlabel('Time [s]')
    plt.ylabel('Acceleration [m/s' + r'$^2$' + ']')
    plt.title('Plot of velocities over time')
    plt.legend(labels)
    plt.show()


MAX_SPEEDS = [1.1, 1.3, 1.5, 1.7]
RESULTS_SHARP = {"wavefront_sharp_turn": [7.47, 7.1, 6.56, 6.43],
                 "rrt_sharp_turn": [7.51, 6.94, 6.27, 6.23]}
RESULTS_SMOOTH = {"wavefront_smooth_turn": [11.22, 10.1, 9.44, 9.23],
                  "rrt_smooth_turn": [11.2, 10.13, 9.45, 9.22]}
RESULTS_CIRCUIT = {"wavefront_circuit": [16.31, 15.4, 14.42, 13.93],
                   "rrt_circuit": [16.89, 15.43, 13.87, 12.97]}


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
    #plt.title('Time to finish the ' + title + 'track')
    plt.legend(labels)
    plt.show()

def max_speed_performance_on_subplots():
    max_speeds = MAX_SPEEDS
    res = [RESULTS_SHARP, RESULTS_SMOOTH, RESULTS_CIRCUIT]

    f, axarr = plt.subplots(3, sharex=True)
    for i, results in enumerate(res):
        #plt.style.use('ggplot')
        labels = []
        for k, v in results.items():
            keywords = k.split("_")
            title = ""
            for kw in keywords[1:]:
                title += kw + " "
            axarr[i].plot(max_speeds, v, color= "r" if "rrt" in keywords[0] else "b")
            labels.append(keywords[0])
        axarr[i].set_title('Time to finish the ' + title + 'track')
        axarr[i].set(ylabel='Time [s]')
    plt.xticks([1.1,1.3,1.5,1.7])
    plt.xlabel('Maximum speed [m/s]')
    axarr[0].legend(labels, loc="upper right")
    plt.show()



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


def get_occupancy_grid(map):
    with open('../rrt/maps/' + map + '.yaml') as fp:
        data = yaml.load(fp)
    img = read_pgm(os.path.join(os.path.dirname('../rrt/maps/' + map), data['image']))
    occupancy_grid = np.empty_like(img, dtype=np.int8)
    occupancy_grid[:] = wavefront.UNKNOWN
    occupancy_grid[img < .1] = wavefront.OCCUPIED
    occupancy_grid[img > .9] = wavefront.FREE
    print(len(occupancy_grid[img > .9]))
    # Transpose (undo ROS processing).
    occupancy_grid = occupancy_grid.T
    # Invert Y-axis.
    occupancy_grid = occupancy_grid[:, ::-1]

    # Invisible wall
    if 'circuit' in map:
        occupancy_grid[170, 144:170] = wavefront.OCCUPIED
        goal = np.array([-1., -2.3],
                                 dtype=np.float32)  # Any orientation is good.
        start = np.array([-2.5, -2.5, np.pi / 2], dtype=np.float32)
    elif 'square' in map:
        occupancy_grid[177, 160:180] = wavefront.OCCUPIED
        goal = np.array([-1., -1.5],
                                 dtype=np.float32)  # Any orientation is good.
        start = np.array([-1.5, -1.5, np.pi / 2], dtype=np.float32)
    elif 'sharp_turn' in map:
        goal = np.array([0.75, -1],
                                 dtype=np.float32)  # Any orientation is good.
        start = np.array([-0.3, -1, np.pi / 2], dtype=np.float32)
    elif 'smooth' in map:
        occupancy_grid[177, 160:180] = wavefront.OCCUPIED
        goal = np.array([-1., -1.5],
                                 dtype=np.float32)  # Any orientation is good.
        start = np.array([-1.5, -1.5, np.pi / 2], dtype=np.float32)

    occupancy_grid = wavefront.OccupancyGrid(occupancy_grid, data['origin'],
                                             data['resolution'])
    
    return occupancy_grid, start, goal


def trajectory_with_occupancy_grid(map, file_ptahs):
    occupancy_grid, start, goal = get_occupancy_grid(map)
    #plt.style.use('ggplot')
    labels = []
    for fp in file_ptahs:
        data_path = np.genfromtxt(fp, delimiter=',')
        plt.plot(data_path[:, 0], data_path[:, 1], color = "r" if "rrt" in fp else "b")
        labels.append("RRT" if "rrt" in fp else "Wavefront")
    #plt.title('Planned trajectories')
    plt.legend(labels)
    occupancy_grid.draw()
    plt.show()


#trajectory_with_occupancy_grid("sharp_turn", ["sharp_turn_rrt_gazebo_race_path.txt","sharp_turn_wavefront_gazebo_race_path.txt"])
#trajectory_with_occupancy_grid("smooth_turn", ["smooth_turn_rrt_gazebo_race_path.txt","smooth_turn_wavefront_gazebo_race_path.txt"])
#trajectory_with_occupancy_grid("circuit", ["circuit_wavefront_gazebo_race_path.txt","../rrt/paths/rrt_path_circuit3.txt"])
# max_speed_performance_on_subplots()
# max_speed_performance(MAX_SPEEDS, RESULTS_SHARP)
# max_speed_performance(MAX_SPEEDS, RESULTS_SMOOTH)
# max_speed_performance(MAX_SPEEDS, RESULTS_CIRCUIT)


# velocity_histogram(['sharp_turn_rrt_gazebo_race_trajectory.txt','sharp_turn_wavefront_gazebo_race_trajectory.txt'])
velocity_over_time(['sharp_turn_rrt_gazebo_race_trajectory.txt','sharp_turn_wavefront_gazebo_race_trajectory.txt'])
# acceleration_over_time(['sharp_turn_rrt_gazebo_race_trajectory.txt','sharp_turn_wavefront_gazebo_race_trajectory.txt'])

# velocity_histogram(['circuit_wavefront_gazebo_race_trajectory.txt'])
# velocity_over_time(
#     ['gazebo_race_trajectory.txt',
#      'circuit_wavefront_gazebo_race_trajectory.txt'])
# acceleration_over_time(
#     ['gazebo_race_trajectory.txt',
#      'circuit_wavefront_gazebo_race_trajectory.txt'])
