from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import matplotlib.pylab as plt
import numpy as np

OUTER_WALL_OFFSET = 2.
INNER_WALL_OFFSET = 1.
INNER_WALLS = np.array([[[1., 1.], [1, -1]], [[1., -1.], [-1, -1]], [[-1., -1.], [-1., 1.]], [[-1., 1.], [1., 1.]]])
OUTER_WALLS = np.array([[[2., 2.], [2, -2]], [[2., -2.], [-2, -2]], [[-2., -2.], [-2., 2.]], [[-2., 2.], [2., 2.]]])
GOAL_POSITION = np.array([0, 1.5], dtype=np.float32)
START_POSITION = np.array([1.5, 1.5], dtype=np.float32)
MAX_SPEED = .5


def is_inside_walls(position, wall_offset):
    xp, yp = position
    if -wall_offset <= xp <= wall_offset and -wall_offset <= yp <= wall_offset:
        return True
    return False


def point_projection(line, point):
    x1, y1 = line[0]
    x2, y2 = line[1]
    x3, y3 = point

    px = x2 - x1
    py = y2 - y1

    d = px * px + py * py
    u = ((x3 - x1) * px + (y3 - y1) * py) / d
    x = x1 + u * px
    y = y1 + u * py

    return [x, y]


def get_velocity_to_reach_goal(position, walls):
    v = np.zeros(2, dtype=np.float32)

    for wall in walls:
        if is_inside_walls(position, INNER_WALL_OFFSET):
            v += np.array([0., 0.])
        elif is_inside_walls(position, OUTER_WALL_OFFSET - 0.2):
            d = point_projection(wall, position)
            distance = np.sqrt(d[0] ** 2 + d[1] ** 2)
            v[0] += d[1] / distance
            v[1] += - d[0] / distance

    v = np.array([min(MAX_SPEED, x) for x in v])
    return v


def get_velocity_to_avoid_obstacles(position, inner_walls):
    v = np.zeros(2, dtype=np.float32)
    # Compute the velocity field needed to avoid the obstacles
    # Both obstacle_positions and obstacle_radii are lists.
    outer_square_width = 0.2

    for wall in inner_walls:
        if is_inside_walls(position, INNER_WALL_OFFSET):
            v += np.array([0., 0.])
        elif is_inside_walls(position, INNER_WALL_OFFSET + outer_square_width):
            d = point_projection(wall, position)
            distance = np.sqrt(d[0] ** 2 + d[1] ** 2)
            v[0] += d[0] / distance
            v[1] += d[1] / distance

    v = np.array([min(MAX_SPEED, x) for x in v])
    return v


def get_velocity_to_avoid_walls(position, outer_walls):
    v = np.zeros(2, dtype=np.float32)
    # Compute the velocity field needed to avoid the obstacles
    # Both obstacle_positions and obstacle_radii are lists.
    outer_square_width = 0.2

    for wall in outer_walls:
        if is_inside_walls(position, INNER_WALL_OFFSET):
            v += np.array([0., 0.])
        elif not is_inside_walls(position, OUTER_WALL_OFFSET - outer_square_width):
            d = point_projection(wall, position)
            distance = np.sqrt(d[0] ** 2 + d[1] ** 2)
            v[0] += - d[0] / (distance ** 2)
            v[1] += - d[1] / (distance ** 2)

    v = np.array([min(MAX_SPEED, x) for x in v])
    return v


def normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-2:
        return np.zeros_like(v)
    return v / n


def cap(v, max_speed):
    n = np.linalg.norm(v)
    if n > max_speed:
        return v / n * max_speed
    return v


def get_velocity(position, mode='all'):
    if mode in ('goal', 'all'):
        v_goal = get_velocity_to_reach_goal(position, OUTER_WALLS)
    else:
        v_goal = np.zeros(2, dtype=np.float32)
    if mode in ('obstacle', 'all'):
        v_avoid = get_velocity_to_avoid_walls(
            position,
            OUTER_WALLS) + get_velocity_to_avoid_obstacles(position, INNER_WALLS)
    else:
        v_avoid = np.zeros(2, dtype=np.float32)
    v = v_goal + v_avoid
    return cap(v, max_speed=MAX_SPEED)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Runs obstacle avoidance with a potential field')
    parser.add_argument('--mode', action='store', default='all', help='Which velocity field to plot.',
                        choices=['obstacle', 'goal', 'all'])
    args, unknown = parser.parse_known_args()

    fig, ax = plt.subplots()
    # Plot field.
    X, Y = np.meshgrid(np.linspace(-OUTER_WALL_OFFSET, OUTER_WALL_OFFSET, 30),
                       np.linspace(-OUTER_WALL_OFFSET, OUTER_WALL_OFFSET, 30))
    U = np.zeros_like(X)
    V = np.zeros_like(X)
    for i in range(len(X)):
        for j in range(len(X[0])):
            velocity = get_velocity(np.array([X[i, j], Y[i, j]]), args.mode)
            U[i, j] = velocity[0]
            V[i, j] = velocity[1]
    plt.quiver(X, Y, U, V, units='width')

    # Plot environment.
    for i in range(len(INNER_WALLS)):
        plt.plot(INNER_WALLS[i][0], INNER_WALLS[i][1], color='black')
    for i in range(len(OUTER_WALLS)):
        plt.plot(OUTER_WALLS[i][0], OUTER_WALLS[i][1], color='black')

    # Plot a simple trajectory from the start position.
    # Uses Euler integration.
    dt = 0.01
    x = START_POSITION
    positions = [x]
    for t in np.arange(0., 50., dt):
        v = get_velocity(x, args.mode)
        x = x + v * dt
        positions.append(x)
    positions = np.array(positions)
    plt.plot(positions[:, 0], positions[:, 1], lw=2, c='r')

    plt.axis('equal')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim([-.5 - OUTER_WALL_OFFSET, OUTER_WALL_OFFSET + .5])
    plt.ylim([-.5 - OUTER_WALL_OFFSET, OUTER_WALL_OFFSET + .5])
    plt.show()
