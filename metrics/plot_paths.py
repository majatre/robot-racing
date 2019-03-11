
def plot_race():
  data_path = np.genfromtxt('gazebo_race_path.txt', delimiter=',')
  data_trajectory = np.genfromtxt('gazebo_race_trajectory.txt', delimiter=',')

  plt.figure()
  plt.plot(data_path[:, 0], data_path[:, 1], 'b', label='path')
  plt.plot(data_trajectory[:, 0], data_trajectory[:, 1], 'g', label='true')

  occ_grid.draw()
  plt.show()