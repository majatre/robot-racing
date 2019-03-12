# Robot Racing


To run gazebo (choose tack out of gazebo_sharp_turn, gazebo_smooth_turn, gazebo_circuit):
```
roslaunch exercises gazebo_sharp_turn.launch 
```

Possibly needed:

```
cd ~/catkin_ws/
catkim_make
export GAZEBO_MODEL_PATH=~/catkin_ws/src/exercises/robot-racing/models/
```

## Path planning
![Path](./img/sharp.png)

![Path](./img/smooth.png)

![Path](./img/circuit.png)


## Steering control
Planned Wavefront path in grey, actual path in blue (colormap indicates velocity)
![Path](./img/blue.png)

Planned RRT* path in grey, actual path in blue (colormap indicates velocity)
![Path](./img/red.png)

## Results
![Path](./img/results.png)

Plots of curvatures, velocities and histogram of accelerations on the sharp turn track
![Path](./img/curvatures.png)
![Path](./img/velocities.png)
![Path](./img/acceleration.png)
