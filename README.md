# Robot Racing


To run gazebo (choose tack out of gazebo_sharp_turn, gazebo_smooth_turn, gazebo_zircuit):
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

![Path](./img/blue.png)
![Path](./img/red.png)

## Results
![Path](./img/results.png)
