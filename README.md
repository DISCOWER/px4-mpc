# px4-mpc
This package contains an MPC using the casadi integrated with with PX4

![mpc_setpoint](https://github.com/Jaeyoung-Lim/px4-mpc/assets/5248102/35dae5bf-626e-4272-a552-5f5d7e3c04cd)

## Setup
To build the code, clone this repository into a ros2 workspace
Dependencies
- [px4_msgs](https://github.com/PX4/px4_msgs/pull/15)
- [px4-offboard](https://github.com/Jaeyoung-Lim/px4-offboard) (Optional): Used for RViz visualization

```
colcon build
```

### Testing demos
```
ros2 run px4_mpc quadrotor_demo
```
```
ros2 run px4_mpc astrobee_demo

```
### Running MPC with PX4 SITL
Run PX4 SITL
```
make px4_sitl gazebo
```

Run the micro-ros-agent
```
micro-ros-agent udp4 --port 8888
```

In order to launch the mpc quadrotor in a ros2 launchfile,
```
ros2 launch px4_mpc mpc_quadrotor_launch.py 
```
