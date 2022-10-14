# fixedwing-mpc
This package contains an MPC example using the casadi with PX4

## Testing demos
```
ros2 run px4_mpc quadrotor_demo
```
```
ros2 run px4_mpc astrobee_demo

```
## Deploying the MPC
In order to launch the mpc quadrotor in a ros2 launchfile,
```
ros2 launch px4_mpc mpc_quadrotor.launch.py 
```
