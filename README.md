# px4-mpc
This package contains an MPC integrated with with [PX4 Autopilot](https://px4.io/) and [ROS 2](https://ros.org/).

The MPC uses the [acados framework](https://github.com/acados/acados)

![px4-mpc](https://github.com/user-attachments/assets/6713b8e6-815f-42fe-b3a0-51708d3416e5)

## Paper
If you find this package useful in an academic context, please consider citing the paper

- Roque, Pedro, Sujet Phodapol, Elias Krantz, Jaeyoung Lim, Joris Verhagen, Frank Jiang, David Dorner, Roland Siegwart, Ivan Stenius, Gunnar Tibert, Huina Mao, Jana Tumova, Christer Fuglesang, Dimos V. Dimarogonas. "Towards Open-Source and Modular Space Systems with ATMOS." arXiv preprint arXiv:2501.16973 (2025).
. [[preprint](https://arxiv.org/abs/2501.16973)]

```
@article{roque2025towards,
  title={Towards Open-Source and Modular Space Systems with ATMOS},
  author={Roque, Pedro and Phodapol, Sujet and Krantz, Elias and Lim, Jaeyoung and Verhagen, Joris and Jiang, Frank and Dorner, David and Siegwart, Roland and Stenius, Ivan and Tibert, Gunnar and others},
  journal={arXiv preprint arXiv:2501.16973},
  year={2025}
}
```

## Setup
The MPC formulation uses acados. In order to install acados, follow the following [instructions](https://docs.acados.org/installation/). After building `acados` remember to install the python interface as described [here](https://docs.acados.org/python_interface/index.html).

To build the code, clone the following repositories into a ros2 workspace
Dependencies
- [px4_msgs](https://github.com/PX4/px4_msgs/pull/15)
- [px4-offboard](https://github.com/Jaeyoung-Lim/px4-offboard) (Optional): Used for RViz visualization

```
colcon build --packages-up-to px4_mpc
```

## Running MPC with PX4 SITL
In order to run the SITL(Software-In-The-Loop) simulation, the PX4 simulation environment and ROS2 needs to be setup.
For instructions, follow the [documentation](https://docs.px4.io/main/en/ros/ros2_comm.html)

Run PX4 SITL
```
make px4_sitl gazebo
```

Run the micro-ros-agent
```
micro-ros-agent udp4 --port 8888
```

### Quadrotor Example
In order to launch the mpc quadrotor in a ros2 launchfile,
```
ros2 launch px4_mpc mpc_quadrotor_launch.py 
```

### Spacecraft Example
First, make sure that you have followed the instructions in the [ATMOS guide](https://atmos.discower.io/pages/Simulation/). To control the vehicle with this package, make sure to switch the vehicle to Offboard mode in QGroundControl (see [QGroundControl in ATMOS guide](https://atmos.discower.io/pages/PX4/#building-qgroundcontrol-for-spacecraft).

The mpc_spacecraft_launch.py file includes optional arguments:

- **mode**: Control mode (wrench by default). Options: wrench, rate, direct_allocation.  
- **namespace**: Spacecraft namespace ('' by default).  
- **setpoint_from_rviz**: Use RViz for setpoints (True by default).

**Example with no namespace:**
```bash
ros2 launch px4_mpc mpc_spacecraft_launch.py mode:=wrench setpoint_from_rviz:=False
```

**Example with namespace:**
For this example to work, make sure you have run the PX4 SITL with the same namespace. Here goes an example
```bash
PX4_UXRCE_DDS_NS=pop make px4_sitl gz_spacecraft_2d
```
On another terminal, run 
```
micro-ros-agent udp4 --port 8888
```

Lastly, start PX4-MPC
```bash
ros2 launch px4_mpc mpc_spacecraft_launch.py mode:=wrench namespace:=pop setpoint_from_rviz:=False
```
