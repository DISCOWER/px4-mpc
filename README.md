# PX4 MPC

Model Predictive Control (MPC) for spacecraft and multirotors running on [PX4 Autopilot](https://px4.io/), integrated with [ROS 2](https://ros.org/).

![px4-mpc](https://github.com/user-attachments/assets/6713b8e6-815f-42fe-b3a0-51708d3416e5)

This package runs MPC controllers as ROS 2 nodes that talk to PX4 through the uXRCE-DDS bridge. The controllers read vehicle states from PX4 and send setpoints back in offboard mode. The optimal control problems are built and solved with the [acados framework](https://github.com/acados/acados).

The main vehicle is the [ATMOS](https://atmos.discower.io/) free-flyer spacecraft. Its controller has several control modes, from body rates and wrench commands down to direct thruster allocation. A quadrotor (`x500`) example is also included. Setpoints can be set interactively in RViz or come from a predefined setpoint node.

## Citation

If you find this package useful in academic work, please cite:

P. Roque, S. Phodapol, E. Krantz, J. Lim, J. Verhagen, F. Jiang, D. Dorner, R. Siegwart, I. Stenius, G. Tibert, H. Mao, J. Tumova, C. Fuglesang and D. V. Dimarogonas,
"Towards Open-Source and Modular Space Systems with ATMOS,"
*arXiv preprint arXiv:2501.16973*, 2025.

[arXiv](https://arxiv.org/abs/2501.16973)

<details>
<summary>BibTeX</summary>

<pre><code>@article{roque2025towards,
  title={Towards Open-Source and Modular Space Systems with ATMOS},
  author={Roque, Pedro and Phodapol, Sujet and Krantz, Elias and Lim, Jaeyoung and Verhagen, Joris and Jiang, Frank and Dorner, David and Siegwart, Roland and Stenius, Ivan and Tibert, Gunnar and others},
  journal={arXiv preprint arXiv:2501.16973},
  year={2025}
}
</code></pre>

</details>

## Quick Start

### Prerequisites

- ROS 2 (e.g. [Humble](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html))
- [acados](https://docs.acados.org/installation/) and its [Python interface](https://docs.acados.org/python_interface/index.html)
- PX4 SITL with ROS 2 support, set up by following the [PX4 ROS 2 guide](https://docs.px4.io/main/en/ros/ros2_comm.html) (this includes the Micro XRCE-DDS Agent)
- For the spacecraft: the ATMOS simulation, set up by following the [ATMOS simulation guide](https://atmos.discower.io/pages/Simulation/)
- (Optional) [QGroundControl](https://docs.qgroundcontrol.com/master/en/qgc-user-guide/getting_started/download_and_install.html)

### Install

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
git clone https://github.com/PX4/px4_msgs.git              # PX4 messages for communication with PX4 DDS
git clone https://github.com/DISCOWER/px4-offboard.git     # RViz interface and setpoint generation
git clone https://github.com/DISCOWER/px4-mpc.git          # this package
cd ..
colcon build --packages-up-to px4_mpc
source install/setup.bash  # use setup.zsh for zsh users
```

For the `propeller` control mode, also clone [atmos_propeller_plate_interface](https://github.com/DISCOWER/atmos_propeller_plate_interface) into `~/ros2_ws/src` and build it:
```bash
colcon build --packages-up-to px4_mpc propeller_plate_iface
```

### PX4 Setup

**DDS topics:** the PX4 topics exposed to ROS 2 are configured in `PX4-Autopilot/src/modules/uxrce_dds_client/dds_topics.yaml`. Edit this file if you need additional topics.

**Arming and mode switching:** by default, PX4 expects RC inputs and a QGroundControl (QGC) connection before it allows arming and mode switching. These are the recommended settings for real deployments. Set up either QGC or a headless configuration before running the examples. Both options below describe how to switch to offboard mode and arm the vehicle.

<details>
<summary>With QGC (recommended)</summary>

1. Open QGroundControl.
2. Click the QGroundControl icon in the top left corner and open "Application Settings".
3. Under the "Fly View" tab, scroll down to "Virtual Joysticks" and enable them by clicking the slider to the right of "Enabled".
4. Make sure that auto-center throttle is disabled.
5. Click "Exit Application Settings" in the top left corner to return to the Fly View screen.

To start the MPC, select "Offboard" in the flight mode menu in the top toolbar, then arm the vehicle.

> **Important:** Before arming the vehicle, make sure that the throttle (left vertical joystick) is at the lowest position.

</details>

<details>
<summary>Headless (no QGC)</summary>

> **Warning:** Do this at your own risk. We recommend using QGC to keep hardware and software setups consistent.

Disable the RC check and the QGC connection requirement by setting the following parameters in the PX4 SITL terminal:
```bash
pxh> param set COM_RC_IN_MODE 1
pxh> param set NAV_DLL_ACT 0
pxh> param set COM_ARM_WO_GPS 1
pxh> param set CBRK_USB_CHK 197848
pxh> param set COM_ARMABLE 1
```
Then restart the PX4 SITL to apply the changes. After rebooting, you should be able to arm and switch vehicle modes.

To start the MPC, switch to offboard mode and arm the vehicle in the PX4 SITL terminal:
```bash
pxh> commander mode offboard
pxh> commander arm
```

</details>

### Run the Spacecraft Example

Terminal 1: Start the DDS agent
```bash
micro-xrce-dds-agent udp4 --port 8888
```

Terminal 2: Start PX4 SITL with Gazebo
```bash
cd ~/PX4-Autopilot
make px4_sitl_spacecraft gz_atmos
```

Terminal 3: Start the MPC
```bash
ros2 launch px4_mpc mpc_spacecraft_launch.py mode:=wrench rviz_mode:=setpoint
```

Switch the vehicle to offboard mode and arm it (see [PX4 Setup](#px4-setup)).

The vehicle should now start following the desired setpoints.

**KTH Space Lab:** to simulate the KTH Space Lab environment, start PX4 SITL with `gz_atmos_kthspacelab` and enable the matching constraints (recommended):
```bash
make px4_sitl_spacecraft gz_atmos_kthspacelab
```
```bash
ros2 launch px4_mpc mpc_spacecraft_launch.py mode:=wrench rviz_mode:=off kthspace_constraints:=true
```

### Spacecraft Control Modes

Set the control mode with `mode:=<mode>` when launching `mpc_spacecraft_launch.py`.

| Mode | Description | PX4 Offboard Setpoint |
|---|---|---|
| `wrench` (default) | MPC that commands body forces and torques | Thrust and torque |
| `rate` | MPC that commands body forces and body rates | Body rate |
| `direct_allocation` | MPC that commands individual thrusters | Direct actuator |
| `offset_free_wrench` | Offset-free wrench MPC. Requires [OpenMPC](https://github.com/mikaelj-kth-se/OpenMPC) | Thrust and torque |
| `lqr_wrench` | LQR that commands body forces and torques | Thrust and torque |
| `propeller` | MPC for the ATMOS propeller plate. Also launches `propeller_plate_iface_node` from [atmos_propeller_plate_interface](https://github.com/DISCOWER/atmos_propeller_plate_interface) | Direct actuator |

### Run the Quadrotor Example

Terminal 1: Start the DDS agent
```bash
micro-xrce-dds-agent udp4 --port 8888
```

Terminal 2: Start PX4 SITL with Gazebo
```bash
cd ~/PX4-Autopilot
make px4_sitl gz_x500
```

Terminal 3: Start the MPC
```bash
ros2 launch px4_mpc mpc_quadrotor_launch.py
```

Switch the vehicle to offboard mode and arm it (see [PX4 Setup](#px4-setup)).

## Launch Arguments

**`mpc_spacecraft_launch.py`:**
| Argument | Default | Description |
|---|---|---|
| `mode` | `wrench` | Control mode (see [Spacecraft Control Modes](#spacecraft-control-modes)) |
| `namespace` | `''` | Namespace for all nodes. Must match the PX4 namespace |
| `rviz_mode` | `setpoint` | `off`: no RViz. `viz`: RViz for visualization only. `setpoint`: RViz for visualization and setting the setpoint pose. In `off` and `viz`, setpoints come from the `test_setpoints` node |
| `px4_uses_ned` | `true` | PX4 uses the NED frame (`true`) or the ENU frame (`false`) |
| `skip_build` | `false` | Skip the acados solver build and reuse the previously generated solver |
| `kthspace_constraints` | `false` | Use KTH Space Lab constraints. Recommended with `gz_atmos_kthspacelab` |
| `sitl` | `false` | Also publish vehicle odometry on `odom` (SITL only) |

**`mpc_quadrotor_launch.py`:**
| Argument | Default | Description |
|---|---|---|
| `namespace` | `''` | Namespace for all nodes. Must match the PX4 namespace |
| `setpoint_from_rviz` | `true` | Set the setpoint pose interactively in RViz |

### Namespaces

To run with a namespace, start PX4 SITL with the same namespace as the MPC:
```bash
PX4_UXRCE_DDS_NS=pop make px4_sitl_spacecraft gz_atmos
```
```bash
ros2 launch px4_mpc mpc_spacecraft_launch.py mode:=wrench rviz_mode:=off namespace:=pop
```

## References

- [ATMOS](https://atmos.discower.io/)
- [PX4 Autopilot](https://px4.io/)
- [PX4 ROS 2 User Guide](https://docs.px4.io/main/en/ros/ros2_comm.html)
- [px4_msgs](https://github.com/PX4/px4_msgs)
- [px4-offboard](https://github.com/E-Krantz/px4-offboard)
- [atmos_propeller_plate_interface](https://github.com/DISCOWER/atmos_propeller_plate_interface)
- [acados](https://docs.acados.org/)
- [ROS 2 Documentation](https://docs.ros.org/)
