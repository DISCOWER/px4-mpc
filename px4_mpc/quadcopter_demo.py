from px4_mpc.models.quadrotor import Quadrotor
from px4_mpc.controllers.setpoint_mpc import SetpointMPC
from px4_mpc.simulation.basic_environment import EmbeddedSimEnvironment


def main():
    # Create Quadrotor and controller objects
    quad = Quadrotor()

    # Instantiate controller
    ulb, uub, xlb, xub = quad.get_limits()

    # Create MPC Solver
    MPC_HORIZON = 15

    # Spawn Controller
    ctl = SetpointMPC(model=quad,
                      dynamics=quad.model,
                      param='P1',
                      N=MPC_HORIZON,
                      ulb=ulb, uub=uub, xlb=xlb, xub=xub)

    # Test 1: Reference tracking
    x_d = quad.get_static_setpoint()
    ctl.set_reference(x_d)
    # Set initial state
    x0 = quad.get_initial_pose()
    sim_env = EmbeddedSimEnvironment(model=quad,
                                     dynamics=quad.model,
                                     controller=ctl.mpc_controller,
                                     time=2)
    t, y, u = sim_env.run(x0)
    sim_env.visualize()  # Visualize state propagation
    # sim_env.visualize_error()


if __name__ == '__main__':
    main()
