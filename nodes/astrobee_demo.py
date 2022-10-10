from px4_mpc.models.astrobee import Astrobee
from px4_mpc.controllers.mpc import MPC
from px4_mpc.simulation.basic_environment import EmbeddedSimEnvironment


def main():
    # Create pendulum and controller objects
    abee = Astrobee()
    abee.test_dynamics()

    # Instantiate controller
    u_lim, x_lim = abee.get_limits()

    # Create MPC Solver
    MPC_HORIZON = 10
    # TODO: inspect the MPC class
    ctl = MPC(model=abee,
              dynamics=abee.model,
              param='P1',
              N=MPC_HORIZON,
              ulb=-u_lim, uub=u_lim,
              xlb=-x_lim, xub=x_lim)

    # Test 1: Reference tracking
    x_d = abee.get_static_setpoint()
    ctl.set_reference(x_d)
    # Set initial state
    x0 = abee.get_initial_pose()
    sim_env = EmbeddedSimEnvironment(model=abee,
                                     dynamics=abee.model,
                                     controller=ctl.mpc_controller,
                                     time=50)
    # t, y, u = sim_env.run(x0)
    # sim_env.visualize()  # Visualize state propagation
    # sim_env.visualize_error()

    # Test 2: Activate Tracking
    tracking_ctl = MPC(model=abee,
                       dynamics=abee.model,
                       param='P1',
                       N=MPC_HORIZON,
                       trajectory_tracking=True,
                       ulb=-u_lim, uub=u_lim,
                       xlb=-x_lim, xub=x_lim)
    sim_env_tracking = EmbeddedSimEnvironment(model=abee,
                                              dynamics=abee.model,
                                              controller=tracking_ctl.mpc_controller,
                                              time=80)
    # t, y, u = sim_env_tracking.run(x0)
    # sim_env_tracking.visualize()  # Visualize state propagation
    # sim_env_tracking.visualize_error()

    # Test 3: Activate forward propagation
    abee.test_forward_propagation()
    tracking_ctl.set_forward_propagation()
    t, y, u = sim_env_tracking.run(x0)
    # sim_env_tracking.visualize()  # Visualize state propagation
    sim_env_tracking.visualize_error()


if __name__ == '__main__':
    main()
