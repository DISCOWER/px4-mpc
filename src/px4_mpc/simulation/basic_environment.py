import numpy as np
import casadi as ca
import matplotlib.pyplot as plt


class EmbeddedSimEnvironment(object):

    def __init__(self, model, dynamics, controller, time=100.0):
        """
        Embedded simulation environment. Simulates the syste given dynamics
        and a control law, plots in matplotlib.

        :param model: model object
        :type model: object
        :param dynamics: system dynamics function (x, u)
        :type dynamics: casadi.DM
        :param controller: controller function (x, r)
        :type controller: casadi.DM
        :param time: total simulation time, defaults to 100 seconds
        :type time: float, optional
        """
        self.model = model
        self.dynamics = dynamics
        self.controller = controller
        self.total_sim_time = time  # seconds
        self.dt = self.model.dt
        self.estimation_in_the_loop = False

    def run(self, x0):
        """
        Run simulator with specified system dynamics and control function.
        """

        print("Running simulation....")
        sim_loop_length = int(self.total_sim_time / self.dt) + 1
        t = np.array([0])
        x_vec = np.array([x0]).reshape(self.model.n, 1)
        u_vec = np.empty((self.model.m, 0))
        e_vec = np.empty((10, 0))

        for i in range(sim_loop_length):

            # Get control input and obtain next state
            x = x_vec[:, -1].reshape(self.model.n, 1)
            u, error = self.controller(x, i * self.dt)
            x_next = self.dynamics(x, u)
            x_next[6:10] = x_next[6:10] / ca.norm_2(x_next[6:10])

            # Store data
            t = np.append(t, t[-1] + self.dt)
            x_vec = np.append(x_vec, np.array(
                x_next).reshape(self.model.n, 1), axis=1)
            u_vec = np.append(u_vec, np.array(
                u).reshape(self.model.m, 1), axis=1)
            e_vec = np.append(e_vec, error, axis=1)

        self.t = t
        self.x_vec = x_vec
        self.u_vec = u_vec
        self.e_vec = e_vec
        self.sim_loop_length = sim_loop_length
        return t, x_vec, u_vec

    def visualize(self):
        """
        Offline plotting of simulation data
        """
        variables = list(
            [self.t, self.x_vec, self.u_vec, self.sim_loop_length])
        if any(elem is None for elem in variables):
            print("Please run the simulation first with the method 'run'.")

        t = self.t
        x_vec = self.x_vec
        u_vec = self.u_vec

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4)
        fig2, (ax5, ax6) = plt.subplots(2)
        ax1.clear()
        ax1.set_title("States")
        ax1.plot(t, x_vec[0, :], 'r--',
                 t, x_vec[1, :], 'g--',
                 t, x_vec[2, :], 'b--')
        ax1.legend(["x1", "x2", "x3"])
        ax1.set_ylabel("Position [m]")
        ax1.grid()

        ax2.clear()
        ax2.plot(t, x_vec[3, :], 'r--',
                 t, x_vec[4, :], 'g--',
                 t, x_vec[5, :], 'b--')
        ax2.legend(["x3", "x4", "x5"])
        ax2.set_ylabel("Velocity [m/s]")
        ax2.grid()

        ax3.clear()
        ax3.plot(t, x_vec[6, :], 'r--',
                 t, x_vec[7, :], 'g--',
                 t, x_vec[8, :], 'b--',
                 t, x_vec[9, :], 'k--')
        ax3.legend(["x6", "x7", "x8"])
        ax3.set_ylabel("Attitude Quaternion")
        ax3.grid()

        ax4.clear()
        ax4.plot(t, x_vec[10, :], 'r--',
                 t, x_vec[11, :], 'g--',
                 t, x_vec[12, :], 'b--')
        ax4.legend(["x9", "x10", "x11"])
        ax4.set_ylabel("Ang. velocity [rad/s]")
        ax4.grid()

        # Plot control input
        ax5.clear()
        ax6.clear()
        if u_vec.shape[0] == 6:
            ax5.set_title("Control inputs")
            ax5.plot(t[:-1], u_vec[0, :], 'r--',
                     t[:-1], u_vec[1, :], 'g--',
                     t[:-1], u_vec[2, :], 'b--')
            ax5.legend(["u0", "u1", "u2"])
            ax5.set_ylabel("Force input [N]")

            ax6.plot(t[:-1], u_vec[3, :], 'r--',
                     t[:-1], u_vec[4, :], 'g--',
                     t[:-1], u_vec[5, :], 'b--')
            ax6.legend(["u3", "u4", "u5"])
            ax6.set_ylabel("Torque input [Nm]")
        else:
            ax5.set_title("Control inputs")
            ax5.plot(t[:-1], u_vec[0, :], 'r--')
            ax5.legend(["u0"])
            ax5.set_ylabel("Force input [N]")

            ax6.plot(t[:-1], u_vec[1, :], 'r--',
                     t[:-1], u_vec[2, :], 'g--',
                     t[:-1], u_vec[3, :], 'b--')
            ax6.legend(["u3", "u4", "u5"])
            ax6.set_ylabel("Torque input [Nm]")
        ax5.grid()
        ax6.grid()

        plt.show()

    def visualize_error(self):
        """
        Offline plotting of simulation data
        """
        variables = list(
            [self.t, self.e_vec, self.u_vec, self.sim_loop_length])
        if any(elem is None for elem in variables):
            print("Please run the simulation first with the method 'run'.")

        t = self.t
        x_vec = self.e_vec
        u_vec = self.u_vec

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4)
        fig2, (ax5, ax6) = plt.subplots(2)
        ax1.clear()
        ax1.set_title("Trajectory Error")
        ax1.plot(t, x_vec[0, :], 'r--',
                 t, x_vec[1, :], 'g--',
                 t, x_vec[2, :], 'b--')
        ax1.legend(["x1", "x2", "x3"])
        ax1.set_ylabel("Position Error [m]")
        ax1.grid()

        ax2.clear()
        ax2.plot(t, x_vec[3, :], 'r--',
                 t, x_vec[4, :], 'g--',
                 t, x_vec[5, :], 'b--')
        ax2.legend(["x3", "x4", "x5"])
        ax2.set_ylabel("Velocity Error [m/s]")
        ax2.grid()

        ax3.clear()
        ax3.plot(t, x_vec[6, :], 'r--')
        # ax3.legend(["u"])
        ax3.set_ylabel("Attitude Error [Distance]")
        ax3.grid()

        ax4.clear()
        ax4.plot(t, x_vec[7, :], 'r--',
                 t, x_vec[8, :], 'g--',
                 t, x_vec[9, :], 'b--')
        ax4.legend(["x9", "x10", "x11"])
        ax4.set_ylabel("Ang. velocity Error [rad/s]")
        ax4.grid()

        # Plot control input
        ax5.clear()
        ax6.clear()
        if u_vec.shape[0] == 6:
            ax5.set_title("Control inputs")
            ax5.plot(t[:-1], u_vec[0, :], 'r--',
                     t[:-1], u_vec[1, :], 'g--',
                     t[:-1], u_vec[2, :], 'b--')
            ax5.legend(["u0", "u1", "u2"])
            ax5.set_ylabel("Force input [N]")

            ax6.plot(t[:-1], u_vec[3, :], 'r--',
                     t[:-1], u_vec[4, :], 'g--',
                     t[:-1], u_vec[5, :], 'b--')
            ax6.legend(["u3", "u4", "u5"])
            ax6.set_ylabel("Torque input [Nm]")
        else:
            ax5.set_title("Control inputs")
            ax5.plot(t[:-1], u_vec[0, :], 'r--')
            ax5.legend(["u0"])
            ax5.set_ylabel("Force input [N]")

            ax6.plot(t[:-1], u_vec[1, :], 'r--',
                     t[:-1], u_vec[2, :], 'g--',
                     t[:-1], u_vec[3, :], 'b--')
            ax6.legend(["u3", "u4", "u5"])
            ax6.set_ylabel("Torque input [Nm]")
        ax5.grid()
        ax6.grid()

        plt.show()
