"""
Model Predictive Control - CasADi interface
Based off Helge-André Langåker work on GP-MPC
Customized by Pedro Roque for EL2700 Model Predictive Countrol Course
"""

from __future__ import absolute_import
import casadi as cs
from px4_mpc.util import *
import casadi as ca
import numpy as np


# class Quadrotor(object):
#     def __init__(self, model, Nx=13, Nu=4, Nz=0, Np=0, dt=0.01):
#         """
#         Initializes a Quadrotors class. By default generates a discrete model
#         with a sampling time of 0.01 seconds.
#         """
#         self.Nx = Nx
#         self.Nu = Nu
#         self.Nz = Nz
#         self.Np = Np
#         self.m = Nu
#         self.n = Nx
#         self.dt = dt

#         # Inertial Parameters
#         self.mass = 1
#         self.J = np.diag([0.01, 0.01, 0.1])

#         # Set CasADi variables
#         x = cs.MX.sym('x', Nx)
#         u = cs.MX.sym('u', Nu)
#         z = cs.MX.sym('z', Nz)
#         p = cs.MX.sym('p', Np)

#         self.nonlinear_model = self.quadrotor_dynamics

#         # Integration method - integrator options an be adjusted
#         options = {"abstol": 1e-5, "reltol": 1e-9, "max_num_steps": 100,
#                    "tf": self.dt}
#         dae = {'x': x, 'ode': self.nonlinear_model(
#             x, u, z, p), 'p': cs.vertcat(u, p)}
#         self.Integrator = cs.integrator('integrator', 'cvodes', dae, options)

#         # Jacobian of continuous system
#         ode_casadi = cs.Function(
#             "ode", [x, u, p], [self.nonlinear_model(x, u, z, p)])
#         self.A = cs.Function('jac_x_A', [x, u, p],
#                              [cs.jacobian(ode_casadi(x, u, p), x)])
#         self.B = cs.Function('jac_x_B', [x, u, p],
#                              [cs.jacobian(ode_casadi(x, u, p), u)])

#         # Jacobian of exact discretization
#         self.Ad = cs.Function('jac_x_Ad', [x, u, p], [cs.jacobian(
#             self.Integrator(x0=x, p=cs.vertcat(u, p))['xf'], x)])
#         self.Bd = cs.Function('jac_u_Bd', [x, u, p], [cs.jacobian(
#             self.Integrator(x0=x, p=cs.vertcat(u, p))['xf'], u)])

#     def simple_integrator(self, x, u, *_):
#         dxdt = [
#             x[1],
#             u[0]
#         ]
#         return cs.vertcat(*dxdt)

#     def quadrotor_dynamics(self, x, u, *_):


#         return cs.vertcat(*dxdt)

#     def rk4_model(self, x_t, u_t):
#         """
#         Runge-Kutta 4th Order discretization.
#         :param x: state
#         :type x: ca.MX
#         :param u: control input
#         :type u: ca.MX
#         :return: state at next step
#         :rtype: ca.MX
#         """

#         return self.Integrator(x0=x_t, p=u_t)['xf']


class Quadrotor(object):
    def __init__(self,
                 mass=1.5,
                 inertia=np.diag([0.001, 0.001, 0.01]),
                 h=0.01,
                 **kwargs):
        """
        Astrobee Robot, NMPC tester class.

        :param mass: mass of the Astrobee
        :type mass: float
        :param inertia: inertia tensor of the Astrobee
        :type inertia: np.diag
        :param h: sampling time of the discrete system, defaults to 0.01
        :type h: float, optional
        :param model: select between 'euler' or 'quat'
        :type model: str
        """

        # Model
        self.nonlinear_model = self.quadrotor_dynamics_quat
        self.n = 13
        self.m = 4
        self.dt = h

        # Model prperties
        self.mass = mass
        self.inertia = inertia

        # Set CasADi functions
        self.set_casadi_options()

        # Set nonlinear model with a RK4 integrator
        self.model = self.rk4_integrator(self.nonlinear_model)

    def set_casadi_options(self):
        """
        Helper function to set casadi options.
        """
        self.fun_options = {
            "jit": False,
            "jit_options": {"flags": ["-O2"]}
        }

    def quadrotor_dynamics_quat(self, x, u):
        """
        Astrobee nonlinear dynamics with Quaternions.

        :param x: state
        :type x: ca.MX
        :param u: control input
        :type u: ca.MX
        :return: state time derivative
        :rtype: ca.MX
        """

        # State extraction
        p = x[0:3]
        v = x[3:6]
        q = x[6:10]
        w = x[10:]

        # 3D Force
        f = u[0]
        gravity = cs.DM.zeros(3, 1)
        gravity[2] = -9.81

        # 3D Torque
        tau = u[1:]

        # Model
        pdot = v
        vdot = ca.mtimes(r_mat_q(q)[:, -1], f) / self.mass + gravity
        qdot = ca.mtimes(xi_mat(q), w) / 2.0
        wdot = ca.mtimes(ca.inv(self.inertia), tau + ca.mtimes(skew(w),
                         ca.mtimes(self.inertia, w)))

        dxdt = [pdot, vdot, qdot, wdot]

        return ca.vertcat(*dxdt)

    def rk4_integrator(self, dynamics):
        """
        Runge-Kutta 4th Order discretization.
        :param x: state
        :type x: ca.MX
        :param u: control input
        :type u: ca.MX
        :return: state at next step
        :rtype: ca.MX
        """
        x0 = ca.MX.sym('x0', self.n, 1)
        u = ca.MX.sym('u', self.m, 1)

        x = x0

        k1 = dynamics(x, u)
        k2 = dynamics(x + self.dt / 2 * k1, u)
        k3 = dynamics(x + self.dt / 2 * k2, u)
        k4 = dynamics(x + self.dt * k3, u)
        xdot = x0 + self.dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        # Normalize quaternion: TODO(Pedro-Roque): check best way to propagate
        rk4 = ca.Function('RK4', [x0, u], [xdot], self.fun_options)

        return rk4

    def get_initial_pose(self):
        """
        Helper function to get a starting state, depending on the dynamics type.

        :return: starting state
        :rtype: np.ndarray
        """
        return np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]).reshape(13, 1)

    def get_static_setpoint(self):
        """
        Helper function to get the initial state of Honey for setpoint stabilization.
        """
        xd = np.array([0, 0, 0.1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]).reshape(13, 1)
        return xd

    def get_limits(self):
        """
        Get Astrobee control and state limits for ISS

        :return: state and control limits
        :rtype: np.ndarray, np.ndarray
        """
        # MPC bounds - control
        ulb = np.array([-0.5, -0.1, -0.1, -0.1])
        uub = np.array([self.mass * 9.81 * 4, 0.1, 0.1, 0.1])
        xlb = np.array([-3, -3, -3,
                        -1, -1, -1,
                        -1, -1, -1, -1,
                        -1, -1, -1])
        xub = -1 * xlb
        return ulb, uub, xlb, xub
