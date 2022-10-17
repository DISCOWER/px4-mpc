"""
Model Predictive Control - CasADi interface
Based off Helge-André Langåker work on GP-MPC
Customized by Pedro Roque for EL2700 Model Predictive Countrol Course
"""

from __future__ import absolute_import
import casadi as cs
from px4_mpc.util import r_mat_q, xi_mat, skew
import casadi as ca
import numpy as np


class Quadrotor(object):
    def __init__(self,
                 mass=1.5,
                 inertia=np.diag([0.001, 0.001, 0.01]),
                 h=0.01,
                 **kwargs):
        """
        Quadrotor Robot, NMPC tester class.

        :param mass: mass of the quadrotor
        :type mass: float
        :param inertia: inertia tensor of the quadrotor
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
        Quadrotor nonlinear dynamics with Quaternions.

        :param x: state
        :type x: ca.MX
        :param u: control input
        :type u: ca.MX
        :return: state time derivative
        :rtype: ca.MX
        """

        # State extraction
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
        xd = np.array([0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]).reshape(13, 1)
        return xd

    def get_limits(self):
        """
        Get Quadrotor control and state bounds

        :return: state and control limits
        :rtype: np.ndarray, np.ndarray
        """
        # MPC bounds - control
        ulb = np.array([0, -0.1, -0.1, -0.1])
        uub = np.array([self.mass * 9.81 * 4, 0.1, 0.1, 0.1])
        xlb = np.array([-np.inf, -np.inf, -np.inf,
                        -np.inf, -np.inf, -np.inf,
                        -1, -1, -1, -1,
                        -1, -1, -1])
        xub = -1 * xlb
        return ulb, uub, xlb, xub
