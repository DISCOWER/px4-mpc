############################################################################
#
#   Copyright (C) 2025 DISCOWER. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the
#    distribution.
# 3. Neither the name PX4 nor the names of its contributors may be
#    used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
# OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
# AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
############################################################################

import numpy as np
import casadi as cs
from px4_mpc.models.spacecraft_wrench_model import SpacecraftWrenchModel
import time


class SpacecraftWrenchMPC:
    """
    Model Predictive Controller for spacecraft attitude control using rate control inputs.
    This controller uses a discrete-time model of the spacecraft and solves an optimal control problem
    to minimize the tracking error of the spacecraft's state to a reference trajectory.
    """

    def __init__(self, _, Tf=1.0, N=10, add_cbf=False):
        """
        Initialize the MPC controller.

        :param model: SpacecraftRateModel instance containing the spacecraft dynamics.
        :param Tf: Final time of the prediction horizon.
        :param N: Number of control intervals in the prediction horizon.
        :param add_cbf: Boolean indicating whether to add a collision avoidance CBF.
        """
        self.model = SpacecraftWrenchModel(six_dof=True)

        self.Tf = Tf
        self.N = N
        self.dt = self.Tf / self.N

        self.add_cbf = add_cbf

        self.x0 = np.array([0.01, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        self.nx = self.model.nx
        self.nu = self.model.nu
        print("Mode: ", self.nu)

        self.u_ub = np.hstack(
            (np.repeat([self.model.max_thrust], 3), np.repeat([self.model.max_torque], 3))
        )
        self.u_lb = -self.u_ub

        self.params = {}
        self.vars = {}

        self.ocp = self.setup()

        self.Q = np.diag([5e0, 5e0, 5e0, 8e-1, 8e-1, 8e-1, 8e3, 10e0, 10e0, 10e0])
        self.Q_e = 10 * self.Q
        self.R = 2 * np.diag([1e-2, 1e-2, 1e-2, 2e0, 2e0, 2e0])

    def setup(self):
        """
        Set up the MPC problem using CasADi.

        :return: ocp object
        """
        # create ocp object to formulate the OCP
        ocp = cs.Opti()
        # set variables: state and control input
        X = ocp.variable(self.nx, self.N + 1)
        U = ocp.variable(self.nu, self.N)
        self.vars["X"] = X
        self.vars["U"] = U
        # set parameters: current state and reference state, cost matrices
        x0 = ocp.parameter(self.nx)
        xref = ocp.parameter(self.nx, self.N + 1)
        Q = ocp.parameter(self.nx - 3, self.nx - 3)
        Q_e = ocp.parameter(self.nx - 3, self.nx - 3)
        R = ocp.parameter(self.nu, self.nu)
        self.params["x0"] = x0
        self.params["xref"] = xref
        self.params["Q"] = Q
        self.params["Q_e"] = Q_e
        self.params["R"] = R

        # --- CONSTRAINTS
        # initial state constraint
        ocp.subject_to(X[:, 0] == x0)

        # dynamics constraints
        for i in range(self.N):
            ocp.subject_to(
                self.model.get_rk4_integration(X[:, i], U[:, i], self.dt) == X[:, i + 1]
            )

        # control input constraints
        for i in range(self.N):
            ocp.subject_to(self.u_lb <= U[:, i])
            ocp.subject_to(U[:, i] <= self.u_ub)

        # --- COST
        # standard trajectory tracking cost
        cost_eq = 0
        for i in range(self.N):
            cost_eq += self.calculate_state_error(X[:, i], xref[:, i], Q)
            cost_eq += U[:, i].T @ R @ U[:, i]
        cost_eq += self.calculate_state_error(X[:, -1], xref[:, -1], Q_e)

        # and minimize the sum of the costs
        ocp.minimize(cost_eq)

        ## SOLVER SETTINGS
        # IPOPT
        # opts = {'ipopt.print_level': 1, 'print_time': 0, 'ipopt.sb': 'yes',
        #         'verbose':False}
        # ocp.solver('ipopt',opts)

        # qrqp
        nlp_options = {
            "qpsol": "qrqp",
            "hessian_approximation": "gauss-newton",
            "max_iter": 100,
            "tol_du": 1e-2,
            "tol_pr": 1e-2,
            "qpsol_options": {
                "sparse": True,
                "hessian_type": "posdef",
                "numRefinementSteps": 1,
            },
        }
        ocp.solver("sqpmethod", nlp_options)

        return ocp

    def calculate_state_error(self, x, xref, Q):
        # state: p, v, q
        es = x - xref
        es = es[0:6]
        cost_es = es.T @ Q[0:6, 0:6] @ es
        cost_ew = x[3:6] - xref[3:6]
        cost_es += cost_ew.T @ Q[7:10, 7:10] @ cost_ew

        # quaternion cost
        q = x[6:10].reshape((4, 1))
        qref = xref[6:10].reshape((4, 1))
        eq = 1 - (q.T @ qref) ** 2
        cost_eq = eq.T @ Q[6, 6].reshape((1, 1)) @ eq

        return cost_eq + cost_es

    def solve(self, x0, ref,
              weights={"Q": None, "Q_e": None, "R": None},
              initial_guess={"X": None, "U": None},
              verbose=False):

        t0 = time.time()

        # set initial guess if we are getting any
        if initial_guess["X"] is not None:
            self.ocp.set_initial(self.vars["X"], initial_guess["X"])
        if initial_guess["U"] is not None:
            self.ocp.set_initial(self.vars["U"], initial_guess["U"])

        # set x0 parameter
        self.ocp.set_value(self.params["x0"], x0)

        # set setpoints parameter
        self.ocp.set_value(self.params["xref"], ref[:13, :])

        # set cost matrices if we are getting any
        Q = self.Q if weights["Q"] is None else weights["Q"]
        Q_e = self.Q_e if weights["Q_e"] is None else weights["Q_e"]
        R = self.R if weights["R"] is None else weights["R"]
        self.ocp.set_value(self.params["Q"], Q)
        self.ocp.set_value(self.params["Q_e"], Q_e)
        self.ocp.set_value(self.params["R"], R)

        try:
            sol = self.ocp.solve()
            X_pred = sol.value(self.vars["X"])
            U_pred = sol.value(self.vars["U"])
        except Exception as e:
            print(f"Optimization failed: {e}") if verbose else None
            X_pred = np.zeros((self.nx, self.N + 1))
            U_pred = np.zeros((self.nu, self.N))

        print(f"time taken: {time.time()-t0}") if verbose else None

        # transpose the obtained state and control to be consistent with the acados setup
        X_pred, U_pred = X_pred.T, U_pred.T
        print("U_PRED: ", U_pred)
        return U_pred, X_pred
