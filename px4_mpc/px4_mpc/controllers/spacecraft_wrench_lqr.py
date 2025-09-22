############################################################################
#
#   Copyright (C) 2024 PX4 Development Team. All rights reserved.
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
import control as ctrl


class SpacecraftWrenchLQR():
    def __init__(self, model, dt=0.1):
        self.model = model
        # x y z vx vy vz qx qy qz qw wx wy wz
        self.x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.dt = dt

        self.gain = self.setup()

    def setup(self):
        # From model, take dynamics
        model = self.model.get_acados_model()

        # Discretize dynamics
        model_disct = self.RK(model, self.dt)

        # Linearize around hover
        Ad = cs.Function('A', [model.x, model.u], [cs.jacobian(model_disct, model.x)])(self.x0, np.zeros(model.u.size()[0]))
        Bd = cs.Function('B', [model.x, model.u], [cs.jacobian(model_disct, model.u)])(self.x0, np.zeros(model.u.size()[0]))

        # Cost matrices
        Q = np.diag([10, 10, 10, 1, 1, 1, 100, 100, 100, 10, 10, 10])
        R = np.diag([1, 1, 1, 10, 10, 10])

        # Compute the infinite-horizon LQR controller using dlqr
        L, _, _ = ctrl.dlqr(Ad, Bd, Q, R)
        return L

    def solve(self, x0, verbose=False, ref=None):

        # Set reference, create zero reference
        x0 = x0.reshape((-1, 1))
        ref = ref.reshape((-1, 1))
        if verbose:
            print("LQR x0: ", x0)
            print("LQR ref: ", ref)
            print("LQR gain: ", self.gain.shape)

        # Compute control action
        output = -self.gain @ (x0 - ref)
        if verbose:
            print("LQR output: ", output.T)

        # zero torque
        output[3] = 0.0
        output[4] = 0.0
        output[5] = 0.0

        return output.T, []

    def RK(self, model, dt):
        """
        Create a Runge-Kutta expression for the given ODE.

        Parameters:
        nlsys (NonlinearSystem): The nonlinear system to integrate.
        dt (float): Time step for the integrator.
        order (int): Order of the Runge-Kutta method (default is 4).

        Returns:
        casadi.MX: CasADi expression for one integration step.
        """
        x = model.x
        u = model.u

        # Create ca function for the ODE
        update_fcn = cs.Function('update_fcn', [x, u], [model.f_expl_expr])
        k1 = update_fcn(x, u)
        k2 = update_fcn(x + 0.5 * dt * k1, u)
        k3 = update_fcn(x + 0.5 * dt * k2, u)
        k4 = update_fcn(x + dt * k3, u)
        rk_step_expr = x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        return rk_step_expr
