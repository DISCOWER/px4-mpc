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
        # x y z vx vy vz qx qy qz wx wy wz
        self.dt = dt

        self.A_fun, self.B_fun = self.model.sym_linearization()

        # Tuning
        self.Q = np.diag([10, 10, 10, 1, 1, 1, 100, 100, 100, 10, 10, 10])
        self.R = np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

    def get_gain(self, x_ref, u_ref=np.zeros((6,))):

        # Get A and B for state and system
        print("LQR x_ref: ", x_ref.shape)
        A = self.A_fun(np.zeros((12,)), u_ref, x_ref, u_ref)
        B = self.B_fun(np.zeros((12,)), u_ref, x_ref, u_ref)
        A = np.array(A).astype(np.float64)
        B = np.array(B).astype(np.float64)

        # Discretize
        sysc = ctrl.ss(A, B, np.eye(12), np.zeros((12, 6)))
        sysd = ctrl.c2d(sysc, self.dt)
        Ad = np.array(sysd.A)
        Bd = np.array(sysd.B)

        # Compute LQR gain
        self.gain, _, _ = ctrl.dlqr(Ad, Bd, self.Q, self.R)
        return self.gain

    def solve(self, x0, verbose=False, ref=None):
        # Set reference, create zero reference
        x0 = x0.reshape((-1, 1))
        ref = ref.reshape((-1, 1))
        if verbose:
            print("LQR x0: ", x0)
            print("LQR ref: ", ref)

        # Compute control action
        K = self.get_gain(ref)
        x_err = self.model.get_error_state(ref, x0).reshape((-1, 1))
        output = -K @ x_err
        output = np.asarray(output).reshape((1, 6))
        if verbose:
            print("LQR output: ", output.T)

        return output, []
