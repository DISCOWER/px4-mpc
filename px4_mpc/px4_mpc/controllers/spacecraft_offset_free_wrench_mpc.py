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

from openmpc.NonlinearMPC import NonlinearSystem, trackingMPC, EKF, create_estimator_model
from openmpc.models.atmos_2d import Atmos2D
import numpy as np


class SpacecraftOffsetFreeWrenchMPC():
    """
    Offset Free MPC implmentation for a freeflyer robot.

    This implementation is based on the OpenMPC library.
    OpenMPC will become publicly available in the near future.
    """
    def __init__(self, model=None):
        if model is not None:
            print("Provided model is ignored, using default model.")

        self.dt = 0.1
        self.model = Atmos2D(dt=self.dt)
        self.rhs = self.model.continuous_dynamics()
        self.states = self.model.x
        self.inputs = self.model.u
        self.disturbances = self.model.d
        print("D: ", self.disturbances.shape)

        self.system = NonlinearSystem(updfcn=self.rhs,
                                      states=self.states,
                                      inputs=self.inputs,
                                      disturbances=self.disturbances,
                                      outfcn=self.states)
        self.prediction_model = self.system.c2d(self.dt)
        self.n = self.states.size()[0]

        # Tuning:
        self.N = 10
        self.Q = np.diag([10, 10, 10, 5, 5, 5, 10, 10, 10, 4, 4, 4])
        self.R = np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]) * 10

        self.umax = np.array([1.0] * 6)
        # self.umax = np.array([1.5, 1.5, 1.5, 0.5, 0.5, 0.5])
        self.umax[3:] *= 0.12
        self.umin = -self.umax

        # Setup LQR and Tracking MPC
        dnom = np.array([0.00, 0.00, 0.0, 0.0, 0.0, 0.0])
        yref = np.array([0.0, 0.0, 0, 0, 0, 0, 0, 0, 0.0, 0, 0, 0])
        (xref, uref) = self.prediction_model.get_target_point(yref, dnom)
        L, P, _ = self.prediction_model.compute_lqr_controller(self.Q, self.R, (xref, uref, dnom))

        mpcProblemData = {
            'N': self.N,
            'dt': self.dt,  # sampling time
            'Q': self.Q,
            'R': self.R,
            'Q_N': P,
            'predictionModel': self.prediction_model,
            'umin': self.umin,
            'umax': self.umax,
            'slackPenaltyWeight': 1e6,  # Slack penalty weight
            'baseController': L,
            'dualModeHorizon': 5,  # Dual mode horizon
            'dualModeController': L
        }

        # Initialize the MPC controller
        self.mpc = trackingMPC(mpcProblemData)

        # Initialize disturbance to nominal value
        self.d_hat = np.zeros((6,))

        # Done with setup
        print("SpacecraftOffsetFreeWrenchMPC initialized.")

    def ekf_setup(self, x0_hat):
        """
        Setup the EKF with the given initial state estimate.
        """
        # Setup EKF
        estimatorModel = create_estimator_model(self.prediction_model)
        # Define the initial state and covariance estimates
        P0 = 1000 * np.eye(18)  # Initial state covariance

        # Define the process noise and measurement noise covariance matrices
        Qnoise = np.diag([0.001] * 18)  # Process noise covariance
        Rnoise = np.diag([0.01] * 12)  # Measurement noise covariance

        # Update xhat with disturbance model
        x0_hat = np.concatenate((x0_hat, np.zeros((6,))), axis=0)

        # Pack the EKF parameters into a struct
        ekfParameters = {
            'predictionModel': estimatorModel,
            'Q': Qnoise,
            'R': Rnoise,
            'x0': x0_hat,
            'P0': P0,
            'dt': self.dt  # Time step
        }
        self.ekf = EKF(ekfParameters)

    def get_disturbance_estimate(self):
        """
        Get disturbance estimate.

        :return: disturbance estimate
        :rtype: np.ndarray
        """
        d_hat = self.ekf.get_state()[12:]
        return d_hat.flatten()

    def remove_quat_scalar(self, x0):
        """
        Remove scalar part of quaternion.

        :param x0: state with scalar component
        :type x0: numpy
        """
        x = np.zeros((12,))
        x[0:3] = x0[0:3].flatten()
        x[3:6] = x0[3:6].flatten()
        x[6:9] = x0[7:10].flatten()
        x[9:12] = x0[10:13].flatten()
        return x

    def add_quat_scalar(self, xPred):
        """
        Add scalar part of quaternion.

        :param uPred: control vector
        :type uPred: ca.DM
        :param xPred: state vector
        :type xPred: ca.DM
        """
        xPredWithScalar = np.zeros((13, self.N))

        for i in range(self.N):
            # Position
            xPredWithScalar[0, i] = xPred[0, i]
            xPredWithScalar[1, i] = xPred[1, i]
            xPredWithScalar[2, i] = xPred[2, i]

            # Velocities
            xPredWithScalar[3, i] = xPred[3, i]
            xPredWithScalar[4, i] = xPred[4, i]
            xPredWithScalar[5, i] = xPred[5, i]

            # Quaternion
            q0_i = np.sqrt(1 - xPred[6, i]**2 - xPred[7, i]**2 - xPred[8, i]**2)
            xPredWithScalar[6, i] = q0_i
            xPredWithScalar[7, i] = xPred[6, i]
            xPredWithScalar[8, i] = xPred[7, i]
            xPredWithScalar[9, i] = xPred[8, i]

            # Angular velocities
            xPredWithScalar[10, i] = xPred[9, i]
            xPredWithScalar[11, i] = xPred[10, i]
            xPredWithScalar[12, i] = xPred[11, i]

        return xPredWithScalar

    def solve(self, x0, verbose=False, ref=None):
        """
        Solve MPC.
        """
        x_meas = self.remove_quat_scalar(x0)

        # Set reference, create zero reference
        if ref is None:
            yref = np.array([0.0, 0.0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0, 0, 0])
        else:
            yref = self.remove_quat_scalar(ref[:, 0])

        # Check if EKF was initialized
        if not hasattr(self, 'ekf'):
            # Append nominal disturbance
            self.ekf_setup(x_meas)

        # Solve MPC:
        x_hat = self.ekf.get_state()

        uPred, xPred = self.mpc.compute_predicted_optimal_controls(x_hat[:12], yref, x_hat[12:])
        u_current = uPred[:, 0]

        # Debugging with verbose:
        if verbose and False:
            print("x_hat: (", x_hat.shape, ") , ", x_hat)
            print("yref: (", yref.shape, ") , ", yref)
            print("x_meas: (", x_meas.shape, ") , ", x_meas)

        self.ekf.prediction_update(u_current)
        self.ekf.measurement_update(x_meas, u_current)

        # Add quaternion scalar part
        xPredWithScalar = self.add_quat_scalar(xPred)

        # Finish instance one?
        if verbose:
            print("\hat(d): ", self.ekf.get_state()[12:])
            print("Done!")

        # Convert to numpy
        uPred = np.array(uPred)
        xPredWithScalar = np.array(xPredWithScalar)

        return uPred.T, xPredWithScalar.T
