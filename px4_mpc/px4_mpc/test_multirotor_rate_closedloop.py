############################################################################
#
#   Copyright (C) 2023 PX4 Development Team. All rights reserved.
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

from visualization import plot_multirotor
from px4_mpc.controllers.multirotor_rate_mpc import MultirotorRateMPC
from px4_mpc.models.multirotor_rate_model import MultirotorRateModel
import numpy as np

def main(use_RTI=False):
    model = MultirotorRateModel()

    mpc_controller = MultirotorRateMPC(model)

    x0 = mpc_controller.x0
    Tf = mpc_controller.Tf
    N_horizon = mpc_controller.N
    Fmax = model.max_thrust
    wmax = model.max_rate

    ocp_solver = mpc_controller.ocp_solver
    integrator = mpc_controller.integrator

    nx = ocp_solver.acados_ocp.dims.nx
    nu = ocp_solver.acados_ocp.dims.nu

    Nsim = 100
    simX = np.ndarray((Nsim+1, nx))
    simU = np.ndarray((Nsim, nu))

    simX[0,:] = x0

    t_feedback = np.zeros((Nsim))

    # closed loop
    for i in range(Nsim):
        input, _ = mpc_controller.solve(simX[i, :])
        simU[i,:] = input[0, :]
        t_feedback[i] = mpc_controller.ocp_solver.get_stats('time_tot')
        # simulate system
        simX[i+1, :] = integrator.simulate(x=simX[i, :], u=simU[i,:])

    # # evaluate timings
    t_feedback *= 1000
    print(f'Computation time in feedback phase in ms:    \
            min {np.min(t_feedback):.3f} median {np.median(t_feedback):.3f} max {np.max(t_feedback):.3f}')

    # plot results
    plot_multirotor(model, np.linspace(0, (Tf/N_horizon)*Nsim, Nsim+1), simU, simX)

    ocp_solver = None


if __name__ == '__main__':
    main(use_RTI=True)
