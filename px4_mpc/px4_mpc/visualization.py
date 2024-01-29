############################################################################
#
#   Copyright (C) 2022 PX4 Development Team. All rights reserved.
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

import casadi as cs
import numpy as np
import matplotlib.pyplot as plt
from acados_template import latexify_plot

def plot_multirotor(model, shooting_nodes, U, X_true, Y_measured=None, latexify=False, X_true_label=None):
    """
    Params:
        shooting_nodes: time values of the discretization
        u_max: maximum absolute value of u
        U: arrray with shape (N_sim-1, nu) or (N_sim, nu)
        X_true: arrray with shape (N_sim, nx)
        Y_measured: array with shape (N_sim, ny)
        latexify: latex style plots
    """

    f_max = model.max_thrust
    w_max = model.max_rate

    if latexify:
        latexify_plot()

    N_sim = X_true.shape[0]
    nx = X_true.shape[1]

    Tf = shooting_nodes[N_sim-1]
    t = shooting_nodes

    Ts = t[1] - t[0]

    num_plots = 5
    
    fig = plt.figure("Information", figsize=(8, 8))

    ax0 = fig.add_subplot(num_plots, 1, 1)
    line0, = ax0.step(t, np.append([U[0, 0]], U[:, 0]), label='F')
    ax0.hlines(f_max, t[0], t[-1], linestyles='dashed', alpha=0.7)
    ax0.hlines(0.0, t[0], t[-1], linestyles='dashed', alpha=0.7)
    ax0.set_ylim([0.0, 1.2*f_max])
    ax0.set_xlim(t[0], t[-1])
    ax0.legend(loc='lower right')

    ax1 = fig.add_subplot(num_plots, 1, 2)
    line1, = ax1.step(t, np.append([U[0, 1]], U[:, 1]), label='w_x')
    line2, = ax1.step(t, np.append([U[0, 2]], U[:, 2]), label='w_y')
    line3, = ax1.step(t, np.append([U[0, 3]], U[:, 3]), label='w_z')
    ax1.hlines(w_max, t[0], t[-1], linestyles='dashed', alpha=0.7)
    ax1.hlines(-w_max, t[0], t[-1], linestyles='dashed', alpha=0.7)
    ax1.set_ylim([-1.2*w_max, 1.2*w_max])
    ax1.set_xlim(t[0], t[-1])
    ax1.legend(loc='lower right')
    ax1.legend(loc='lower right')

    ax1.set_ylabel('$u$')
    ax1.set_xlabel('$t$')

    ax1.grid()

    ax2 = fig.add_subplot(num_plots, 1, 3)
    ax2.plot(t, X_true[:, 0], label='p_x')
    ax2.plot(t, X_true[:, 1], label='p_y')
    ax2.plot(t, X_true[:, 2], label='p_z')
    ax2.set_ylabel('X [m]')
    ax2.set_xlabel('$t$')
    ax2.grid()
    ax2.legend(loc=1)
    ax2.set_xlim(t[0], t[-1])

    ax3 = fig.add_subplot(num_plots, 1, 4)
    ax3.plot(t, X_true[:, 3], label='v_x')
    ax3.plot(t, X_true[:, 4], label='v_y')
    ax3.plot(t, X_true[:, 5], label='v_z')
    ax3.set_ylabel('V [m/s]')
    ax3.set_xlabel('$t$')
    ax3.grid()
    ax3.legend(loc=1)
    ax3.set_xlim(t[0], t[-1])    

    ax4 = fig.add_subplot(num_plots, 1, 5)
    ax4.plot(t, X_true[:, 6], label=r'$q_w$')
    ax4.plot(t, X_true[:, 7], label=r'$q_x$')
    ax4.plot(t, X_true[:, 8], label=r'$q_y$')
    ax4.plot(t, X_true[:, 9], label=r'$q_z$')
    ax4.set_ylabel(r'\theta')
    ax4.set_xlabel('$t$')
    ax4.grid()
    ax4.legend(loc=1)
    ax4.set_xlim(t[0], t[-1])

    plt.tight_layout()
    plt.show()
