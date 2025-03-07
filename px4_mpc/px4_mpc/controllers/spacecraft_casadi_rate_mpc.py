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
from px4_mpc.models.spacecraft_rate_model import SpacecraftRateModel
import time

class SpacecraftCasadiRateMPC():
    def __init__(self, model:SpacecraftRateModel, Tf=1.0, N=10, add_cbf=False):
        self.model = model

        self.Tf = Tf
        self.N = N
        self.dt = self.Tf/self.N

        self.add_cbf = add_cbf

        self.x0 = np.array([0.01, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0])

        self.nx = model.nx
        self.nu = model.nu
        self.ny = self.nx + self.nu
        self.ny_e = self.nx

        self.params = {}
        self.vars = {}

        self.ocp = self.setup()

        self.Q = np.diag([5e0, 5e0, 5e0, 8e-1, 8e-1, 8e-1, 8e3])
        self.Q_e = 10 * self.Q
        self.R = 2*np.diag([1e-2, 1e-2, 1e-2, 2e0, 2e0, 2e0])

        if self.add_cbf:
            p_r, v_r = cs.SX.sym('p_r', 3), cs.SX.sym('v_r', 3)
            q_r, u_r = cs.SX.sym('q_r', 4), cs.SX.sym('u_r', self.nu)
            p_o, v_o = cs.SX.sym('p_o', 3), cs.SX.sym('v_o', 3)
            q_o, u_o = cs.SX.sym('q_o', 4), cs.SX.sym('u_o', self.nu)
            h = cs.sumsqr(p_r[0:2] - p_o[0:2]) - (0.2+0.2+0.1)**2 # 2 times the radius of the object + 0.1 m
            x = cs.vertcat(p_r,p_o)
            dx = cs.vertcat(v_r,v_o)

            X_r = cs.vertcat(p_r,v_r,q_r)
            X_o = cs.vertcat(p_o,v_o,q_o)
            U_r = u_r
            U_o = u_o

            # dh = dh/dx \dot{x}
            # ddh = ddh/dx \dot{x} + ddh/ddx \ddot{x}
            dh = cs.jacobian(h,x) @ dx
            ddh = cs.jacobian(dh,x) @ dx + cs.jacobian(dh,dx) @ cs.vertcat(self.model.get_dx(X_r,u_r)[3:6],
                                                                           self.model.get_dx(X_o,u_o)[3:6])

            # now create functions out of these
            # the 2nd order CBF constraint: ddh + alpha*dh + beta*h >= 0
            self.h = cs.Function('h', [X_r,X_o], [h])
            self.dh = cs.Function('dh', [X_r,X_o], [dh])
            self.ddh = cs.Function('ddh', [X_r,X_o,U_r,U_o], [ddh])
            self.beta = 1e0 # h
            self.alpha  = 2e0 # dh

    def setup(self):
        # create ocp object to formulate the OCP
        ocp = cs.Opti()
        # set variables: state and control input
        X = ocp.variable(self.nx,self.N+1)
        U = ocp.variable(self.nu,self.N)
        self.vars['X'] = X
        self.vars['U'] = U
        # set parameters: current state and reference state, cost matrices
        x0 = ocp.parameter(self.nx)
        xref = ocp.parameter(self.nx,self.N+1)
        Q = ocp.parameter(self.nx-3,self.nx-3)
        Q_e = ocp.parameter(self.nx-3,self.nx-3)
        R = ocp.parameter(self.nu,self.nu)
        self.params['x0'] = x0
        self.params['xref'] = xref
        self.params['Q'] = Q
        self.params['Q_e'] = Q_e
        self.params['R'] = R

        ## CONSTRAINTS
        # initial state constraint
        ocp.subject_to(X[:,0] == x0)

        # dynamics constraints
        for i in range(self.N):
            # ocp.subject_to(self.model.get_euler_integration(X[:,i],U[:,i],self.dt) == X[:,i+1])
            ocp.subject_to(self.model.get_rk4_integration(X[:,i],U[:,i],self.dt) == X[:,i+1])
        
        # control input constraints
        u_ub = np.hstack((np.repeat([self.model.max_thrust],3),np.repeat([self.model.max_rate],3)))
        u_lb = -u_ub
        for i in range(self.N):
            ocp.subject_to(u_lb <= U[:,i])
            ocp.subject_to(U[:,i] <= u_ub)
 
        # potential collision avoidance CBF constraint
        if self.add_cbf:
            # CBF variables: slack variable
            delta = ocp.variable(1)
            self.vars['delta'] = delta
            # CBF parameters: object state and control, offswitch (big slack)
            X_o = ocp.parameter(self.nx)
            U_o = ocp.parameter(self.nu)
            OffSwitch = ocp.parameter(1)
            self.params['X_o'] = X_o
            self.params['U_o'] = U_o
            self.params['OffSwitch'] = OffSwitch

            # the CBF gets assigned to the first time instance (linear constraint)
            X_r = X[:,0]
            U_r = U[:,0]
            ocp.subject_to(self.ddh(X_r,X_o,U_r,U_o) + self.alpha*self.dh(X_r,X_o) + \
                           self.beta*self.h(X_r,X_o) >= -delta - OffSwitch)
            ocp.subject_to(delta >= 0)            

        ## COST
        # standard trajectory tracking cost
        cost_eq = 0
        for i in range(self.N):
            cost_eq += self.calculate_state_error(X[:,i], xref[:,i], Q)
            cost_eq += U[:,i].T @ R @ U[:,i]
        cost_eq += self.calculate_state_error(X[:,-1], xref[:,-1], Q_e)
        
        # potential CBF slack cost
        if self.add_cbf:
            cost_eq += 100*delta

        # and minimize the sum of the costs
        ocp.minimize(cost_eq)

        ## SOLVER SETTINGS
        # IPOPT
        # opts = {'ipopt.print_level': 1, 'print_time': 0, 'ipopt.sb': 'yes',
        #         'verbose':False}
        # ocp.solver('ipopt',opts)
        
        # qpoases
        nlp_options = {
            "qpsol": "qrqp",
            "hessian_approximation": "gauss-newton",
            "max_iter": 100,
            "tol_du": 1e-2,
            "tol_pr": 1e-2,
            "qpsol_options": {"sparse":True, "hessian_type": "posdef", "numRefinementSteps":1}
        }
        ocp.solver('sqpmethod',nlp_options)

        return ocp

    def calculate_state_error(self, x, xref, Q):
        # state: p, v, q
        es = x - xref
        es = es[0:6]
        cost_es = es.T @ Q[0:6,0:6] @ es

        # quaternion cost
        q = x[6:10].reshape((4,1))
        qref = xref[6:10].reshape((4,1))
        eq = 1 - (q.T @ qref)**2 
        cost_eq = eq.T @ Q[6,6].reshape((1, 1)) @ eq

        return cost_eq + cost_es
    
    def solve(self, x0, ref,
              weights={'Q': None, 'Q_e': None, 'R': None},
              initial_guess={'X': None, 'U': None},
              xobj=None, enable_cbf=True,
              verbose=False):
        
        t0 = time.time()

        # set initial guess if we are getting any
        if initial_guess['X'] is not None:
            self.ocp.set_initial(self.vars['X'], initial_guess['X'])
        if initial_guess['U'] is not None:
            self.ocp.set_initial(self.vars['U'], initial_guess['U'])

        # set x0 parameter
        self.ocp.set_value(self.params['x0'], x0)

        # set setpoints parameter
        self.ocp.set_value(self.params['xref'], ref[:10,:])

        # set cost matrices if we are getting any
        Q = self.Q if weights['Q'] is None else weights['Q']
        Q_e = self.Q_e if weights['Q_e'] is None else weights['Q_e']
        R = self.R if weights['R'] is None else weights['R']
        self.ocp.set_value(self.params['Q'], Q)
        self.ocp.set_value(self.params['Q_e'], Q_e)
        self.ocp.set_value(self.params['R'], R)

        # set other object parameters if we should add the cbf (otherwise these
        # parameters do not exist)
        if xobj is not None and self.add_cbf:
            self.ocp.set_value(self.params['X_o'], xobj)
            self.ocp.set_value(self.params['U_o'], np.zeros((self.nu,1)))
            # sometimes we might wish to disable the cbf under certain conditions
            # to enable this, we add a significant slack to the CBF constraint
            if enable_cbf:
                self.ocp.set_value(self.params['OffSwitch'], 0)
            else:
                self.ocp.set_value(self.params['OffSwitch'], 10000) # make it a trivial constraint

        try:
            sol = self.ocp.solve()
            X_pred = sol.value(self.vars['X'])
            U_pred = sol.value(self.vars['U'])
        except Exception as e:
            print(f"Optimization failed: {e}") if verbose else None
            X_pred = np.zeros((self.nx, self.N+1))
            U_pred = np.zeros((self.nu, self.N))

        print(f"time taken: {time.time()-t0}") if verbose else None

        # transpose the obtained state and control to be consistent with the acados setup
        X_pred, U_pred = X_pred.T, U_pred.T
        return U_pred, X_pred
