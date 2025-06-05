import casadi as cs

class CollisionAvoidanceCBF:
    def __init__(self, model):
        """ Initialize the Collision Avoidance Control Barrier Function (CBF).
        :param model: The model of the system for which the CBF is being defined.
        """
        self.model = model

    def setup(self, ocp, X, U):
        """
        Setup the CBF constraints in the OCP.
        """
        p_r, v_r = cs.SX.sym('p_r', 3), cs.SX.sym('v_r', 3)
        q_r, u_r = cs.SX.sym('q_r', 4), cs.SX.sym('u_r', self.model.nu)
        p_o, v_o = cs.SX.sym('p_o', 3), cs.SX.sym('v_o', 3)
        q_o, u_o = cs.SX.sym('q_o', 4), cs.SX.sym('u_o', self.model.nu)
        h = cs.sumsqr(p_r[0:2] - p_o[0:2]) - (0.2 + 0.2 + 0.1) ** 2  # 2 times the radius of the object + 0.1 m
        x = cs.vertcat(p_r, p_o)
        dx = cs.vertcat(v_r, v_o)

        X_r = cs.vertcat(p_r, v_r, q_r)
        X_o = cs.vertcat(p_o, v_o, q_o)
        U_r = u_r
        U_o = u_o

        # dh = dh/dx \dot{x}
        # ddh = ddh/dx \dot{x} + ddh/ddx \ddot{x}
        dh = cs.jacobian(h, x) @ dx
        ddh = cs.jacobian(dh, x) @ dx + cs.jacobian(dh, dx) @ cs.vertcat(self.model.get_dx(X_r, u_r)[3:6],
                                                                         self.model.get_dx(X_o, u_o)[3:6])

        # now create functions out of these
        # the 2nd order CBF constraint: ddh + alpha*dh + beta*h >= 0
        self.h = cs.Function('h', [X_r, X_o], [h])
        self.dh = cs.Function('dh', [X_r, X_o], [dh])
        self.ddh = cs.Function('ddh', [X_r, X_o, U_r, U_o], [ddh])
        self.beta = 1e0    # h
        self.alpha  = 2e0  # dh

        # CBF variables: slack variable
        delta = ocp.variable(1)
        # CBF parameters: object state and control, offswitch (big slack)
        X_o = ocp.parameter(self.nx)
        U_o = ocp.parameter(self.nu)
        OffSwitch = ocp.parameter(1)
        params = {'X_o': X_o, 'U_o': U_o, 'OffSwitch': OffSwitch}

        # the CBF gets assigned to the first time instance (linear constraint)
        X_r = X[:, 0]
        U_r = U[:, 0]
        ocp.subject_to(self.ddh(X_r, X_o, U_r, U_o) + self.alpha * self.dh(X_r, X_o)
                       + self.beta * self.h(X_r, X_o) >= -delta - OffSwitch)
        ocp.subject_to(delta >= 0)
        return ocp, params, delta
