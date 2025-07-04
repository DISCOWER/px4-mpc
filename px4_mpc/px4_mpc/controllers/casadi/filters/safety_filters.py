import casadi as cs
import numpy as np
from abc import ABC, abstractmethod

from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from px4_msgs.msg import VehicleLocalPosition, VehicleAttitude


class SafetyFilter(Node):

    def __init__(self, name: str):
        super().__init__(name)

    @abstractmethod
    def set_h_constants(self):
        """
        Set the constants (e.g. A matrix) for the safety filter.
        This method should be implemented in subclasses to define specific constants.
        """
        pass

    @abstractmethod
    def set_controller_constants(self):
        """
        Set the controller constants (e.g. alpha, beta, gamma) for the safety filter.
        This method can be overridden in subclasses to define specific constants.
        """
        pass

    @abstractmethod
    def setup(self, model, ocp:cs.Opti, X:cs.MX, U:cs.MX):
        """
        Setup the safety filter constraints in the OCP.
        :param ocp: The optimal control problem to which the constraints will be added.
        :param X: The state variables of the system.
        :param U: The control inputs of the system.
        :return: The updated OCP and slack variable.
        """
        pass

    @abstractmethod
    def update_params(self, ocp:cs.Opti):
        """
        Update the parameters of the safety filter.
        params are parameters in the OCP and contain e.g. the position of the other robot.
        Hence, every controller call, these need to be updated.
        """
        pass

class HalfSpaceSafetyFilter(SafetyFilter):
    def __init__(self):
        """ Initialize the Half Space Safety Filter.
        :param model: The model of the system for which the safety filter is being defined.
        """
        super().__init__('half_space_safety_filter')
    
    def set_h_constants(self, A: np.ndarray, b: np.ndarray):
        """
        Set the constants for the half-space safety filter.
        :param A: The matrix defining the half-space constraints.
        :param b: The vector defining the half-space constraints.
        """
        assert A is not None, "Matrix A must be provided for half-space constraints."
        assert b is not None, "Vector b must be provided for half-space constraints."
        assert A.shape[0] == b.shape[0], "Matrix A and vector b must have compatible dimensions."
        self.A = cs.SX(A) if isinstance(A, np.ndarray) else A
        self.b = cs.SX(b) if isinstance(b, np.ndarray) else b
    
    def set_controller_constants(self, alpha: float = 2.0, beta: float = 1.0):
        """
        Set the controller constants for the half-space safety filter.
        :param alpha: CBF coefficient (default: 2.0): ddh + alpha*dh + beta*h >= 0.
        :param beta: CBF coefficient (default: 1.0): ddh + alpha*dh + beta*h >= 0.
        """
        self.alpha = alpha
        self.beta = beta

    def setup(self, model, ocp: cs.Opti, X: cs.MX, U: cs.MX):
        """
        Setup the half-space safety filter constraints in the OCP.
        :param ocp: The optimal control problem to which the constraints will be added.
        :param X: The state variables of the system.
        :param U: The control inputs of the system.
        :return: The updated OCP and slack variable.
        """
        if not hasattr(self, 'A') or not hasattr(self, 'b'):
            raise ValueError("Constants A and b must be set before calling setup.")
        if not hasattr(self, 'alpha') or not hasattr(self, 'beta'):
            self.set_controller_constants()  # Set default constants if not set

        self.model = model

        # Inequality constraints: A * x <= b
        p_r, v_r = cs.SX.sym('p_r', 3), cs.SX.sym('v_r', 3)
        q_r, u_r = cs.SX.sym('q_r', 4), cs.SX.sym('u_r', self.model.nu)
        x = cs.vertcat(p_r)
        dx = cs.vertcat(v_r)

        X_r = cs.vertcat(p_r, v_r, q_r)
        U_r = u_r

        # Create the half-space CBF constraints
        h = self.b - self.A @ x  # Half-space constraint
        dh = cs.jacobian(h, x) @ dx  # First derivative of h
        ddh = cs.jacobian(dh, x) @ dx + cs.jacobian(dh, dx) @ self.model.get_dx(X_r, u_r)[3:6]

        # Create functions for h, dh, and ddh
        self.h = cs.Function('h', [X_r], [h])
        self.dh = cs.Function('dh', [X_r], [dh])
        self.ddh = cs.Function('ddh', [X_r, U_r], [ddh])

        # Slack variable for the half-space safety filter
        delta = ocp.variable(1)
        self.params = {}
        # TODO: what if X and U are not the first time instance?
        # Constrain the CBF with the variables in the optimization
        X_r = X[:, 0]  # Use the first time instance for the CBF
        U_r = U[:, 0]  # Use the first control input for the CBF
        ocp.subject_to(self.ddh(X_r, U_r) + self.alpha * self.dh(X_r) + self.beta * self.h(X_r) >= -delta)
        ocp.subject_to(delta >= 0)

        return ocp, delta

    def update_params(self, ocp: cs.Opti):
        """
        Half-space is currently not parameterized, so this method does nothing.
        """
        pass

class ObjectAvoidanceFilter(SafetyFilter):
    def __init__(self, object_ns:str):
        super().__init__('object_avoidance_filter')
        # As the object avoidance filter is a ros2 node,
        # we let it subscribe to the object state and update its parameters
        # for the optimizer internally. As such, create a subscriber to the position
        qos_profile_sub = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=0
        )
        self.local_position_sub = self.create_subscription(
            VehicleLocalPosition,
            f'/{object_ns}/fmu/out/vehicle_local_position',
            self.vehicle_local_position_callback,
            qos_profile=qos_profile_sub
        )
        self.attitude_sub = self.create_subscription(
            VehicleAttitude,
            f'/{object_ns}/fmu/out/vehicle_attitude',
            self.vehicle_attitude_callback,
            qos_profile_sub)
        self.vehicle_local_position = np.array([0.0, 0.0, 0.0])  # Initialize with zeros
        self.vehicle_local_velocity = np.array([0.0, 0.0, 0.0])  # Initialize with zeros
        self.vehicle_attitude = np.array([1.0, 0.0, 0.0, 0.0])  # Initialize with identity quaternion

    def vehicle_local_position_callback(self, msg):
        # TODO: handle NED->ENU transformation
        self.vehicle_local_position[0] = msg.x
        self.vehicle_local_position[1] = -msg.y
        self.vehicle_local_position[2] = -msg.z
        self.vehicle_local_velocity[0] = msg.vx
        self.vehicle_local_velocity[1] = -msg.vy
        self.vehicle_local_velocity[2] = -msg.vz
    def vehicle_attitude_callback(self, msg):
        # TODO: handle NED->ENU transformation
        self.vehicle_attitude[0] = msg.q[0]
        self.vehicle_attitude[1] = msg.q[1]
        self.vehicle_attitude[2] = -msg.q[2]
        self.vehicle_attitude[3] = -msg.q[3]

    def set_h_constants(self, object_radius: float = 0.2, margin: float = 0.1,):
        """
        Set the constants for the object avoidance filter.
        :param object_radius: The radius of the object to avoid (default: 0.2).
        :param margin: The margin to add to the object radius (default: 0.1).
        """
        self.object_radius = object_radius
        self.margin = margin

    def set_controller_constants(self, alpha: float = 2.0, beta: float = 1.0):
        """
        Set the controller constants for the object avoidance filter.
        :param alpha: CBF coefficient (default: 2.0): ddh + alpha*dh + beta*h >= 0.
        :param beta: CBF coefficient (default: 1.0): ddh + alpha*dh + beta*h >= 0.
        """
        self.alpha = alpha
        self.beta = beta
    

    def setup(self, model, ocp: cs.Opti, X: cs.MX, U: cs.MX):
        """
        Setup the object avoidance filter constraints in the OCP.
        :param ocp: The optimal control problem to which the constraints will be added.
        :param X: The state variables of the system.
        :param U: The control inputs of the system.
        :return: The updated OCP and slack variable.
        """
        if not hasattr(self, 'object_radius') or not hasattr(self, 'margin'):
            self.set_h_constants()  # Set default constants if not set
        if not hasattr(self, 'alpha') or not hasattr(self, 'beta'):
            self.set_controller_constants()
            
        self.model = model

        p_r, v_r = cs.SX.sym('p_r', 3), cs.SX.sym('v_r', 3)
        q_r, u_r = cs.SX.sym('q_r', 4), cs.SX.sym('u_r', self.model.nu)
        p_o, v_o = cs.SX.sym('p_o', 3), cs.SX.sym('v_o', 3)
        q_o, u_o = cs.SX.sym('q_o', 4), cs.SX.sym('u_o', self.model.nu)
        x = cs.vertcat(p_r, p_o)
        dx = cs.vertcat(v_r, v_o)

        X_r = cs.vertcat(p_r, v_r, q_r)
        X_o = cs.vertcat(p_o, v_o, q_o)
        U_r = u_r
        U_o = u_o

        # Create the object avoidance CBF constraints
        # the 2nd order CBF constraint: ddh + alpha*dh + beta*h >= 0
        h = cs.sumsqr(p_r[0:2] - p_o[0:2]) - (2*self.object_radius+self.margin) ** 2
        dh = cs.jacobian(h, x) @ dx
        ddh = cs.jacobian(dh, x) @ dx + cs.jacobian(dh, dx) @ cs.vertcat(self.model.get_dx(X_r, u_r)[3:6],
                                                                         self.model.get_dx(X_o, u_o)[3:6])

        # Create functions for h, dh, and ddh
        self.h = cs.Function('h', [X_r, X_o], [h])
        self.dh = cs.Function('dh', [X_r, X_o], [dh])
        self.ddh = cs.Function('ddh', [X_r, X_o, U_r, U_o], [ddh])

        # CBF variables: slack variable
        delta = ocp.variable(1)
        # CBF parameters: object state and control, offswitch (big slack)
        X_o = ocp.parameter(self.nx)
        U_o = ocp.parameter(self.nu)
        self.params = {'X_o': X_o, 'U_o': U_o}

        # the CBF gets assigned to the first time instance (linear constraint)
        # TODO: what if X and U are not the first time instance?
        X_r = X[:, 0]
        U_r = U[:, 0]
        ocp.subject_to(self.ddh(X_r, X_o, U_r, U_o) + self.alpha * self.dh(X_r, X_o)
                       + self.beta * self.h(X_r, X_o) >= -delta)
        ocp.subject_to(delta >= 0)
        return ocp, delta

    def update_params(self, ocp: cs.Opti):
        """
        Update the parameters of the object avoidance filter.
        This method updates the object state and control inputs in the OCP.
        """
        X_o = np.hstack((self.vehicle_local_position, self.vehicle_local_velocity, self.vehicle_attitude))
        U_o = np.zeros(self.model.nu)  # Assuming the object has no control input
        # Update the parameters with the current state of the object
        ocp.set_value(self.params['X_o'], X_o)
        ocp.set_value(self.params['U_o'], U_o)

