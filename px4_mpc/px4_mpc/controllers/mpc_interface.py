import casadi as ca
import importlib
from typing import List
from px4_mpc.controllers.casadi.filters.safety_filters import SafetyFilter

class MPCInterface:
    def __init__(self, vehicle, mode:str, framework:str, safety_filters: List[SafetyFilter]=[]):
        self.vehicle = vehicle
        self.mode = mode
        self.framework = framework
        self.safety_filters = safety_filters

        # TODO: dictionary of supported vehicles, frameworks and modes
        if self.vehicle not in ["spacecraft", "quadrotor"]:
            raise ValueError("Vehicle must be either 'spacecraft' or 'quadrotor'")

        if framework not in ["acados", "casadi"]:
            raise ValueError("Framework must be either 'acados' or 'casadi'")

        if mode not in ["rate", "wrench", "direct_allocation"]:
            raise ValueError("Mode must be either 'rate', 'wrench', or 'direct_allocation'")

        # Dynamically import model module based on vehicle and framework
        try:
            model_module_path = f"px4_mpc.models.{vehicle}_{mode}_model"
            model_module = importlib.import_module(model_module_path)
            model_class_name = f"{vehicle.capitalize()}{mode.capitalize()}Model"
            model_class = getattr(model_module, model_class_name)

            # Dynamically import the controller module based on vehicle, mode, and framework
            module_path = f"px4_mpc.controllers.{framework}.{vehicle}_{mode}_mpc"
            control_module = importlib.import_module(module_path)
            controller_class_name = f"{vehicle.capitalize()}{mode.capitalize()}MPC"
            controller_class = getattr(control_module, controller_class_name)
        except Exception as e:
            raise ValueError(f"Could not import model or controller for {vehicle} in {framework} with mode {mode}.") from e

        # Instantiate the model and controller classes
        self.model = model_class()
        try:
            self.mpc = controller_class(self.model, safety_filters=self.safety_filters)
        except TypeError as e:
            if 'safety_filters' in str(e):
                raise TypeError(f"Can currently not instantiate safety filters for {vehicle} in {framework} with mode {mode}.") from e
            else:
                raise 
        
    

    def __getattr__(self, name):
        # Pass upwards any mpc-related attributes or methods
        return getattr(self.mpc, name)
