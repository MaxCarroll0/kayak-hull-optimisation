import GPy
from .interfaces import KernelStrategy
from typing import Dict, Any

class StandardMaternKernel(KernelStrategy):
    """
    Apparently a good default if we dont know anything but we will hopefully outperform this.
    ARD is so the kernel can learn different lengthscales for each input
    """
    def __init__(self, ard: bool = True):
        super().__init__(name="Standard Matern 5/2", config={"ard": ard})

    def build(self, input_dim: int, column_map: Dict[str, Any] = None) -> GPy.kern.Kern:

        self.validate_dimensions(input_dim, 1)
        
        return GPy.kern.Matern52(
            input_dim=input_dim, 
            ARD=self.config["ard"],
            name='matern_base'
        )

class HydroPhysicsKernel(KernelStrategy):

    def __init__(self):
        super().__init__(name="Hydrodynamic Interaction Kernel")

    def build(self, input_dim: int, column_map: Dict[str, Any]) -> GPy.kern.Kern:
        """
        Requires column_map to have keys: 'speed', 'angles', 'shape'
        """
        # 1. See if we got all that we need
        required_keys = ['speed', 'angles', 'shape']
        if not all(k in column_map for k in required_keys):
            raise ValueError(f"{self.name} requires column_map with keys: {required_keys}")

        # 2. We get the indices for each aspect. SOo, we can apply the periodic kernel to the angle input, then RBF to speed, Matern to shape
        idx_speed = column_map['speed']  
        idx_angles = column_map['angles']
        idx_shape = column_map['shape']   

        self.validate_dimensions(input_dim, max(idx_shape) + 1)


        k_speed = GPy.kern.RBF(
            input_dim=len(idx_speed), 
            active_dims=idx_speed, 
            name='speed_trend'
        )
        

        k_heel = GPy.kern.StdPeriodic(input_dim=1, active_dims=[idx_angles[0]], name='heel_p')
        k_kayak = GPy.kern.StdPeriodic(input_dim=1, active_dims=[idx_angles[1]], name='kayak_p')
        k_angles = k_heel * k_kayak 
        k_shape = GPy.kern.Matern52(
            input_dim=len(idx_shape), 
            active_dims=idx_shape, 
            ARD=True, 
            name='geom_shape'
        )

        return k_speed * k_angles * k_shape