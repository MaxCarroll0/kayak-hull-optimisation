import GPy
from .interfaces import KernelStrategy
from typing import Dict, Any, List

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
        Dynamically builds kernel based on available keys in column_map.
        Skips missing components without failing.
        """
        
        # We will collect valid sub-kernels here
        kernels: List[GPy.kern.Kern] = []

        # --- 1. Speed (RBF) ---
        if 'speed' in column_map and column_map['speed']:
            idx_speed = column_map['speed']
            k_speed = GPy.kern.RBF(
                input_dim=len(idx_speed), 
                active_dims=idx_speed, 
                name='speed_trend'
            )
            kernels.append(k_speed)
        
        # --- 2. Angles (Periodic) ---
        # Handle flexible number of angles (e.g. just heel, or heel+yaw)
        if 'angles' in column_map and column_map['angles']:
            idx_angles = column_map['angles']
            k_angles_prod = None
            
            for i, idx in enumerate(idx_angles):
                k_p = GPy.kern.StdPeriodic(
                    input_dim=1, 
                    active_dims=[idx], 
                    name=f'angle_{i}_p'
                )
                if k_angles_prod is None:
                    k_angles_prod = k_p
                else:
                    k_angles_prod = k_angles_prod * k_p
            
            if k_angles_prod:
                kernels.append(k_angles_prod)

        # --- 3. Shape (Matern) ---
        if 'shape' in column_map and column_map['shape']:
            idx_shape = column_map['shape']
            k_shape = GPy.kern.Matern52(
                input_dim=len(idx_shape), 
                active_dims=idx_shape, 
                ARD=True, 
                name='geom_shape'
            )
            kernels.append(k_shape)

        if not kernels:

            print(f"Warning: {self.name} found no recognizable columns. Defaulting to Matern.")
            return GPy.kern.Matern52(input_dim=input_dim, ARD=True)

        # Multiply all collected kernels: k1 * k2 * k3...
        final_kernel = kernels[0]
        for k in kernels[1:]:
            final_kernel = final_kernel * k

        return final_kernel