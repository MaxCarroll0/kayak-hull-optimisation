
import GPy
from .interfaces import KernelStrategy
from typing import Dict, Any, List, Union
from collections import defaultdict
class ConfigurablePhysicsKernel(KernelStrategy):
    """
    This is my newest attempt. Ideally you should be able to pass a config dict to use all the different kernels in the kernel registry.
    Important: If the input is not in the config it will not be tracked! 
    """
    
    KERNEL_REGISTRY = {
        'rbf': GPy.kern.RBF,
        'matern52': GPy.kern.Matern52,
        'matern32': GPy.kern.Matern32,
        'periodic': GPy.kern.StdPeriodic,
        'linear': GPy.kern.Linear,
        'white': GPy.kern.White,
        'bias': GPy.kern.Bias,
        'cosine': GPy.kern.Cosine
    }

    def __init__(self, config_map: Dict[str, str]):
        """
        :param config_map: Dictionary mapping each fucking column_map keys to kernel types.
                           Example: {'speed': 'rbf', 'angles': 'periodic', 'shape': 'matern52'}
        """
        super().__init__(name="Configurable Physics Kernel", config=config_map)

    def build(self, input_dim: int, parameter_order: List[str]) -> GPy.kern.Kern:
        """
        Just goes through the column_map. If the key exists in self.config,
        it builds that specific kernel for those specific columns.
        """
        column_map = defaultdict(list)
        for idx, param_name in enumerate(parameter_order):
            column_map[param_name].append(idx)
        kernels: List[GPy.kern.Kern] = []
        
        for phys_key, indices in column_map.items():
            
            if phys_key not in self.config:
                continue

            k_type_str = self.config[phys_key].lower()
            if k_type_str not in self.KERNEL_REGISTRY:
                raise ValueError(f"Unknown kernel type '{k_type_str}' requested for '{phys_key}'. Supported: {list(self.KERNEL_REGISTRY.keys())}")
            
            kern_cls = self.KERNEL_REGISTRY[k_type_str]


            # Special handling for Periodic: usually applied per-dimension (1D) 
            # to allow different periods for different angles (yaw vs heel)
            if k_type_str in ['periodic', 'cosine']:
                for i, idx in enumerate(indices):
                    sub_k = kern_cls(
                        input_dim=1, 
                        active_dims=[idx], 
                        name=f"{phys_key}_{i}"
                    )
                    kernels.append(sub_k)
            
            # Standard handling for RBF/Matern (Multivariate)
            else:
                sub_k = kern_cls(
                    input_dim=len(indices),
                    active_dims=indices,
                    ARD=True, # Allow different lengthscales for different shape params. Important guys we keep this to True
                    name=phys_key
                )
                kernels.append(sub_k)

        if not kernels:
            raise ValueError(f"Config resulted in no kernels! \nConfig: {self.config.keys()} \nMap: {column_map.keys()}")


        final_kernel = kernels[0]
        for k in kernels[1:]:
            final_kernel = final_kernel * k

        return final_kernel

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