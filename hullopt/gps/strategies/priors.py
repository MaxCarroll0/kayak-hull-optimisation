import GPy
import numpy as np
from .interfaces import PriorStrategy
from typing import List

class ZeroMeanPrior(PriorStrategy):
    '''
    We know nothing. But 0 righting moment is not a great guess so ideally the other prior attempts should outperform this.
    '''
    def __init__(self):
        super().__init__(name="Zero Mean")

    def get_mean_function(self, input_dim: int, output_dim: int = 1):

        return None


class HydrostaticMapping(GPy.core.Mapping):
    def __init__(self, input_dim, output_dim, indices):
        super().__init__(input_dim=input_dim, output_dim=output_dim)
        self.indices = indices

    def f(self, X: np.ndarray) -> np.ndarray:

        N = X.shape[0]
        result = np.zeros((N, self.output_dim))

        idx_heel = self.indices['angles'][0] 
        idx_beam = self.indices['shape'][0]

        phi = X[:, idx_heel:idx_heel+1]
        beam = X[:, idx_beam:idx_beam+1]
        physics_val = 9.81 * 1025 * beam * np.sin(phi)
        result[:] = physics_val 
        
        return result

    def update_gradients(self, dL_dF, X):
        pass

class HydrostaticBaselinePrior(PriorStrategy):

    def __init__(self, col_map: dict):
        super().__init__(name="Hydrostatic Formula")
        self.col_map = col_map

    def get_mean_function(self, input_dim: int, output_dim: int = 1):
    
        return HydrostaticMapping(
            input_dim=input_dim, 
            output_dim=output_dim, 
            indices=self.col_map, 

        )

