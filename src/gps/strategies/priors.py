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
        """
        In this version we use a chatgpt formula for physics to estimate the righting moement.
        This could be a good starting point to get the prior mean but can definitely be improved upon.
        """

        phi = X[:, self.indices['heel']:self.indices['heel']+1]
        beam = X[:, self.indices['beam']:self.indices['beam']+1]
        

        return 9.81 * 1025 * beam * np.sin(phi)

    def update_gradients(self, dL_dF, X):
        pass

class HydrostaticBaselinePrior(PriorStrategy):

    def __init__(self, speed_idx: int, heel_idx: int, beam_idx: int):
        super().__init__(name="Hydrostatic Formula")
        self.idx_speed = speed_idx
        self.idx_heel = heel_idx
        self.idx_beam = beam_idx

    def get_mean_function(self, input_dim: int, output_dim: int = 1):
        indices = {'speed': self.idx_speed, 'heel': self.idx_heel, 'beam': self.idx_beam}

        return HydrostaticMapping(
            input_dim=input_dim, 
            output_dim=output_dim, 
            indices=indices, 

        )

