from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import GPy
import numpy as np

class BaseStrategy(ABC):
    """
    Common functionality for all Strategies
    """
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}

    def __repr__(self):
        return f"<{self.__class__.__name__}: {self.name}>"

    def get_config(self) -> Dict[str, Any]:
        return self.config

class KernelStrategy(BaseStrategy):
    """
    Base class for Kernel construction.
    Adds validation to ensure input dimensions match physical expectations.
    """
    @abstractmethod
    def build(self, input_dim: int, column_map: Dict[str, Any]) -> GPy.kern.Kern:
        """
        Constructs the GPy Kernel object.
        :param input_dim: Total columns in X
        :param column_map: Dict mapping physics to indices, e.g. {'speed': [0], 'shape': [3,4,5]}
        """
        pass

    def validate_dimensions(self, input_dim: int, required_dims: int):
        """Helper to ensure we have enough data columns for the strategy."""
        if input_dim < required_dims:
            raise ValueError(f"Strategy {self.name} requires at least {required_dims} input dimensions, got {input_dim}.")

class PriorStrategy(BaseStrategy):
    """
    Base class for Mean Function construction.
    Adds ability to define analytical formulas.
    """
    @abstractmethod
    def get_mean_function(self, input_dim: int, output_dim: int = 1) -> GPy.core.Mapping:
        pass

    def _check_columns_exist(self, X: np.ndarray, indices: List[int]):
        """Helper for concrete priors to validate data at runtime."""
        if X.shape[1] <= max(indices):
            raise IndexError(f"Prior {self.name} tries to access index {max(indices)} but input only has {X.shape[1]} columns.")