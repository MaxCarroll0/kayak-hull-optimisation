import os
import pickle
import numpy as np
import GPy
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split 

from typing import Dict, Any, Tuple, Optional, List
from strategies.interfaces import KernelStrategy, PriorStrategy



class GaussianProcessSurrogate:
    """
    Wrapper for GPy to handle boring shit
    """
    def __init__(self, 
                 kernel_strat: KernelStrategy, 
                 prior_strat: PriorStrategy):
        self.k_strat = kernel_strat
        self.p_strat = prior_strat
        self.model: Optional[GPy.models.GPRegression] = None

    def fit(self, X: np.ndarray, y: np.ndarray, column_order: List[str]) -> None:
        """
        Constructs the kernel/prior from strategies and optimizes the GP.
        """
        if self.model is not None:
            print("Warning: Overwriting existing model.")

        input_dim = X.shape[1]
        

        kernel = self.k_strat.build(input_dim, column_order)
        mean_func = self.p_strat.get_mean_function(input_dim, output_dim=y.shape[1])
        

        self.model = GPy.models.GPRegression(X, y, kernel=kernel, mean_function=mean_func, normalizer=True)
        self.model.kern.constrain_bounded(1e-3, 1000.0, warning=False)
        self.model.optimize(messages=True)

    def predict(self, X_new: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (mean, variance) for new inputs.
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained or loaded.")
        return self.model.predict(X_new)

    def save(self, filepath: str) -> None:
        """Serializes the trained model to a pickle file."""
        if self.model is None:
            raise RuntimeError("No model to save.")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"GP Model saved to: {filepath}")

    def load(self, filepath: str) -> bool:
        """
        Loads model if file exists. Returns True if successful, False otherwise.
        """
        if not os.path.exists(filepath):
            return False
            
        try:
            with open(filepath, 'rb') as f:
                self.model = pickle.load(f)
            print(f"GP Model loaded from: {filepath}")
            return True
        except (pickle.PickleError, EOFError) as e:
            print(f"Failed to load model from {filepath}: {e}")
            return False
        
        
if __name__ == "__main__":   
    from strategies.compare import compare_models    
    from strategies.kernels import HydroPhysicsKernel, StandardMaternKernel, ConfigurablePhysicsKernel
    from strategies.priors import HydrostaticBaselinePrior, ZeroMeanPrior
    from utils import load_simulation_data

    DATA_FILE = "gp_data.pkl"
    MODEL_PATH = "models/boat_gp.pkl"


    X_full, y_full, column_order = load_simulation_data(DATA_FILE)

    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full, test_size=0.2, random_state=42
    )


    p_strat = HydrostaticBaselinePrior(column_order)
    z_strat = ZeroMeanPrior()

    config_dict = {"rocker_bow": "matern52", "heel": "periodic" }

    models_to_compare = {
        "HydroPhysics": GaussianProcessSurrogate(ConfigurablePhysicsKernel(config_dict), p_strat),
        "Standard Everything": GaussianProcessSurrogate(StandardMaternKernel(), z_strat),

    }

    compare_models(models_to_compare, X_train, y_train, X_test, y_test, column_order)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)


    print(f"\nSaving fully trained HydroPhysics model to {MODEL_PATH}...")
    final_gp = models_to_compare["HydroPhysics"]
    # final_gp.fit(X_train, y_train, col_map) Gets already fitted inside the compare models function
    final_gp.save(MODEL_PATH)