import os
import pickle
import numpy as np
import GPy
from typing import Dict, Any, Tuple, Optional
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

    def fit(self, X: np.ndarray, y: np.ndarray, col_map: Dict[str, Any]) -> None:
        """
        Constructs the kernel/prior from strategies and optimizes the GP.
        """
        if self.model is not None:
            print("Warning: Overwriting existing model.")

        input_dim = X.shape[1]
        

        kernel = self.k_strat.build(input_dim, col_map)
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
    from strategies.kernels import HydroPhysicsKernel
    from strategies.priors import HydrostaticBaselinePrior



    rho = 1025
    g = 9.81
    MODEL_PATH = "models/boat_gp.pkl"
    
    X_train = np.random.rand(20, 5)
    

    y_train = (rho * g) * np.sin(X_train[:, 1:2]) * X_train[:, 0:1]
    
    col_map = {'speed': [0], 'angles': [1, 2], 'shape': [3, 4]}

   
    k_strat = HydroPhysicsKernel()
    p_strat = HydrostaticBaselinePrior(speed_idx=0, heel_idx=1, beam_idx=3)
    
    gp = GaussianProcessSurrogate(k_strat, p_strat)

    if False: #gp.load(MODEL_PATH):
        print("Skipping training, using loaded model.")
    else:
        print("No model found. Training...")
        gp.fit(X_train, y_train, col_map)
        gp.save(MODEL_PATH)

    # 5. Predict
    X_new = np.random.rand(3, 5)
    mu, var = gp.predict(X_new)
    print("\nPredictions (Mean):\n", mu)