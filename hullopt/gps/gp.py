import os
import pickle
import numpy as np
import GPy
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error
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
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    
    from strategies.kernels import HydroPhysicsKernel, StandardMaternKernel
    from strategies.priors import HydrostaticBaselinePrior
    from utils import load_simulation_data

    DATA_FILE = "gp_data.pkl"
    MODEL_PATH = "models/boat_gp.pkl"

    # 1. Load Data
    print("Loading simulation data...")
    X_full, y_full, col_map = load_simulation_data(DATA_FILE)

    # 2. Split Data (80% Train, 20% Test)
    # We need a held-out test set to fairly calculate "accuracy"
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X_full, y_full, test_size=0.2, random_state=42
    )

    # 3. Initialize the two strategies/models you defined
    k_strat = HydroPhysicsKernel()
    p_strat = HydrostaticBaselinePrior(col_map)
    
    # The two models to compare
    gp1 = GaussianProcessSurrogate(k_strat, p_strat)              # Hydro Physics
    gp2 = GaussianProcessSurrogate(StandardMaternKernel(), p_strat) # Standard Matern

    # 4. Tracking Loop (Learning Curve)
    # We test with 20%, 40%, 60%, 80%, and 100% of the training data
    ratios = [0.2, 0.4, 0.6, 0.8, 1.0]
    rmse_gp1 = []
    rmse_gp2 = []

    print(f"\n--- Starting Comparison (Test Set Size: {len(X_test)}) ---")

    for ratio in ratios:
        # Create a subset of the training data
        n = int(len(X_train_full) * ratio)
        X_sub = X_train_full[:n]
        y_sub = y_train_full[:n]
        
        print(f"Training on {n} samples ({int(ratio*100)}%)...")

        # --- Fit & Evaluate GP1 (Hydro) ---
        gp1.fit(X_sub, y_sub, col_map)
        mu1, _ = gp1.predict(X_test)
        err1 = np.sqrt(mean_squared_error(y_test, mu1))
        rmse_gp1.append(err1)

        # --- Fit & Evaluate GP2 (Matern) ---
        gp2.fit(X_sub, y_sub, col_map)
        mu2, _ = gp2.predict(X_test)
        err2 = np.sqrt(mean_squared_error(y_test, mu2))
        rmse_gp2.append(err2)

    # 5. Plotting
    
    plt.figure(figsize=(10, 6))
    
    # Plot GP1
    plt.plot(ratios, rmse_gp1, label='GP1: HydroPhysics Kernel', 
             marker='o', linestyle='-', linewidth=2, color='blue')
    
    # Plot GP2
    plt.plot(ratios, rmse_gp2, label='GP2: Standard Matern', 
             marker='s', linestyle='--', linewidth=2, color='orange')

    plt.title('Model Accuracy Tracking: RMSE vs Data Size')
    plt.xlabel('Fraction of Training Data')
    plt.ylabel('RMSE (Lower is better)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    print("\nDisplaying plot...")
    plt.show()

    # Optional: Save the best GP1 (Hydro) model using 100% of data
    print(f"Saving fully trained GP1 to {MODEL_PATH}...")
    gp1.save(MODEL_PATH)