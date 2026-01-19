import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List
from hullopt.gps.gp import GaussianProcessSurrogate
from sklearn.metrics import mean_squared_error





def compare_models(
    models: Dict[str, GaussianProcessSurrogate],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    column_order: List[str],
    ratios: List[float] = [0.2, 0.4, 0.6, 0.8, 1.0]
) -> None:
    """
    Trains multiple GP models on increasing subsets of data and plots the RMSE comparison.
    """
    print(f"\n--- Starting Comparison (Test Set Size: {len(X_test)}) ---")
    
    # Dictionary to store results for each model: {'ModelName': [rmse_20, rmse_40...]}
    results = {name: [] for name in models.keys()}
    valid_ratios = []
    
    for name, gp in models.items():
        for ratio in ratios:
            n = int(len(X_train) * ratio)
        
            # Safety check for very small datasets
            if n < 1:
                print(f"Skipping {int(ratio*100)}% (0 samples calculated)")
                continue
            
            valid_ratios.append(ratio)
            X_sub = X_train[:n]
            y_sub = y_train[:n]
        
            print(f"Training on {n} samples ({int(ratio*100)}%)...")

            try:
                # We catch errors here so one failing model doesn't crash the whole loop
                gp.fit(X_sub, y_sub, column_order)
                mu, _ = gp.predict(X_test)
                rmse = np.sqrt(mean_squared_error(y_test, mu))
                results[name].append(rmse)
            except Exception as e:
                print(f"  Err training {name}: {e}")
                results[name].append(np.nan) # Append NaN to maintain list length

    # --- Plotting ---
    if not valid_ratios:
        print("Error: Not enough data to train models.")
        return

    plt.figure(figsize=(10, 6))
    
    markers = ['o', 's', '^', 'D'] # Different markers for different lines
    for i, (name, rmse_scores) in enumerate(results.items()):
        plt.plot(valid_ratios, rmse_scores, 
                 label=name, 
                 marker=markers[i % len(markers)], 
                 linewidth=2)

    plt.title(f'Model Accuracy Tracking (N_total={len(X_train) + len(X_test)})')
    plt.xlabel('Fraction of Training Data')
    plt.ylabel('RMSE (Lower is better)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plot_path = "models/gp_comparison_plot.png"
    plt.savefig(plot_path)
    print(f"\nComparison plot saved to {plot_path}")
    plt.show()
