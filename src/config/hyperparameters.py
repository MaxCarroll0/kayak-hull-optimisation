"""
Values/functions of hyperparameters
"""
import numpy as np
# Analytic Simulator
draught_threshold: float = 0.0001  # 99.99% accuracy in draught level
draught_max_iterations: int = 100

# Simulation cost weightings & functions
cost_analytic_weight: float = 1
cost_static_weight: float = 2  # TODO: Set hyperparams

def cost_analytic(iterations: int) -> float:
    return iterations * cost_analytic_weight

def cost_static(iterations: int, discretisation: float) -> float:
    return 0 * cost_static_weight # TODO
