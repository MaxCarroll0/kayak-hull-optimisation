# Example usage of tool
import pickle
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataclasses import dataclass
from hullopt.hull.constraints import Constraints
from hullopt.gps.strategies.compare import compare_models
from hullopt.gps.strategies.kernels import HydroPhysicsKernel, StandardMaternKernel, ConfigurablePhysicsKernel
from hullopt.gps.strategies.priors import HydrostaticBaselinePrior, ZeroMeanPrior
from hullopt.gps.aggregator import Aggregator
from hullopt.gps.utils import load_simulation_data
from gui import WeightSelector, ResultVisualizer
from sklearn.model_selection import train_test_split
from hullopt.gps.base_functions import create_gp, update_gp
from hullopt.gps.gp import GaussianProcessSurrogate
from hullopt.optimise import optimise
from hullopt.hull import Hull
import numpy as np
import hullopt


# Configuration variables here
DATA_PATH = "gp_data.pkl"
BUOYANCY_MODEL_PATH = "models/boat_buoyancy_gp.pkl"
RIGHTING_MODEL_PATH = "models/boat_righting_gp.pkl"
KERNEL_CONFIG_HYDRO_PROD = {"length": "rbf",
                 "beam": "rbf",
                 "depth": "rbf",
                 "cross_section_exponent": "matern52",
                 "beam_position": "matern52",
                 "rocker_bow": "matern52",
                 "rocker_sterm": "matern52",
                 "rocker_position": "matern52",
                 "rocker_exponent": "matern52",
                 "heel": "periodic_matern",
}

KERNEL_CONFIG_HYDRO_SUM = {"length": "rbf",
                 "beam": "rbf",
                 "depth": "rbf",
                 "cross_section_exponent": "matern52",
                 "beam_position": "matern52",
                 "rocker_bow": "matern52",
                 "rocker_sterm": "matern52",
                 "rocker_position": "matern52",
                 "rocker_exponent": "matern52",
                 "heel": "sum_periodic_matern" }
KERNEL_CONFIG_HYDRO_PERIODIC = {"length": "rbf",
                 "beam": "rbf",
                 "depth": "rbf",
                 "cross_section_exponent": "matern52",
                 "beam_position": "matern52",
                 "rocker_bow": "matern52",
                 "rocker_sterm": "matern52",
                 "rocker_position": "matern52",
                 "rocker_exponent": "matern52",
                 "heel": "periodic" }
KERNEL_CONFIG_MATERN = {"length": "rbf",
                 "beam": "rbf",
                 "depth": "rbf",
                 "cross_section_exponent": "matern52",
                 "beam_position": "matern52",
                 "rocker_bow": "matern52",
                 "rocker_sterm": "matern52",
                 "rocker_position": "matern52",
                 "rocker_exponent": "matern52",
                 "heel": "matern52" }

KERNEL_CONFIG_RBF = {"length": "rbf",
                 "beam": "rbf",
                 "depth": "rbf",
                 "cross_section_exponent": "rbf",
                 "beam_position": "rbf",
                 "rocker_bow": "rbf",
                 "rocker_sterm": "rbf",
                 "rocker_position": "rbf",
                 "rocker_exponent": "rbf",
                 "heel": "rbf" }

KERNEL_CONFIG_LINEAR = {"length": "linear",
                 "beam": "linear",
                 "depth": "linear",
                 "cross_section_exponent": "linear",
                 "beam_position": "linear",
                 "rocker_bow": "linear",
                 "rocker_sterm": "linear",
                 "rocker_position": "linear",
                 "rocker_exponent": "linear",
                 "heel": "linear" }


# Initial data gathering for GP
if not os.path.exists(DATA_PATH):
    print(os.getcwd())
    from hullopt.hull.utils import generate_random_hulls
    from hullopt.config.defaults import dummy_hull
    
    from hullopt.simulations.params import Params
    from hullopt.simulations.analytic import run
    hulls = generate_random_hulls(n=10, cockpit_opening=True, seed=42)
    # Second step: We run a simulation for a given heel angle:
    for idx, hull in enumerate(hulls):
        print("Simulating random hull: " + str(idx))
        r = np.random.random()*62
        for k in range(int(r)):
            heel = np.random.random()*5*np.pi - 3*np.pi
            result = run(hull, Params(heel=heel))

    
X_full, y_full, column_order = load_simulation_data(DATA_PATH)

X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full, test_size=0.2, random_state=42
    )

def remove_costs(Xs):
    i = column_order.index("cost")
    print(i)
    return np.delete(Xs, i, axis=1)

#X_train = remove_costs(X_train)
#X_text = remove_costs(X_test)

# --- Batch 1: Righting (First 3 cols) ---
if os.path.exists(RIGHTING_MODEL_PATH):
    print(f"Loading {RIGHTING_MODEL_PATH}...")
    with open(RIGHTING_MODEL_PATH, 'rb') as f:
        gp_righting = GaussianProcessSurrogate(model=pickle.load(f))
else:
    print("Training Batch 1 (Righting)...")
    gps = [GaussianProcessSurrogate(ConfigurablePhysicsKernel(KC), ZeroMeanPrior()) for KC in (KERNEL_CONFIG_HYDRO_PROD, KERNEL_CONFIG_HYDRO_SUM, KERNEL_CONFIG_HYDRO_PERIODIC, KERNEL_CONFIG_MATERN, KERNEL_CONFIG_RBF, KERNEL_CONFIG_LINEAR)]
    
    compare_models({"HYDRO_PROD": gps[0], "HYDRO_SUM": gps[1], "HYDRO_PERIODIC": gps[2], "MATERN": gps[3], "RBF": gps[4], "LINEAR": gps[5]},
        X_train, y_train[:, :1], X_test, y_test[:, :1], column_order, file_name="righting_gp_comparison_plot.png")
        
    gp_righting = gps[0]
    gp_righting.save(RIGHTING_MODEL_PATH)

# --- Batch 2: Buoyancy (Last 2 cols) ---
if os.path.exists(BUOYANCY_MODEL_PATH):
    print(f"Loading {BUOYANCY_MODEL_PATH}...")
    with open(BUOYANCY_MODEL_PATH, 'rb') as f:
        gp_buoyancy = GaussianProcessSurrogate(model=pickle.load(f))
else:
    print("Training Batch 2 (Buoyancy)...")
    gps = [GaussianProcessSurrogate(ConfigurablePhysicsKernel(KC), ZeroMeanPrior()) for KC in (KERNEL_CONFIG_HYDRO_PROD, KERNEL_CONFIG_HYDRO_SUM, KERNEL_CONFIG_HYDRO_PERIODIC, KERNEL_CONFIG_MATERN, KERNEL_CONFIG_RBF, KERNEL_CONFIG_LINEAR)]
    
    compare_models({"HYDRO_PROD": gps[0], "HYDRO_SUM": gps[1], "HYDRO_PERIODIC": gps[2], "MATERN": gps[3], "RBF": gps[4], "LINEAR": gps[5]},
        X_train, y_train[:, -2:], X_test, y_test[:, -2:], column_order, file_name="buoyancy_gp_comparison_plot.png")
        
    gp_buoyancy = gps[0]
    gp_buoyancy.save(BUOYANCY_MODEL_PATH)


@dataclass
class GP_Result:
    overall_stability: float
    initial_stability: float
    diminishing_stability: float
    tipping_point: float
    righting_energy: float
    overall_buoyancy: float
    initial_buoyancy: float

user_weights = WeightSelector(GP_Result).run()
time = user_weights["time"]
del user_weights["time"]
aggregator = Aggregator(user_weights, gp_righting, gp_buoyancy, column_order)
f = aggregator.f

best_params = optimise(f, Constraints(), time=time)
print("Optimised!!")

visualizer = ResultVisualizer(best_params, hullopt.optimise.best_dict, hullopt.optimise.best_score, Hull)
visualizer.run()
