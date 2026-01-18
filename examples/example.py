# Example usage of tool
import pickle
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from hullopt.gps.strategies.compare import compare_models
from hullopt.gps.strategies.kernels import HydroPhysicsKernel, StandardMaternKernel, ConfigurablePhysicsKernel
from hullopt.gps.strategies.priors import HydrostaticBaselinePrior, ZeroMeanPrior
from hullopt.gps.utils import load_simulation_data
from gui import WeightSelector
from sklearn.model_selection import train_test_split
from hullopt.gps.base_functions import create_gp, update_gp
from hullopt.gps.gp import GaussianProcessSurrogate




# Configuration variables here
DATA_PATH = "gp_data.pkl"
MODEL_PATH = "models/boat_gp.pkl"
KERNEL_CONFIG = {"rocker_bow": "matern52", "heel": "periodic" }



# Initial data gathering for GP
if not os.path.exists(DATA_PATH):
    from hullopt.hull.utils import generate_random_hulls
    from hullopt.config.defaults import dummy_hull
    print("this shouldnt happen anymore")
    from hullopt.simulations.params import Params
    from hullopt.simulations.analytic import run
    hulls = generate_random_hulls(n=20, cockpit_opening=False, seed=42)
    # Second step: We run a simulation for a given heel angle:
    i = 0
    for hull in hulls[:1]:
        print("Simulating random hull: " + str(i))
        for k in range(301):
            result = run(hull, Params(heel=0.1*k))
else:
    with open("gp_data.pkl", "rb") as f_read:
        while True:
            try:
                entry = pickle.load(f_read)
                print("Read back:", entry)
                break
            except EOFError:
                break
    exit()
    
X_full, y_full, column_order = load_simulation_data(DATA_PATH)

X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full, test_size=0.2, random_state=42
    )

rmse = compare_models({"cool":
    GaussianProcessSurrogate(ConfigurablePhysicsKernel(KERNEL_CONFIG), ZeroMeanPrior())},
    X_train,
    y_train,
    X_test,
    y_test,
    column_order,
)
print(f"Initial GP RMSE: {rmse}")








