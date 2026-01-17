# Example usage of tool
import pickle
from hullopt.simulations.analytic import run
from hullopt.simulations.params import Params
from hullopt.config.defaults import dummy_hull
<<<<<<< HEAD
import os
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

    for k in range(30):
        result = run(dummy_hull, Params(heel=0.1*k))

    
X_full, y_full, column_order = load_simulation_data(DATA_PATH)

X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full, test_size=0.2, random_state=42
    )

rmse = create_gp(
    GaussianProcessSurrogate(ConfigurablePhysicsKernel(KERNEL_CONFIG), ZeroMeanPrior()),
    X_train,
    y_train,
    column_order,
    X_test,
    y_test
)
print(f"Initial GP RMSE: {rmse}")
=======
from hullopt.hull.utils import generate_random_hulls
>>>>>>> a0d4fdc790bc7271cb96b08e680b430daae34043


# First step: we generate random hulls with consraints
hulls = generate_random_hulls(n=20, cockpit_opening=False, seed=42)

<<<<<<< HEAD
=======
# Second step: We run a simulation for a given heel angle:
for hull in hulls[:1]:
    for k in range(31):
        result = run(hull, Params(heel=0.1*k))

with open("gp_data.pkl", "rb") as f_read:
    while True:
        try:
            entry = pickle.load(f_read)
            print("Read back:", entry)
        except EOFError:
            break


>>>>>>> a0d4fdc790bc7271cb96b08e680b430daae34043

