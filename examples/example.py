# Example usage of tool
import pickle
from hullopt.simulations.analytic import run
from hullopt.simulations.params import Params
from hullopt.hull.utils import generate_random_hulls
from hullopt.hull import Hull
from hullopt.config.defaults import dummy_hull
import os
from hullopt.gps.strategies.compare import compare_models    
from hullopt.gps.strategies.kernels import HydroPhysicsKernel, StandardMaternKernel, ConfigurablePhysicsKernel
from hullopt.gps.strategies.priors import HydrostaticBaselinePrior, ZeroMeanPrior
from hullopt.gps.utils import load_simulation_data
from sklearn.model_selection import train_test_split
from hullopt.gps.base_functions import create_gp, update_gp
from hullopt.gps.gp import GaussianProcessSurrogate
from tqdm import tqdm
import warnings


# Configuration variables here
DATA_PATH = "gp_data.pkl"
MODEL_PATH = "models/boat_gp.pkl"
KERNEL_CONFIG = {"rocker_bow": "matern52", "heel": "periodic"}

# Redirect warnings to file
warnings.filterwarnings('default')
logging_file = open('warnings.log', 'w')
def warning_handler(message, category, filename, lineno, file=None, line=None):
    logging_file.write(f'{category.__name__}: {message}\n')
warnings.showwarning = warning_handler


# Initial data gathering for GP
if not os.path.exists(DATA_PATH):
    # First step: we generate random hulls with constraints
    hulls = generate_random_hulls(n=2, cockpit_opening=False, seed=42)
    print(f"Generated {len(hulls)} hulls for initial data.")

    for idx, hull in enumerate(hulls):        
        # Second step: Run simulations at different heel angles
        for k in tqdm(range(31), desc=f"Simulating hull {idx+1}/{len(hulls)}"):
            result = run(hull, Params(heel=0.1*k))

    
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

# Close the logging file at the end
logging_file.close()

# with open("gp_data.pkl", "rb") as f_read:
#     while True:
#         try:
#             entry = pickle.load(f_read)
#             print("Read back:", entry)
#         except EOFError:
#             break



