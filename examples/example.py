# Example usage of tool
import pickle
from hullopt.simulations.analytic import run
from hullopt.simulations.params import Params
from hullopt.config.defaults import dummy_hull
from hullopt.hull.utils import generate_random_hulls


# First step: we generate random hulls with consraints
hulls = generate_random_hulls(n=20, cockpit_opening=False, seed=42)

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



