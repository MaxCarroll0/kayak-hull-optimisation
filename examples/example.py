# Example usage of tool
import pickle
from hullopt.simulations.analytic import run
from hullopt.simulations.params import Params
from hullopt.config.defaults import dummy_hull




# First step: We run a simulation for a given heel angle:
for k in range(31):
    result = run(dummy_hull, Params(heel=0.1*k))



with open("gp_data.pkl", "rb") as f_read:
    while True:
        try:
            entry = pickle.load(f_read)
            print("Read back:", entry)
        except EOFError:
            break



