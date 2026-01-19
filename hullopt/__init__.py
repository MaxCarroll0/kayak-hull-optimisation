from .hull import Hull # Directly export Hull class (must be done before config)
from hullopt import hull, config, simulations, gps, optimise, graphing

# Aliases
ParamsSim = simulations.Params
ParamsHull = hull.Params

__all__ = ["config", "hull", "gps", "simulations", "graphing", "optimise", "Hull", "ParamsSim", "ParamsHull"]
