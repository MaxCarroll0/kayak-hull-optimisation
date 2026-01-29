import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import mpl_axes_aligner
from collections import defaultdict
from hullopt.gps.utils import load_simulation_data
import hullopt
from hullopt import Hull

from hullopt import simulations

def plot_heels(ps, rs):
    """
    ps: input simulation parameters
    rs: output simulation results
    """
    def remove_discontinuities(ys):
        ys = np.asarray(ys).reshape(-1)
        threshold = 25 * np.median(np.abs(np.diff(ys)))
        jumps = np.abs(np.diff(ys)) > threshold
        ys[:-1][jumps] = np.nan
        return ys

    xs = [p.heel for p in ps]
    rs_pos = list(map(lambda x: x[1], filter(lambda xr: xr[0] >= 0, zip(xs, rs))))
    rs_neg = list(map(lambda x: x[1], filter(lambda xr: xr[0] < 0, zip(xs, rs))))
    
    fig, ax1 = plt.subplots(figsize=(10,5))
    ax1.grid(True)
    ax1.set_xlabel("Heel angle (rad)")
    ax1.set_xticks(np.arange(-np.pi, np.pi+0.01, 1/8*np.pi))
    plt.title("Righting Moments and Reserve Buoyancies for Heel Angles")

    # Moment curves
    def cleanup_helper(f):
        # To ensure discontinuities are removed at symmetrical points
        return list(reversed(remove_discontinuities(list(reversed([f(r) for r in rs_neg]))))) + list(remove_discontinuities([f(r) for r in rs_pos]))
    
    ms_heel = cleanup_helper(lambda x: x.righting_moment_heel())
    ms_pitch = cleanup_helper(lambda x: x.righting_moment_pitch())
    ms_yaw = cleanup_helper(lambda x: x.righting_moment_yaw())

    print(ms_heel)
    
    ax1.plot(xs, ms_heel, label="Heel righting moment")
    ax1.plot(xs, ms_pitch, label="Pitch righting moment")
    ax1.plot(xs, ms_yaw, label="Yaw righting moment")

    # Mark discontinuities
    first = True
    discontinuities = []
    for idm, y in enumerate(ms_heel + ms_pitch + ms_yaw):
        if np.isnan(y) and idm < len(xs):
            plt.axvline(xs[idm], color='red', linestyle=':', label=('Discontinuities (hull flooded)' if not discontinuities else None))
            discontinuities += [idm]

    ax1.set_ylabel("Righting Moment (Nm)")

    # Buoyancy curve
    ax2 = ax1.twinx()
    bs = np.asarray([r.reserve_buoyancy for r in rs])
    bhs = np.asarray([r.reserve_buoyancy_hull for r in rs])
    bs[discontinuities] = np.nan
    bhs[discontinuities] = np.nan
    ax2.plot(xs, bhs, color='wheat', label="Reserve buoyancy (from unsubmerged hull)")
    ax2.plot(xs, bs, color='grey', label="Reserve buoyancy")


    ax2.set_ylabel("Reserve buoyancy (kg)")

    # Align x-axes in center of figure
    mpl_axes_aligner.align.yaxes(ax1, 0, ax2, 0, 0.5)
    
    fig.legend()
    plt.savefig("righting_moments.png")
    plt.show()

def plot_simulation(simulation, hull, lower = -np.pi, upper = np.pi, resolution = 101):
    """
    simulation: simulation (simulation.analytic, simulation.static, etc.)
    lower: heel angle (rads) lower bound
    upper: heel angle (rads) upper bound
    resolution: number of samples
    """
    params = list(map(simulations.Params, np.linspace(lower, upper, resolution)))
    results = list(map(partial(simulation.run, hull), params))
    plot_heels(params, results)

def plot_pickle(hull_index = 0):
    xs, ys, column_order = load_simulation_data("./gp_data.pkl")
    i = column_order.index("heel")
    j = column_order.index("cost") # TODO: Make cost an output!
    print(j)
    split = list(zip(map(lambda x:
                         (list(x[:j]) + list(x[j+1:i]) + list(x[i+1:]) if j < i else
                          list(x[:i]) + list(x[i+1:j]) + list(x[j+1:]),
                          (x[i])),
                         xs),
                     ys))

    seen = []
    j = 0
    for (hull, _heel), _y in split:
        if not any(np.array_equal(hull, hull2) for hull2 in seen):
            seen += [hull]
            j += 1
        if j > hull_index:
            break

    def wrap(x):
        return x if x <= np.pi else x - 2 * np.pi
        
    data = list(map(lambda x: (wrap(x[0][1]), x[1]), filter(lambda x: np.array_equal(x[0][0], seen[-1]), split)))
    data.sort(key=lambda x: x[0])
    
    heels = [simulations.Params(d[0]) for d in data]
    rs = [simulations.Result(righting_moment = (d[1][0], d[1][1], d[1][2]),
                             reserve_buoyancy = d[1][3],
                             reserve_buoyancy_hull = d[1][4],
                             cost = 0,
                             scene = None)
          for d in data]
    print(f"Hull params: {seen[-1]}")
    print(f"Column Order: {column_order}")
    print(f"{len(heels)} samples")
    plot_heels(heels, rs)

def view_pickle(hull_index = 0, heel = 0):
    xs, ys, column_order = load_simulation_data("./gp_data.pkl")
    i = column_order.index("heel")
    j = column_order.index("cost") # TODO: Make cost an output!
    split = list(zip(map(lambda x:
                         (list(x[:j]) + list(x[j+1:i]) + list(x[i+1:]) if j < i else
                          list(x[:i]) + list(x[i+1:j]) + list(x[j+1:]),
                          (x[i])),
                         xs),
                     ys))

    seen = []
    j = 0
    for (hull, _heel), _y in split:
        if not any(np.array_equal(hull, hull2) for hull2 in seen):
            seen += [hull]
            j += 1
        if j > hull_index:
            break

    param_names = list(filter(lambda x: x != "cost" and x != "heel", column_order))
    hull_params = {k: v for (k,v) in zip(param_names, seen[-1])}
    hull_params["cockpit_opening"] = hull_params["cockpit_opening"] == 1.0    # Why is this stored / loading as a float?
    hull_obj = Hull(hullopt.ParamsHull(**hull_params))
    print(f"Hull params: {hullopt.ParamsHull(**hull_params)}")
    simulations.analytic.run(hull_obj, simulations.Params(heel)).scene.show()

def resimulate_pickle(hull_index = 0):
    xs, ys, column_order = load_simulation_data("./gp_data.pkl")
    i = column_order.index("heel")
    j = column_order.index("cost") # TODO: Make cost an output!
    split = list(zip(map(lambda x:
                         (list(x[:j]) + list(x[j+1:i]) + list(x[i+1:]) if j < i else
                          list(x[:i]) + list(x[i+1:j]) + list(x[j+1:]),
                          (x[i])),
                         xs),
                     ys))

    seen = []
    j = 0
    for (hull, _heel), _y in split:
        if not any(np.array_equal(hull, hull2) for hull2 in seen):
            seen += [hull]
            j += 1
        if j > hull_index:
            break

    param_names = list(filter(lambda x: x != "cost" and x != "heel", column_order))
    hull_params = {k: v for (k,v) in zip(param_names, seen[-1])}
    hull_params["cockpit_opening"] = hull_params["cockpit_opening"] == 1.0
    # Why is this stored / loading as a float?
    hull_obj = Hull(hullopt.ParamsHull(**hull_params))
    plot_simulation(simulations.analytic, hull_obj)
