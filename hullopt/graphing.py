import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import mpl_axes_aligner

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
        ys[1:][jumps] = np.nan
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
        if np.isnan(y):
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

def plot_simulation(simulation, hull, lower = -np.pi, upper = np.pi, resolution = 100):
    """
    simulation: simulation (simulation.analytic, simulation.static, etc.)
    lower: heel angle (rads) lower bound
    upper: heel angle (rads) upper bound
    resolution: number of samples
    """
    params = list(map(simulations.Params, np.linspace(lower, upper, resolution)))
    results = list(map(partial(simulation.run, hull), params))
    plot_heels(params, results)
