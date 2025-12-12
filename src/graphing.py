import numpy as np
import matplotlib.pyplot as plt
from . import *

def plot_RM(xs, ys):
    """
    x: input heel angles
    y: 3d righting moments
    """
    def remove_discontinuities(ys):
        ys = np.asarray(ys).reshape(-1)
        threshold = 25 * np.median(np.abs(np.diff(ys)))
        jumps = np.abs(np.diff(ys)) > threshold
        ys[1:][jumps] = np.nan
        return ys
    
    plt.figure(figsize=(10,5))
    # Heel curves
    ys_heel = remove_discontinuities([y.righting_moment_heel() for y in ys])
    ys_pitch = remove_discontinuities([y.righting_moment_pitch() for y in ys])
    ys_yaw = remove_discontinuities([y.righting_moment_yaw() for y in ys])
    plt.plot(xs, ys_heel, label="Heel righting moment")
    plt.plot(xs, ys_pitch, label="Pitch righting moment")
    plt.plot(xs, ys_yaw, label="Yaw righting moment")

    # Mark discontinuities
    first = True
    for idy, y in enumerate(ys_heel + ys_pitch + ys_yaw):
        if np.isnan(y):
            plt.axvline(xs[idy], color='red', linestyle=':', label=('Discontinuities (hull flooded)' if first else None))
            first = False
    
    plt.xlabel("Heel angle (rad)")
    plt.ylabel("Righting Moment (Nm)")
    plt.title("Righting Moments for Heel Angles")
    plt.legend()
    plt.show()

def plot_simulation(run, hull, lower = -np.pi, upper = np.pi, resolution = 100):
    """
    run: simulation runner
    lower: heel angle (rads) lower bound
    upper: heel angle (rads) upper bound
    resolution: number of samples
    """
    heel_angles = np.linspace(lower, upper, resolution)
    results = [run(hull, simulations.Params(heel=heel)) for heel in heel_angles]
    plot_RM(heel_angles, results)
