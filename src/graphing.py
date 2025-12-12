import numpy as np
import matplotlib.pyplot as plt
from . import *

def plot_RM(xs, ys):
    """
    x: input heel angles
    y: 3d righting moments
    """
    plt.figure(figsize=(8,5))
    plt.plot(xs, [y.righting_moment_heel() for y in ys], label="Heel righting moment")
    plt.plot(xs, [y.righting_moment_pitch() for y in ys], label="Pitch righting moment")
    plt.plot(xs, [y.righting_moment_yaw() for y in ys], label="Heel righting moment")

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
