# kayak-hull-optimisation
Efficiently optimising a kayak hull under constant fluid flows for a customisable compound metric assessing hydrodynamicity, stability, and buoyancy using Gaussian Processes. 

## Method
### User Inputs
- Hull Parameter Constraints
- Fluid Flow Direction
- Fluid Flow Speed (by Galilean Relativity, this theoretically accounts also for the Kayak speed)
- Weight of the Kayaker

### Outputs
**Optimised hull parameters** maximising the quality of the hull by a compound metric taking into account:
#### Hydrodynamicity:
- Drag opposite the forward motion of the hull
#### Stability:
- Tipping Point
- Total Stability
- Diminishing Stability Point
- Initial Stability (Stiffness)
#### Buoyancy
- Reserve Buoyancy

## Hull Parameterisation
TODO

## Hull Constraints
TODO

## Simulation
4 simulation models of increasing compute power and fidelity. 

All work from a **mesh** approximation of the hull, with reserve buoyancy calculated directly by the analytic method.

The hybrid and dynamic approaches are theoretically both as _accurate / high fidelity_ as each other, but differ greatly in terms of numerical stability and cost. If doing a multi-fidelity model, the multi-fidelity acquisition function can be balanced based on expected **cost** and **numerical stability**.

### Analytic
Ignores fluid flow and calculates:
- Hull mass
- Hull moments of inertia
- Hull Draught
- Reserve Buoyancy
- Righting modment for a given heel angle based on the hull geometry.

**Fidelity** is low due to non-account of fluid flow. But, still is a good baseline/sanity check, being the most accurate, stable, and efficient for low fluid flows.

**Cost** is proportional to the accuracy of the mesh.

**Numirical Stability** is perfect.

### Static
Using interFoam with a fixed hull.

1) Calculates the analytic results for some heel angle
2) Fixes the hull in a fluid at its expected draught in calm water
3) Simulates drag and moment forces upon the hull until a rough steady-state is reached

**Fidelity** is higher as it accounts for fluid flow. But, the hull is fixed so the fluid flow may cause excess buoyancy which is unable to take effect, rresulting in an incorrect steady-state draught.

**Cost** is proportional to the product of mesh accuracy and the discretisation of the fluid simulation.

**Numerical Stability** is fine except with high fluid flows, where turbuence may significantly vary and not settle into a steady state.

### Hybrid
Iterating the static method.

1) Calculate the excess buoyancy in the static method
2) Adjust the draught accordingly
3) Iterate entire static simulations until negligible buoyancy
4) Use drag / moment force results from the final iteration

**Fidelity** now accounts for excess buoyancy and should be theoretically correct under a **steady-state** assumption.

**Cost** is proportional to the product of the iterations, mesh accuracy, and the discretisation of the fluid simulation.

**Numerical Stability** compounds the possibility of non-convergence of draught (**i.e. oscillations**) with the inherent Static method instability. Setting hyperparameters carefully alleviates this.

### (Optional) Dynamic
A single _dynamic_ fluid simulation with interDyMFoam accounting for 6DoF transformations of the mesh in the fluid. i.e. like my6DoFFoam.

1) Calculate the analytic results for some heel angle
2) Place the hull in an initial position based on the analytic results
3) Applying a (variable) torque to try stabilise the hull at the desired heel angle. Initial value will be the analytic righting moment.
4) Iterate through one simulation until righting moment stabilises
5) Calculate hydrodynamicity in final iteration and use stabilised righting moment

**Fidelity** should be similar to that of the Hybrid method. 

**Cost** is proportional to only the mesh accuracy and fluid discretisation plus the extra cost in running interDyMFoam over interFoam. (This is likely lower than the Hybrid method for high fluid flows)

**Numerical Stability** is inherently more unstable compared to the static method, and also risks oscillations or total collapse in the righting moment. In particular, stability will be much worse for greater heel angles. Requires even more careful setting of hyperparameters.

## Acquisition Functions and the basic Gaussian Process
### Model
For **constant hull parameters**, learn how the value of the compound output metric (hydrodynamicity, stability, buoyancy) varys over heel angles and fluid flows.

One way to create an acquisition function on this is to run **two GPs** modelling the hydrodynamicity and stability curves, but determine points to sample on _both_ the curves by an acquisition function on the compound metric directly. Creating an intuitive kernel for this seems difficult, but we could just throw RBF or Matern at it.

### Compount Acquisition Functions
Splitting the GP in two (on the hydrodynamicity curve and stability curve) we can get more intuitive results and design more effect acquisition functions and kernels. But there is additional complexity on deciding how to effectively balance the multiple models.

Differing acquisition functions make sense for each part of the compound metric, we wish to optimise all of the following over varying heel angles and fluid flows, each of which will have a *differing acquisition function*:
- Integral of the stability curve up to the tipping point _(force required to capsize from flat)_
- Maximum of the stability curve _(point of diminishing stability)_
- Positive root of the stability curve _(tipping point)_
- Gradient of the stability curve at zero _(stiffness)_
- Integral of the hydrodynamic foward drag weighted by small angles heel angle between up to the point of diminishing stability _(total drag over 'realistic' heel angles)_
  
Note that _reserve buoyancy_ is modelled as just a constant, we don't need to sample the functions.

These compound acquisition functions are simply balanced in exactly the same ratios that output metric is composed of. i.e. if we care 50% about tipping point and 50% about hydrodynamicity, we would randomly sample 50% of the time from the tipping point acquisition functions and 50% of the time from the hydrodynamicity acquisition function.

**TODO:** Research if this simple method has mathematical grounding. It seems unlikely as we don't take into account how sure each acquisition function is that it has already optimised its respective output. For example, suppose we have found the tipping point by exploring extreme heel angles, but have not yet found the hydrodynamicity, we would still keep on sampling extreme heel angles when this is no longer needed. 

### Multi-fidelities
We can either sample from one simulation fidelity or take into account all 4 and estimate cost and numerical stability of each fidelity for improved efficiency. The simulation cost can be estimated and iteratively refined by taking previous results and intuition about the inherent cost and stability of the simulations (noted down above). 

The fidelities are **NOT** linearly related in general. But, for a given fluid flow, a linear relationship should hold proved the hull shape is not super weird. So the join-Gaussian property will be preserved for specified fluid flows. Equally, small variations in fluid flow, especially for calm fluids should be roughly linearly related.

## Contrained Bayesian Optimisation of the Gaussian Process
Optimise the hull parameters under the input constraints to maximise the output metric for the given fluid flow (and kayaker weight). The fluid flows to consider could just be determined by the user (i.e. a 'calm' scenario vs a 'white water rafting' scenario).

**TODO**

## (Optional) Sensitivity Analysis
**TODO** Probably not worth doing...
