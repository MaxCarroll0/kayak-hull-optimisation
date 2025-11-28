# kayak-hull-optimisation
Efficiently optimising a kayak hull under constant fluid flows for a customisable compound metric assessing hydrodynamicity, stability, and buoyancy using Gaussian Processes. 

## Method
### Inputs
- Hull Parameter Constraints
- Fluid Flow Direction
- Fluid Flow Speed (by Galilean Relativity, this theoretically accounts also for the Kayak speed)
- Weight of the Kayaker

### Outputs
Optimised hull parameters maximising the quality of the hull by a compound metric taking into account:
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

The hybrid and dynamic approaches are theoretically both as _accurate / high fidelity_ as each other, but differ greatly in terms of numerical stability and cost. The acquisition functions are balanced taking to samples from multiple fidelities with based on expected cost and numerical stability.

### Analytic
1) Ignores fluid flow and calculates draught, reserve buoyancy, and righting modment for a given heel angle based on the geometry of the hull.

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

### Dynamic
A single _dynamic_ fluid simulation with interDyMFoam accounting for 6DoF transformations of the mesh in the fluid. i.e. like my6DoFFoam.

1) Calculate the analytic results for some heel angle
2) Place the hull in an initial position based on the analytic results
3) Applying a (variable) torque to try stabilise the hull at the desired heel angle. Initial value will be the analytic righting moment.
4) Iterate through one simulation until righting moment stabilises
5) Calculate hydrodynamicity in final iteration and use stabilised righting moment

**Fidelity** should be similar to that of the Hybrid method. 

**Cost** is proportional to only the mesh accuracy and fluid discretisation plus the extra cost in running interDyMFoam over interFoam. (This is likely lower than the Hybrid method for high fluid flows)

**Numerical Stability** is inherently more unstable compared to the static method, and also risks oscillations or total collapse in the righting moment. Requires even more careful setting of hyperparameters.

## Acquisition Functions

## Contrained Bayesian Optimisation

## (Optional) Sensitivity Analysis
