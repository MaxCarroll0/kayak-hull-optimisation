from hullopt import Hull, simulations, config
from hullopt.gps.gp import GaussianProcessSurrogate
from hullopt.gps.base_functions import update_gp

from typing import Tuple
from scipy.stats import norm
import numpy as np

# Expected Improvement to find the maximum
def a_EI_max(f_star, Xs, mu, varSigma):
    alpha = np.zeros(mu[0].shape)
    for i in range(0,Xs.shape[0]):
        alpha[i] = (mu[i][0] - f_star)*norm.cdf(f_star,mu[i][0],np.sqrt(varSigma[i][0])) + varSigma[i][0]*norm.pdf(f_star,mu[i][0],np.sqrt(varSigma[i][0]))
    return alpha

# 'Sign-change' acquisition to find roots
# SC(x) = p(y = 0) i.e. for y ~ N(mu(x), varSigma(x))
# Only consider if LARGER than point of diminishing stability (maximum)
def a_SC(dim, Xs, mu, varSigma):
    return [norm.pdf(0, m[0], s[0]).item() if x > dim else 0 for (x, (m, s)) in zip(Xs, zip(mu, varSigma))]

# Integrals
# Maximum variance sampling (up to point)
# TODO: Explore Quadrature acquisition functions
def a_INT(bounds, Xs, mu, varSigma):
    return [varSigma[x][0] if bounds[0] <= x < bounds[1] else 0 for x in Xs]

class Aggregator:
    def __init__(self, user_weights, gp_righting: GaussianProcessSurrogate, gp_buoyancy: GaussianProcessSurrogate, column_order):
        self.weights = {}
        tot = 0
        for k in user_weights.keys():
            prev_tot = tot
            tot += user_weights[k]
            self.weights[k] = [prev_tot, tot]
        self.tot = tot
        self.gp_righting = gp_righting
        self.gp_buoyancy = gp_buoyancy
        self.column_order = column_order

    def f(self, hull: Hull, budget: int = 180) -> Tuple[float, dict]:
        def add_hull_params(x):
            def f(k):
                match k:
                    case "cost": return 0
                    case "heel": return x
                    case k: return getattr(hull.params, k)
            return np.asarray([f(k) for k in self.column_order])
        X_heels = np.linspace(0, np.pi, 180)
        X_grid = np.asarray(list(map(add_hull_params, X_heels)))

        mx = (0, 0) # Max val
        root_estimate = np.pi / 2 # Estimated root based on mu
        initial_stability = 0
        initial_buoyancy = 0

        def update(x, sample, righting=True):
            update_gp(self.gp_righting if righting else self.gp_buoyancy,
                      np.asarray([[x if k == "heel" else (getattr(hull.params, k) if k != "cost" else 0) for k in self.column_order]]),
                      np.asarray([sample.righting_moment]) if righting else\
                      np.asarray([[sample.reserve_buoyancy, sample.reserve_buoyancy_hull]]),
                      self.column_order)

        def adjust_budgets(budgets, k, cost):
            budgets[k] -= cost
            if budgets[k] <= 0:
                diff = self.weights[k][1] - self.weights[k][0]
                self.tot -= diff
                for k2 in budgets.keys():
                    if self.weights[k2][0] >= self.weights[k][0]:
                        if not k2 == k: self.weights[k2][0] -= diff
                        self.weights[k2][1] -= diff

        mu_r, varSigma_r = self.gp_righting.predict(X_grid)
        mu_b, varSigma_b = self.gp_buoyancy.predict(X_grid)
        heel_index = self.column_order.index("heel")
        r = np.random.random() * self.tot
        budgets = {k: (self.weights[k][1]-self.weights[k][0])/self.tot * budget for k in self.weights.keys()}
            
        while any(budget > 0 for budget in budgets.values()):
            print(f"Aggregator Budgets: {budgets}")
            root_estimate = X_heels[np.where(mu_r[:-1]*mu_r[1:] < 0)[0][0]]
            for k in self.weights.keys():
                # TODO: WORK OUT HOW SURE EACH ACQUISITION FUNCTION IS ON ITS RESULT (to optimise by terminating early)
                match k:
                    case "diminishing_stability" if self.weights[k][0] <= r < self.weights[k][1]:
                        a = a_EI_max(mx[1], X_heels, mu_r, varSigma_r)
                        x = X_heels[np.argmax(a)]
                        sample = simulations.analytic.run(hull, simulations.Params(x))
                        mx = (x, sample.righting_moment_heel())
                        update(mx[0], sample)
                        adjust_budgets(budgets, k, sample.cost)
                            
                    case "tipping_point" if self.weights[k][0] <= r < self.weights[k][1]:
                        a = a_SC(mx[0], X_heels, mu_r, varSigma_r)
                        x = X_heels[np.argmax(a)]
                        sample = simulations.analytic.run(hull, simulations.Params(x))
                        y = sample.righting_moment_heel()
                        update(x, sample)
                        adjust_budgets(budgets, k, sample.cost)
                            
                    case "overall_stability" | "righting_energy" | "overall_buoyancy"\
                         if self.weights[k][0] <= r < self.weights[k][1]:
                        bounds = (0,0)
                        moments = True
                        match k:
                            case "overall_stability": bounds = (0, root_estimate)
                            case "righting_energy": bounds = (root_estimate, np.pi)
                            case "overall_buoyancy":
                                moments = False
                                bounds = (0, np.pi)
                        a = a_INT(bounds, X_heels, mu_r if moments else mu_b, varSigma_r if moments else varSigma_b)
                        x = X_heels[argmax(a)]
                        sample = simulations.analytic.run(hull, simulations.Params(x))
                        update(x, sample)
                        adjust_budgets(budgets, k, sample.cost)
                            
                    case "initial_stability":
                        x = X_heels[1]
                        initial_stability = mu_r[x][0] if varSigma_r[1][0] == 0 else \
                            simulations.analytic.run(hull, simulations.Params(x)).righting_moment_heel()
                        adjust_budgets(budgets, k, budgets[k])
                            
                    case "initial_buoyancy":
                        x = 0
                        initial_buoyancy = mu_b[x][0] if varSigma_b[1][0] == 0 else \
                            simulations.analytic.run(hull, simulations.Params(x)).reserve_buoyancy
                        adjust_budgets(budgets, k, budgets[k])

        # I use root_estimate here because, root may be wildly inaccurate for low budgets or when tipping point is not a priority
        overall_stability = sum(mu_r[np.where(X_heels < root_estimate)][:,0])*X_heels[1]
        righting_energy = sum(mu_r[np.where(X_heels > root_estimate)][:,0])*X_heels[1]
        overall_buoyancy = sum(mu_b[:,0])*X_heels[1]
        result = {
            "overall_stability": overall_stability,
            "initial_stability": initial_stability,
            "righting_energy": righting_energy,
            "tipping_point": root_estimate,
            "overall_buoyancy": overall_buoyancy,
            "initial_buoyancy": initial_buoyancy
        }
        print(result)
        aggregate = 0
        for k, norm in config.hyperparameters.weight_normalisers.items():
            aggregate += np.abs(result[k]) * (self.weights[k][1] - self.weights[k][0]) / (norm * self.tot)
        print(aggregate)
        return aggregate, result
                        
