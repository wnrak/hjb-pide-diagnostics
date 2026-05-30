"""Neural HJB-PIDE Solver for Lévy-driven Stochastic Control.

This module implements Deep Galerkin Method (DGM) style solvers for
Hamilton-Jacobi-Bellman Partial Integro-Differential Equations (PIDEs)
arising in optimal control under Lévy dynamics.

The key equation:
    ∂_t V + sup_u { L^u_diff V + f(x,u) + ∫[V(t,x+z) - V(t,x) - ∇V·z 1_{|z|<1}] ν^u(dz) } = 0

Components:
    - ValueNetwork: Neural network approximating V(t,x)
    - PolicyNetwork: Neural network approximating optimal u(t,x)
    - LevyIntegral: Monte Carlo approximation of the jump integral
    - HJBSolver: Main solver combining all components
"""

from levy_flows.hjb.networks import ValueNetwork, PolicyNetwork
from levy_flows.hjb.levy_integral import LevyIntegralMC, ImportanceSampledLevyIntegral
from levy_flows.hjb.solver import LevyHJBSolver
from levy_flows.hjb.problems import MertonPortfolioProblem, CVaRControlProblem

__all__ = [
    "ValueNetwork",
    "PolicyNetwork",
    "LevyIntegralMC",
    "ImportanceSampledLevyIntegral",
    "LevyHJBSolver",
    "MertonPortfolioProblem",
    "CVaRControlProblem",
]
