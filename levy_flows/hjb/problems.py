"""Stochastic control problems for Lévy-HJB solver.

Defines specific control problems with their dynamics, costs, and terminal conditions.
"""

from typing import Optional, Tuple, Callable
from abc import ABC, abstractmethod
import math

import torch
import torch.nn as nn
from torch import Tensor


class ControlProblem(ABC):
    """Abstract base class for stochastic control problems."""

    @property
    @abstractmethod
    def state_dim(self) -> int:
        """Dimension of state space."""
        pass

    @property
    @abstractmethod
    def control_dim(self) -> int:
        """Dimension of control space."""
        pass

    @property
    @abstractmethod
    def T(self) -> float:
        """Terminal time."""
        pass

    @abstractmethod
    def drift(self, t: Tensor, x: Tensor, u: Tensor) -> Tensor:
        """Drift coefficient b(t, x, u)."""
        pass

    @abstractmethod
    def diffusion(self, t: Tensor, x: Tensor, u: Tensor) -> Tensor:
        """Diffusion coefficient σ(t, x, u)."""
        pass

    @abstractmethod
    def running_cost(self, t: Tensor, x: Tensor, u: Tensor) -> Tensor:
        """Running cost f(t, x, u)."""
        pass

    @abstractmethod
    def terminal_cost(self, x: Tensor) -> Tensor:
        """Terminal cost g(x)."""
        pass

    def control_bounds(self) -> Optional[Tuple[float, float]]:
        """Optional bounds on control."""
        return None

    def simplex_constraint(self) -> bool:
        """Whether control must sum to 1."""
        return False


class MertonPortfolioProblem(ControlProblem):
    """Merton optimal portfolio problem with Lévy jumps.

    State: X_t = wealth
    Control: u_t = fraction invested in risky asset

    Dynamics:
        dX_t = X_t * [r + u_t(μ - r)]dt + X_t * u_t * σ dW_t + X_t * u_t * dJ_t

    where J_t is a Lévy process (jumps in returns).

    Objective: Maximize expected utility of terminal wealth
        max E[U(X_T)]

    where U(x) = x^(1-γ) / (1-γ) for CRRA utility (γ > 0, γ ≠ 1)
    or U(x) = log(x) for log utility (γ = 1).
    """

    def __init__(
        self,
        r: float = 0.02,        # Risk-free rate
        mu: float = 0.08,       # Expected return of risky asset
        sigma: float = 0.2,     # Volatility
        gamma: float = 2.0,     # Risk aversion
        terminal_time: float = 1.0,
        x0: float = 1.0,        # Initial wealth
    ):
        self.r = r
        self.mu = mu
        self.sigma = sigma
        self.gamma = gamma
        self._T = terminal_time
        self.x0 = x0

    @property
    def state_dim(self) -> int:
        return 1

    @property
    def control_dim(self) -> int:
        return 1

    @property
    def T(self) -> float:
        return self._T

    def drift(self, t: Tensor, x: Tensor, u: Tensor) -> Tensor:
        """Drift: X * [r + u(μ - r)]."""
        return x * (self.r + u * (self.mu - self.r))

    def diffusion(self, t: Tensor, x: Tensor, u: Tensor) -> Tensor:
        """Diffusion: X * u * σ."""
        return x * u * self.sigma

    def running_cost(self, t: Tensor, x: Tensor, u: Tensor) -> Tensor:
        """No running cost for Merton problem."""
        return torch.zeros_like(x[:, 0:1])

    def terminal_cost(self, x: Tensor) -> Tensor:
        """CRRA utility U(x) = x^(1-γ)/(1-γ).

        Max convention: V(t,x) = sup E[U(X_T)|X_t=x], terminal V(T,x) = U(x).
        For γ > 1, U(x) ∈ (-∞, 0); for γ < 1, U(x) > 0; for γ = 1 use log.
        The HJB residual ∂_t V + sup_u H = 0 is consistent with this when the
        policy network is trained to maximize H (see Solver.policy_argmax_loss).
        """
        x_safe = torch.clamp(x, min=1e-8)

        if abs(self.gamma - 1.0) < 1e-6:
            utility = torch.log(x_safe)
        else:
            utility = (x_safe ** (1 - self.gamma)) / (1 - self.gamma)

        return utility.squeeze(-1)

    def control_bounds(self) -> Optional[Tuple[float, float]]:
        """Long-only with no leverage: u ∈ [0, 1].

        Paired with a Lévy support truncated to |z| < 1 (see VarianceGammaMeasure
        with truncation_max < 1.0), this guarantees 1 + uz > 0 for all admissible
        (u, z), so post-jump wealth never crosses zero. If a caller widens the
        jump support, they must also tighten u_max accordingly.
        """
        return (0.0, 1.0)

    def analytical_value(self, t: Tensor, x: Tensor) -> Tensor:
        """Analytical Merton value for the diffusion-only case (max convention).

        V(t,x) = (x^(1-γ)/(1-γ)) * exp(A(T-t)), with
        A = (1-γ)[r + (μ-r)²/(2γσ²)].
        For γ > 1 this is negative; for log utility V = log x + A(T-t).
        """
        x_safe = torch.clamp(x, min=1e-8)
        tau = self.T - t

        if abs(self.gamma - 1.0) < 1e-6:
            A = self.r + 0.5 * ((self.mu - self.r) / self.sigma) ** 2
            V = torch.log(x_safe) + A * tau.unsqueeze(-1)
        else:
            A = (1 - self.gamma) * (
                self.r + 0.5 * ((self.mu - self.r) ** 2) / (self.gamma * self.sigma ** 2)
            )
            V = (x_safe ** (1 - self.gamma)) / (1 - self.gamma) * torch.exp(A * tau.unsqueeze(-1))

        return V

    def analytical_policy(self, t: Tensor, x: Tensor) -> Tensor:
        """Analytical optimal policy for diffusion-only case.

        u* = (μ - r) / (γ σ²) (constant Merton ratio)
        """
        merton_ratio = (self.mu - self.r) / (self.gamma * self.sigma ** 2)
        return torch.full_like(x[:, 0:1], merton_ratio)


class MultiAssetPortfolioProblem(ControlProblem):
    """Multi-asset portfolio optimization with correlated jumps.

    State: X_t ∈ R^n = vector of wealth in each asset
    Control: u_t ∈ R^n = portfolio weights (sum to 1)

    Dynamics for each asset i:
        dX_t^i = X_t^i * [r + u_t^i(μ_i - r)]dt + X_t^i * u_t^i * Σ_i dW_t + X_t^i * u_t^i * dJ_t^i
    """

    def __init__(
        self,
        n_assets: int = 2,
        r: float = 0.02,
        mu: Optional[Tensor] = None,
        sigma: Optional[Tensor] = None,
        correlation: Optional[Tensor] = None,
        gamma: float = 2.0,
        terminal_time: float = 1.0,
    ):
        self.n_assets = n_assets
        self.r = r
        self.gamma = gamma
        self._T = terminal_time

        # Default parameters
        if mu is None:
            mu = torch.linspace(0.05, 0.12, n_assets)
        if sigma is None:
            sigma = torch.linspace(0.15, 0.30, n_assets)
        if correlation is None:
            correlation = torch.eye(n_assets) * 0.5 + 0.5  # Moderate correlation

        self.mu = mu
        self.sigma = sigma
        self.correlation = correlation

        # Covariance matrix
        self.cov = torch.outer(sigma, sigma) * correlation

    @property
    def state_dim(self) -> int:
        return self.n_assets

    @property
    def control_dim(self) -> int:
        return self.n_assets

    @property
    def T(self) -> float:
        return self._T

    def drift(self, t: Tensor, x: Tensor, u: Tensor) -> Tensor:
        """Drift for multi-asset case."""
        excess_return = self.mu.to(x.device) - self.r
        return x * (self.r + u * excess_return)

    def diffusion(self, t: Tensor, x: Tensor, u: Tensor) -> Tensor:
        """Diffusion matrix."""
        return x * u * self.sigma.to(x.device)

    def running_cost(self, t: Tensor, x: Tensor, u: Tensor) -> Tensor:
        return torch.zeros(x.shape[0], 1, device=x.device)

    def terminal_cost(self, x: Tensor) -> Tensor:
        """CRRA on total wealth, max convention (terminal V(T,x) = U(x))."""
        total_wealth = x.sum(dim=-1, keepdim=True)
        total_wealth = torch.clamp(total_wealth, min=1e-8)

        if abs(self.gamma - 1.0) < 1e-6:
            utility = torch.log(total_wealth)
        else:
            utility = (total_wealth ** (1 - self.gamma)) / (1 - self.gamma)

        return utility.squeeze(-1)

    def simplex_constraint(self) -> bool:
        return True  # Weights sum to 1


class CVaRControlProblem(ControlProblem):
    """CVaR (Conditional Value-at-Risk) minimization problem.

    Objective: Minimize CVaR_α(X_T) = E[X_T | X_T ≤ VaR_α(X_T)]

    This is a tail-risk-aware objective that penalizes extreme losses.

    We use the Rockafellar-Uryasev reformulation:
        CVaR_α(X) = min_ξ { ξ + (1/α) E[(−X − ξ)^+] }

    The problem is solved by augmenting the state with ξ.
    """

    def __init__(
        self,
        base_problem: ControlProblem,
        alpha: float = 0.05,  # CVaR level (5% = CVaR_0.05)
    ):
        """
        Args:
            base_problem: Underlying portfolio problem
            alpha: CVaR quantile level
        """
        self.base_problem = base_problem
        self.alpha = alpha

    @property
    def state_dim(self) -> int:
        return self.base_problem.state_dim + 1  # Augmented with VaR level ξ

    @property
    def control_dim(self) -> int:
        return self.base_problem.control_dim + 1  # Also optimize ξ

    @property
    def T(self) -> float:
        return self.base_problem.T

    def drift(self, t: Tensor, x: Tensor, u: Tensor) -> Tensor:
        """Drift for original state; ξ is constant."""
        x_base = x[:, :-1]
        u_base = u[:, :-1]

        drift_base = self.base_problem.drift(t, x_base, u_base)

        # ξ has zero drift
        drift_xi = torch.zeros(x.shape[0], 1, device=x.device)

        return torch.cat([drift_base, drift_xi], dim=-1)

    def diffusion(self, t: Tensor, x: Tensor, u: Tensor) -> Tensor:
        x_base = x[:, :-1]
        u_base = u[:, :-1]

        diff_base = self.base_problem.diffusion(t, x_base, u_base)

        # ξ has zero diffusion
        diff_xi = torch.zeros(x.shape[0], 1, device=x.device)

        return torch.cat([diff_base, diff_xi], dim=-1)

    def running_cost(self, t: Tensor, x: Tensor, u: Tensor) -> Tensor:
        return torch.zeros(x.shape[0], 1, device=x.device)

    def terminal_cost(self, x: Tensor) -> Tensor:
        """CVaR objective at terminal time.

        CVaR_α(−X_T) = ξ + (1/α) E[(−X_T − ξ)^+]

        For terminal cost, we use:
        g(x, ξ) = ξ + (1/α) max(−x − ξ, 0)
        """
        x_base = x[:, :-1]
        xi = x[:, -1:]

        # Total wealth (for multi-asset, sum up)
        if x_base.shape[-1] > 1:
            wealth = x_base.sum(dim=-1, keepdim=True)
        else:
            wealth = x_base

        # CVaR: ξ + (1/α) * (−wealth − ξ)^+
        shortfall = torch.relu(-wealth - xi)
        cvar = xi + shortfall / self.alpha

        return cvar.squeeze(-1)


class LQRLevyProblem(ControlProblem):
    """Linear-Quadratic Regulator with Lévy noise.

    Simple problem with analytical structure for benchmarking.

    Dynamics: dX = (AX + Bu)dt + σ dW + dJ
    Cost: ∫[X'QX + u'Ru]dt + X_T' Q_T X_T
    """

    def __init__(
        self,
        state_dim: int = 2,
        control_dim: int = 1,
        A: Optional[Tensor] = None,
        B: Optional[Tensor] = None,
        Q: Optional[Tensor] = None,
        R: Optional[Tensor] = None,
        Q_terminal: Optional[Tensor] = None,
        sigma: float = 0.1,
        terminal_time: float = 1.0,
    ):
        self._state_dim = state_dim
        self._control_dim = control_dim
        self._T = terminal_time
        self._sigma = sigma

        # Default system matrices
        if A is None:
            A = -0.1 * torch.eye(state_dim)  # Stable drift
        if B is None:
            B = torch.randn(state_dim, control_dim) * 0.5
        if Q is None:
            Q = torch.eye(state_dim)
        if R is None:
            R = 0.1 * torch.eye(control_dim)
        if Q_terminal is None:
            Q_terminal = torch.eye(state_dim)

        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.Q_terminal = Q_terminal

    @property
    def state_dim(self) -> int:
        return self._state_dim

    @property
    def control_dim(self) -> int:
        return self._control_dim

    @property
    def T(self) -> float:
        return self._T

    def drift(self, t: Tensor, x: Tensor, u: Tensor) -> Tensor:
        """Linear drift: Ax + Bu."""
        A = self.A.to(x.device)
        B = self.B.to(x.device)
        return x @ A.T + u @ B.T

    def diffusion(self, t: Tensor, x: Tensor, u: Tensor) -> Tensor:
        """Constant diffusion."""
        return torch.full_like(x, self._sigma)

    def running_cost(self, t: Tensor, x: Tensor, u: Tensor) -> Tensor:
        """Quadratic running cost: x'Qx + u'Ru."""
        Q = self.Q.to(x.device)
        R = self.R.to(x.device)

        cost_x = (x @ Q * x).sum(dim=-1, keepdim=True)
        cost_u = (u @ R * u).sum(dim=-1, keepdim=True)

        return cost_x + cost_u

    def terminal_cost(self, x: Tensor) -> Tensor:
        """Quadratic terminal cost: x'Q_T x."""
        Q_T = self.Q_terminal.to(x.device)
        return (x @ Q_T * x).sum(dim=-1)
