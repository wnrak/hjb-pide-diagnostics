"""Scalar/vector closed-form-up-to-quadrature baseline for CRRA-Merton-VG.

Why this file exists
====================
For the problem the paper actually tests — CRRA utility, multiplicative
wealth dynamics dX/X = (r + u(μ-r))dt + uσ dW + u dJ, constant
parameters, no consumption, no transaction costs, no state-dependent
constraints — the HJB collapses by homogeneity. With the ansatz
        V(t, x) = A(t) · x^(1-γ) / (1-γ),
substitution into the HJB and division by A(t)·x^(1-γ) gives, for γ ≠ 1,

    A'(t)/[(1-γ)·A(t)] + sup_{u ∈ U} F(u) = 0,
    A(T) = 1,

where the per-period objective is

    F(u) = r + u(μ-r) - (γ/2)·σ²·u²
           + ∫ [((1+uz)^(1-γ) - 1)/(1-γ) - uz] · ν(dz).

The optimal control u* is independent of (t, x), so the entire control
problem reduces to one scalar maximization.

This file implements that scalar maximization for the 1D portfolio
problem and a vector version for the 2D coupled portfolio with simplex
constraint. Both use deterministic quadrature against the Lévy density —
no neural network, no PDE solver, no Monte Carlo. The result serves as
an independent third reference for the §4.5 audit (alongside FD v1 and
FD v2) and as a sanity check on the §4.8 2D number.

Caveat: this is a baseline for the *current* paper benchmarks only.
It does not generalize to problems where homogeneity does not collapse
(state-dependent coefficients, consumption, transaction costs, regime
switching). Those problems are where neural HJB-PIDE solvers earn
their keep.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# 1D scalar baseline
# ---------------------------------------------------------------------------

@dataclass
class ScalarBaselineResult:
    """Result of the 1D CRRA-Merton-Lévy scalar maximization."""
    u_star: float
    F_at_u_star: float
    u_grid: np.ndarray
    F_grid: np.ndarray
    levy_integral_at_u_star: float


def crra_jump_integrand(z: np.ndarray, u: float, gamma: float) -> np.ndarray:
    """Integrand of the Lévy term inside F(u):
        h(z, u) = [(1+uz)^(1-γ) - 1] / (1-γ) - uz
    Defined for 1+uz > 0 (admissibility). For γ = 1 we return the
    log-utility limit log(1+uz) - uz.
    """
    if abs(gamma - 1.0) < 1e-12:
        # Log-utility limit: ((1+uz)^(1-γ)-1)/(1-γ) → log(1+uz)
        return np.log(np.maximum(1.0 + u * z, 1e-12)) - u * z
    safe = 1.0 + u * z > 0
    out = np.full_like(z, -1e10, dtype=float)
    pwr = np.where(safe, np.maximum(1.0 + u * z, 1e-12) ** (1.0 - gamma), 0.0)
    out = np.where(safe, (pwr - 1.0) / (1.0 - gamma) - u * z, out)
    return out


def scalar_baseline_1d(
    *,
    r: float,
    mu: float,
    sigma: float,
    gamma: float,
    levy_density: Callable[[np.ndarray], np.ndarray],
    z_min: float,
    z_max: float,
    n_quad_per_side: int = 2001,
    u_min: float = 0.0,
    u_max: float = 1.0,
    n_u_fine: int = 1001,
) -> ScalarBaselineResult:
    """Solve sup_{u ∈ [u_min, u_max]} F(u) for the 1D CRRA-Merton-Lévy problem.

    Quadrature uses composite Simpson's rule on |z| ∈ [z_min, z_max], split
    into negative and positive sides. The optimization uses two passes:
    a fine linspace search for the basin of attraction, then SciPy's
    bounded scalar minimizer for refinement.

    Parameters
    ----------
    r, mu, sigma, gamma : float
        Standard Merton problem parameters.
    levy_density : callable(z) -> ν(z)
        Lévy density, vectorized; evaluated at the quadrature nodes.
    z_min, z_max : float
        Truncation: |z| ∈ [z_min, z_max] with 0 < z_min < z_max.
    n_quad_per_side : int
        Number of Simpson nodes per side (must be odd; will be adjusted).
    u_min, u_max : float
        Admissible-set endpoints.
    n_u_fine : int
        Coarse u-grid resolution used to seed the refinement.
    """
    if not (z_min > 0 and z_max > z_min):
        raise ValueError("Need 0 < z_min < z_max")
    if n_quad_per_side % 2 == 0:
        n_quad_per_side += 1

    # Simpson nodes and weights on [z_min, z_max]
    z_pos = np.linspace(z_min, z_max, n_quad_per_side)
    h = (z_max - z_min) / (n_quad_per_side - 1)
    w_simp = np.empty(n_quad_per_side)
    w_simp[0] = w_simp[-1] = h / 3.0
    w_simp[1:-1:2] = 4.0 * h / 3.0
    w_simp[2:-2:2] = 2.0 * h / 3.0
    z_neg = -z_pos[::-1]
    w_neg = w_simp[::-1].copy()  # Simpson weights are symmetric on uniform grid

    nu_pos = levy_density(z_pos)
    nu_neg = levy_density(z_neg)

    def levy_integral(u: float) -> float:
        I = (crra_jump_integrand(z_pos, u, gamma) * nu_pos * w_simp).sum()
        I += (crra_jump_integrand(z_neg, u, gamma) * nu_neg * w_neg).sum()
        return float(I)

    def F(u: float) -> float:
        return (
            r + u * (mu - r) - 0.5 * gamma * sigma ** 2 * u ** 2
            + levy_integral(u)
        )

    # Coarse u-grid pass
    u_grid = np.linspace(u_min, u_max, n_u_fine)
    F_grid = np.array([F(u) for u in u_grid])
    k = int(np.argmax(F_grid))

    # Refine via bounded scalar minimizer of -F around the coarse argmax
    if k == 0:
        u_star = u_min
    elif k == n_u_fine - 1:
        u_star = u_max
    else:
        lo, hi = u_grid[max(0, k - 2)], u_grid[min(n_u_fine - 1, k + 2)]
        res = minimize_scalar(lambda u: -F(u), bounds=(lo, hi), method="bounded",
                              options={"xatol": 1e-9})
        u_star = float(res.x)
        # Compare to the boundary just in case the refinement landed slightly worse
        for u_b in (u_min, u_max):
            if F(u_b) > F(u_star):
                u_star = u_b

    return ScalarBaselineResult(
        u_star=u_star,
        F_at_u_star=F(u_star),
        u_grid=u_grid,
        F_grid=F_grid,
        levy_integral_at_u_star=levy_integral(u_star),
    )


# ---------------------------------------------------------------------------
# 2D scalar (= simplex) baseline
# ---------------------------------------------------------------------------

@dataclass
class ScalarBaselineResult2D:
    """Result of the 2D CRRA-Merton-Lévy simplex maximization."""
    u_star: np.ndarray            # shape (2,)
    cash_star: float
    G_at_u_star: float


def crra_2asset_objective(
    u: np.ndarray,
    *,
    r: float,
    mu: np.ndarray,
    sigma: np.ndarray,
    rho: float,
    gamma: float,
    levy_density_per_asset: list,
    z_quad: dict,
) -> float:
    """G(u_1, u_2) for two correlated risky assets with independent VG jumps.

    G(u) = r + Σ_i u_i (μ_i − r)
           - (γ/2) · u^T Σ u
           + Σ_i ∫ [((1+u_i z)^(1-γ) − 1)/(1-γ) − u_i z] · ν_i(dz)
    """
    excess = mu - r
    cov = np.array(
        [[sigma[0] ** 2, rho * sigma[0] * sigma[1]],
         [rho * sigma[0] * sigma[1], sigma[1] ** 2]],
        dtype=float,
    )
    drift_diff = float(excess @ u) - 0.5 * gamma * float(u @ cov @ u)
    levy_total = 0.0
    for i in range(2):
        z_pos = z_quad["z_pos"]
        z_neg = z_quad["z_neg"]
        w_pos = z_quad["w_pos"]
        w_neg = z_quad["w_neg"]
        nu_pos_i = z_quad[f"nu_pos_{i+1}"]
        nu_neg_i = z_quad[f"nu_neg_{i+1}"]
        levy_total += (
            (crra_jump_integrand(z_pos, u[i], gamma) * nu_pos_i * w_pos).sum()
            + (crra_jump_integrand(z_neg, u[i], gamma) * nu_neg_i * w_neg).sum()
        )
    return r + drift_diff + levy_total


def scalar_baseline_2d(
    *,
    r: float,
    mu: np.ndarray,
    sigma: np.ndarray,
    rho: float,
    gamma: float,
    levy_density_1: Callable,
    levy_density_2: Callable,
    z_min: float,
    z_max: float,
    n_quad_per_side: int = 2001,
) -> ScalarBaselineResult2D:
    """Solve sup_{u ∈ Δ²} G(u_1, u_2) over the simplex u_i ≥ 0, u_1 + u_2 ≤ 1."""
    if n_quad_per_side % 2 == 0:
        n_quad_per_side += 1

    z_pos = np.linspace(z_min, z_max, n_quad_per_side)
    h = (z_max - z_min) / (n_quad_per_side - 1)
    w_simp = np.empty(n_quad_per_side)
    w_simp[0] = w_simp[-1] = h / 3.0
    w_simp[1:-1:2] = 4.0 * h / 3.0
    w_simp[2:-2:2] = 2.0 * h / 3.0
    z_neg = -z_pos[::-1]
    w_neg = w_simp[::-1].copy()

    z_quad = dict(
        z_pos=z_pos, z_neg=z_neg, w_pos=w_simp, w_neg=w_neg,
        nu_pos_1=levy_density_1(z_pos),
        nu_neg_1=levy_density_1(z_neg),
        nu_pos_2=levy_density_2(z_pos),
        nu_neg_2=levy_density_2(z_neg),
    )

    obj = lambda u_arr: -crra_2asset_objective(
        u_arr, r=r, mu=mu, sigma=sigma, rho=rho, gamma=gamma,
        levy_density_per_asset=[levy_density_1, levy_density_2],
        z_quad=z_quad,
    )

    bounds = [(0.0, 1.0), (0.0, 1.0)]
    constraints = [{"type": "ineq", "fun": lambda u: 1.0 - u[0] - u[1]}]

    # Multi-start for robustness on the simplex
    best = None
    for u0 in [(0.1, 0.1), (0.4, 0.4), (0.5, 0.0), (0.0, 0.5), (0.34, 0.26)]:
        res = minimize(obj, np.array(u0), method="SLSQP",
                       bounds=bounds, constraints=constraints,
                       options={"ftol": 1e-12, "maxiter": 500})
        if best is None or res.fun < best.fun:
            best = res

    u_star = np.clip(best.x, 0.0, 1.0)
    cash = max(0.0, 1.0 - float(u_star.sum()))
    return ScalarBaselineResult2D(
        u_star=u_star,
        cash_star=cash,
        G_at_u_star=-float(best.fun),
    )


# ---------------------------------------------------------------------------
# Convenience: VG Lévy density (matches the implementation in
# levy_flows/hjb/levy_integral.py and fd_pide_solver.py).
# ---------------------------------------------------------------------------

def vg_density(
    z: np.ndarray, *, sigma_vg: float = 0.2, theta: float = -0.1, nu: float = 0.3,
) -> np.ndarray:
    """VG Lévy density:
        ν(z) = (C/|z|) · exp(-M|z|)   for z > 0,
        ν(z) = (C/|z|) · exp(-G|z|)   for z < 0,
        C = 1/ν,    M = √(2/ν + θ²/σ⁴)/σ - θ/σ²,
        G = √(2/ν + θ²/σ⁴)/σ + θ/σ².
    """
    z = np.asarray(z, dtype=float)
    C = 1.0 / nu
    temp = np.sqrt(theta ** 2 / sigma_vg ** 4 + 2.0 / (nu * sigma_vg ** 2))
    M = temp - theta / sigma_vg ** 2
    G = temp + theta / sigma_vg ** 2
    out = np.zeros_like(z, dtype=float)
    pos = z > 0
    neg = z < 0
    out[pos] = C / np.abs(z[pos]) * np.exp(-M * np.abs(z[pos]))
    out[neg] = C / np.abs(z[neg]) * np.exp(-G * np.abs(z[neg]))
    return out


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def _test_diffusion_only_recovers_merton():
    """For ν ≡ 0 the scalar maximizer must give u* = (μ-r)/(γσ²)."""
    res = scalar_baseline_1d(
        r=0.02, mu=0.08, sigma=0.2, gamma=2.0,
        levy_density=lambda z: np.zeros_like(z),
        z_min=0.01, z_max=0.99,
    )
    expected = 0.06 / (2.0 * 0.04)  # 0.75
    err = abs(res.u_star - expected)
    print(f"  diffusion-only Merton: u*={res.u_star:.6f}, expected {expected:.4f}, "
          f"err={err:.2e}  -> {'PASS' if err < 1e-4 else 'FAIL'}")
    return err < 1e-4


def _test_log_utility_limit():
    """At γ = 1 the log limit and γ → 1 should agree to high precision.

    For γ exactly 1, F(u) = r + u(μ-r) - σ²u²/2 + ∫[log(1+uz) - uz] ν(dz).
    We just check the sub-routine doesn't error out and the result is sane.
    """
    res = scalar_baseline_1d(
        r=0.02, mu=0.08, sigma=0.2, gamma=1.0,
        levy_density=lambda z: np.zeros_like(z),
        z_min=0.01, z_max=0.99,
    )
    # Diffusion-only at γ=1: u* = (μ-r)/σ²
    expected = 0.06 / 0.04  # = 1.5; clipped to u_max = 1.0
    print(f"  log utility (γ=1, diffusion-only, U=[0,1]): u*={res.u_star:.4f}, "
          f"expected min(1.5, 1.0)={min(expected, 1.0):.4f}  -> "
          f"{'PASS' if abs(res.u_star - 1.0) < 1e-3 else 'FAIL'}")
    return abs(res.u_star - 1.0) < 1e-3


def run_unit_tests() -> None:
    print("scalar_baseline unit tests:")
    ok = True
    ok &= _test_diffusion_only_recovers_merton()
    ok &= _test_log_utility_limit()
    if not ok:
        raise AssertionError("scalar_baseline unit tests failed")


if __name__ == "__main__":
    run_unit_tests()
