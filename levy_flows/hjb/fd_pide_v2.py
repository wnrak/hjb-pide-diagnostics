"""Independent FD-PIDE reference implementation (Phase 3a).

This is a from-scratch reimplementation of the 1D control HJB-PIDE solver,
written deliberately to share as little as possible with `fd_pide_solver.py`
so that any persistent neural-vs-FD gap cannot be blamed on a shared bug
in the FD code path. The two implementations differ on every numerical
choice that could plausibly hide a bug:

   axis                    fd_pide_solver.py (v1)        fd_pide_v2.py (this file)
   ---------               -------------------------     ---------------------------
   wealth coordinate       log-wealth y = log(x)         linear x
   time stepping           implicit Euler                Crank-Nicolson
   z-quadrature            np.trapz on linspace          composite Simpson on linspace
   compensator form        [V(x(1+uz)) - V - uxz V_x]    same value, but computed as
                                                         [V(x(1+uz)) - V] - uxz V_x · w_z,
                                                         summed separately so the
                                                         compensator is exposed
   u-optimization          linspace argmax (n_u points)  Brent root-find on dH/du with
                                                         explicit boundary checks at
                                                         u_min, u_max
   boundary handling       Neumann, then power-law       Dirichlet using the diffusion-
                           extrapolation                 only Merton value (closed form)

What is shared by design (same equation, same calibration):
   - admissible set U = [0, u_max] passed in at construction
   - Lévy support truncation [-z_max, z_max] passed in
   - multiplicative wealth jump shift x → x(1+uz)
   - VG Lévy density formula

Diagnostics: the function `hamiltonian_breakdown(x_i, u, V, V_x, V_xx)`
returns a dict with drift_term, diffusion_term, levy_integral_compensated,
levy_integral_uncompensated, and compensator separately, so the per-component
agreement with v1 / neural can be inspected at any (x, u).

Reference for the equation
    ∂_t V + sup_u { x(r + u(μ-r)) V_x + ½ u² σ² x² V_xx
                    + ∫ [V(x(1+uz)) - V - uxz V_x] ν(dz) } = 0,
    V(T, x) = U(x).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
from scipy.optimize import brentq
from scipy.sparse import diags as sp_diags
from scipy.sparse.linalg import spsolve as sp_solve


# ---------------------------------------------------------------------------
# Quadrature on the truncated z-domain
# ---------------------------------------------------------------------------

def _simpson_weights(n: int, h: float) -> np.ndarray:
    """Composite Simpson's-rule weights for n equally spaced points (n must be odd).

    For n odd, integral ≈ (h/3) * (f0 + 4 f1 + 2 f2 + 4 f3 + ... + 4 f_{n-2} + f_{n-1}).
    For n even, fall back to (h/3) * Simpson on first n-1 points + trapezoid on last.
    """
    if n < 3:
        # Fall back to trapezoid for tiny n
        w = np.full(n, h)
        if n >= 2:
            w[0] = w[-1] = 0.5 * h
        return w
    w = np.empty(n)
    if n % 2 == 1:
        w[0] = w[-1] = h / 3.0
        w[1:-1:2] = 4.0 * h / 3.0   # odd interior indices
        w[2:-2:2] = 2.0 * h / 3.0   # even interior indices
    else:
        # Simpson on first n-1 (odd), trapezoid on the last interval.
        w_simp = _simpson_weights(n - 1, h)
        w[: n - 1] = w_simp
        w[-1] = 0.0
        w[-2] += 0.5 * h
        w[-1] += 0.5 * h
    return w


@dataclass
class ZQuadrature:
    """Pre-computed Lévy-integral quadrature on the truncated z-domain.

    Splits z into negative and positive halves so that |z|=0 (where ν has a
    1/|z| singularity) is excluded by construction. Each half uses composite
    Simpson's rule.
    """

    z_neg: np.ndarray              # (n_neg,) z < 0 quadrature nodes
    w_neg: np.ndarray              # (n_neg,) Simpson weights
    nu_neg: np.ndarray             # (n_neg,) Lévy density evaluated at z_neg
    z_pos: np.ndarray              # (n_pos,) z > 0 nodes
    w_pos: np.ndarray
    nu_pos: np.ndarray


def build_z_quadrature(
    levy_density: Callable[[float], float],
    z_min: float,
    z_max: float,
    n_per_side: int,
) -> ZQuadrature:
    """Build composite-Simpson quadrature over [-z_max, -z_min] ∪ [z_min, z_max].

    The 1/|z| factor in ν_VG makes uniform-grid trapezoidal weakly singular at
    the inner edge z = ±z_min. Simpson's rule converges faster on smooth
    integrands but is still subject to that singularity; we resolve it by
    requiring z_min > 0 and using sufficiently many nodes.
    """
    if not (z_min > 0 and z_max > z_min):
        raise ValueError("Need 0 < z_min < z_max")
    z_pos = np.linspace(z_min, z_max, n_per_side)
    h_pos = (z_max - z_min) / (n_per_side - 1)
    w_pos = _simpson_weights(n_per_side, h_pos)
    nu_pos = np.array([levy_density(z) for z in z_pos])

    z_neg = -z_pos[::-1]
    h_neg = h_pos
    w_neg = _simpson_weights(n_per_side, h_neg)
    nu_neg = np.array([levy_density(z) for z in z_neg])

    return ZQuadrature(z_neg, w_neg, nu_neg, z_pos, w_pos, nu_pos)


def integrate_against_nu(quad: ZQuadrature, integrand_neg, integrand_pos) -> float:
    """Compute ∫ f(z) ν(dz) on the truncated z-domain using Simpson weights.

    Pass the integrand evaluated at the negative and positive nodes separately;
    the function multiplies by ν(z) and the Simpson weights and sums.
    """
    return float(
        (integrand_neg * quad.nu_neg * quad.w_neg).sum()
        + (integrand_pos * quad.nu_pos * quad.w_pos).sum()
    )


# ---------------------------------------------------------------------------
# Hamiltonian, with explicit per-component breakdown
# ---------------------------------------------------------------------------

@dataclass
class HBreakdown:
    """Per-component decomposition of H(t, x, u, V, V_x, V_xx)."""

    drift_term: float
    diffusion_term: float
    levy_integral_compensated: float       # ∫ [V(x(1+uz)) - V(x) - uxz V_x] ν(dz)
    levy_integral_uncompensated: float     # ∫ [V(x(1+uz)) - V(x)] ν(dz)
    compensator: float                     # ux V_x · ∫ z ν(dz)
    H_total: float


def hamiltonian_breakdown(
    *,
    r: float,
    mu: float,
    sigma: float,
    x: float,
    u: float,
    V_at_x: float,
    V_x_at_x: float,
    V_xx_at_x: float,
    V_at_jump: Callable[[float], float],   # V(x(1+uz)) interpolator
    quad: ZQuadrature,
) -> HBreakdown:
    """Evaluate H(t, x, u) component by component.

    The Hamiltonian we are computing (max convention, multiplicative jump):
        H = x(r + u(μ−r)) V_x + ½ u² σ² x² V_xx
            + ∫ [V(x(1+uz)) − V(x) − uxz V_x] ν(dz)

    The compensated integrand is computed as
        [V(x(1+uz)) − V(x)] − uxz V_x
    so the "uncompensated" piece and the "compensator" piece are tracked
    separately. The two should sum (modulo numerical error) to the
    compensated integral; this is verified by callers.
    """
    drift_term = x * (r + u * (mu - r)) * V_x_at_x
    diffusion_term = 0.5 * (u ** 2) * (sigma ** 2) * (x ** 2) * V_xx_at_x

    # Evaluate V at jump targets x*(1+u*z) on the quadrature grid.
    Vj_neg = np.array([V_at_jump(x * (1.0 + u * z)) for z in quad.z_neg])
    Vj_pos = np.array([V_at_jump(x * (1.0 + u * z)) for z in quad.z_pos])

    # Uncompensated jump integral: ∫ [V(x(1+uz)) - V(x)] ν(dz)
    levy_uncomp = integrate_against_nu(
        quad,
        Vj_neg - V_at_x,
        Vj_pos - V_at_x,
    )

    # Compensator: ux V_x · ∫ z ν(dz)
    int_z_nu = integrate_against_nu(quad, quad.z_neg, quad.z_pos)
    compensator = u * x * V_x_at_x * int_z_nu

    levy_comp = levy_uncomp - compensator

    H_total = drift_term + diffusion_term + levy_comp
    return HBreakdown(
        drift_term=drift_term,
        diffusion_term=diffusion_term,
        levy_integral_compensated=levy_comp,
        levy_integral_uncompensated=levy_uncomp,
        compensator=compensator,
        H_total=H_total,
    )


# ---------------------------------------------------------------------------
# Argmax over u via Brent root-finding on dH/du with boundary checks
# ---------------------------------------------------------------------------

def argmax_H_over_u(
    H_at_u: Callable[[float], float],
    u_min: float,
    u_max: float,
    n_bracket: int = 33,
    tol: float = 1e-6,
) -> float:
    """Find argmax_{u ∈ [u_min, u_max]} H(u).

    Strategy: evaluate H on `n_bracket` coarse points, find the bracket where
    the maximum lies, refine inside that bracket with Brent on the central
    difference of H. Then check the two boundary values explicitly. This is
    deliberately a different code path from v1's linspace-argmax.
    """
    us = np.linspace(u_min, u_max, n_bracket)
    Hs = np.array([H_at_u(u) for u in us])
    k = int(np.argmax(Hs))

    # If the max sits on the boundary of the bracket, return that boundary.
    if k == 0:
        u_best = u_min
        H_best = Hs[0]
    elif k == n_bracket - 1:
        u_best = u_max
        H_best = Hs[-1]
    else:
        # Refine via Brent on central difference of H around (us[k-1], us[k+1])
        lo, hi = us[k - 1], us[k + 1]

        def dH_du(u: float) -> float:
            eps = 1e-4 * max(abs(hi - lo), 1e-3)
            return (H_at_u(u + eps) - H_at_u(u - eps)) / (2 * eps)

        try:
            u_best = brentq(dH_du, lo, hi, xtol=tol)
            H_best = H_at_u(u_best)
        except ValueError:
            # Brent fails if dH/du has same sign at both endpoints; fall back
            # to the densest local sample.
            u_best, H_best = us[k], Hs[k]

    # Final boundary check: compare to the explicit endpoints
    H_lo = H_at_u(u_min)
    H_hi = H_at_u(u_max)
    if H_lo > H_best:
        u_best, H_best = u_min, H_lo
    if H_hi > H_best:
        u_best, H_best = u_max, H_hi
    return u_best


# ---------------------------------------------------------------------------
# Linear-x grid + Crank-Nicolson tridiagonal builder (drift + diffusion)
# ---------------------------------------------------------------------------

def _build_tridiag_for_step(
    *,
    x: np.ndarray,
    u_opt: np.ndarray,
    r: float,
    mu: float,
    sigma: float,
    dt: float,
    theta_cn: float = 0.5,
):
    """Build (A, B) for the implicit step: A V^n = B V^{n+1} + Lévy_explicit.

    Uses Crank-Nicolson with weight `theta_cn` (0.5 = standard CN). On a
    non-uniform x-grid we use central differences for V_x and three-point
    stencil for V_xx. Drift/diffusion are evaluated at u_opt at each x.

    Boundary: Dirichlet — first and last rows are identity, RHS will set the
    boundary values from the analytical Merton diffusion-limit value. This
    differs from v1's Neumann + power-law extrapolation.
    """
    n = x.size
    dx_fwd = np.empty(n)
    dx_bwd = np.empty(n)
    dx_fwd[:-1] = x[1:] - x[:-1]
    dx_bwd[1:] = x[1:] - x[:-1]
    dx_fwd[-1] = dx_bwd[-1]
    dx_bwd[0] = dx_fwd[0]

    # First-derivative stencil (central, non-uniform)
    a_x = -dx_fwd / (dx_bwd * (dx_bwd + dx_fwd))    # coefficient on V_{i-1}
    c_x = dx_bwd / (dx_fwd * (dx_bwd + dx_fwd))     # coefficient on V_{i+1}
    b_x = -(a_x + c_x)                               # coefficient on V_i

    # Second-derivative stencil (non-uniform 3-point)
    a_xx = 2.0 / (dx_bwd * (dx_bwd + dx_fwd))
    c_xx = 2.0 / (dx_fwd * (dx_bwd + dx_fwd))
    b_xx = -(a_xx + c_xx)

    drift_coef = x * (r + u_opt * (mu - r))
    diff_coef = 0.5 * (u_opt ** 2) * (sigma ** 2) * (x ** 2)

    L_lower = drift_coef * a_x + diff_coef * a_xx
    L_diag = drift_coef * b_x + diff_coef * b_xx
    L_upper = drift_coef * c_x + diff_coef * c_xx

    # Crank-Nicolson:
    #   (I - θ dt L) V^n = (I + (1-θ) dt L) V^{n+1} + dt * Lévy_explicit
    A_lower = -theta_cn * dt * L_lower[1:]
    A_diag = 1.0 - theta_cn * dt * L_diag
    A_upper = -theta_cn * dt * L_upper[:-1]

    B_lower = (1.0 - theta_cn) * dt * L_lower[1:]
    B_diag = 1.0 + (1.0 - theta_cn) * dt * L_diag
    B_upper = (1.0 - theta_cn) * dt * L_upper[:-1]

    # Dirichlet rows (rows 0 and n-1): identity on A, zero on B.
    A_diag[0] = 1.0
    A_diag[-1] = 1.0
    A_upper[0] = 0.0
    A_lower[-1] = 0.0
    B_diag[0] = 0.0
    B_diag[-1] = 0.0
    B_upper[0] = 0.0
    B_lower[-1] = 0.0

    A = sp_diags([A_lower, A_diag, A_upper], offsets=[-1, 0, 1], format="csr")
    B = sp_diags([B_lower, B_diag, B_upper], offsets=[-1, 0, 1], format="csr")
    return A, B


# ---------------------------------------------------------------------------
# Main solver class
# ---------------------------------------------------------------------------

class FDPIDESolverV2:
    """Independent FD-PIDE reference solver. See module docstring."""

    def __init__(
        self,
        *,
        r: float,
        mu: float,
        sigma: float,
        gamma: float,
        T: float,
        # Spatial grid (linear x with optional logarithmic concentration near x=1)
        x_min: float = 0.05,
        x_max: float = 5.0,
        n_x: int = 401,
        # Time grid
        n_t: int = 200,
        # Lévy
        levy_density: Callable[[float], float],
        z_max: float = 0.99,
        z_min: float = 0.005,
        n_z_per_side: int = 201,        # Simpson likes odd
        # Admissible set and argmax
        u_min: float = 0.0,
        u_max: float = 1.0,
        n_bracket: int = 41,
    ):
        self.r = r
        self.mu = mu
        self.sigma = sigma
        self.gamma = gamma
        self.T = T

        # Linear x grid concentrated near x=1 via tanh stretch.
        s = np.linspace(-1.0, 1.0, n_x)
        stretch = np.tanh(2.0 * s) / np.tanh(2.0)
        self.x = 0.5 * (x_min + x_max) + 0.5 * (x_max - x_min) * stretch
        self.x[0] = x_min
        self.x[-1] = x_max
        # Sort just to be safe (tanh stretch is monotonic so this is a no-op).
        self.x = np.sort(self.x)
        self.n_x = self.x.size

        self.n_t = n_t
        self.dt = T / n_t
        self.t = np.linspace(0.0, T, n_t + 1)

        # Lévy quadrature, built once
        if n_z_per_side % 2 == 0:
            n_z_per_side += 1   # Simpson prefers odd
        self.quad = build_z_quadrature(
            levy_density=levy_density,
            z_min=z_min, z_max=z_max,
            n_per_side=n_z_per_side,
        )

        self.u_min = u_min
        self.u_max = u_max
        self.n_bracket = n_bracket

        # Storage
        self.V = None
        self.u_opt = None

    # -- Boundary values: analytical diffusion-only Merton --
    def _diffusion_merton_value(self, t: float, x: float) -> float:
        """V_diff(t, x) = (x^(1-γ) / (1-γ)) · exp(A (T - t)),
        A = (1-γ)[r + (μ-r)²/(2γσ²)]. Closed-form Merton diffusion value.
        Negative for γ > 1."""
        A = (1 - self.gamma) * (self.r + 0.5 * ((self.mu - self.r) ** 2)
                                / (self.gamma * self.sigma ** 2))
        tau = self.T - t
        return (x ** (1 - self.gamma)) / (1 - self.gamma) * np.exp(A * tau)

    def _terminal_value(self, x: np.ndarray) -> np.ndarray:
        return (x ** (1 - self.gamma)) / (1 - self.gamma)

    # -- V interpolator on a fixed grid (linear interpolation, with extrapolation
    #    by holding the diffusion-Merton value at the boundary) --
    def _make_V_interpolator(self, V: np.ndarray, t: float) -> Callable[[float], float]:
        x_arr = self.x

        def interp(x_query: float) -> float:
            if x_query <= x_arr[0]:
                # Below grid: use diffusion-Merton extrapolation
                return self._diffusion_merton_value(t, max(x_query, 1e-8))
            if x_query >= x_arr[-1]:
                return self._diffusion_merton_value(t, x_query)
            # Linear interpolation
            j = np.searchsorted(x_arr, x_query) - 1
            j = max(0, min(j, len(x_arr) - 2))
            w = (x_query - x_arr[j]) / (x_arr[j + 1] - x_arr[j])
            return float((1 - w) * V[j] + w * V[j + 1])
        return interp

    def _Vx_Vxx_at_grid(self, V: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute V_x, V_xx on the (non-uniform) x-grid by central differences."""
        n = self.n_x
        dx_fwd = np.empty(n); dx_bwd = np.empty(n)
        dx_fwd[:-1] = self.x[1:] - self.x[:-1]
        dx_bwd[1:] = self.x[1:] - self.x[:-1]
        dx_fwd[-1] = dx_bwd[-1]; dx_bwd[0] = dx_fwd[0]

        # First derivative (non-uniform central)
        V_x = np.empty(n)
        V_x[1:-1] = (
            -dx_fwd[1:-1] / (dx_bwd[1:-1] * (dx_bwd[1:-1] + dx_fwd[1:-1])) * V[:-2]
            + (dx_fwd[1:-1] - dx_bwd[1:-1])
              / (dx_bwd[1:-1] * dx_fwd[1:-1]) * V[1:-1]
            + dx_bwd[1:-1] / (dx_fwd[1:-1] * (dx_bwd[1:-1] + dx_fwd[1:-1])) * V[2:]
        )
        V_x[0] = (V[1] - V[0]) / dx_fwd[0]
        V_x[-1] = (V[-1] - V[-2]) / dx_bwd[-1]

        # Second derivative (non-uniform 3-point)
        V_xx = np.empty(n)
        V_xx[1:-1] = (
            2.0 * V[:-2] / (dx_bwd[1:-1] * (dx_bwd[1:-1] + dx_fwd[1:-1]))
            - 2.0 * V[1:-1] / (dx_bwd[1:-1] * dx_fwd[1:-1])
            + 2.0 * V[2:] / (dx_fwd[1:-1] * (dx_bwd[1:-1] + dx_fwd[1:-1]))
        )
        V_xx[0] = V_xx[1]
        V_xx[-1] = V_xx[-2]
        return V_x, V_xx

    def _levy_explicit_rhs(self, V: np.ndarray, t: float) -> np.ndarray:
        """Compute the *compensated* Lévy integral at each grid point at time t.
        Returns an array of length n_x. The integral uses u = u_opt at the
        previous step or the current step's argmax once it's available."""
        # Note: this method is called inside the time-step loop after u_opt is
        # known on the grid. It uses self.u_opt as the per-x policy.
        V_x, _ = self._Vx_Vxx_at_grid(V)
        V_interp = self._make_V_interpolator(V, t)
        out = np.zeros(self.n_x)
        int_z_nu = integrate_against_nu(self.quad, self.quad.z_neg, self.quad.z_pos)
        for i, x_i in enumerate(self.x):
            u = self.u_opt[i]
            Vj_neg = np.array([V_interp(x_i * (1 + u * z)) for z in self.quad.z_neg])
            Vj_pos = np.array([V_interp(x_i * (1 + u * z)) for z in self.quad.z_pos])
            uncomp = integrate_against_nu(
                self.quad, Vj_neg - V[i], Vj_pos - V[i]
            )
            comp = u * x_i * V_x[i] * int_z_nu
            out[i] = uncomp - comp
        return out

    def _optimize_control_step(self, V: np.ndarray, t: float) -> np.ndarray:
        """Return u_opt on the x-grid by evaluating H at each x via Brent."""
        V_x, V_xx = self._Vx_Vxx_at_grid(V)
        V_interp = self._make_V_interpolator(V, t)
        u_opt = np.zeros(self.n_x)
        for i, x_i in enumerate(self.x):
            def H_at_u(u: float, _i=i, _x=x_i,
                       _Vi=V[i], _Vxi=V_x[i], _Vxxi=V_xx[i],
                       _interp=V_interp) -> float:
                bk = hamiltonian_breakdown(
                    r=self.r, mu=self.mu, sigma=self.sigma,
                    x=_x, u=u, V_at_x=_Vi,
                    V_x_at_x=_Vxi, V_xx_at_x=_Vxxi,
                    V_at_jump=_interp, quad=self.quad,
                )
                return bk.H_total

            u_opt[i] = argmax_H_over_u(
                H_at_u, self.u_min, self.u_max, n_bracket=self.n_bracket,
            )
        return u_opt

    # -- Public solve: backward in time --
    def solve(self, *, verbose: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        n_x, n_t = self.n_x, self.n_t
        V = np.zeros((n_t + 1, n_x))
        u_opt = np.zeros((n_t + 1, n_x))
        V[-1, :] = self._terminal_value(self.x)
        # Fill u_opt[T] with diffusion Merton ratio (placeholder; not used)
        u_opt[-1, :] = (self.mu - self.r) / (self.gamma * self.sigma ** 2)

        for n in range(n_t - 1, -1, -1):
            t_n = self.t[n]
            t_np1 = self.t[n + 1]

            # Step 1: optimize control given V^{n+1}
            self.u_opt = u_opt[n + 1, :]
            u_new = self._optimize_control_step(V[n + 1, :], t_np1)
            self.u_opt = u_new

            # Step 2: build CN tridiagonal for drift + diffusion at u_new
            A, B = _build_tridiag_for_step(
                x=self.x, u_opt=u_new,
                r=self.r, mu=self.mu, sigma=self.sigma,
                dt=self.dt,
            )

            # Step 3: explicit Lévy integral evaluated at V^{n+1}
            levy_rhs = self._levy_explicit_rhs(V[n + 1, :], t_np1)

            rhs = B @ V[n + 1, :] + self.dt * levy_rhs
            # Dirichlet boundary at x_min, x_max using diffusion-Merton value at t_n
            rhs[0] = self._diffusion_merton_value(t_n, self.x[0])
            rhs[-1] = self._diffusion_merton_value(t_n, self.x[-1])

            V[n, :] = sp_solve(A, rhs)
            u_opt[n, :] = u_new

            if verbose and n % max(1, n_t // 10) == 0:
                print(f"  v2 step {n}/{n_t}: V(0,1)={np.interp(1.0, self.x, V[n,:]):.4f}, "
                      f"u(0,1)={np.interp(1.0, self.x, u_new):.4f}")

        self.V = V
        self._u_opt_full = u_opt
        return V, u_opt

    # -- Public diagnostic: H(u) at (t=0, x=1) on a u-grid, with breakdown --
    def hamiltonian_breakdown_at(
        self,
        *,
        t: float,
        x: float,
        u_grid: np.ndarray,
        V_override: Optional[np.ndarray] = None,
    ) -> dict:
        """Return per-component H(u) at (t, x) over a u-grid. If V_override is
        provided (shape (n_t+1, n_x)), use it instead of the solver's own V."""
        V_full = self.V if V_override is None else V_override
        if V_full is None:
            raise RuntimeError("Solve first or pass V_override.")
        n_idx = int(np.argmin(np.abs(self.t - t)))
        V = V_full[n_idx, :]
        V_x, V_xx = self._Vx_Vxx_at_grid(V)
        V_interp = self._make_V_interpolator(V, t)
        # Interpolate V, V_x, V_xx at x
        V_at_x = float(np.interp(x, self.x, V))
        V_x_at_x = float(np.interp(x, self.x, V_x))
        V_xx_at_x = float(np.interp(x, self.x, V_xx))

        rows = []
        for u in u_grid:
            bk = hamiltonian_breakdown(
                r=self.r, mu=self.mu, sigma=self.sigma,
                x=x, u=float(u), V_at_x=V_at_x,
                V_x_at_x=V_x_at_x, V_xx_at_x=V_xx_at_x,
                V_at_jump=V_interp, quad=self.quad,
            )
            rows.append({
                "u": float(u),
                "drift": bk.drift_term,
                "diffusion": bk.diffusion_term,
                "levy_uncomp": bk.levy_integral_uncompensated,
                "compensator": bk.compensator,
                "levy_comp": bk.levy_integral_compensated,
                "H": bk.H_total,
            })
        return {
            "t": t, "x": x,
            "V_at_x": V_at_x, "V_x_at_x": V_x_at_x, "V_xx_at_x": V_xx_at_x,
            "rows": rows,
        }

    def get_control_at_t0(self, x: float = 1.0) -> float:
        if not hasattr(self, "_u_opt_full") or self._u_opt_full is None:
            raise RuntimeError("Solve first.")
        return float(np.interp(x, self.x, self._u_opt_full[0, :]))


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def _vg_levy_density(z, sigma=0.2, theta=-0.1, nu=0.3):
    """VG Lévy density (independent reimplementation; same formula as fd_pide_solver)."""
    if abs(z) < 1e-12:
        return 0.0
    C = 1.0 / nu
    temp = np.sqrt(theta ** 2 / sigma ** 4 + 2.0 / (nu * sigma ** 2))
    M = temp - theta / sigma ** 2
    G = temp + theta / sigma ** 2
    if z > 0:
        return C / abs(z) * np.exp(-M * abs(z))
    return C / abs(z) * np.exp(-G * abs(z))


def _test_compensated_integral_for_linear_V():
    """For V(x) = x: compensated integrand is V(x(1+uz)) - V(x) - uxz·V_x = 0
    identically. Therefore ∫ [...] ν(dz) = 0 to machine precision (modulo
    quadrature error on a non-singular integrand: it's exactly zero pointwise)."""
    quad = build_z_quadrature(_vg_levy_density, z_min=0.005, z_max=0.99, n_per_side=201)
    bk = hamiltonian_breakdown(
        r=0.02, mu=0.08, sigma=0.2, x=1.0, u=0.5,
        V_at_x=1.0, V_x_at_x=1.0, V_xx_at_x=0.0,
        V_at_jump=lambda x: x,                # linear V(x) = x
        quad=quad,
    )
    err = abs(bk.levy_integral_compensated)
    print(f"  test 1 (V(x)=x, expect compensated integral = 0): "
          f"got {bk.levy_integral_compensated:.3e}  -> {'PASS' if err < 1e-10 else 'FAIL'}")
    return err < 1e-10


def _test_compensated_integral_for_quadratic_V():
    """For V(x) = x²: compensated integrand = (x(1+uz))² - x² - uxz·2x
    = x²(1 + 2uz + u²z²) - x² - 2x²uz = x²·u²·z²,
    so ∫ [...] ν(dz) = x²·u² · ∫ z² ν(dz) =: x²·u²·M2.

    We compute M2 by running the same Simpson quadrature on integrand z² and
    compare to the breakdown's compensated integral."""
    quad = build_z_quadrature(_vg_levy_density, z_min=0.005, z_max=0.99, n_per_side=201)
    M2 = integrate_against_nu(quad, quad.z_neg ** 2, quad.z_pos ** 2)

    x, u = 1.7, 0.4
    expected = (x ** 2) * (u ** 2) * M2

    bk = hamiltonian_breakdown(
        r=0.02, mu=0.08, sigma=0.2, x=x, u=u,
        V_at_x=x ** 2, V_x_at_x=2 * x, V_xx_at_x=2.0,
        V_at_jump=lambda x_: x_ ** 2,
        quad=quad,
    )
    err = abs(bk.levy_integral_compensated - expected) / abs(expected)
    print(f"  test 2 (V(x)=x², expect x²u²·M2 = {expected:.6f}): "
          f"got {bk.levy_integral_compensated:.6f}, "
          f"rel err = {err:.3e}  -> {'PASS' if err < 1e-6 else 'FAIL'}")
    return err < 1e-6


def _test_compensator_split_consistency():
    """Verify that levy_uncomp − compensator equals levy_comp identically
    (this is just a check on the breakdown bookkeeping, not on quadrature)."""
    quad = build_z_quadrature(_vg_levy_density, z_min=0.005, z_max=0.99, n_per_side=101)
    bk = hamiltonian_breakdown(
        r=0.02, mu=0.08, sigma=0.2, x=1.0, u=0.4,
        V_at_x=-1.0, V_x_at_x=1.0, V_xx_at_x=-2.0,
        V_at_jump=lambda x: -1.0 / max(x, 1e-9),
        quad=quad,
    )
    err = abs(bk.levy_integral_uncompensated - bk.compensator - bk.levy_integral_compensated)
    print(f"  test 3 (split-form consistency): err = {err:.3e}  -> "
          f"{'PASS' if err < 1e-12 else 'FAIL'}")
    return err < 1e-12


def run_unit_tests() -> None:
    print("FDPIDESolverV2 unit tests:")
    ok = True
    ok &= _test_compensated_integral_for_linear_V()
    ok &= _test_compensated_integral_for_quadratic_V()
    ok &= _test_compensator_split_consistency()
    if not ok:
        raise AssertionError("FD v2 unit tests failed")


if __name__ == "__main__":
    run_unit_tests()
