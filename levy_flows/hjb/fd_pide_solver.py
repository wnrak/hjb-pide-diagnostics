"""Finite Difference PIDE Solver for HJB with Lévy Jumps.

Classical baseline for comparison with neural solver.
Uses implicit Euler time-stepping + trapezoidal quadrature for jump integral.

Reference: Cont & Voltchkova (2005), "A finite difference scheme for option
pricing in jump diffusion and exponential Lévy models"
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from typing import Tuple, Callable, Optional
import time


class FDPIDESolver:
    """Finite difference solver for 1D HJB-PIDE with Lévy jumps.

    Solves:
        ∂_t V + sup_u { L^u V + I^u V } = 0
        V(T, x) = U(x)

    where:
        L^u V = x(r + u(μ-r))V_x + 0.5 u²σ²x² V_xx
        I^u V = ∫ [V(x(1+uz)) - V(x) - uxz V_x] ν(dz)
    """

    def __init__(
        self,
        r: float = 0.02,
        mu: float = 0.08,
        sigma: float = 0.2,
        gamma: float = 2.0,
        T: float = 1.0,
        # Grid parameters
        x_min: float = 0.1,
        x_max: float = 5.0,
        n_x: int = 200,
        n_t: int = 100,
        # Jump parameters
        levy_density: Optional[Callable] = None,
        z_min: float = -0.5,
        z_max: float = 0.5,
        n_z: int = 100,
        # Control optimization
        n_u_grid: int = 20,
        u_min: float = 0.0,
        u_max: float = 1.0,
        use_v_interpolation: bool = False,
    ):
        self.r = r
        self.mu = mu
        self.sigma = sigma
        self.gamma = gamma
        self.T = T

        # Spatial grid (log-wealth for stability)
        self.n_x = n_x
        self.log_x_min = np.log(x_min)
        self.log_x_max = np.log(x_max)
        self.log_x = np.linspace(self.log_x_min, self.log_x_max, n_x)
        self.x = np.exp(self.log_x)
        self.dx = self.log_x[1] - self.log_x[0]

        # Time grid
        self.n_t = n_t
        self.dt = T / n_t
        self.t = np.linspace(0, T, n_t + 1)

        # Jump grid
        self.levy_density = levy_density
        self.z_min = z_min
        self.z_max = z_max
        self.n_z = n_z
        self.z = np.linspace(z_min, z_max, n_z)
        self.dz = self.z[1] - self.z[0]

        # Precompute Lévy weights if density provided
        if levy_density is not None:
            self.levy_weights = np.array([levy_density(zi) for zi in self.z])
        else:
            self.levy_weights = np.zeros(n_z)

        # Control optimization: u-grid resolution and whether to use proper
        # V interpolation when evaluating H(u) (vs the cheap Taylor expansion
        # used by the original FD-PIDE code, which is only accurate near u=0).
        self.n_u_grid = n_u_grid
        self.u_min = u_min
        self.u_max = u_max
        self.use_v_interpolation = use_v_interpolation

        # Storage
        self.V = None
        self.u_opt = None

    def terminal_condition(self, x: np.ndarray) -> np.ndarray:
        """CRRA utility U(x) = x^(1-γ)/(1-γ)."""
        return x ** (1 - self.gamma) / (1 - self.gamma)

    def _build_diffusion_matrix(self, u: float) -> sparse.csr_matrix:
        """Build tridiagonal matrix for diffusion operator at control u.

        In log-wealth coordinates y = log(x):
            L^u V = (r + u(μ-r) - 0.5u²σ²) V_y + 0.5 u²σ² V_yy
        """
        n = self.n_x
        drift = self.r + u * (self.mu - self.r) - 0.5 * u**2 * self.sigma**2
        diffusion = 0.5 * u**2 * self.sigma**2

        # Centered differences
        # V_y ≈ (V_{i+1} - V_{i-1}) / (2dx)
        # V_yy ≈ (V_{i+1} - 2V_i + V_{i-1}) / dx²

        alpha = diffusion / self.dx**2 - drift / (2 * self.dx)  # lower diagonal
        beta = -2 * diffusion / self.dx**2  # main diagonal
        gamma = diffusion / self.dx**2 + drift / (2 * self.dx)  # upper diagonal

        # Build sparse matrix
        diagonals = [
            alpha * np.ones(n - 1),
            beta * np.ones(n),
            gamma * np.ones(n - 1),
        ]
        L = sparse.diags(diagonals, [-1, 0, 1], format='csr')

        # Boundary conditions (Neumann: V_y = 0 at boundaries)
        L = L.tolil()
        L[0, 0] = beta + alpha  # reflect lower boundary
        L[-1, -1] = beta + gamma  # reflect upper boundary

        return L.tocsr()

    def _compute_jump_integral(self, V: np.ndarray, u: float) -> np.ndarray:
        """Compute compensated jump integral via trapezoidal quadrature.

        I^u V(x) = ∫ [V(x(1+uz)) - V(x) - uxz V_x(x)] ν(dz)

        In log-wealth: x(1+uz) → y + log(1+uz)
        """
        n = self.n_x
        integral = np.zeros(n)

        if np.sum(np.abs(self.levy_weights)) < 1e-10:
            return integral

        # Compute V_y (derivative in log-wealth)
        V_y = np.zeros(n)
        V_y[1:-1] = (V[2:] - V[:-2]) / (2 * self.dx)
        V_y[0] = (V[1] - V[0]) / self.dx
        V_y[-1] = (V[-1] - V[-2]) / self.dx

        for i, y_i in enumerate(self.log_x):
            x_i = self.x[i]
            integrand = np.zeros(self.n_z)

            for j, z_j in enumerate(self.z):
                if abs(z_j) < 1e-10 or self.levy_weights[j] < 1e-10:
                    continue

                # New log-wealth after jump
                jump_arg = 1 + u * z_j
                if jump_arg <= 0:
                    # Bankruptcy - large negative value
                    V_new = -1e10
                else:
                    y_new = y_i + np.log(jump_arg)

                    # Interpolate V at y_new
                    if y_new < self.log_x_min:
                        # Extrapolate using power law
                        V_new = V[0] * np.exp((1 - self.gamma) * (y_new - self.log_x_min))
                    elif y_new > self.log_x_max:
                        V_new = V[-1] * np.exp((1 - self.gamma) * (y_new - self.log_x_max))
                    else:
                        # Linear interpolation
                        idx = (y_new - self.log_x_min) / self.dx
                        idx_low = int(np.floor(idx))
                        idx_high = min(idx_low + 1, n - 1)
                        weight = idx - idx_low
                        V_new = (1 - weight) * V[idx_low] + weight * V[idx_high]

                # Compensated integrand: V(x(1+uz)) - V(x) - uxz V_x
                # In log-wealth: V_x = V_y / x, so uxz V_x = uz V_y
                compensation = u * z_j * V_y[i]
                integrand[j] = (V_new - V[i] - compensation) * self.levy_weights[j]

            # Trapezoidal quadrature
            integral[i] = np.trapz(integrand, dx=self.dz)

        return integral

    def _optimize_control(self, V: np.ndarray, V_y: np.ndarray, V_yy: np.ndarray) -> np.ndarray:
        """Find optimal control at each grid point via discrete argmax over a u-grid.

        For CRRA without jumps the FOC u* = -(μ-r)V_y/(σ²V_yy) is interior, but
        the same code path also handles the constrained jump case where the
        argmax may sit on the boundary. The u-grid resolution is `n_u_grid`
        (default 20; legacy paper used this); in the audit we pass
        n_u_grid >= 200 to remove the discrete-argmax quantization that was
        making "FD vs neural agreement" depend on linspace fence-posts.

        When ``use_v_interpolation`` is True we evaluate H(u) by linearly
        interpolating V(x(1+uz)) on the spatial grid (same scheme as the
        residual integral in `_compute_jump_integral`). The original code
        used a degree-2 Taylor expansion in log(1+uz) around the grid point,
        which is only accurate for small |uz|; for |uz| > ~0.3 the Taylor
        truncation error competes with the actual jump contribution.
        """
        n = self.n_x
        u_grid = np.linspace(self.u_min, self.u_max, self.n_u_grid)
        n_u = u_grid.size
        u_opt = np.zeros(n)
        has_jumps = np.sum(np.abs(self.levy_weights)) > 1e-10

        # Pre-build masks for the active z indices (skip near-zero z and zero weights)
        z = self.z
        weights = self.levy_weights
        if has_jumps:
            active = (np.abs(z) >= 1e-10) & (weights >= 1e-10)
            z_active = z[active]
            w_active = weights[active]
            # jump_args[k, j] = 1 + u_k * z_j, shape (n_u, n_z_active)
            jump_args = 1.0 + np.outer(u_grid, z_active)
            safe = jump_args > 0
            # log(1+uz) computed only where safe; zeroed elsewhere (gets masked later)
            log_jumps = np.where(safe, np.log(np.where(safe, jump_args, 1.0)), 0.0)
        else:
            z_active = np.zeros(0)
            w_active = np.zeros(0)
            jump_args = np.zeros((n_u, 0))
            safe = np.zeros((n_u, 0), dtype=bool)
            log_jumps = np.zeros((n_u, 0))

        # Diffusion-side coefficients vectorized over u
        drift_u = self.r + u_grid * (self.mu - self.r) - 0.5 * u_grid ** 2 * self.sigma ** 2
        diff_u = 0.5 * u_grid ** 2 * self.sigma ** 2

        for i in range(n):
            y_i = self.log_x[i]
            H_diff = drift_u * V_y[i] + diff_u * V_yy[i]   # (n_u,)

            if has_jumps:
                if self.use_v_interpolation:
                    # y_new = y_i + log(1 + u_k z_j) where safe; bankruptcy elsewhere
                    y_new = y_i + log_jumps                  # (n_u, n_z_active)
                    # In-grid: linear interpolate V on log-wealth grid
                    idx = (y_new - self.log_x_min) / self.dx
                    idx_low = np.floor(idx).astype(int)
                    in_grid = (y_new >= self.log_x_min) & (y_new <= self.log_x_max)
                    idx_low_clamped = np.clip(idx_low, 0, n - 2)
                    w_lo = idx - idx_low_clamped
                    V_new_in = (1 - w_lo) * V[idx_low_clamped] + w_lo * V[idx_low_clamped + 1]
                    # Below grid: power-law extrapolation
                    V_new_below = V[0] * np.exp(
                        (1 - self.gamma) * (y_new - self.log_x_min)
                    )
                    # Above grid: power-law extrapolation
                    V_new_above = V[-1] * np.exp(
                        (1 - self.gamma) * (y_new - self.log_x_max)
                    )
                    V_new = np.where(
                        y_new < self.log_x_min, V_new_below,
                        np.where(y_new > self.log_x_max, V_new_above, V_new_in),
                    )
                    # Bankruptcy: heavy penalty
                    V_new = np.where(safe, V_new, -1e10)
                else:
                    # Legacy 2nd-order Taylor expansion in log(1+uz)
                    V_new = V[i] + V_y[i] * log_jumps + 0.5 * V_yy[i] * log_jumps ** 2
                    V_new = np.where(safe, V_new, -1e10)

                compensation = np.outer(u_grid, z_active) * V_y[i]   # (n_u, n_z_active)
                integrand = (V_new - V[i] - compensation) * w_active * self.dz
                H_jump = integrand.sum(axis=1)
            else:
                H_jump = 0.0

            H_total = H_diff + H_jump
            u_opt[i] = u_grid[int(np.argmax(H_total))]

        return u_opt

    def solve(self, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, float]:
        """Solve HJB-PIDE using implicit Euler + policy iteration.

        Returns:
            V: Value function on grid (n_t+1, n_x)
            u_opt: Optimal control on grid (n_t+1, n_x)
            elapsed: Wall-clock time in seconds
        """
        start_time = time.time()

        n_x = self.n_x
        n_t = self.n_t

        # Initialize storage
        V = np.zeros((n_t + 1, n_x))
        u_opt = np.zeros((n_t + 1, n_x))

        # Terminal condition
        V[n_t, :] = self.terminal_condition(self.x)

        # Merton ratio as initial guess
        merton_u = (self.mu - self.r) / (self.gamma * self.sigma**2)
        u_opt[n_t, :] = np.clip(merton_u, 0.01, 1.0)

        # Backward iteration
        for k in range(n_t - 1, -1, -1):
            if verbose and k % 20 == 0:
                print(f"  Time step {n_t - k}/{n_t}")

            # Policy iteration
            V_curr = V[k + 1, :].copy()
            u_curr = u_opt[k + 1, :].copy()

            for policy_iter in range(5):  # Max 5 policy iterations
                # Use average control for now (simplification)
                u_avg = np.mean(u_curr)

                # Build system matrix: (I - dt*L) V_new = V_old + dt*I
                L = self._build_diffusion_matrix(u_avg)
                I_matrix = sparse.eye(n_x)
                A = I_matrix - self.dt * L

                # Jump integral at current V
                jump_integral = self._compute_jump_integral(V_curr, u_avg)

                # RHS
                b = V[k + 1, :] + self.dt * jump_integral

                # Solve
                V_new = spsolve(A, b)

                # Update control
                V_y = np.zeros(n_x)
                V_y[1:-1] = (V_new[2:] - V_new[:-2]) / (2 * self.dx)
                V_y[0] = (V_new[1] - V_new[0]) / self.dx
                V_y[-1] = (V_new[-1] - V_new[-2]) / self.dx

                V_yy = np.zeros(n_x)
                V_yy[1:-1] = (V_new[2:] - 2*V_new[1:-1] + V_new[:-2]) / self.dx**2
                V_yy[0] = V_yy[1]
                V_yy[-1] = V_yy[-2]

                u_new = self._optimize_control(V_new, V_y, V_yy)

                # Check convergence
                if np.max(np.abs(u_new - u_curr)) < 1e-4:
                    break

                V_curr = V_new
                u_curr = u_new

            V[k, :] = V_new
            u_opt[k, :] = u_new

        elapsed = time.time() - start_time

        self.V = V
        self.u_opt = u_opt

        if verbose:
            print(f"FD-PIDE solver completed in {elapsed:.2f}s")

        return V, u_opt, elapsed

    def get_control_at_t0(self, x: float = 1.0) -> float:
        """Get optimal control at t=0 for given wealth."""
        if self.u_opt is None:
            raise ValueError("Must call solve() first")

        log_x = np.log(x)
        if log_x < self.log_x_min or log_x > self.log_x_max:
            # Extrapolate
            return self.u_opt[0, self.n_x // 2]

        # Interpolate
        idx = (log_x - self.log_x_min) / self.dx
        idx_low = int(np.floor(idx))
        idx_high = min(idx_low + 1, self.n_x - 1)
        weight = idx - idx_low

        return (1 - weight) * self.u_opt[0, idx_low] + weight * self.u_opt[0, idx_high]


def vg_levy_density(z: float, sigma: float = 0.2, theta: float = -0.1, nu: float = 0.3) -> float:
    """Variance Gamma Lévy density.

    ν(z) = C/|z| * exp(-M|z|) for z > 0
    ν(z) = C/|z| * exp(-G|z|) for z < 0

    where C = 1/nu, M = sqrt(theta²/(sigma⁴) + 2/(nu*sigma²)) - theta/sigma²
          G = sqrt(theta²/(sigma⁴) + 2/(nu*sigma²)) + theta/sigma²
    """
    if abs(z) < 1e-6:
        return 0.0

    C = 1.0 / nu
    temp = np.sqrt(theta**2 / sigma**4 + 2.0 / (nu * sigma**2))
    M = temp - theta / sigma**2
    G = temp + theta / sigma**2

    if z > 0:
        return C / abs(z) * np.exp(-M * abs(z))
    else:
        return C / abs(z) * np.exp(-G * abs(z))


def compare_neural_vs_fd(neural_u: float, verbose: bool = True) -> dict:
    """Compare neural solver result against FD-PIDE baseline."""

    # Merton parameters
    r, mu, sigma, gamma, T = 0.02, 0.08, 0.2, 2.0, 1.0
    merton_u = (mu - r) / (gamma * sigma**2)

    results = {
        'merton_analytical': merton_u,
        'neural_solver': neural_u,
    }

    # 1. Diffusion-only FD
    if verbose:
        print("=" * 60)
        print("FD-PIDE Baseline: Diffusion Only")
        print("=" * 60)

    fd_diff = FDPIDESolver(
        r=r, mu=mu, sigma=sigma, gamma=gamma, T=T,
        levy_density=None,
        n_x=100, n_t=50,
    )
    V_diff, u_diff, time_diff = fd_diff.solve(verbose=verbose)
    fd_u_diff = fd_diff.get_control_at_t0(1.0)

    results['fd_diffusion'] = fd_u_diff
    results['fd_diffusion_time'] = time_diff
    results['fd_diffusion_error'] = abs(fd_u_diff - merton_u) / merton_u

    if verbose:
        print(f"  FD control (diffusion): {fd_u_diff:.4f}")
        print(f"  Merton analytical: {merton_u:.4f}")
        print(f"  Error: {results['fd_diffusion_error']*100:.2f}%")

    # 2. VG jumps FD
    if verbose:
        print("\n" + "=" * 60)
        print("FD-PIDE Baseline: VG Jumps")
        print("=" * 60)

    vg_density = lambda z: vg_levy_density(z, sigma=0.2, theta=-0.1, nu=0.3)

    fd_vg = FDPIDESolver(
        r=r, mu=mu, sigma=sigma, gamma=gamma, T=T,
        levy_density=vg_density,
        z_min=-0.5, z_max=0.5, n_z=50,
        n_x=100, n_t=50,
    )
    V_vg, u_vg, time_vg = fd_vg.solve(verbose=verbose)
    fd_u_vg = fd_vg.get_control_at_t0(1.0)

    results['fd_vg'] = fd_u_vg
    results['fd_vg_time'] = time_vg
    results['fd_vg_reduction'] = (merton_u - fd_u_vg) / merton_u

    if verbose:
        print(f"  FD control (VG): {fd_u_vg:.4f}")
        print(f"  Reduction from Merton: {results['fd_vg_reduction']*100:.1f}%")

    # 3. Comparison summary
    if verbose:
        print("\n" + "=" * 60)
        print("Comparison Summary")
        print("=" * 60)
        print(f"{'Method':<25} {'Control u':<12} {'Time (s)':<12}")
        print("-" * 50)
        print(f"{'Analytical (Merton)':<25} {merton_u:<12.4f} {'--':<12}")
        print(f"{'FD-PIDE (diffusion)':<25} {fd_u_diff:<12.4f} {time_diff:<12.2f}")
        print(f"{'FD-PIDE (VG)':<25} {fd_u_vg:<12.4f} {time_vg:<12.2f}")
        print(f"{'Neural solver (VG)':<25} {neural_u:<12.4f} {'~300':<12}")

    return results


if __name__ == "__main__":
    # Test FD solver standalone
    print("Testing FD-PIDE solver...")

    # Diffusion only
    fd = FDPIDESolver(
        r=0.02, mu=0.08, sigma=0.2, gamma=2.0, T=1.0,
        levy_density=None,
        n_x=100, n_t=50,
    )
    V, u, elapsed = fd.solve()

    merton_u = (0.08 - 0.02) / (2.0 * 0.2**2)
    fd_u = fd.get_control_at_t0(1.0)

    print(f"\nMerton ratio: {merton_u:.4f}")
    print(f"FD solution: {fd_u:.4f}")
    print(f"Error: {abs(fd_u - merton_u)/merton_u*100:.2f}%")
    print(f"Time: {elapsed:.2f}s")
