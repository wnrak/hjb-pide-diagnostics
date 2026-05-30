"""2D Portfolio HJB-PIDE Solver with Lévy Jumps.

Extends the 1D solver to 2-asset portfolios to demonstrate scalability.
State: (t, x) where x is scalar wealth, but control is 2D: u = (u1, u2).

Phase 1 corrections (2026-04):
- ``merton_u`` is now the *correlated* unconstrained solution
  u* = (1/γ) Σ⁻¹ (μ - r·1), not the per-asset formula. The per-asset
  version is only correct for ρ = 0 and was producing inflated
  "reduction vs Merton" numbers.
- A simplex-sampled argmax-target loss is added to the training step
  so that u_φ is actually trained to maximize H, not just to make the
  HJB residual zero (which is satisfiable with any u as V adapts).

Phase 3a corrections (2026-04):
- The Lévy sampler now uses the corrected 1D ``VarianceGammaMeasure``
  per asset, with the same 50/50-mixture proposal (rates M/2, G/2 plus
  the missing 0.5 mixture probability factor that was halving the
  integral in the previous draft) and the same per-batch median weight
  cap. The previous ``_sample_vg_jumps`` used a uniform proposal on
  [-0.5, 0.5] together with ``weights /= weights.mean()`` self-
  normalization, which (i) hardcoded an inconsistent truncation and
  (ii) returned an estimator of E_ν-normalized[f] rather than the
  required ∫f(z) ν(dz). Both are fixed here.
- Truncation is now an explicit constructor argument matching the 1D
  default ``|z| ∈ [0.01, 0.99]``; bankruptcy guard 1+u·z>0 holds for
  every admissible (u,z).
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict
import time

from levy_flows.hjb.levy_integral import VarianceGammaMeasure


class PolicyNetwork2D(nn.Module):
    """Policy network for 2-asset portfolio.

    Outputs (u1, u2) ∈ [0, 1]² with constraint u1 + u2 ≤ 1.
    """

    def __init__(self, hidden_dim: int = 128, n_layers: int = 4):
        super().__init__()

        layers = []
        prev_dim = 2  # (t, log_x)

        for i in range(n_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.SiLU())
            prev_dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, 2))  # Output (u1, u2)

        self.net = nn.Sequential(*layers)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Time tensor (batch,)
            x: Wealth tensor (batch, 1)

        Returns:
            u: Control tensor (batch, 2) with u1, u2 ∈ [0, 1] and u1 + u2 ≤ 1
        """
        log_x = torch.log(x + 1e-6)
        inp = torch.cat([t.unsqueeze(-1), log_x], dim=-1)
        raw = self.net(inp)

        # Softmax to ensure u1 + u2 ≤ 1 (allocations to risky assets)
        # Add a third "cash" component
        raw_with_cash = torch.cat([raw, torch.zeros_like(raw[:, :1])], dim=-1)
        weights = torch.softmax(raw_with_cash, dim=-1)

        return weights[:, :2]  # Return only (u1, u2)


class ValueNetwork2D(nn.Module):
    """Value function network for 2D problem."""

    def __init__(self, hidden_dim: int = 128, n_layers: int = 4):
        super().__init__()

        layers = []
        prev_dim = 2  # (t, log_x)

        for i in range(n_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.SiLU())
            if i > 0:
                layers.append(nn.LayerNorm(hidden_dim))
            prev_dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        log_x = torch.log(x + 1e-6)
        inp = torch.cat([t.unsqueeze(-1), log_x], dim=-1)
        return self.net(inp)


class TwoAssetLevyHJBSolver:
    """HJB-PIDE solver for 2-asset portfolio with Lévy jumps.

    Dynamics:
        dX/X = (r + u1(μ1-r) + u2(μ2-r))dt + u1 σ1 dW1 + u2 σ2 dW2 + u1 dJ1 + u2 dJ2

    where J1, J2 are independent VG jump processes (can be extended to correlated).
    """

    def __init__(
        self,
        r: float = 0.02,
        mu1: float = 0.10,
        mu2: float = 0.06,
        sigma1: float = 0.25,
        sigma2: float = 0.15,
        rho: float = 0.3,  # Correlation between Brownian motions
        gamma: float = 2.0,
        T: float = 1.0,
        # VG parameters for each asset
        vg_theta1: float = -0.10,
        vg_theta2: float = -0.05,
        vg_sigma: float = 0.2,
        vg_nu: float = 0.3,
        # Lévy truncation, matched to the corrected 1D default
        truncation_min: float = 0.01,
        truncation_max: float = 0.99,
        # Network params
        hidden_dim: int = 128,
        n_layers: int = 4,
        n_levy_samples: int = 32,
    ):
        self.r = r
        self.mu1 = mu1
        self.mu2 = mu2
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.rho = rho
        self.gamma = gamma
        self.T = T

        self.vg_theta1 = vg_theta1
        self.vg_theta2 = vg_theta2
        self.vg_sigma = vg_sigma
        self.vg_nu = vg_nu
        self.truncation_min = truncation_min
        self.truncation_max = truncation_max

        self.n_levy_samples = n_levy_samples

        # Per-asset Lévy measures using the corrected 1D IS proposal.
        # Sharing the 1D class guarantees identical proposal density,
        # weight conventions, and clip threshold across the 1D and 2D
        # solvers — no separate code path to maintain.
        self.measure1 = VarianceGammaMeasure(
            sigma=vg_sigma, theta=vg_theta1, nu=vg_nu,
            truncation_min=truncation_min, truncation_max=truncation_max,
            intensity_scale=1.0,
        )
        self.measure2 = VarianceGammaMeasure(
            sigma=vg_sigma, theta=vg_theta2, nu=vg_nu,
            truncation_min=truncation_min, truncation_max=truncation_max,
            intensity_scale=1.0,
        )

        # Networks
        self.value_net = ValueNetwork2D(hidden_dim, n_layers)
        self.policy_net = PolicyNetwork2D(hidden_dim, n_layers)

        # Merton ratios for comparison.
        # Correlated unconstrained solution: u* = (1/γ) Σ⁻¹ (μ - r·1)
        # where Σ_ij = ρ_ij σ_i σ_j is the asset return covariance.
        # The per-asset formula u_i = (μ_i - r)/(γ σ_i^2) is only correct
        # for ρ = 0 and was previously producing inflated "reduction vs
        # Merton" numbers in the §4.8 table.
        cov = np.array(
            [
                [sigma1 ** 2, rho * sigma1 * sigma2],
                [rho * sigma1 * sigma2, sigma2 ** 2],
            ],
            dtype=float,
        )
        excess = np.array([mu1 - r, mu2 - r], dtype=float)
        u_correlated = (1.0 / gamma) * np.linalg.solve(cov, excess)
        self.merton_u1 = float(u_correlated[0])
        self.merton_u2 = float(u_correlated[1])
        # Per-asset values kept under explicit names for any caller that
        # actually wanted the uncorrelated reference.
        self.merton_u1_uncorrelated = (mu1 - r) / (gamma * sigma1 ** 2)
        self.merton_u2_uncorrelated = (mu2 - r) / (gamma * sigma2 ** 2)

    def terminal_utility(self, x: torch.Tensor) -> torch.Tensor:
        """CRRA utility."""
        return (x ** (1 - self.gamma)) / (1 - self.gamma)

    def _sample_vg_jumps(self, n_samples: int, asset: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample from the Lévy measure for asset ``asset`` (1 or 2).

        Delegates to the corrected 1D ``VarianceGammaMeasure``: 50/50
        mixture of one-sided shifted exponentials with rates M/2 and G/2,
        proposal density including the 0.5 mixture probability factor,
        weights w = ν/q with the per-batch median cap inherited from the
        1D class. No self-normalization. Returns ``z`` of shape
        ``(n_samples,)`` and ``weights`` of shape ``(n_samples,)`` for
        compatibility with the existing ``_hamiltonian_at_u`` loop.
        """
        if asset == 1:
            measure = self.measure1
        elif asset == 2:
            measure = self.measure2
        else:
            raise ValueError(f"asset must be 1 or 2, got {asset}")
        z, w = measure.sample(n_samples, torch.device("cpu"))
        return z.squeeze(-1), w

    def _hamiltonian_at_u(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        u1: torch.Tensor,
        u2: torch.Tensor,
        V: torch.Tensor,
        V_x: torch.Tensor,
        V_xx: torch.Tensor,
        z1: torch.Tensor,
        w1: torch.Tensor,
        z2: torch.Tensor,
        w2: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the Hamiltonian H(t, x, u, V, V_x, V_xx) at a given (u1, u2).

        The Hamiltonian is the bracket inside ``sup_u`` in the HJB-PIDE; the
        residual ``V_t + H`` should be zero at the optimum. Splitting H out
        from the residual lets the argmax-target loss evaluate it at multiple
        candidate u-values with shared (V, V_x, V_xx) and shared Lévy samples.
        """
        drift = x * (self.r + u1 * (self.mu1 - self.r) + u2 * (self.mu2 - self.r))

        var1 = (u1 * self.sigma1) ** 2
        var2 = (u2 * self.sigma2) ** 2
        cov = 2 * u1 * u2 * self.sigma1 * self.sigma2 * self.rho
        diffusion = 0.5 * (var1 + var2 + cov) * x ** 2

        L_V = drift * V_x + diffusion * V_xx

        batch_size = x.shape[0]
        I1 = torch.zeros(batch_size, 1, device=x.device)
        for k in range(self.n_levy_samples):
            x_jump = torch.clamp(x * (1 + u1 * z1[k]), min=1e-6)
            V_jump = self.value_net(t, x_jump)
            compensation = u1 * z1[k] * x * V_x
            I1 = I1 + w1[k] * (V_jump - V - compensation) / self.n_levy_samples

        I2 = torch.zeros(batch_size, 1, device=x.device)
        for k in range(self.n_levy_samples):
            x_jump = torch.clamp(x * (1 + u2 * z2[k]), min=1e-6)
            V_jump = self.value_net(t, x_jump)
            compensation = u2 * z2[k] * x * V_x
            I2 = I2 + w2[k] * (V_jump - V - compensation) / self.n_levy_samples

        return L_V + I1 + I2

    def compute_hjb_residual(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute HJB residual at the current policy: residual = V_t + H(u_φ).

        Returns the residual and V (so callers can reuse V/V_x/V_xx).
        """
        t = t.requires_grad_(True)
        x = x.requires_grad_(True)

        V = self.value_net(t, x)
        u = self.policy_net(t, x)
        u1, u2 = u[:, 0:1], u[:, 1:2]

        V_t = torch.autograd.grad(V.sum(), t, create_graph=True)[0]
        V_x = torch.autograd.grad(V.sum(), x, create_graph=True)[0]
        V_xx = torch.autograd.grad(V_x.sum(), x, create_graph=True)[0]

        z1, w1 = self._sample_vg_jumps(self.n_levy_samples, asset=1)
        z2, w2 = self._sample_vg_jumps(self.n_levy_samples, asset=2)

        H = self._hamiltonian_at_u(t, x, u1, u2, V, V_x, V_xx, z1, w1, z2, w2)
        residual = V_t + H
        return residual, V

    def policy_argmax_loss(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        n_candidates: int = 8,
    ) -> torch.Tensor:
        """MSE between u_φ and the simplex argmax of H over n_candidates samples.

        Without this term the residual loss alone is satisfied by *any* u
        once V adapts to it: ``V_t + H = 0`` is one equation in two unknowns
        (V and u). The argmax-target loss enforces u = argmax_u H so that
        the (V, u) pair actually solves the HJB. Candidates are drawn from
        Dirichlet(1, 1, 1) over (u_1, u_2, cash), which is the same simplex
        the policy network outputs through its softmax.
        """
        # V_x and V_xx need autograd to be enabled even though we'll detach
        # them before the argmax. Compute them outside no_grad, detach, then
        # do the candidate sweep without graph.
        t_grad = t.detach().clone().requires_grad_(True)
        x_grad = x.detach().clone().requires_grad_(True)
        V = self.value_net(t_grad, x_grad)
        V_x = torch.autograd.grad(V.sum(), x_grad, create_graph=True)[0]
        V_xx = torch.autograd.grad(V_x.sum(), x_grad, create_graph=False)[0]
        V = V.detach()
        V_x = V_x.detach()
        V_xx = V_xx.detach()

        with torch.no_grad():
            z1, w1 = self._sample_vg_jumps(self.n_levy_samples, asset=1)
            z2, w2 = self._sample_vg_jumps(self.n_levy_samples, asset=2)

            batch = x.shape[0]
            dirichlet = torch.distributions.Dirichlet(
                torch.ones(3, device=x.device, dtype=x.dtype)
            )
            cand = dirichlet.sample((batch, n_candidates))  # (batch, K, 3)

            H_all = torch.zeros(batch, n_candidates, device=x.device, dtype=x.dtype)
            for k in range(n_candidates):
                u1k = cand[:, k, 0:1]
                u2k = cand[:, k, 1:2]
                Hk = self._hamiltonian_at_u(
                    t.detach(), x.detach(),
                    u1k, u2k, V, V_x, V_xx, z1, w1, z2, w2,
                )
                H_all[:, k] = Hk.squeeze(-1)
            argmax_idx = H_all.argmax(dim=1)
            u_target = torch.gather(
                cand[:, :, :2], 1,
                argmax_idx.view(batch, 1, 1).expand(batch, 1, 2),
            ).squeeze(1).detach()

        u_pred = self.policy_net(t, x)
        return ((u_pred - u_target) ** 2).mean()

    def fit(
        self,
        n_epochs: int = 500,
        batch_size: int = 256,
        lr: float = 1e-3,
        lambda_terminal: float = 10.0,
        lambda_argmax: float = 1.0,
        n_argmax_candidates: int = 8,
        argmax_frequency: int = 2,
        verbose: bool = True,
    ) -> Dict:
        """Train the solver.

        Loss has three terms: terminal, HJB residual at the current policy,
        and an argmax-target loss for the policy. The argmax loss is what
        actually trains u_φ toward the maximizer; the residual loss alone
        is satisfiable with any u as V adapts.
        """
        optimizer = torch.optim.Adam(
            list(self.value_net.parameters()) + list(self.policy_net.parameters()),
            lr=lr,
        )

        history = {'loss': [], 'u1': [], 'u2': []}

        for epoch in range(n_epochs):
            # Sample collocation points
            t = torch.rand(batch_size) * self.T
            x = torch.exp(torch.randn(batch_size, 1) * 0.5)  # Log-normal wealth

            # Terminal condition
            t_term = torch.full((batch_size,), self.T)
            x_term = torch.exp(torch.randn(batch_size, 1) * 0.5)
            V_term = self.value_net(t_term, x_term)
            U_term = self.terminal_utility(x_term)
            loss_term = ((V_term - U_term)**2).mean()

            # HJB residual (after warmup)
            if epoch > 50:
                residual, _ = self.compute_hjb_residual(t, x)
                loss_hjb = (residual**2).mean()
            else:
                loss_hjb = torch.tensor(0.0)

            # Argmax-target policy loss (after warmup, amortized).
            # Only include it in total loss on epochs where it's freshly
            # computed; otherwise its tensor would carry stale graph state
            # from the previous batch and break backward().
            if epoch > 50 and (epoch % argmax_frequency == 0 or epoch == n_epochs - 1):
                loss_argmax = self.policy_argmax_loss(
                    t, x, n_candidates=n_argmax_candidates,
                )
                loss = lambda_terminal * loss_term + loss_hjb + lambda_argmax * loss_argmax
            else:
                loss_argmax = torch.tensor(0.0)
                loss = lambda_terminal * loss_term + loss_hjb

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track
            with torch.no_grad():
                t_test = torch.zeros(100)
                x_test = torch.ones(100, 1)
                u_test = self.policy_net(t_test, x_test)
                u1_mean = u_test[:, 0].mean().item()
                u2_mean = u_test[:, 1].mean().item()

            history['loss'].append(loss.item())
            history['u1'].append(u1_mean)
            history['u2'].append(u2_mean)

            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}: loss={loss.item():.4f}, u1={u1_mean:.4f}, u2={u2_mean:.4f}")

        return history

    def get_policy(self) -> Tuple[float, float]:
        """Get learned policy at t=0, x=1."""
        with torch.no_grad():
            t = torch.zeros(100)
            x = torch.ones(100, 1)
            u = self.policy_net(t, x)
            return u[:, 0].mean().item(), u[:, 1].mean().item()


def run_2d_experiment():
    """Run 2D portfolio experiment."""
    print("=" * 60)
    print("2-ASSET PORTFOLIO EXPERIMENT")
    print("=" * 60)

    # Asset parameters
    r = 0.02
    params = {
        'Asset 1 (High vol)': {'mu': 0.10, 'sigma': 0.25, 'vg_theta': -0.10},
        'Asset 2 (Low vol)': {'mu': 0.06, 'sigma': 0.15, 'vg_theta': -0.05},
    }

    gamma = 2.0

    print("\nAsset Parameters:")
    for name, p in params.items():
        merton = (p['mu'] - r) / (gamma * p['sigma']**2)
        print(f"  {name}: mu={p['mu']}, sigma={p['sigma']}, theta_VG={p['vg_theta']}, Merton_u={merton:.4f}")

    # 2D solver
    print("\n--- Training 2D Solver ---")
    solver = TwoAssetLevyHJBSolver(
        r=r,
        mu1=0.10, mu2=0.06,
        sigma1=0.25, sigma2=0.15,
        rho=0.3,
        gamma=gamma,
        vg_theta1=-0.10, vg_theta2=-0.05,
        hidden_dim=128,
        n_layers=4,
        n_levy_samples=32,
    )

    start = time.time()
    history = solver.fit(n_epochs=500, batch_size=256, verbose=True)
    elapsed = time.time() - start

    u1, u2 = solver.get_policy()

    print(f"\n--- Results ---")
    print(f"Training time: {elapsed:.1f}s")
    print(f"Merton ratios: u1*={solver.merton_u1:.4f}, u2*={solver.merton_u2:.4f}")
    print(f"Learned policy: u1={u1:.4f}, u2={u2:.4f}")
    print(f"Reduction from Merton: u1: {(solver.merton_u1-u1)/solver.merton_u1*100:.1f}%, u2: {(solver.merton_u2-u2)/solver.merton_u2*100:.1f}%")

    # Economic interpretation
    print("\n--- Economic Interpretation ---")
    print("Asset 1 has higher return but larger negative jumps -> larger reduction")
    print("Asset 2 has lower return but smaller negative jumps -> smaller reduction")

    return {
        'merton_u1': solver.merton_u1,
        'merton_u2': solver.merton_u2,
        'learned_u1': u1,
        'learned_u2': u2,
        'time': elapsed,
    }


if __name__ == "__main__":
    torch.manual_seed(42)
    run_2d_experiment()
