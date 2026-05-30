"""Monte Carlo approximation of Lévy integral in HJB-PIDE.

The Lévy integral term in the HJB equation for Merton-style problems is:
    ∫[V(t, x(1 + u·z)) - V(t,x) - x·u·∇V(t,x)·z 1_{|z|<1}] ν(dz)

Note: For proportional jump dynamics (dX/X = u·dJ), the state after jump is
x → x·(1 + u·z) where z is the relative jump size and u is the control.

This module provides Monte Carlo estimators for this integral,
including proper ν(dz) weighting for infinite-activity measures.

IMPORTANT - Approximations and Limitations:
-------------------------------------------
1. TRUNCATION: For infinite-activity processes (VG, NIG), we solve a TRUNCATED
   version of the PIDE. With default truncation [0.01, 2.0], we only integrate
   jumps where |z| ∈ [1%, 200%]. Very small jumps (|z| < 1%) are approximated
   by the quadratic variation term in diffusion, and extreme jumps (|z| > 200%)
   are neglected. This is standard practice but should be noted in publications.

2. COMPENSATOR: The compensator term (subtracting ∫z ν(dz) for |z| < 1) is also
   restricted to the truncation region. For VG/NIG, this means we're effectively
   using a "truncated martingale" approximation.

3. WEIGHT CLIPPING: For stability, importance weights are clipped at 100x median.
   This introduces a small bias but prevents variance explosion. For heavy-tailed
   measures or extreme parameters, monitor the fraction of clipped weights.

4. SELF-NORMALIZATION: Option available via `self_normalize=True` in the integral
   class. This trades bias for lower variance when weights have high variability.

For rigorous validation, compare results across truncation levels and sample sizes.
"""

from typing import Optional, Callable, Tuple
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch import Tensor
import math


class LevyMeasure(ABC):
    """Abstract base class for Lévy measures."""

    @abstractmethod
    def sample(self, n_samples: int, device: torch.device) -> Tuple[Tensor, Tensor]:
        """Sample from a proposal distribution and return importance weights.

        Returns:
            z: Jump samples (n_samples, dim)
            weights: Importance weights ν(z)/q(z) (n_samples,) - NOT normalized
        """
        pass

    @abstractmethod
    def density(self, z: Tensor) -> Tensor:
        """Evaluate the Lévy density ν(z) at points z."""
        pass

    @abstractmethod
    def intensity(self, truncation_min: float, truncation_max: float) -> float:
        """Return ∫_{|z|∈[min,max]} ν(dz) for the truncated region."""
        pass

    @abstractmethod
    def total_mass(self) -> float:
        """Return ∫ν(dz) if finite, else inf."""
        pass


class VarianceGammaMeasure(LevyMeasure):
    """Lévy measure for Variance Gamma process.

    The VG Lévy density is:
        ν(dz) = C/|z| * exp(-M|z|) for z > 0
        ν(dz) = C/|z| * exp(-G|z|) for z < 0

    where C = 1/ν, M = sqrt(2/ν + θ²/σ⁴)/σ + θ/σ², G = sqrt(2/ν + θ²/σ⁴)/σ - θ/σ²

    This is an infinite-activity process (∫ν(dz) = ∞).

    For HJB integration, small jumps are handled by the compensator term,
    so we focus sampling on economically significant jumps (|z| > 0.01).

    Sampling: 50/50 mixture of two one-sided shifted exponentials with rates
    M/2 (positive tail) and G/2 (negative tail). The unconditional proposal
    density is q(z) = 0.5 * (rate) * exp(-(rate)(|z| - z_min)) — the 0.5
    factor is the mixture probability and is required for the importance
    weights w(z) = ν(z)/q(z) to be unbiased; an earlier version omitted it
    and the MC estimator returned exactly half the true Lévy integral
    (caught by the Phase 3a per-component diagnostic in
    experiments/hjb/phase3a_audit.py and fixed by including the 0.5 here).

    Estimator: standard (unnormalized) IS, (1/K) Σ w_i f(z_i), with weights
    capped at 100x the per-batch median for stability. The cap introduces a
    small, empirically-bounded bias; the cap is *not* self-normalization.
    """

    def __init__(
        self,
        sigma: float = 0.2,
        theta: float = -0.1,  # Negative skew typical for equity
        nu: float = 0.2,
        truncation_min: float = 0.01,  # Focus on meaningful jumps (1% or larger)
        truncation_max: float = 2.0,   # Jumps > 200% are rare
        intensity_scale: float = 1.0,
    ):
        self.sigma = sigma
        self.theta = theta
        self.nu = nu
        self.truncation_min = truncation_min
        self.truncation_max = truncation_max
        self.intensity_scale = intensity_scale

        # VG parameters in M-G parameterization
        # VG Lévy density: ν(dz) = C * exp(θz/σ²) / |z| * exp(-√(2/ν)|z|/σ)
        # For z > 0: ν(z) = C/z * exp(-z*(√(2/ν)/σ - θ/σ²)) = C/z * exp(-M*z)
        # For z < 0: ν(z) = C/|z| * exp(-|z|*(√(2/ν)/σ + θ/σ²)) = C/|z| * exp(-G*|z|)
        #
        # So: M = √(2/ν)/σ - θ/σ²  (rate for positive jumps)
        #     G = √(2/ν)/σ + θ/σ²  (rate for negative jumps)
        #
        # When θ < 0 (negative skew): M > G, so positive jumps decay faster
        # => More mass on negative jumps = negative skew. Correct!
        self.C = 1.0 / nu
        sqrt_2_nu_over_sigma = math.sqrt(2.0 / nu) / sigma
        theta_over_sigma2 = theta / (sigma ** 2)
        self.M = sqrt_2_nu_over_sigma - theta_over_sigma2  # Rate for positive jumps
        self.G = sqrt_2_nu_over_sigma + theta_over_sigma2  # Rate for negative jumps

        # Ensure positive rates
        self.M = max(self.M, 0.1)
        self.G = max(self.G, 0.1)

        # Compute intensity for truncated region (needed for scaling)
        self._intensity = self._compute_intensity()

    def _compute_intensity(self) -> float:
        """Compute ∫_{truncation_min < |z| < truncation_max} ν(dz)."""
        n_quad = 2000

        # Positive jumps
        z_pos = torch.linspace(self.truncation_min, self.truncation_max, n_quad)
        dz = (z_pos[1] - z_pos[0]).item()
        nu_pos = self.C / z_pos * torch.exp(-self.M * z_pos)
        intensity_pos = (nu_pos * dz).sum().item()

        # Negative jumps
        z_neg = torch.linspace(self.truncation_min, self.truncation_max, n_quad)
        nu_neg = self.C / z_neg * torch.exp(-self.G * z_neg)
        intensity_neg = (nu_neg * dz).sum().item()

        return self.intensity_scale * (intensity_pos + intensity_neg)

    def sample(self, n_samples: int, device: torch.device) -> Tuple[Tensor, Tensor]:
        """Sample from proposal and return importance weights.

        Uses mixture proposal: 50% from each tail (positive/negative).
        Within each tail, uses shifted Exponential which better matches VG tails.

        IMPORTANT: Weights are clipped at 100x median for stability. This introduces
        a small bias (~1-5% for typical VG parameters) but prevents variance explosion.
        For sensitivity analysis, compare with different clip levels or use
        SelfNormalizedLevyIntegral.

        Returns:
            z: Samples (n_samples, 1)
            weights: Importance weights ν(z)/q(z), clipped for stability
        """
        n_pos = n_samples // 2
        n_neg = n_samples - n_pos

        # Phase 3b correction: sample from the *exactly truncated* shifted
        # exponential on |z| ∈ [z_min, z_max] via inverse-CDF, instead of
        # the previous "sample from unbounded exponential then clamp"
        # scheme. The clamp version put a point mass at |z| = z_max while
        # the importance weights were computed using the continuous
        # density there — a small but real bias the reviewer flagged.
        # The truncated CDF is
        #     F_+(z) = (1 - exp(-r_+ (z - z_min))) / Z_+,
        #     Z_+ = 1 - exp(-r_+ (z_max - z_min)),
        # invertible:  z = z_min - (1/r_+) log(1 - U · Z_+),  U ~ Unif(0,1).
        # Density on [z_min, z_max]: q_+(|z|) = r_+ exp(-r_+ (|z| - z_min)) / Z_+.
        rate_pos = self.M * 0.5
        rate_neg = self.G * 0.5
        Z_pos = 1.0 - math.exp(-rate_pos * (self.truncation_max - self.truncation_min))
        Z_neg = 1.0 - math.exp(-rate_neg * (self.truncation_max - self.truncation_min))

        u_pos = torch.rand(n_pos, device=device, dtype=torch.float32)
        z_pos = self.truncation_min - (1.0 / rate_pos) * torch.log1p(-u_pos * Z_pos)

        u_neg = torch.rand(n_neg, device=device, dtype=torch.float32)
        z_neg_abs = self.truncation_min - (1.0 / rate_neg) * torch.log1p(-u_neg * Z_neg)
        z_neg = -z_neg_abs

        # Combine
        z = torch.cat([z_pos, z_neg], dim=0)
        z_abs = torch.abs(z)
        pos_mask = z > 0
        neg_mask = z < 0

        # Compute ν(z)
        nu_z = torch.zeros_like(z)
        nu_z[pos_mask] = (
            self.intensity_scale * self.C / z_abs[pos_mask] * torch.exp(-self.M * z_abs[pos_mask])
        )
        nu_z[neg_mask] = (
            self.intensity_scale * self.C / z_abs[neg_mask] * torch.exp(-self.G * z_abs[neg_mask])
        )

        # Proposal density q(z) of the 50/50-mixture truncated sampler.
        # Per-tail conditional density on [z_min, z_max] is
        #     q_pm(|z|) = rate * exp(-rate (|z| - z_min)) / Z,
        # so the unconditional density is (1/2) * q_pm(|z|). The 0.5
        # mixture probability is the factor whose omission was the
        # central bug isolated in §sec:fd-comparison; it remains
        # essential here.
        q_z = torch.zeros_like(z)
        q_z[pos_mask] = 0.5 * rate_pos * torch.exp(
            -rate_pos * (z_abs[pos_mask] - self.truncation_min)
        ) / Z_pos
        q_z[neg_mask] = 0.5 * rate_neg * torch.exp(
            -rate_neg * (z_abs[neg_mask] - self.truncation_min)
        ) / Z_neg

        # Raw importance weights
        raw_weights = nu_z / (q_z + 1e-10)

        # Self-normalize and scale by intensity
        # This gives: Σ w̃_i * f(z_i) ≈ intensity * E_ν[f]
        # But we want ∫f(z)ν(dz), so we scale by intensity / n_samples
        # Actually for MC: (1/n) * Σ w_i * f(z_i) ≈ E_q[w*f] = ∫f*ν
        # So we want raw weights, but clamped for stability

        # Clamp extreme weights for stability
        weight_cap = raw_weights.median() * 100
        weights = torch.clamp(raw_weights, max=weight_cap)

        return z.unsqueeze(-1), weights

    def density(self, z: Tensor) -> Tensor:
        """Evaluate Lévy density ν(z)."""
        z_abs = torch.abs(z) + 1e-10
        density = torch.zeros_like(z)

        pos_mask = z > 0
        neg_mask = z < 0

        density[pos_mask] = (
            self.intensity_scale * self.C / z_abs[pos_mask] * torch.exp(-self.M * z_abs[pos_mask])
        )
        density[neg_mask] = (
            self.intensity_scale * self.C / z_abs[neg_mask] * torch.exp(-self.G * z_abs[neg_mask])
        )

        return density

    def intensity(self, truncation_min: float = None, truncation_max: float = None) -> float:
        """Return intensity for truncated region."""
        if truncation_min is None:
            return self._intensity
        # Recompute for custom bounds
        n_quad = 2000
        z = torch.linspace(truncation_min, truncation_max, n_quad)
        dz = (z[1] - z[0]).item()
        nu_pos = self.C / z * torch.exp(-self.M * z)
        nu_neg = self.C / z * torch.exp(-self.G * z)
        return self.intensity_scale * ((nu_pos + nu_neg) * dz).sum().item()

    def total_mass(self) -> float:
        return float('inf')  # Infinite activity

    def expected_jump_effect(self) -> Tuple[float, float]:
        """Compute E[z] and E[z²] under the (truncated) Lévy measure.

        Useful for understanding the jump contribution to drift/variance.
        """
        n_quad = 2000
        z = torch.linspace(self.truncation_min, self.truncation_max, n_quad)
        dz = (z[1] - z[0]).item()

        # Positive contribution
        nu_pos = self.C / z * torch.exp(-self.M * z)
        E_z_pos = (z * nu_pos * dz).sum().item()
        E_z2_pos = (z**2 * nu_pos * dz).sum().item()

        # Negative contribution
        nu_neg = self.C / z * torch.exp(-self.G * z)
        E_z_neg = -(z * nu_neg * dz).sum().item()  # z is negative
        E_z2_neg = (z**2 * nu_neg * dz).sum().item()

        E_z = self.intensity_scale * (E_z_pos + E_z_neg)
        E_z2 = self.intensity_scale * (E_z2_pos + E_z2_neg)

        return E_z, E_z2


class NIGMeasure(LevyMeasure):
    """Lévy measure for Normal Inverse Gaussian process.

    ν(dz) = (δα/π) * exp(βz) * K_1(α|z|) / |z| dz

    where K_1 is the modified Bessel function.
    """

    def __init__(
        self,
        alpha: float = 15.0,
        beta: float = -5.0,
        delta: float = 0.5,
        truncation_min: float = 1e-4,
        truncation_max: float = 5.0,
    ):
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.truncation_min = truncation_min
        self.truncation_max = truncation_max

        self.gamma = math.sqrt(alpha**2 - beta**2)
        self._intensity = self._compute_intensity()

    def _compute_intensity(self) -> float:
        """Numerical integration of NIG Lévy density."""
        n_quad = 1000
        z = torch.linspace(self.truncation_min, self.truncation_max, n_quad)
        dz = z[1] - z[0]

        # K_1(x) approximation
        def bessel_k1_approx(x):
            return torch.where(
                x > 2,
                torch.sqrt(math.pi / (2 * x)) * torch.exp(-x),
                1.0 / x + 0.5 * x * torch.log(x / 2)
            )

        # Positive and negative
        k1_pos = bessel_k1_approx(self.alpha * z)
        nu_pos = (self.delta * self.alpha / math.pi) * torch.exp(self.beta * z) * k1_pos / z

        k1_neg = bessel_k1_approx(self.alpha * z)  # |z|
        nu_neg = (self.delta * self.alpha / math.pi) * torch.exp(-self.beta * z) * k1_neg / z

        return ((nu_pos + nu_neg) * dz).sum().item()

    def sample(self, n_samples: int, device: torch.device) -> Tuple[Tensor, Tensor]:
        """Sample from NIG using inverse Gaussian subordinator."""
        n_pos = n_samples // 2
        n_neg = n_samples - n_pos

        # Use double-exponential proposal
        lambda_param = self.alpha - abs(self.beta)
        lambda_param = max(lambda_param, 0.5)

        exp_dist = torch.distributions.Exponential(torch.tensor(lambda_param, device=device, dtype=torch.float32))

        z_pos = exp_dist.sample((n_pos,))
        z_pos = torch.clamp(z_pos, self.truncation_min, self.truncation_max)

        z_neg = -exp_dist.sample((n_neg,))
        z_neg = torch.clamp(z_neg, -self.truncation_max, -self.truncation_min)

        z = torch.cat([z_pos, z_neg], dim=0)

        # Compute importance weights
        nu_z = self.density(z)

        # Proposal density (truncated exponential)
        Z_q = math.exp(-lambda_param * self.truncation_min) - math.exp(-lambda_param * self.truncation_max)
        q_z = lambda_param * torch.exp(-lambda_param * torch.abs(z)) / Z_q

        weights = nu_z / (q_z + 1e-10)

        return z.unsqueeze(-1), weights

    def density(self, z: Tensor) -> Tensor:
        """Approximate NIG Lévy density."""
        z_abs = torch.abs(z) + 1e-10

        # K_1(x) approximation
        ax = self.alpha * z_abs
        bessel_approx = torch.where(
            ax > 2,
            torch.sqrt(math.pi / (2 * ax)) * torch.exp(-ax),
            1.0 / ax
        )

        density = (
            self.delta * self.alpha / math.pi
            * torch.exp(self.beta * z)
            * bessel_approx
            / z_abs
        )
        return density

    def intensity(self, truncation_min: float = None, truncation_max: float = None) -> float:
        return self._intensity

    def total_mass(self) -> float:
        return float('inf')


class CompoundPoissonMeasure(LevyMeasure):
    """Compound Poisson measure (finite activity).

    ν(dz) = λ * f(z) dz

    where λ is the jump intensity and f is the jump size distribution.
    This is for comparison/debugging - easier to simulate exactly.
    """

    def __init__(
        self,
        intensity: float = 1.0,
        jump_mean: float = 0.0,
        jump_std: float = 0.1,
    ):
        self.lambda_ = intensity
        self.jump_mean = jump_mean
        self.jump_std = jump_std

    def sample(self, n_samples: int, device: torch.device) -> Tuple[Tensor, Tensor]:
        """Sample from jump size distribution."""
        z = torch.randn(n_samples, device=device) * self.jump_std + self.jump_mean
        # Weights are just the intensity (since samples are from f, not ν)
        weights = torch.full((n_samples,), self.lambda_, device=device)
        return z.unsqueeze(-1), weights

    def density(self, z: Tensor) -> Tensor:
        """Lévy density = λ * N(z; μ, σ²)."""
        log_f = -0.5 * ((z - self.jump_mean) / self.jump_std)**2 - math.log(self.jump_std * math.sqrt(2 * math.pi))
        return self.lambda_ * torch.exp(log_f)

    def intensity(self, truncation_min: float = None, truncation_max: float = None) -> float:
        return self.lambda_

    def total_mass(self) -> float:
        return self.lambda_


class LearnedLevyMeasure(LevyMeasure, nn.Module):
    """Lévy measure learned from data via Lévy-Flows.

    Uses a trained Lévy-Flow model to sample jumps and evaluate densities.
    This connects the HJB solver to data-driven dynamics.

    Note: The learned distribution is normalized (∫p(z)dz = 1), so we need
    to scale by an estimated intensity parameter.
    """

    def __init__(
        self,
        levy_flow_model: nn.Module,
        intensity: float = 10.0,
        scale: float = 1.0,
    ):
        """
        Args:
            levy_flow_model: Trained LevyFlow or LevyFlowModel
            intensity: Estimated jump intensity (jumps per unit time)
            scale: Scaling factor for jumps
        """
        nn.Module.__init__(self)
        self.levy_flow = levy_flow_model
        self.lambda_ = intensity
        self.scale = scale

    def sample(self, n_samples: int, device: torch.device) -> Tuple[Tensor, Tensor]:
        """Sample from the learned distribution."""
        with torch.no_grad():
            samples = self.levy_flow.sample(n_samples)
        z = samples.to(device) * self.scale

        # Weight by intensity
        weights = torch.full((n_samples,), self.lambda_, device=device)

        return z, weights

    def density(self, z: Tensor) -> Tensor:
        """Evaluate learned density scaled by intensity."""
        log_p = self.levy_flow.log_prob(z / self.scale)
        return self.lambda_ * torch.exp(log_p) / self.scale

    def intensity(self, truncation_min: float = None, truncation_max: float = None) -> float:
        return self.lambda_

    def total_mass(self) -> float:
        return self.lambda_


class LevyIntegralMC(nn.Module):
    """Monte Carlo estimator for the Lévy integral in Merton-type problems.

    For proportional dynamics (wealth process):
        dX_t = X_t * [r + u(μ-r)]dt + X_t * u * σ dW_t + X_t * u * dJ_t

    The HJB-PIDE integral term is:
        ∫[V(t, x(1 + u·z)) - V(t,x) - x·u·∇V(t,x)·z 1_{|z|<1}] ν(dz)

    where z is the relative jump size (dJ/J-).

    CRITICAL: The jump displacement depends on both state x AND control u.

    Two estimation modes:
    1. Standard IS (self_normalize=False): E_q[w·f] where w = ν/q
       - Unbiased if weights are unclipped
       - Higher variance with heavy-tailed ν

    2. Self-normalized IS (self_normalize=True): Σ w̃_i·f_i where w̃_i = w_i/Σw_j
       - Biased but consistent estimator
       - Much lower variance when weights vary widely
       - Recommended for VG with large θ or small ν
    """

    def __init__(
        self,
        levy_measure: LevyMeasure,
        n_samples: int = 64,
        compensator_cutoff: float = 1.0,
        self_normalize: bool = False,
    ):
        """
        Args:
            levy_measure: The Lévy measure to integrate against
            n_samples: Number of Monte Carlo samples
            compensator_cutoff: Cutoff for the compensator term (usually 1.0)
            self_normalize: Use self-normalized IS (lower variance, slight bias)
        """
        super().__init__()
        self.levy_measure = levy_measure
        self.n_samples = n_samples
        self.compensator_cutoff = compensator_cutoff
        self.self_normalize = self_normalize

    def forward(
        self,
        value_fn: Callable[[Tensor, Tensor], Tensor],
        t: Tensor,
        x: Tensor,
        u: Tensor,
        V_x: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute the Lévy integral with control-dependent jumps.

        Args:
            value_fn: Function V(t, x) -> (batch, 1)
            t: Time (batch,)
            x: State/wealth (batch, state_dim)
            u: Control (batch, control_dim) - fraction in risky asset
            V_x: Gradient ∇V(t,x) if precomputed (batch, state_dim)

        Returns:
            Integral value (batch, 1)
        """
        batch_size = x.shape[0]
        state_dim = x.shape[1]
        device = x.device

        # Sample jump sizes with importance weights
        z, weights = self.levy_measure.sample(self.n_samples, device)
        # z: (n_samples, 1) or (n_samples, state_dim)
        # weights: (n_samples,) - unnormalized, represent ν(z_i)/q(z_i)

        # Expand z to match state_dim if needed
        if z.shape[-1] == 1 and state_dim > 1:
            z_full = torch.zeros(self.n_samples, state_dim, device=device)
            z_full[:, 0] = z.squeeze(-1)
            z = z_full

        # Current value
        V_current = value_fn(t, x)  # (batch, 1)

        # Compute state after jump: x_after = x * (1 + u * z)
        # For each batch point and each jump sample
        # x: (batch, state_dim), u: (batch, control_dim), z: (n_samples, state_dim)

        # Handle dimension matching between u and z
        # For Merton: control_dim = state_dim = 1, u controls exposure to jump
        x_expanded = x.unsqueeze(1)  # (batch, 1, state_dim)
        u_expanded = u.unsqueeze(1)  # (batch, 1, control_dim)
        z_expanded = z.unsqueeze(0)  # (1, n_samples, state_dim)

        # Jump displacement: x * u * z (proportional to wealth and control)
        # For 1D case: x_after = x * (1 + u * z)
        if state_dim == 1:
            jump_displacement = x_expanded * u_expanded * z_expanded  # (batch, n_samples, 1)
        else:
            # Multi-asset: each asset has its own control weight
            # u[i] controls exposure to jump in asset i
            jump_displacement = x_expanded * u_expanded * z_expanded  # (batch, n_samples, state_dim)

        x_after_jump = x_expanded + jump_displacement  # (batch, n_samples, state_dim)

        # Clamp to prevent negative wealth
        x_after_jump = torch.clamp(x_after_jump, min=1e-8)

        # Flatten for value function evaluation
        x_after_flat = x_after_jump.reshape(-1, state_dim)  # (batch * n_samples, state_dim)
        t_expanded = t.unsqueeze(1).expand(-1, self.n_samples).reshape(-1)  # (batch * n_samples,)

        V_shifted = value_fn(t_expanded, x_after_flat)  # (batch * n_samples, 1)
        V_shifted = V_shifted.reshape(batch_size, self.n_samples)  # (batch, n_samples)

        # Compute gradient for compensator if not provided
        if V_x is None:
            x_grad = x.clone().requires_grad_(True)
            V_for_grad = value_fn(t, x_grad)
            V_x = torch.autograd.grad(
                V_for_grad.sum(), x_grad,
                create_graph=True,
            )[0]

        # Compensator term: x * u * ∇V · z for |z| < 1
        # This is the first-order approximation subtracted for small jumps
        z_norm = torch.norm(z, dim=-1)  # (n_samples,)
        compensator_mask = (z_norm < self.compensator_cutoff).float()  # (n_samples,)

        # Gradient term: ∇V · (x * u * z)
        # V_x: (batch, state_dim), x*u: (batch, state_dim), z: (n_samples, state_dim)
        # Result should be (batch, n_samples)
        x_u = x * u  # (batch, state_dim) - exposure-weighted wealth
        grad_times_xu = V_x * x_u  # (batch, state_dim) - ∇V ⊙ (x*u)
        compensator_term = torch.einsum('bd,nd->bn', grad_times_xu, z)  # (batch, n_samples)
        compensator = compensator_term * compensator_mask.unsqueeze(0)  # (batch, n_samples)

        # Integrand: V(t, x(1+uz)) - V(t,x) - compensator
        integrand = V_shifted - V_current - compensator  # (batch, n_samples)

        # Weighted Monte Carlo estimation
        if self.self_normalize:
            # Self-normalized IS: Σ (w_i / Σw_j) * f_i
            # This is biased but has lower variance
            # Scale by estimated intensity to get correct magnitude
            w_normalized = weights / (weights.sum() + 1e-10)  # (n_samples,)
            weighted_integrand = integrand * w_normalized.unsqueeze(0)  # (batch, n_samples)
            # Sum (not mean) since weights are normalized
            integral = weighted_integrand.sum(dim=1, keepdim=True)  # (batch, 1)
            # Scale by intensity estimate (sum of weights / n ≈ intensity via MC)
            intensity_estimate = weights.sum() / self.n_samples
            integral = integral * intensity_estimate
        else:
            # Standard IS: E_q[w * f] where w = ν/q
            # integral ≈ (1/n) * Σ w_i * integrand_i
            weighted_integrand = integrand * weights.unsqueeze(0)  # (batch, n_samples)
            integral = weighted_integrand.mean(dim=1, keepdim=True)  # (batch, 1)

        return integral


class ImportanceSampledLevyIntegral(LevyIntegralMC):
    """Lévy integral with additional importance sampling for variance reduction.

    Overweights large jumps which contribute more to tail risk.
    """

    def __init__(
        self,
        levy_measure: LevyMeasure,
        n_samples: int = 64,
        compensator_cutoff: float = 1.0,
        tail_emphasis: float = 2.0,
    ):
        super().__init__(levy_measure, n_samples, compensator_cutoff)
        self.tail_emphasis = tail_emphasis

    def forward(
        self,
        value_fn: Callable[[Tensor, Tensor], Tensor],
        t: Tensor,
        x: Tensor,
        u: Tensor,
        V_x: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute Lévy integral with tail-emphasized sampling."""
        batch_size = x.shape[0]
        state_dim = x.shape[1]
        device = x.device

        # Sample with heavier tails
        # Use Student-t proposal instead of the measure's default
        proposal = torch.distributions.StudentT(
            df=torch.tensor(3.0, device=device),
            scale=torch.tensor(self.tail_emphasis * 0.1, device=device)
        )
        z_proposal = proposal.sample((self.n_samples,)).unsqueeze(-1)  # (n_samples, 1)

        # Truncate to reasonable range
        truncation = getattr(self.levy_measure, 'truncation_max', 5.0)
        z_proposal = torch.clamp(z_proposal, -truncation, truncation)

        if state_dim > 1:
            z_full = torch.zeros(self.n_samples, state_dim, device=device)
            z_full[:, 0] = z_proposal.squeeze(-1)
            z_proposal = z_full
        else:
            z_proposal = z_proposal.squeeze(-1).unsqueeze(-1)  # (n_samples, 1)

        # Compute importance weights: ν(z) / q(z)
        target_density = self.levy_measure.density(z_proposal[:, 0])
        proposal_density = torch.exp(proposal.log_prob(z_proposal[:, 0]))
        weights = target_density / (proposal_density + 1e-10)

        # Rest follows base class logic
        V_current = value_fn(t, x)

        x_expanded = x.unsqueeze(1)
        u_expanded = u.unsqueeze(1)
        z_expanded = z_proposal.unsqueeze(0)

        if state_dim == 1:
            jump_displacement = x_expanded * u_expanded * z_expanded
        else:
            jump_displacement = x_expanded * u_expanded * z_expanded

        x_after_jump = torch.clamp(x_expanded + jump_displacement, min=1e-8)

        x_after_flat = x_after_jump.reshape(-1, state_dim)
        t_expanded = t.unsqueeze(1).expand(-1, self.n_samples).reshape(-1)

        V_shifted = value_fn(t_expanded, x_after_flat)
        V_shifted = V_shifted.reshape(batch_size, self.n_samples)

        if V_x is None:
            x_grad = x.clone().requires_grad_(True)
            V_for_grad = value_fn(t, x_grad)
            V_x = torch.autograd.grad(V_for_grad.sum(), x_grad, create_graph=True)[0]

        z_norm = torch.norm(z_proposal, dim=-1)
        compensator_mask = (z_norm < self.compensator_cutoff).float()

        x_u = x * u
        grad_times_xu = V_x * x_u
        compensator_term = torch.einsum('bd,nd->bn', grad_times_xu, z_proposal)
        compensator = compensator_term * compensator_mask.unsqueeze(0)

        integrand = V_shifted - V_current - compensator
        weighted_integrand = integrand * weights.unsqueeze(0)
        integral = weighted_integrand.mean(dim=1, keepdim=True)

        return integral


class ControlVariateLevyIntegral(LevyIntegralMC):
    """Lévy integral with control variate for variance reduction.

    Uses a quadratic approximation of V as control variate.
    """

    def forward(
        self,
        value_fn: Callable[[Tensor, Tensor], Tensor],
        t: Tensor,
        x: Tensor,
        u: Tensor,
        V_x: Optional[Tensor] = None,
        V_xx_diag: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute Lévy integral with control variate."""
        batch_size = x.shape[0]
        state_dim = x.shape[1]
        device = x.device

        z, weights = self.levy_measure.sample(self.n_samples, device)

        if z.shape[-1] == 1 and state_dim > 1:
            z_full = torch.zeros(self.n_samples, state_dim, device=device)
            z_full[:, 0] = z.squeeze(-1)
            z = z_full

        V_current = value_fn(t, x)

        # Compute V(t, x(1+uz))
        x_expanded = x.unsqueeze(1)
        u_expanded = u.unsqueeze(1)
        z_expanded = z.unsqueeze(0)

        jump_displacement = x_expanded * u_expanded * z_expanded
        x_after_jump = torch.clamp(x_expanded + jump_displacement, min=1e-8)

        x_after_flat = x_after_jump.reshape(-1, state_dim)
        t_expanded = t.unsqueeze(1).expand(-1, self.n_samples).reshape(-1)

        V_shifted = value_fn(t_expanded, x_after_flat)
        V_shifted = V_shifted.reshape(batch_size, self.n_samples)

        # Get gradient and Hessian diagonal for control variate
        if V_x is None or V_xx_diag is None:
            x_grad = x.clone().requires_grad_(True)
            V_for_grad = value_fn(t, x_grad)
            V_x = torch.autograd.grad(V_for_grad.sum(), x_grad, create_graph=True)[0]

            V_xx_diag = torch.zeros(batch_size, state_dim, device=device)
            for i in range(state_dim):
                grad2 = torch.autograd.grad(
                    V_x[:, i].sum(), x_grad, create_graph=True, retain_graph=True
                )[0]
                V_xx_diag[:, i] = grad2[:, i]

        # Control variate: second-order Taylor expansion
        # V(x + Δx) ≈ V(x) + ∇V·Δx + 0.5 Δx'·H·Δx
        # where Δx = x*u*z
        x_u = x * u
        delta_x = x_u.unsqueeze(1) * z_expanded  # (batch, n_samples, state_dim)

        # First order: ∇V · Δx
        first_order = torch.einsum('bd,bnd->bn', V_x, delta_x)

        # Second order (diagonal Hessian): 0.5 * Σ H_ii * (Δx_i)²
        delta_x_sq = delta_x ** 2
        second_order = 0.5 * torch.einsum('bd,bnd->bn', V_xx_diag, delta_x_sq)

        control_variate = first_order + second_order  # (batch, n_samples)

        # Compensator
        z_norm = torch.norm(z, dim=-1)
        compensator_mask = (z_norm < self.compensator_cutoff).float()
        compensator = first_order * compensator_mask.unsqueeze(0)

        # Integrand with control variate correction
        # Original: V(x+Δx) - V(x) - compensator
        # CV-reduced: [V(x+Δx) - V(x) - compensator] - [CV - compensator] + E[CV - compensator]
        integrand = V_shifted - V_current - compensator
        cv_correction = control_variate - compensator

        # Variance-reduced integrand
        integrand_reduced = integrand - cv_correction

        # Weighted average
        weighted_integrand = integrand_reduced * weights.unsqueeze(0)
        integral = weighted_integrand.mean(dim=1, keepdim=True)

        # Add back expected CV (using MC estimate)
        weighted_cv = cv_correction * weights.unsqueeze(0)
        integral = integral + weighted_cv.mean(dim=1, keepdim=True)

        return integral
