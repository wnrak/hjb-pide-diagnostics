"""Deep Galerkin Method (DGM) solver for Lévy HJB-PIDE.

Main solver that combines value/policy networks with Monte Carlo Lévy integration
to solve the Hamilton-Jacobi-Bellman PIDE arising in optimal control under
Lévy dynamics.

The HJB-PIDE for Merton-style portfolio problems:
    ∂_t V + sup_u { L^u_diff V + f(x,u) + ∫[V(t,x(1+uz)) - V(t,x) - xu∇V·z 1_{|z|<1}] ν(dz) } = 0
    V(T, x) = g(x)

where L^u_diff V = b(x,u)·∇V + 0.5 Tr[σσ'∇²V]

Key implementation details:
1. Jump dynamics are proportional: x → x(1 + u·z) where u is control
2. Lévy integral uses importance sampling with proper ν(dz) weighting
3. Optimality via Hamiltonian maximization (not just residual minimization)

IMPORTANT - Approximations:
---------------------------
For infinite-activity Lévy processes (VG, NIG), this solver uses a TRUNCATED
version of the PIDE where only jumps with |z| in [truncation_min, truncation_max]
are integrated. Default: [0.01, 2.0] = [1%, 200%].

This is standard practice (see e.g. Cont & Voltchkova 2005) but should be noted:
- Very small jumps (|z| < 1%) are approximated by the diffusion term
- Extreme jumps (|z| > 200%) are neglected
- The compensator only covers |z| < 1 within the truncation region

For sensitivity analysis, compare results across different truncation levels.
"""

from typing import Optional, Dict, List, Callable, Tuple
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from levy_flows.hjb.networks import ValueNetwork, PolicyNetwork, ActorCriticNetwork
from levy_flows.hjb.levy_integral import LevyIntegralMC, LevyMeasure
from levy_flows.hjb.problems import ControlProblem


class LevyHJBSolver(nn.Module):
    """Neural solver for Lévy HJB-PIDE.

    Uses Deep Galerkin Method with actor-critic style training and
    Hamiltonian optimality enforcement.
    """

    def __init__(
        self,
        problem: ControlProblem,
        levy_measure: LevyMeasure,
        hidden_dim: int = 128,
        n_layers: int = 4,
        n_levy_samples: int = 64,
        use_shared_network: bool = True,
        device: str = "cpu",
    ):
        """
        Args:
            problem: The control problem to solve
            levy_measure: Lévy measure for the jump integral
            hidden_dim: Hidden dimension for networks
            n_layers: Number of layers
            n_levy_samples: MC samples for Lévy integral
            use_shared_network: Whether to use shared value/policy backbone
            device: Device to use
        """
        super().__init__()

        self.problem = problem
        self.levy_measure = levy_measure
        self.device = torch.device(device)

        # Initialize networks
        if use_shared_network:
            self.network = ActorCriticNetwork(
                state_dim=problem.state_dim,
                control_dim=problem.control_dim,
                hidden_dim=hidden_dim,
                n_shared_layers=n_layers // 2,
                n_value_layers=n_layers // 2,
                n_policy_layers=n_layers // 2,
                control_bounds=problem.control_bounds(),
                simplex_constraint=problem.simplex_constraint(),
            ).to(self.device)
            self.value_net = None
            self.policy_net = None
        else:
            self.network = None
            self.value_net = ValueNetwork(
                state_dim=problem.state_dim,
                hidden_dim=hidden_dim,
                n_layers=n_layers,
            ).to(self.device)
            self.policy_net = PolicyNetwork(
                state_dim=problem.state_dim,
                control_dim=problem.control_dim,
                hidden_dim=hidden_dim,
                n_layers=n_layers,
                control_bounds=problem.control_bounds(),
                simplex_constraint=problem.simplex_constraint(),
            ).to(self.device)

        # Lévy integral estimator
        self.levy_integral = LevyIntegralMC(
            levy_measure=levy_measure,
            n_samples=n_levy_samples,
        ).to(self.device)

    def value(self, t: Tensor, x: Tensor) -> Tensor:
        """Evaluate value function V(t, x)."""
        if self.network is not None:
            return self.network.forward_value(t, x)
        else:
            return self.value_net(t, x)

    def policy(self, t: Tensor, x: Tensor) -> Tensor:
        """Evaluate policy u(t, x)."""
        if self.network is not None:
            return self.network.forward_policy(t, x)
        else:
            return self.policy_net(t, x)

    def hamiltonian(
        self,
        t: Tensor,
        x: Tensor,
        u: Tensor,
        V_x: Tensor,
        V_xx_diag: Tensor,
        include_levy: bool = True,
    ) -> Tensor:
        """Compute the Hamiltonian H(t, x, u, V, ∇V, ∇²V).

        H = b(x,u)·∇V + 0.5 Tr[σσ'∇²V] + f(x,u) + LévyIntegral(u)

        The HJB equation is: ∂_t V + sup_u H = 0

        Args:
            t: Time (batch,)
            x: State (batch, state_dim)
            u: Control (batch, control_dim)
            V_x: Gradient ∇V (batch, state_dim)
            V_xx_diag: Hessian diagonal (batch, state_dim)
            include_levy: Whether to include Lévy term

        Returns:
            Hamiltonian value (batch, 1)
        """
        # Problem coefficients
        b = self.problem.drift(t, x, u)      # (batch, state_dim)
        sigma = self.problem.diffusion(t, x, u)  # (batch, state_dim)
        f = self.problem.running_cost(t, x, u)   # (batch, 1)

        # Diffusion operator: b·∇V + 0.5 Tr[σσ'∇²V]
        drift_term = (b * V_x).sum(dim=-1, keepdim=True)
        diffusion_term = 0.5 * (sigma ** 2 * V_xx_diag).sum(dim=-1, keepdim=True)

        H = drift_term + diffusion_term + f

        # Lévy integral term (control-dependent!)
        if include_levy:
            levy_term = self.levy_integral(
                lambda t_, x_: self.value(t_, x_),
                t, x, u, V_x,  # Pass control u to Lévy integral
            )
            H = H + levy_term

        return H

    def hjb_residual(
        self,
        t: Tensor,
        x: Tensor,
        include_levy: bool = True,
    ) -> Tensor:
        """Compute HJB-PIDE residual.

        Residual = ∂_t V + H(t, x, u(t,x), V, ∇V, ∇²V)

        where u(t,x) is the current policy.

        Args:
            t: Time points (batch,)
            x: State points (batch, state_dim)
            include_levy: Whether to include Lévy integral term

        Returns:
            Residual (batch, 1)
        """
        batch_size = x.shape[0]

        # Get control from policy network
        u = self.policy(t, x)

        # Get value and derivatives
        if self.network is not None:
            t_grad = t.clone().requires_grad_(True)
            x_grad = x.clone().requires_grad_(True)
            V = self.network.forward_value(t_grad, x_grad)

            # Compute gradients
            grad_outputs = torch.ones_like(V)
            grads = torch.autograd.grad(
                V, [t_grad, x_grad],
                grad_outputs=grad_outputs,
                create_graph=True,
            )
            V_t = grads[0].unsqueeze(-1)
            V_x = grads[1]
        else:
            t_grad = t.clone().requires_grad_(True)
            x_grad = x.clone().requires_grad_(True)
            V, V_t, V_x = self.value_net.forward_with_gradients(t_grad, x_grad)

        # Compute Hessian diagonal for diffusion term
        V_xx_diag = torch.zeros_like(V_x)
        for i in range(self.problem.state_dim):
            grad2 = torch.autograd.grad(
                V_x[:, i].sum(), x_grad,
                create_graph=True,
                retain_graph=True,
            )[0]
            V_xx_diag[:, i] = grad2[:, i]

        # Compute Hamiltonian
        H = self.hamiltonian(t, x, u, V_x, V_xx_diag, include_levy=include_levy)

        residual = V_t + H

        return residual

    def optimality_loss(
        self,
        t: Tensor,
        x: Tensor,
        V_x: Tensor,
        V_xx_diag: Tensor,
        include_levy: bool = True,
        n_control_samples: int = 16,
    ) -> Tensor:
        """Compute loss encouraging policy to maximize Hamiltonian.

        For Merton problem, the optimal control satisfies:
            u* = argmax_u H(t, x, u, V, ∇V)

        We use a soft penalty: encourage H(u_policy) ≥ H(u_perturbed).

        Args:
            t: Time (batch,)
            x: State (batch, state_dim)
            V_x: Gradient (batch, state_dim)
            V_xx_diag: Hessian diagonal (batch, state_dim)
            include_levy: Whether to include Lévy term
            n_control_samples: Number of control perturbations to try

        Returns:
            Optimality loss (scalar)
        """
        batch_size = x.shape[0]

        # Current policy
        u_current = self.policy(t, x)
        H_current = self.hamiltonian(t, x, u_current, V_x, V_xx_diag, include_levy)

        # Sample alternative controls and compute their Hamiltonians
        bounds = self.problem.control_bounds()
        if bounds is not None:
            lo, hi = bounds
        else:
            lo, hi = 0.0, 1.0

        total_violation = torch.zeros(1, device=self.device)

        for _ in range(n_control_samples):
            # Random perturbation
            u_perturbed = torch.rand_like(u_current) * (hi - lo) + lo

            H_perturbed = self.hamiltonian(
                t, x, u_perturbed, V_x, V_xx_diag, include_levy
            )

            # Penalize if perturbed control gives higher Hamiltonian
            # (policy should maximize, so current should be best)
            violation = torch.relu(H_perturbed - H_current)
            total_violation = total_violation + violation.mean()

        return total_violation / n_control_samples

    def policy_argmax_loss(
        self,
        t: Tensor,
        x: Tensor,
        V_x: Tensor,
        V_xx_diag: Tensor,
        include_levy: bool = True,
        n_candidates: int = 21,
    ) -> Tensor:
        """Train policy to match argmax_u H over the admissible set U.

        For each (t_i, x_i), we evaluate H(t_i, x_i, u_k, V, ∇V, ∇²V) at K
        candidate u-values laid out as a deterministic linspace over U, take the
        argmax to obtain u_target_i (detached), and minimize MSE(u_φ, u_target).

        This is the constrained-optimization-correct counterpart of the
        first-order-condition loss |∂_u H|² used previously: the FOC vanishes
        only at *interior* optima, so it gives the wrong target whenever the
        true argmax sits on the boundary of U. The argmax-target loss has no
        such failure mode, and it returns zero only when u_φ matches the global
        maximizer over the discretized U.

        For multi-D control or simplex-constrained problems we fall back to
        i.i.d. random samples drawn from the admissible set (n_candidates
        samples per collocation point).
        """
        if self.problem.simplex_constraint() or self.problem.control_dim > 1:
            return self._policy_argmax_loss_multid(
                t, x, V_x, V_xx_diag, include_levy, n_candidates
            )

        bounds = self.problem.control_bounds()
        if bounds is None:
            lo, hi = 0.0, 1.0
        else:
            lo, hi = bounds
        batch = x.shape[0]
        K = n_candidates

        u_grid = torch.linspace(
            lo, hi, K, device=self.device, dtype=x.dtype
        ).view(K, 1)  # (K, 1)

        # Tile (batch, K) for vectorized H evaluation
        t_tile = t.unsqueeze(1).expand(batch, K).reshape(-1)
        x_tile = x.unsqueeze(1).expand(batch, K, -1).reshape(batch * K, -1)
        V_x_tile = V_x.unsqueeze(1).expand(batch, K, -1).reshape(batch * K, -1)
        V_xx_tile = V_xx_diag.unsqueeze(1).expand(batch, K, -1).reshape(batch * K, -1)
        u_tile = u_grid.unsqueeze(0).expand(batch, K, -1).reshape(batch * K, -1)

        with torch.no_grad():
            H_all = self.hamiltonian(
                t_tile, x_tile, u_tile, V_x_tile, V_xx_tile, include_levy=include_levy
            )
            H_all = H_all.view(batch, K)
            argmax_idx = H_all.argmax(dim=1)
            u_target = u_grid[argmax_idx].detach()

        u_pred = self.policy(t, x)
        return ((u_pred - u_target) ** 2).mean()

    def _policy_argmax_loss_multid(
        self,
        t: Tensor,
        x: Tensor,
        V_x: Tensor,
        V_xx_diag: Tensor,
        include_levy: bool,
        n_candidates: int,
    ) -> Tensor:
        """Multi-D control fallback: i.i.d. samples from U, MSE to argmax target.

        For simplex-constrained problems we sample from the (closed) probability
        simplex via Dirichlet(1, ..., 1); for box-constrained problems we sample
        uniformly in the box. Variance is higher than the 1D grid version, but
        the loss is still consistent: as n_candidates grows, the empirical
        argmax converges to the true argmax over U.
        """
        batch = x.shape[0]
        K = n_candidates
        cdim = self.problem.control_dim

        if self.problem.simplex_constraint():
            dirichlet = torch.distributions.Dirichlet(
                torch.ones(cdim, device=self.device, dtype=x.dtype)
            )
            u_cand = dirichlet.sample((batch, K))  # (batch, K, cdim)
        else:
            bounds = self.problem.control_bounds()
            if bounds is None:
                lo, hi = 0.0, 1.0
            else:
                lo, hi = bounds
            u_cand = lo + (hi - lo) * torch.rand(
                batch, K, cdim, device=self.device, dtype=x.dtype
            )

        t_tile = t.unsqueeze(1).expand(batch, K).reshape(-1)
        x_tile = x.unsqueeze(1).expand(batch, K, -1).reshape(batch * K, -1)
        V_x_tile = V_x.unsqueeze(1).expand(batch, K, -1).reshape(batch * K, -1)
        V_xx_tile = V_xx_diag.unsqueeze(1).expand(batch, K, -1).reshape(batch * K, -1)
        u_tile = u_cand.reshape(batch * K, cdim)

        with torch.no_grad():
            H_all = self.hamiltonian(
                t_tile, x_tile, u_tile, V_x_tile, V_xx_tile, include_levy=include_levy
            )
            H_all = H_all.view(batch, K)
            argmax_idx = H_all.argmax(dim=1)
            # gather argmax-row from u_cand
            u_target = torch.gather(
                u_cand, 1, argmax_idx.view(batch, 1, 1).expand(batch, 1, cdim)
            ).squeeze(1).detach()

        u_pred = self.policy(t, x)
        return ((u_pred - u_target) ** 2).mean()

    def terminal_loss(self, x: Tensor) -> Tensor:
        """Compute terminal condition loss.

        Loss = E[(V(T, x) - g(x))²]
        """
        t_terminal = torch.full((x.shape[0],), self.problem.T, device=self.device)
        V_T = self.value(t_terminal, x)
        g = self.problem.terminal_cost(x)

        return ((V_T.squeeze(-1) - g) ** 2).mean()

    def sample_domain(
        self,
        n_samples: int,
        state_sampler: Optional[Callable] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Sample points from the time-state domain.

        Args:
            n_samples: Number of points to sample
            state_sampler: Optional custom state sampler

        Returns:
            t: Time samples (n_samples,)
            x: State samples (n_samples, state_dim)
        """
        # Sample time uniformly
        t = torch.rand(n_samples, device=self.device) * self.problem.T

        # Sample state
        if state_sampler is not None:
            x = state_sampler(n_samples)
        else:
            # Default: log-normal for wealth (positive, centered around 1)
            log_x = torch.randn(n_samples, self.problem.state_dim, device=self.device) * 0.3
            x = torch.exp(log_x)  # Log-normal: mostly in [0.5, 2.0]

        return t, x

    def fit(
        self,
        n_epochs: int = 1000,
        batch_size: int = 256,
        lr: float = 1e-3,
        lambda_terminal: float = 10.0,
        lambda_optimality: float = 1.0,
        warmup_epochs: int = 100,
        use_foc: bool = True,
        foc_frequency: int = 1,
        n_argmax_candidates: int = 21,
        state_sampler: Optional[Callable] = None,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """Train the HJB solver.

        Uses curriculum learning: first train without Lévy term (warmup),
        then add the Lévy integral. The policy is trained against an explicit
        argmax target over the admissible control set (see policy_argmax_loss);
        the legacy ``use_foc`` flag is retained as the on/off switch for that
        loss (kept for back-compat with existing call sites).

        Args:
            n_epochs: Total training epochs
            batch_size: Batch size
            lr: Learning rate
            lambda_terminal: Terminal loss weight
            lambda_optimality: Argmax-target loss weight
            warmup_epochs: Epochs without Lévy term (for stability)
            use_foc: If True, apply the policy argmax-target loss (default).
                Name kept for back-compat; the underlying loss is the
                constrained argmax loss, not the unconstrained FOC residual.
            foc_frequency: Compute the argmax loss every N epochs. The argmax
                evaluation runs the Hamiltonian K=n_argmax_candidates extra
                times per collocation point; set foc_frequency>1 to amortize.
            n_argmax_candidates: Grid resolution for the policy argmax target.
            state_sampler: Optional custom state sampler
            verbose: Whether to print progress

        Returns:
            Training history
        """
        # Optimizer
        if self.network is not None:
            optimizer = optim.Adam(self.network.parameters(), lr=lr)
        else:
            optimizer = optim.Adam(
                list(self.value_net.parameters()) +
                list(self.policy_net.parameters()),
                lr=lr,
            )

        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

        history = {
            "total": [],
            "interior": [],
            "terminal": [],
            "optimality": [],
        }

        # Cache for FOC loss when foc_frequency > 1
        last_optimality_loss = torch.tensor(0.0, device=self.device)

        for epoch in range(n_epochs):
            # Sample training points
            t, x = self.sample_domain(batch_size, state_sampler)
            _, x_terminal = self.sample_domain(batch_size // 4, state_sampler)

            # Curriculum: no Lévy term during warmup
            include_levy = epoch >= warmup_epochs

            # Forward pass
            optimizer.zero_grad()

            # HJB residual loss
            residual = self.hjb_residual(t, x, include_levy=include_levy)
            interior_loss = (residual ** 2).mean()

            # Terminal condition loss
            terminal_loss = self.terminal_loss(x_terminal)

            # Policy argmax-target loss (constrained-correct replacement for
            # the unconstrained ∂_uH=0 FOC). Expensive in the Lévy-on phase
            # because the Hamiltonian is evaluated at K candidate u-values per
            # collocation point — amortize with foc_frequency.
            compute_argmax = (
                use_foc and
                epoch >= warmup_epochs // 2 and
                (epoch % foc_frequency == 0 or epoch == n_epochs - 1)
            )
            if compute_argmax:
                x_grad = x.clone().requires_grad_(True)
                V_for_grad = self.value(t, x_grad)
                V_x = torch.autograd.grad(
                    V_for_grad.sum(), x_grad, create_graph=True
                )[0]
                # Hessian diagonal (needed by H for the diffusion term)
                V_xx_diag = torch.zeros_like(V_x)
                for i in range(self.problem.state_dim):
                    grad2 = torch.autograd.grad(
                        V_x[:, i].sum(), x_grad,
                        create_graph=True, retain_graph=True,
                    )[0]
                    V_xx_diag[:, i] = grad2[:, i]
                optimality_loss = self.policy_argmax_loss(
                    t, x, V_x, V_xx_diag,
                    include_levy=include_levy,
                    n_candidates=n_argmax_candidates,
                )
                last_optimality_loss = optimality_loss.detach()
            elif use_foc and epoch >= warmup_epochs // 2:
                optimality_loss = last_optimality_loss
            else:
                optimality_loss = torch.tensor(0.0, device=self.device)

            # Total loss
            total_loss = (
                interior_loss
                + lambda_terminal * terminal_loss
                + lambda_optimality * optimality_loss
            )

            # Backward pass
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.network.parameters() if self.network else
                list(self.value_net.parameters()) + list(self.policy_net.parameters()),
                1.0,
            )
            optimizer.step()
            scheduler.step()

            # Log
            history["total"].append(total_loss.item())
            history["interior"].append(interior_loss.item())
            history["terminal"].append(terminal_loss.item())
            history["optimality"].append(optimality_loss.item() if torch.is_tensor(optimality_loss) else optimality_loss)

            if verbose and (epoch + 1) % 100 == 0:
                levy_str = "ON" if include_levy else "OFF"
                opt_str = f"{optimality_loss.item():.6f}" if torch.is_tensor(optimality_loss) else "N/A"
                print(
                    f"Epoch {epoch + 1}/{n_epochs} | "
                    f"Loss: {total_loss.item():.6f} | "
                    f"Interior: {interior_loss.item():.6f} | "
                    f"Terminal: {terminal_loss.item():.6f} | "
                    f"Opt: {opt_str} | "
                    f"Lévy: {levy_str}"
                )

        return history

    def simulate_controlled(
        self,
        x0: Tensor,
        n_steps: int = 100,
        n_paths: int = 1000,
        include_jumps: bool = True,
    ) -> Dict[str, Tensor]:
        """Simulate controlled paths using the learned policy.

        Uses the CORRECT SDE dynamics matching the PIDE:
            dX_t = X_t * [r + u(μ-r)]dt + X_t * u * σ dW_t + X_t * u * dJ_t

        where dJ_t is the compensated jump process with measure ν.

        Args:
            x0: Initial state (state_dim,) or (n_paths, state_dim)
            n_steps: Number of time steps
            n_paths: Number of paths to simulate
            include_jumps: Whether to include Lévy jumps

        Returns:
            Dictionary with paths, times, controls, and values
        """
        dt = self.problem.T / n_steps

        if x0.dim() == 1:
            x0 = x0.unsqueeze(0).expand(n_paths, -1)

        x = x0.clone()
        times = torch.zeros(n_paths, n_steps + 1, device=self.device)
        paths = torch.zeros(n_paths, n_steps + 1, self.problem.state_dim, device=self.device)
        controls = torch.zeros(n_paths, n_steps, self.problem.control_dim, device=self.device)
        values = torch.zeros(n_paths, n_steps + 1, device=self.device)

        paths[:, 0] = x
        values[:, 0] = self.value(torch.zeros(n_paths, device=self.device), x).squeeze(-1)

        # Get intensity from Lévy measure
        intensity = self.levy_measure.intensity()
        if math.isinf(intensity):
            # For infinite-activity, use effective intensity from truncation
            intensity = getattr(self.levy_measure, '_intensity', 10.0)

        with torch.no_grad():
            for i in range(n_steps):
                t = torch.full((n_paths,), i * dt, device=self.device)
                times[:, i] = t

                # Get control
                u = self.policy(t, x)
                controls[:, i] = u

                # Simulate one step of the SDE
                # dX = X[r + u(μ-r)]dt + Xuσ dW + Xu dJ
                b = self.problem.drift(t, x, u)      # X[r + u(μ-r)]
                sigma = self.problem.diffusion(t, x, u)  # Xuσ

                # Brownian increment
                dW = torch.randn_like(x) * math.sqrt(dt)

                # Euler step for diffusion part
                x_new = x + b * dt + sigma * dW

                # Add jumps with CORRECT dynamics: X_after = X * (1 + u * z)
                if include_jumps:
                    # Poisson thinning: sample number of jumps in [t, t+dt]
                    n_jumps = torch.poisson(
                        torch.full((n_paths,), intensity * dt, device=self.device)
                    ).long()

                    # Process jumps for paths that have them
                    max_jumps = n_jumps.max().item()
                    if max_jumps > 0:
                        for j in range(int(max_jumps)):
                            jump_mask = n_jumps > j
                            if jump_mask.any():
                                n_jumping = jump_mask.sum().item()
                                # Sample jump sizes
                                z, _ = self.levy_measure.sample(n_jumping, self.device)
                                if z.shape[-1] == 1 and self.problem.state_dim > 1:
                                    z_full = torch.zeros(n_jumping, self.problem.state_dim, device=self.device)
                                    z_full[:, 0] = z.squeeze(-1)
                                    z = z_full
                                else:
                                    z = z.squeeze(-1) if z.dim() > 1 else z

                                # Apply jump: X_new = X * (1 + u * z)
                                # Note: u[jump_mask] controls exposure to jump
                                u_jumping = u[jump_mask]
                                x_jumping = x_new[jump_mask]

                                # Handle dimension matching
                                if z.dim() == 1:
                                    z = z.unsqueeze(-1)
                                if u_jumping.dim() == 1:
                                    u_jumping = u_jumping.unsqueeze(-1)

                                jump_mult = 1.0 + u_jumping * z
                                x_new[jump_mask] = x_jumping * jump_mult

                x = torch.clamp(x_new, min=1e-6)  # Prevent negative wealth
                paths[:, i + 1] = x
                values[:, i + 1] = self.value(
                    torch.full((n_paths,), (i + 1) * dt, device=self.device), x
                ).squeeze(-1)

            times[:, -1] = self.problem.T

        return {
            "paths": paths,
            "times": times,
            "controls": controls,
            "values": values,
            "terminal_wealth": paths[:, -1].sum(dim=-1) if paths.shape[-1] > 1 else paths[:, -1, 0],
        }

    def evaluate(
        self,
        n_paths: int = 10000,
        x0: Optional[Tensor] = None,
    ) -> Dict[str, float]:
        """Evaluate the learned policy.

        Args:
            n_paths: Number of paths for Monte Carlo evaluation
            x0: Initial state (default: problem default)

        Returns:
            Evaluation metrics
        """
        if x0 is None:
            x0 = torch.ones(self.problem.state_dim, device=self.device)

        results = self.simulate_controlled(x0, n_paths=n_paths)

        terminal_wealth = results["terminal_wealth"]

        # Compute metrics
        metrics = {
            "mean_terminal_wealth": terminal_wealth.mean().item(),
            "std_terminal_wealth": terminal_wealth.std().item(),
            "var_5": torch.quantile(terminal_wealth, 0.05).item(),
            "cvar_5": terminal_wealth[terminal_wealth <= torch.quantile(terminal_wealth, 0.05)].mean().item(),
            "min_wealth": terminal_wealth.min().item(),
            "max_wealth": terminal_wealth.max().item(),
            "mean_control": results["controls"].mean().item(),
            "std_control": results["controls"].std().item(),
        }

        # Expected utility
        if hasattr(self.problem, 'gamma'):
            gamma = self.problem.gamma
            if abs(gamma - 1.0) < 1e-6:
                utility = torch.log(terminal_wealth.clamp(min=1e-8))
            else:
                utility = (terminal_wealth.clamp(min=1e-8) ** (1 - gamma)) / (1 - gamma)
            metrics["expected_utility"] = utility.mean().item()

        return metrics

    def compare_diffusion_vs_levy(
        self,
        n_paths: int = 5000,
        x0: Optional[Tensor] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Compare policy performance with and without jumps.

        This validates whether the learned policy properly accounts for
        jump risk compared to the diffusion-only Merton solution.

        Returns:
            Dictionary with metrics for both cases
        """
        if x0 is None:
            x0 = torch.ones(self.problem.state_dim, device=self.device)

        # Simulate with jumps
        results_levy = self.simulate_controlled(x0, n_paths=n_paths, include_jumps=True)
        wealth_levy = results_levy["terminal_wealth"]

        # Simulate without jumps
        results_diff = self.simulate_controlled(x0, n_paths=n_paths, include_jumps=False)
        wealth_diff = results_diff["terminal_wealth"]

        # Compute expected utility for both
        gamma = getattr(self.problem, 'gamma', 2.0)

        def compute_utility(wealth):
            if abs(gamma - 1.0) < 1e-6:
                return torch.log(wealth.clamp(min=1e-8)).mean().item()
            else:
                return ((wealth.clamp(min=1e-8) ** (1 - gamma)) / (1 - gamma)).mean().item()

        return {
            "with_jumps": {
                "mean_wealth": wealth_levy.mean().item(),
                "std_wealth": wealth_levy.std().item(),
                "var_5": torch.quantile(wealth_levy, 0.05).item(),
                "cvar_5": wealth_levy[wealth_levy <= torch.quantile(wealth_levy, 0.05)].mean().item(),
                "expected_utility": compute_utility(wealth_levy),
                "mean_control": results_levy["controls"].mean().item(),
            },
            "without_jumps": {
                "mean_wealth": wealth_diff.mean().item(),
                "std_wealth": wealth_diff.std().item(),
                "var_5": torch.quantile(wealth_diff, 0.05).item(),
                "cvar_5": wealth_diff[wealth_diff <= torch.quantile(wealth_diff, 0.05)].mean().item(),
                "expected_utility": compute_utility(wealth_diff),
                "mean_control": results_diff["controls"].mean().item(),
            },
        }
