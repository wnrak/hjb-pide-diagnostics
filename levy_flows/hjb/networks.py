"""Neural network architectures for HJB-PIDE solving.

Value and Policy networks with automatic differentiation support
for computing gradients and Hessians needed in the HJB residual.
"""

from typing import Optional, Tuple, List
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal encoding for time variable (improves temporal resolution)."""

    def __init__(self, dim: int, max_period: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, t: Tensor) -> Tensor:
        """
        Args:
            t: Time tensor of shape (batch,) or (batch, 1)

        Returns:
            Encoded time of shape (batch, dim)
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)

        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half_dim, device=t.device, dtype=t.dtype)
            / half_dim
        )
        args = t * freqs
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ResidualBlock(nn.Module):
    """Residual block with layer normalization."""

    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.net(x)


class ValueNetwork(nn.Module):
    """Neural network approximating the value function V(t, x).

    Architecture designed for smooth approximation with easy gradient/Hessian computation.
    Uses sinusoidal time encoding and residual blocks.
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 128,
        n_layers: int = 4,
        time_embed_dim: int = 32,
        dropout: float = 0.0,
    ):
        """
        Args:
            state_dim: Dimension of state space x
            hidden_dim: Hidden layer dimension
            n_layers: Number of residual blocks
            time_embed_dim: Dimension of time encoding
            dropout: Dropout rate
        """
        super().__init__()
        self.state_dim = state_dim

        # Time encoding
        self.time_encoder = SinusoidalPositionalEncoding(time_embed_dim)

        # Input projection
        self.input_proj = nn.Linear(state_dim + time_embed_dim, hidden_dim)

        # Residual blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout) for _ in range(n_layers)
        ])

        # Output projection (scalar value)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Initialize output to be near zero
        nn.init.zeros_(self.output_proj[-1].weight)
        nn.init.zeros_(self.output_proj[-1].bias)

    def forward(self, t: Tensor, x: Tensor) -> Tensor:
        """
        Compute V(t, x).

        Args:
            t: Time tensor (batch,) or (batch, 1)
            x: State tensor (batch, state_dim)

        Returns:
            Value tensor (batch, 1)
        """
        # Encode time
        t_embed = self.time_encoder(t)

        # Concatenate time and state
        h = torch.cat([t_embed, x], dim=-1)

        # Forward through network
        h = self.input_proj(h)
        h = F.gelu(h)

        for block in self.blocks:
            h = block(h)

        return self.output_proj(h)

    def forward_with_gradients(
        self, t: Tensor, x: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute V(t,x), ∂V/∂t, and ∇_x V in one pass.

        Uses torch.autograd.grad for efficient computation.

        Args:
            t: Time tensor (batch,) with requires_grad=True
            x: State tensor (batch, state_dim) with requires_grad=True

        Returns:
            V: Value (batch, 1)
            V_t: Time derivative (batch, 1)
            V_x: Spatial gradient (batch, state_dim)
        """
        # Ensure gradients are tracked
        t = t.requires_grad_(True)
        x = x.requires_grad_(True)

        V = self.forward(t, x)

        # Compute gradients
        grad_outputs = torch.ones_like(V)

        grads = torch.autograd.grad(
            V, [t, x],
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
        )

        V_t = grads[0].unsqueeze(-1) if grads[0].dim() == 1 else grads[0]
        V_x = grads[1]

        return V, V_t, V_x

    def forward_with_hessian(
        self, t: Tensor, x: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Compute V, ∂V/∂t, ∇_x V, and ∇²_x V (Hessian).

        The Hessian is needed for the diffusion term in HJB.

        Args:
            t: Time tensor (batch,)
            x: State tensor (batch, state_dim)

        Returns:
            V: Value (batch, 1)
            V_t: Time derivative (batch, 1)
            V_x: Spatial gradient (batch, state_dim)
            V_xx: Spatial Hessian (batch, state_dim, state_dim)
        """
        batch_size = x.shape[0]

        t = t.requires_grad_(True)
        x = x.requires_grad_(True)

        V = self.forward(t, x)

        # First derivatives
        grad_outputs = torch.ones_like(V)
        grads = torch.autograd.grad(
            V, [t, x],
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
        )

        V_t = grads[0].unsqueeze(-1) if grads[0].dim() == 1 else grads[0]
        V_x = grads[1]

        # Compute Hessian (second derivatives w.r.t. x)
        # More efficient: compute diagonal + off-diagonal if needed
        V_xx = torch.zeros(batch_size, self.state_dim, self.state_dim, device=x.device)

        for i in range(self.state_dim):
            grad2 = torch.autograd.grad(
                V_x[:, i].sum(), x,
                create_graph=True,
                retain_graph=True,
            )[0]
            V_xx[:, i, :] = grad2

        return V, V_t, V_x, V_xx


class PolicyNetwork(nn.Module):
    """Neural network approximating the optimal control u(t, x).

    Outputs control in a constrained set U (e.g., [0,1] for portfolio weights).
    """

    def __init__(
        self,
        state_dim: int,
        control_dim: int,
        hidden_dim: int = 128,
        n_layers: int = 3,
        time_embed_dim: int = 32,
        control_bounds: Optional[Tuple[float, float]] = None,
        simplex_constraint: bool = False,
    ):
        """
        Args:
            state_dim: Dimension of state space
            control_dim: Dimension of control space
            hidden_dim: Hidden layer dimension
            n_layers: Number of layers
            time_embed_dim: Time encoding dimension
            control_bounds: Optional (min, max) bounds for control
            simplex_constraint: If True, output sums to 1 (for portfolio weights)
        """
        super().__init__()
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.control_bounds = control_bounds
        self.simplex_constraint = simplex_constraint

        # Time encoding
        self.time_encoder = SinusoidalPositionalEncoding(time_embed_dim)

        # Network layers
        layers = [nn.Linear(state_dim + time_embed_dim, hidden_dim), nn.GELU()]
        for _ in range(n_layers - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            ])
        layers.append(nn.Linear(hidden_dim, control_dim))

        self.net = nn.Sequential(*layers)

        # Initialize near zero for stable start
        nn.init.normal_(self.net[-1].weight, std=0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, t: Tensor, x: Tensor) -> Tensor:
        """
        Compute optimal control u(t, x).

        Args:
            t: Time tensor (batch,)
            x: State tensor (batch, state_dim)

        Returns:
            Control tensor (batch, control_dim)
        """
        # Encode time
        t_embed = self.time_encoder(t)

        # Concatenate and forward
        h = torch.cat([t_embed, x], dim=-1)
        u = self.net(h)

        # Apply constraints
        if self.simplex_constraint:
            # Softmax for portfolio weights (sum to 1, non-negative)
            u = F.softmax(u, dim=-1)
        elif self.control_bounds is not None:
            # Sigmoid scaled to bounds
            lo, hi = self.control_bounds
            u = lo + (hi - lo) * torch.sigmoid(u)

        return u


class ActorCriticNetwork(nn.Module):
    """Combined value and policy network with shared representation.

    More parameter-efficient than separate networks.
    """

    def __init__(
        self,
        state_dim: int,
        control_dim: int,
        hidden_dim: int = 128,
        n_shared_layers: int = 2,
        n_value_layers: int = 2,
        n_policy_layers: int = 2,
        time_embed_dim: int = 32,
        control_bounds: Optional[Tuple[float, float]] = None,
        simplex_constraint: bool = False,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.control_bounds = control_bounds
        self.simplex_constraint = simplex_constraint

        # Time encoding
        self.time_encoder = SinusoidalPositionalEncoding(time_embed_dim)

        # Shared backbone
        self.shared_proj = nn.Linear(state_dim + time_embed_dim, hidden_dim)
        self.shared_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(n_shared_layers)
        ])

        # Value head
        self.value_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(n_value_layers)
        ])
        self.value_out = nn.Linear(hidden_dim, 1)

        # Policy head
        self.policy_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(n_policy_layers)
        ])
        self.policy_out = nn.Linear(hidden_dim, control_dim)

        # Initialize outputs
        nn.init.zeros_(self.value_out.weight)
        nn.init.zeros_(self.value_out.bias)
        nn.init.normal_(self.policy_out.weight, std=0.01)
        nn.init.zeros_(self.policy_out.bias)

    def _shared_forward(self, t: Tensor, x: Tensor) -> Tensor:
        """Compute shared representation."""
        t_embed = self.time_encoder(t)
        h = torch.cat([t_embed, x], dim=-1)
        h = F.gelu(self.shared_proj(h))
        for block in self.shared_blocks:
            h = block(h)
        return h

    def forward_value(self, t: Tensor, x: Tensor) -> Tensor:
        """Compute V(t, x)."""
        h = self._shared_forward(t, x)
        for block in self.value_blocks:
            h = block(h)
        return self.value_out(h)

    def forward_policy(self, t: Tensor, x: Tensor) -> Tensor:
        """Compute u(t, x)."""
        h = self._shared_forward(t, x)
        for block in self.policy_blocks:
            h = block(h)
        u = self.policy_out(h)

        if self.simplex_constraint:
            u = F.softmax(u, dim=-1)
        elif self.control_bounds is not None:
            lo, hi = self.control_bounds
            u = lo + (hi - lo) * torch.sigmoid(u)

        return u

    def forward(self, t: Tensor, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute both V(t,x) and u(t,x)."""
        h = self._shared_forward(t, x)

        # Value
        h_v = h
        for block in self.value_blocks:
            h_v = block(h_v)
        V = self.value_out(h_v)

        # Policy
        h_u = h
        for block in self.policy_blocks:
            h_u = block(h_u)
        u = self.policy_out(h_u)

        if self.simplex_constraint:
            u = F.softmax(u, dim=-1)
        elif self.control_bounds is not None:
            lo, hi = self.control_bounds
            u = lo + (hi - lo) * torch.sigmoid(u)

        return V, u
