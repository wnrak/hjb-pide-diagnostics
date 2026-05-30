"""Phase 3a: independent FD-PIDE reference + per-component H(u) diagnostic.

Three solvers on the §4.5 VG benchmark (γ=2, μ=0.08, σ=0.2, r=0.02; VG with
σ_VG=0.2, θ=-0.1, ν=0.3; truncation |z|∈[0.01, 0.99]; U=[0, 1]):

  - FD v1 (existing log-wealth solver)
  - FD v2 (new linear-x solver, this script)
  - Neural (best Phase 1 audit seed, deeper-trained from Phase 2.1)

For each, evaluate H(0, 1, u) on a u-grid and decompose into
   drift_term, diffusion_term, levy_uncomp, compensator, levy_comp, H_total.

The goal is *not* to declare a winner from the headline u value alone, but
to localize where the H(u) curves diverge component-wise.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from levy_flows.hjb.fd_pide_solver import FDPIDESolver, vg_levy_density as v1_vg_density
from levy_flows.hjb.fd_pide_v2 import FDPIDESolverV2, _vg_levy_density as v2_vg_density
from levy_flows.hjb.levy_integral import VarianceGammaMeasure
from levy_flows.hjb.problems import MertonPortfolioProblem
from levy_flows.hjb.solver import LevyHJBSolver


PROBLEM = dict(r=0.02, mu=0.08, sigma=0.2, gamma=2.0, T=1.0)
VG = dict(sigma=0.2, theta=-0.1, nu=0.3)
TRUNCATION = (0.01, 0.99)


# ---------------------------------------------------------------------------
# FD v1 with breakdown helper (replicates v1's _optimize_control logic explicitly)
# ---------------------------------------------------------------------------

def fd_v1_breakdown_at_x1(
    fd: FDPIDESolver,
    V: np.ndarray,                 # shape (n_t+1, n_x), V on log-wealth grid
    u_grid: np.ndarray,
    t_idx: int = 0,
) -> list[dict]:
    """Compute v1's H(u) decomposition at (t_idx, x=1).

    v1 works in log-wealth y = log(x). At y=0 (i.e. x=1), V_y = x V_x and
    V_yy = x² V_xx + x V_x. We extract V at y=0 and the local derivatives,
    then re-evaluate v1's Hamiltonian formula component by component.
    """
    log_x = fd.log_x
    V0 = V[t_idx, :]
    # V_y on log-grid via central differences
    V_y = np.zeros_like(V0)
    V_y[1:-1] = (V0[2:] - V0[:-2]) / (2 * fd.dx)
    V_y[0] = (V0[1] - V0[0]) / fd.dx
    V_y[-1] = (V0[-1] - V0[-2]) / fd.dx
    V_yy = np.zeros_like(V0)
    V_yy[1:-1] = (V0[2:] - 2 * V0[1:-1] + V0[:-2]) / fd.dx ** 2
    V_yy[0] = V_yy[1]
    V_yy[-1] = V_yy[-2]

    V_at_1 = float(np.interp(0.0, log_x, V0))
    Vy_at_1 = float(np.interp(0.0, log_x, V_y))
    Vyy_at_1 = float(np.interp(0.0, log_x, V_yy))

    # In linear-x derivatives at x=1: V_x = V_y, V_xx = V_yy - V_y
    Vx_at_1 = Vy_at_1
    Vxx_at_1 = Vyy_at_1 - Vy_at_1

    rows = []
    n_x = fd.n_x

    # Pre-collect the active z indices and ν weights as v1 uses them.
    z = fd.z
    weights = fd.levy_weights
    active = (np.abs(z) >= 1e-10) & (weights >= 1e-10)
    z_act = z[active]; w_act = weights[active]

    for u in u_grid:
        # Diffusion term in linear x: 0.5 * u² σ² x² V_xx with x=1
        diff_term = 0.5 * (u ** 2) * (PROBLEM["sigma"] ** 2) * (1.0 ** 2) * Vxx_at_1
        # Drift term in linear x: x(r + u(μ-r)) V_x with x=1
        drift_term = (PROBLEM["r"] + u * (PROBLEM["mu"] - PROBLEM["r"])) * Vx_at_1

        # Lévy: v1's _optimize_control uses Taylor V approximation by default.
        # For the audit we rebuild it using the SAME interpolation v1 uses
        # in `_compute_jump_integral` (linear interp in log-wealth + power-law
        # extrap), to make the comparison apples-to-apples.
        log_jumps = np.log(np.where(1 + u * z_act > 0, 1 + u * z_act, 1.0))
        safe = (1 + u * z_act) > 0
        y_new = 0.0 + log_jumps   # y at x=1 is 0
        # Linear interp on log_x grid + power-law extrap
        idx = (y_new - fd.log_x_min) / fd.dx
        idx_low = np.clip(np.floor(idx).astype(int), 0, n_x - 2)
        w_lo = idx - idx_low
        V_in = (1 - w_lo) * V0[idx_low] + w_lo * V0[idx_low + 1]
        V_below = V0[0] * np.exp((1 - PROBLEM["gamma"]) * (y_new - fd.log_x_min))
        V_above = V0[-1] * np.exp((1 - PROBLEM["gamma"]) * (y_new - fd.log_x_max))
        V_jump = np.where(y_new < fd.log_x_min, V_below,
                          np.where(y_new > fd.log_x_max, V_above, V_in))
        V_jump = np.where(safe, V_jump, -1e10)

        # Lévy integrand:
        #   uncompensated: (V_jump - V_at_1) * ν(z) * dz
        #   compensator at x=1: u·1·V_x · ∫ z ν(dz)
        uncomp = float(((V_jump - V_at_1) * w_act * fd.dz).sum())
        int_z_nu = float((z_act * w_act * fd.dz).sum())
        comp = u * 1.0 * Vx_at_1 * int_z_nu
        levy_comp = uncomp - comp

        rows.append({
            "u": float(u),
            "drift": drift_term,
            "diffusion": diff_term,
            "levy_uncomp": uncomp,
            "compensator": comp,
            "levy_comp": levy_comp,
            "H": drift_term + diff_term + levy_comp,
        })
    return rows


# ---------------------------------------------------------------------------
# Neural breakdown at x=1
# ---------------------------------------------------------------------------

def neural_breakdown_at_x1(
    solver: LevyHJBSolver,
    measure: VarianceGammaMeasure,
    u_grid: np.ndarray,
) -> list[dict]:
    """Decompose neural H(0, 1, u) using the solver's own machinery.

    For a fair component split, we compute the diffusion-side terms
    (drift, diffusion) deterministically from the autodiff V, V_x, V_xx
    at x=1, and use the solver's importance-weighted MC for the Lévy
    integral. Compensator is reported as the ν-integral of z times u·V_x
    (also estimated by IS MC, with a large sample for accuracy)."""
    t = torch.zeros(1)
    x = torch.ones(1, 1).clone().requires_grad_(True)
    V = solver.value(t, x)
    V_x = torch.autograd.grad(V.sum(), x, create_graph=True)[0]
    V_xx = torch.autograd.grad(V_x.sum(), x, create_graph=False)[0]
    V_at_1 = float(V.item())
    Vx_at_1 = float(V_x.item())
    Vxx_at_1 = float(V_xx.item())

    # IS estimate of ∫ z ν(dz) using a large sample
    K = 16384
    with torch.no_grad():
        z_samples, w = measure.sample(K, device=torch.device("cpu"))
        z_samples = z_samples.squeeze(-1).numpy()
        w = w.numpy()
    int_z_nu = float((w * z_samples).mean())

    rows = []
    sigma = PROBLEM["sigma"]
    for u in u_grid:
        drift = (PROBLEM["r"] + u * (PROBLEM["mu"] - PROBLEM["r"])) * Vx_at_1
        diffusion = 0.5 * (u ** 2) * (sigma ** 2) * (1.0 ** 2) * Vxx_at_1
        # Lévy via the solver's own LevyIntegralMC, using a large K for accuracy
        with torch.no_grad():
            t_b = torch.zeros(1)
            x_b = torch.ones(1, 1)
            u_b = torch.tensor([[u]], dtype=torch.float32)
            V_x_b = torch.tensor([[Vx_at_1]], dtype=torch.float32)
            # Re-instantiate a high-K integrator for diagnostic accuracy
            from levy_flows.hjb.levy_integral import LevyIntegralMC
            big_int = LevyIntegralMC(levy_measure=measure, n_samples=4096)
            levy_comp = float(big_int(
                lambda t_, x_: solver.value(t_, x_),
                t_b, x_b, u_b, V_x_b,
            ).item())
        # The solver's LevyIntegralMC returns the *compensated* integral.
        # Reconstruct uncompensated = compensated + compensator.
        compensator = u * 1.0 * Vx_at_1 * int_z_nu
        levy_uncomp = levy_comp + compensator
        rows.append({
            "u": float(u),
            "drift": drift,
            "diffusion": diffusion,
            "levy_uncomp": levy_uncomp,
            "compensator": compensator,
            "levy_comp": levy_comp,
            "H": drift + diffusion + levy_comp,
        })
    return rows


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> None:
    out_dir = Path(__file__).resolve().parents[2] / "results" / "phase3a"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Phase 3a: independent FD-PIDE + per-component H(u) ===\n")

    # ---- 1. FD v2 grid refinement on the same VG problem ----
    # We keep the finest solver instance after the loop to use for the
    # H(u) breakdown (no double-solve). The 401-grid case is dropped from
    # the sweep because it would take ~1 hour and the 101→201 jump already
    # shows the convergence direction; if reviewers want a 401 row we can
    # add it later.
    print("--- FD v2 grid refinement ---")
    fd_v2_runs = []
    fd2 = None
    for (n_x, n_t, n_z) in [(101, 50, 101), (201, 100, 201)]:
        fd2 = FDPIDESolverV2(
            **PROBLEM,
            levy_density=v2_vg_density,
            n_x=n_x, n_t=n_t, n_z_per_side=n_z,
            z_min=TRUNCATION[0], z_max=TRUNCATION[1],
            u_min=0.0, u_max=1.0, n_bracket=41,
        )
        t0 = time.time()
        fd2.solve(verbose=False)
        elapsed = time.time() - t0
        u0 = fd2.get_control_at_t0(1.0)
        print(f"  n_x={n_x}, n_t={n_t}, n_z_per_side={n_z}: "
              f"u(0,1) = {u0:.4f}  ({elapsed:.0f}s)")
        fd_v2_runs.append({
            "n_x": n_x, "n_t": n_t, "n_z_per_side": n_z,
            "u_at_0_1": u0, "elapsed_sec": elapsed,
        })
    u_v2_finest = fd2.get_control_at_t0(1.0)
    print(f"  finest FD v2 u(0,1) = {u_v2_finest:.4f}\n")

    # ---- 2. FD v1 reference (re-solve at refined Phase-2 setting) ----
    print("--- FD v1 reference ---")
    fd1 = FDPIDESolver(
        r=PROBLEM["r"], mu=PROBLEM["mu"], sigma=PROBLEM["sigma"],
        gamma=PROBLEM["gamma"], T=PROBLEM["T"],
        levy_density=lambda z: v1_vg_density(z, **VG),
        z_min=-TRUNCATION[1], z_max=TRUNCATION[1], n_z=200,
        n_x=200, n_t=100,
        n_u_grid=200, u_min=0.0, u_max=1.0, use_v_interpolation=True,
    )
    t0 = time.time()
    V_v1, u_field_v1, _ = fd1.solve(verbose=False)
    elapsed_v1 = time.time() - t0
    u_v1 = fd1.get_control_at_t0(1.0)
    print(f"  FD v1 u(0,1) = {u_v1:.4f}  ({elapsed_v1:.0f}s)\n")

    # ---- 3. Neural reference (best seed = lowest loss from Phase 2.1) ----
    print("--- Neural reference (deeper-trained, best Phase 2.1 seed) ---")
    np.random.seed(42); torch.manual_seed(42)
    problem = MertonPortfolioProblem(
        r=PROBLEM["r"], mu=PROBLEM["mu"], sigma=PROBLEM["sigma"],
        gamma=PROBLEM["gamma"], terminal_time=PROBLEM["T"],
    )
    measure = VarianceGammaMeasure(
        **VG,
        truncation_min=TRUNCATION[0], truncation_max=TRUNCATION[1],
        intensity_scale=1.0,
    )
    solver = LevyHJBSolver(
        problem=problem, levy_measure=measure,
        hidden_dim=192, n_layers=4, n_levy_samples=96,
    )
    t0 = time.time()
    solver.fit(
        n_epochs=1000, batch_size=256, warmup_epochs=200,
        use_foc=True, foc_frequency=2, verbose=False,
    )
    elapsed_n = time.time() - t0
    with torch.no_grad():
        u_n = float(solver.policy(torch.zeros(1), torch.ones(1, 1)).item())
    print(f"  Neural u(0,1) = {u_n:.4f}  ({elapsed_n:.0f}s)\n")

    # ---- 4. Per-component H(u) breakdown at x=1, t=0 ----
    print("--- Per-component H(u) breakdown at (t=0, x=1) ---")
    u_grid = np.linspace(0.0, 1.0, 21)
    bk_v1 = fd_v1_breakdown_at_x1(fd1, V_v1, u_grid, t_idx=0)
    bk_v2 = fd2.hamiltonian_breakdown_at(t=0.0, x=1.0, u_grid=u_grid)["rows"]
    bk_n = neural_breakdown_at_x1(solver, measure, u_grid)

    # Find argmax of each H curve over u_grid
    def argmax_u(rows):
        H = np.array([r["H"] for r in rows])
        return float(rows[int(np.argmax(H))]["u"]), float(H.max())
    u_argmax_v1, H_v1 = argmax_u(bk_v1)
    u_argmax_v2, H_v2 = argmax_u(bk_v2)
    u_argmax_n, H_n = argmax_u(bk_n)
    print(f"  argmax_u H_FDv1   = {u_argmax_v1:.3f}  (max H = {H_v1:.4f})")
    print(f"  argmax_u H_FDv2   = {u_argmax_v2:.3f}  (max H = {H_v2:.4f})")
    print(f"  argmax_u H_neural = {u_argmax_n:.3f}  (max H = {H_n:.4f})\n")

    # ---- 5. Save full breakdowns ----
    summary = {
        "problem": PROBLEM, "vg_params": VG, "truncation": list(TRUNCATION),
        "fd_v1_u_at_0_1": u_v1,
        "fd_v2_grid_refinement": fd_v2_runs,
        "fd_v2_u_at_0_1_finest": u_v2_finest,
        "neural_u_at_0_1": u_n,
        "argmax_u_FDv1": u_argmax_v1,
        "argmax_u_FDv2": u_argmax_v2,
        "argmax_u_neural": u_argmax_n,
        "breakdown_FDv1": bk_v1,
        "breakdown_FDv2": bk_v2,
        "breakdown_neural": bk_n,
    }
    out_path = out_dir / "phase3a_audit.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"saved: {out_path}")

    # Print a tight comparison table at a few u values
    print("\n--- H components at selected u (FD v1 / FD v2 / neural) ---")
    print(f"{'u':>5}  {'drift v1':>11} {'v2':>11} {'n':>11}   "
          f"{'diff v1':>11} {'v2':>11} {'n':>11}   "
          f"{'levy_c v1':>11} {'v2':>11} {'n':>11}   "
          f"{'comp v1':>11} {'v2':>11} {'n':>11}")
    for i in (0, 4, 8, 10, 12, 16, 20):
        r1, r2, rn = bk_v1[i], bk_v2[i], bk_n[i]
        print(f"{r1['u']:>5.2f}  "
              f"{r1['drift']:+.4f} {r2['drift']:+.4f} {rn['drift']:+.4f}   "
              f"{r1['diffusion']:+.4f} {r2['diffusion']:+.4f} {rn['diffusion']:+.4f}   "
              f"{r1['levy_comp']:+.4f} {r2['levy_comp']:+.4f} {rn['levy_comp']:+.4f}   "
              f"{r1['compensator']:+.4f} {r2['compensator']:+.4f} {rn['compensator']:+.4f}")


if __name__ == "__main__":
    main()
