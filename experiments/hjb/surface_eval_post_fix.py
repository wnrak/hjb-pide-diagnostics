"""Post-IS-fix surface comparison: ||V_NN - V_ref|| over (t, x).

Trains one seed of the corrected neural solver on the §4.5 1D VG benchmark
and evaluates V_NN, V_x_NN, V_xx_NN on a (t, x) grid against the closed-form
V_ref(t, x) = x^(1-γ)/(1-γ) · exp((1-γ)·F(u*)·(T-t)) from the scalar baseline
(§sec:homogeneity). Saves JSON to results/surface_eval_post_fix/.

Addresses reviewer concern #7: "experiments report only u_φ(0,1); need
surface comparisons over (t, x)."
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from levy_flows.hjb.levy_integral import VarianceGammaMeasure
from levy_flows.hjb.problems import MertonPortfolioProblem
from levy_flows.hjb.solver import LevyHJBSolver
from levy_flows.hjb.scalar_baseline import scalar_baseline_1d, vg_density


def main() -> None:
    out_dir = Path(__file__).resolve().parents[2] / "results" / "surface_eval_post_fix"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Train one seed of the post-IS-fix neural solver
    print("=== training corrected neural solver (1 seed, 500 epochs) ===")
    np.random.seed(42); torch.manual_seed(42)
    problem = MertonPortfolioProblem(
        r=0.02, mu=0.08, sigma=0.2, gamma=2.0, terminal_time=1.0,
    )
    measure = VarianceGammaMeasure(
        sigma=0.2, theta=-0.1, nu=0.3,
        truncation_min=0.01, truncation_max=0.99, intensity_scale=1.0,
    )
    solver = LevyHJBSolver(
        problem=problem, levy_measure=measure,
        hidden_dim=128, n_layers=4, n_levy_samples=64,
    )
    t0 = time.time()
    solver.fit(n_epochs=500, batch_size=256, warmup_epochs=100,
                use_foc=True, foc_frequency=2, verbose=False)
    elapsed = time.time() - t0
    with torch.no_grad():
        u_at_01 = float(solver.policy(torch.zeros(1), torch.ones(1, 1)).item())
    print(f"  trained ({elapsed:.0f}s); u_NN(0,1) = {u_at_01:.4f}")

    # Closed-form reference
    res = scalar_baseline_1d(
        r=0.02, mu=0.08, sigma=0.2, gamma=2.0,
        levy_density=lambda z: vg_density(z, sigma_vg=0.2, theta=-0.1, nu=0.3),
        z_min=0.01, z_max=0.99,
        n_quad_per_side=4001, n_u_fine=2001,
    )
    gamma = 2.0
    T = 1.0
    F_star = res.F_at_u_star
    u_ref = res.u_star
    print(f"  scalar baseline u* = {u_ref:.6f}, F(u*) = {F_star:.6f}")

    def V_ref(t_arr, x_arr):
        # V(t, x) = x^(1-γ)/(1-γ) · exp((1-γ) F(u*) (T-t))
        A = np.exp((1 - gamma) * F_star * (T - t_arr))
        return x_arr ** (1 - gamma) / (1 - gamma) * A

    def Vx_ref(t_arr, x_arr):
        A = np.exp((1 - gamma) * F_star * (T - t_arr))
        return x_arr ** (-gamma) * A

    def Vxx_ref(t_arr, x_arr):
        A = np.exp((1 - gamma) * F_star * (T - t_arr))
        return -gamma * x_arr ** (-gamma - 1) * A

    # Evaluation grid: x ∈ [0.5, 1.5], t ∈ [0, T]
    x_grid = np.linspace(0.5, 1.5, 21)
    t_grid = np.linspace(0.0, 0.95, 8)  # avoid t=T to dodge degenerate terminal at exact boundary

    rows = []
    for t_val in t_grid:
        t_t = torch.full((len(x_grid),), float(t_val), dtype=torch.float32)
        x_t = torch.tensor(x_grid, dtype=torch.float32).unsqueeze(-1).clone().requires_grad_(True)
        V_n = solver.value(t_t, x_t)
        Vx_n = torch.autograd.grad(V_n.sum(), x_t, create_graph=True)[0]
        Vxx_n = torch.autograd.grad(Vx_n.sum(), x_t, create_graph=False)[0]
        with torch.no_grad():
            u_n = solver.policy(t_t, x_t).squeeze(-1).numpy()
        V_n_np = V_n.detach().squeeze(-1).numpy()
        Vx_n_np = Vx_n.detach().squeeze(-1).numpy()
        Vxx_n_np = Vxx_n.detach().squeeze(-1).numpy()
        V_r = V_ref(t_val, x_grid)
        Vx_r = Vx_ref(t_val, x_grid)
        Vxx_r = Vxx_ref(t_val, x_grid)
        for i, x_val in enumerate(x_grid):
            rows.append({
                "t": float(t_val), "x": float(x_val),
                "V_NN": float(V_n_np[i]), "V_ref": float(V_r[i]),
                "Vx_NN": float(Vx_n_np[i]), "Vx_ref": float(Vx_r[i]),
                "Vxx_NN": float(Vxx_n_np[i]), "Vxx_ref": float(Vxx_r[i]),
                "u_NN": float(u_n[i]), "u_ref": float(u_ref),
            })

    # Aggregate metrics
    arr = np.array([(r["V_NN"], r["V_ref"], r["Vx_NN"], r["Vx_ref"],
                       r["Vxx_NN"], r["Vxx_ref"], r["u_NN"], r["u_ref"]) for r in rows])
    V_n, V_r, Vx_n, Vx_r, Vxx_n, Vxx_r, u_n, u_r = (arr[:, i] for i in range(8))

    def relmax(a, b):
        return float(np.max(np.abs(a - b) / (np.abs(b) + 1e-12)))

    metrics = {
        "n_eval_points": len(rows),
        "t_min": float(t_grid.min()),
        "t_max": float(t_grid.max()),
        "x_min": float(x_grid.min()),
        "x_max": float(x_grid.max()),
        "V_inf": float(np.max(np.abs(V_n - V_r))),
        "V_l2": float(np.sqrt(np.mean((V_n - V_r) ** 2))),
        "V_relmax": relmax(V_n, V_r),
        "Vx_inf": float(np.max(np.abs(Vx_n - Vx_r))),
        "Vx_l2": float(np.sqrt(np.mean((Vx_n - Vx_r) ** 2))),
        "Vx_relmax": relmax(Vx_n, Vx_r),
        "Vxx_inf": float(np.max(np.abs(Vxx_n - Vxx_r))),
        "Vxx_l2": float(np.sqrt(np.mean((Vxx_n - Vxx_r) ** 2))),
        "Vxx_relmax": relmax(Vxx_n, Vxx_r),
        "u_inf": float(np.max(np.abs(u_n - u_r))),
        "u_l2": float(np.sqrt(np.mean((u_n - u_r) ** 2))),
        "u_relmax": relmax(u_n, u_r),
    }

    print("\n=== surface metrics (post-IS-fix neural vs closed-form V_ref) ===")
    print(f"  grid: {len(t_grid)}t × {len(x_grid)}x = {metrics['n_eval_points']} points")
    print(f"  t ∈ [{metrics['t_min']:.2f}, {metrics['t_max']:.2f}], x ∈ [{metrics['x_min']:.2f}, {metrics['x_max']:.2f}]")
    print(f"  ||V   - V_ref||_inf  = {metrics['V_inf']:.4f}     rel max = {100*metrics['V_relmax']:.2f}%")
    print(f"  ||V_x - V_x_ref||_inf = {metrics['Vx_inf']:.4f}    rel max = {100*metrics['Vx_relmax']:.2f}%")
    print(f"  ||V_xx- V_xx_ref||_inf = {metrics['Vxx_inf']:.4f}  rel max = {100*metrics['Vxx_relmax']:.2f}%")
    print(f"  ||u   - u_ref||_inf  = {metrics['u_inf']:.4f}      rel max = {100*metrics['u_relmax']:.2f}%")

    summary = {
        "config": {
            "r": 0.02, "mu": 0.08, "sigma": 0.2, "gamma": 2.0, "T": 1.0,
            "vg_sigma": 0.2, "vg_theta": -0.1, "vg_nu": 0.3,
            "truncation_min": 0.01, "truncation_max": 0.99,
            "n_epochs": 500, "seed": 42,
        },
        "scalar_baseline_u_star": u_ref,
        "scalar_baseline_F_at_u_star": F_star,
        "neural_u_at_0_1": u_at_01,
        "metrics": metrics,
        "rows": rows,
    }
    out_path = out_dir / "surface_eval_post_fix.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()
