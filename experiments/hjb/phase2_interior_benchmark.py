"""Phase 2.2 — interior-optimum benchmark.

The §4.7 ablation under MLE-VG is uninformative for importance weighting
and weight clipping because the unconstrained Merton ratio is ~2.4, well
above U=[0,1], and every variant saturates at the upper bound. Here we
pick parameters whose unconstrained Merton ratio lies inside U=[0,1],
so the ablation has room to differentiate.

Specification: γ=2, μ=0.08, σ=0.25, r=0.02 → Merton ratio = 0.48.
Synthetic VG params: σ_VG=0.2, θ=-0.1, ν=0.3, truncation z ∈ [0.01, 0.99].

The script runs:
- 5-seed audit of the full method (baseline)
- 3-seed ablation × 4 variants
- comparison against FD-PIDE at n_u=200, V_interp, n_z=200, n_x=200, n_t=100

Outputs results/phase2_interior/.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from levy_flows.hjb.fd_pide_solver import FDPIDESolver, vg_levy_density
from levy_flows.hjb.levy_integral import VarianceGammaMeasure
from levy_flows.hjb.problems import MertonPortfolioProblem
from levy_flows.hjb.solver import LevyHJBSolver

# Ablation measure classes live in run_method_ablation.py; re-import.
from experiments.hjb.run_method_ablation import (
    EqualWeightMeasure,
    UnclippedVarianceGammaMeasure,
)


# Interior-optimum spec: Merton = (0.08-0.02) / (2 * 0.25^2) = 0.48
PROBLEM = dict(r=0.02, mu=0.08, sigma=0.25, gamma=2.0, T=1.0)
VG = dict(sigma=0.2, theta=-0.1, nu=0.3)
TRUNCATION = (0.01, 0.99)
MERTON = (PROBLEM["mu"] - PROBLEM["r"]) / (PROBLEM["gamma"] * PROBLEM["sigma"] ** 2)


def make_problem() -> MertonPortfolioProblem:
    return MertonPortfolioProblem(
        r=PROBLEM["r"], mu=PROBLEM["mu"],
        sigma=PROBLEM["sigma"], gamma=PROBLEM["gamma"],
        terminal_time=PROBLEM["T"],
    )


def make_measure(unclipped: bool = False, equal_weight: bool = False):
    cls = UnclippedVarianceGammaMeasure if unclipped else VarianceGammaMeasure
    base = cls(
        **VG,
        truncation_min=TRUNCATION[0], truncation_max=TRUNCATION[1],
        intensity_scale=1.0,
    )
    return EqualWeightMeasure(base) if equal_weight else base


def measure_u(solver: LevyHJBSolver) -> float:
    with torch.no_grad():
        return float(solver.policy(torch.zeros(1), torch.ones(1, 1)).item())


def run_seed(
    seed: int,
    *,
    measure_factory,
    n_epochs: int = 500,
    fixed_control: float = None,
    disable_compensation: bool = False,
) -> dict:
    np.random.seed(seed)
    torch.manual_seed(seed)
    problem = make_problem()
    measure = measure_factory()
    solver = LevyHJBSolver(
        problem=problem, levy_measure=measure,
        hidden_dim=128, n_layers=4, n_levy_samples=64,
    )
    if disable_compensation:
        solver.levy_integral.compensator_cutoff = 0.0
    if fixed_control is not None:
        c = float(fixed_control)
        def constant_policy(t, x):
            return torch.full((x.shape[0], 1), c, device=x.device)
        solver.policy = constant_policy
    t0 = time.time()
    history = solver.fit(
        n_epochs=n_epochs, batch_size=256,
        warmup_epochs=min(100, n_epochs // 3),
        use_foc=fixed_control is None, foc_frequency=2,
        lambda_optimality=0.0 if fixed_control is not None else 1.0,
        verbose=False,
    )
    elapsed = time.time() - t0
    return {
        "seed": seed,
        "u_at_0_1": measure_u(solver),
        "elapsed_sec": elapsed,
        "final_total_loss": history["total"][-1],
    }


def block_audit_full(seeds: list[int]) -> dict:
    print(f"\n--- Full method, {len(seeds)} seeds ---")
    out = []
    for s in seeds:
        r = run_seed(s, measure_factory=lambda: make_measure())
        print(f"  seed {s}: u={r['u_at_0_1']:.4f} ({r['elapsed_sec']:.0f}s)")
        out.append(r)
    arr = np.array([r["u_at_0_1"] for r in out])
    print(f"  -> u = {arr.mean():.4f} ± {arr.std(ddof=1):.4f}")
    return {
        "label": "full_method",
        "u_mean": float(arr.mean()), "u_std": float(arr.std(ddof=1)),
        "u_min": float(arr.min()), "u_max": float(arr.max()),
        "per_seed": out,
    }


def block_ablation(seeds: list[int]) -> list[dict]:
    print(f"\n--- Ablation variants, {len(seeds)} seeds each ---")
    variants = [
        ("no_importance_weighting", lambda: make_measure(equal_weight=True), {}),
        ("no_compensation", lambda: make_measure(), {"disable_compensation": True}),
        ("no_weight_clipping", lambda: make_measure(unclipped=True), {}),
        ("fixed_merton_control", lambda: make_measure(),
         {"fixed_control": MERTON}),
    ]
    out = []
    for name, factory, kwargs in variants:
        per_seed = []
        for s in seeds:
            r = run_seed(s, measure_factory=factory, n_epochs=500, **kwargs)
            per_seed.append(r)
        arr = np.array([r["u_at_0_1"] for r in per_seed])
        summary = {
            "variant": name,
            "u_mean": float(arr.mean()),
            "u_std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "per_seed": per_seed,
        }
        print(f"  {name}: u = {arr.mean():.4f} ± {summary['u_std']:.4f}")
        out.append(summary)
    return out


def block_fd() -> dict:
    print("\n--- FD-PIDE reference at matched truncation ---")
    fd = FDPIDESolver(
        **PROBLEM,
        levy_density=lambda z: vg_levy_density(z, **VG),
        z_min=-TRUNCATION[1], z_max=TRUNCATION[1], n_z=200,
        n_x=200, n_t=100,
        n_u_grid=200, u_min=0.0, u_max=1.0,
        use_v_interpolation=True,
    )
    t0 = time.time()
    fd.solve(verbose=False)
    elapsed = time.time() - t0
    u0 = fd.get_control_at_t0(1.0)
    print(f"  FD u(0,1) = {u0:.4f} ({elapsed:.1f}s)")
    return {
        "u_at_0_1": float(u0),
        "elapsed_sec": elapsed,
        "n_x": 200, "n_t": 100, "n_z": 200, "n_u_grid": 200,
        "z_max": TRUNCATION[1], "use_v_interpolation": True,
    }


def main() -> None:
    out_dir = Path(__file__).resolve().parents[2] / "results" / "phase2_interior"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Interior-optimum benchmark: γ=2, μ=0.08, σ=0.25, Merton ratio = {MERTON:.4f}\n")

    full = block_audit_full([42, 123, 999, 7, 2024])
    ablation = block_ablation([42, 123, 999])
    fd = block_fd()

    summary = {
        "problem": PROBLEM,
        "vg_params": VG,
        "truncation": list(TRUNCATION),
        "merton_ratio": MERTON,
        "full_method_audit": full,
        "ablation": ablation,
        "fd_reference": fd,
    }
    with open(out_dir / "phase2_interior.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsaved: {out_dir/'phase2_interior.json'}")
    print(f"\nHeadline:")
    print(f"  Merton (analytical):  {MERTON:.4f}")
    print(f"  FD reference:         {fd['u_at_0_1']:.4f}")
    print(f"  Neural (5 seeds):     {full['u_mean']:.4f} ± {full['u_std']:.4f}")
    for v in ablation:
        print(f"  {v['variant']:<28} {v['u_mean']:.4f} ± {v['u_std']:.4f}")


if __name__ == "__main__":
    main()
