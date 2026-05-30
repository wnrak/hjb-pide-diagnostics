"""Phase 1.4 audit: re-run paper headline experiments under the corrected solver.

Covers:
- §4.1 diffusion-only Merton (u* = 0.75 reference).
- §4.2 VG jumps with the paper's stated parameters σ_VG=0.2, θ=-0.1, ν=0.3.
- §4.5 FD-PIDE comparison (defers to existing FD-PIDE; just records neural side).

Each setting is run with 5 seeds × 500 epochs; we report mean ± std and
flag the difference from the original paper number. Results land in
results/phase1_audit/.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from levy_flows.hjb.levy_integral import (
    CompoundPoissonMeasure,
    VarianceGammaMeasure,
)
from levy_flows.hjb.problems import MertonPortfolioProblem
from levy_flows.hjb.solver import LevyHJBSolver


SEEDS = [42, 123, 999, 7, 2024]
N_EPOCHS = 500
WARMUP = 100


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def fresh_problem() -> MertonPortfolioProblem:
    return MertonPortfolioProblem(
        r=0.02, mu=0.08, sigma=0.2, gamma=2.0, terminal_time=1.0,
    )


def measure_u(solver: LevyHJBSolver) -> float:
    """Single canonical scalar: u_φ(0, 1)."""
    with torch.no_grad():
        t = torch.zeros(1)
        x = torch.ones(1, 1)
        return float(solver.policy(t, x).item())


def run_diffusion_seed(seed: int) -> dict:
    set_seed(seed)
    problem = fresh_problem()
    measure = CompoundPoissonMeasure(intensity=0.0, jump_mean=0.0, jump_std=0.01)
    solver = LevyHJBSolver(
        problem=problem, levy_measure=measure,
        hidden_dim=128, n_layers=4, n_levy_samples=32,
    )
    t0 = time.time()
    history = solver.fit(
        n_epochs=N_EPOCHS, batch_size=256, warmup_epochs=WARMUP,
        use_foc=True, lambda_terminal=10.0, lambda_optimality=1.0,
        verbose=False,
    )
    elapsed = time.time() - t0
    u = measure_u(solver)
    merton = (problem.mu - problem.r) / (problem.gamma * problem.sigma**2)
    return {
        "seed": seed,
        "u_at_0_1": u,
        "merton_ratio": merton,
        "rel_error": (u - merton) / merton,
        "elapsed_sec": elapsed,
        "final_total_loss": history["total"][-1],
    }


def run_vg_seed(seed: int) -> dict:
    set_seed(seed)
    problem = fresh_problem()
    # Paper §4.2 VG parameters: σ_VG = 0.2, θ = -0.1, ν = 0.3. We tighten
    # truncation_max to 0.99 so the bankruptcy guard 1+uz>0 holds for u in [0,1].
    measure = VarianceGammaMeasure(
        sigma=0.2, theta=-0.1, nu=0.3,
        truncation_min=0.01, truncation_max=0.99,
        intensity_scale=1.0,
    )
    solver = LevyHJBSolver(
        problem=problem, levy_measure=measure,
        hidden_dim=128, n_layers=4, n_levy_samples=64,
    )
    t0 = time.time()
    history = solver.fit(
        n_epochs=N_EPOCHS, batch_size=256, warmup_epochs=WARMUP,
        use_foc=True, foc_frequency=2,
        lambda_terminal=10.0, lambda_optimality=1.0,
        verbose=False,
    )
    elapsed = time.time() - t0
    u = measure_u(solver)
    merton = (problem.mu - problem.r) / (problem.gamma * problem.sigma**2)
    return {
        "seed": seed,
        "u_at_0_1": u,
        "merton_ratio": merton,
        "reduction_pct": 100.0 * (merton - u) / merton,
        "vg_intensity_truncated": measure._intensity,
        "elapsed_sec": elapsed,
        "final_total_loss": history["total"][-1],
    }


def summarize(seeds: list, label: str) -> dict:
    arr = np.array([s["u_at_0_1"] for s in seeds])
    return {
        "label": label,
        "n_seeds": len(seeds),
        "u_mean": float(arr.mean()),
        "u_std": float(arr.std(ddof=1)),
        "u_min": float(arr.min()),
        "u_max": float(arr.max()),
        "per_seed": seeds,
    }


def main() -> None:
    out_dir = Path(__file__).resolve().parents[2] / "results" / "phase1_audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Phase 1.4 audit: {len(SEEDS)} seeds × {N_EPOCHS} epochs ===\n")

    print("--- §4.1 diffusion-only Merton ---")
    diff_seeds = []
    for s in SEEDS:
        r = run_diffusion_seed(s)
        print(f"  seed {s}: u(0,1) = {r['u_at_0_1']:.4f}  rel_err = {r['rel_error']:+.2%}  ({r['elapsed_sec']:.0f}s)")
        diff_seeds.append(r)
    diff_summary = summarize(diff_seeds, "diffusion_only_merton")
    print(f"  -> u = {diff_summary['u_mean']:.4f} ± {diff_summary['u_std']:.4f} (target 0.75)\n")

    print("--- §4.2 VG jumps (σ_VG=0.2, θ=-0.1, ν=0.3) ---")
    vg_seeds = []
    for s in SEEDS:
        r = run_vg_seed(s)
        print(f"  seed {s}: u(0,1) = {r['u_at_0_1']:.4f}  reduction = {r['reduction_pct']:.1f}%  ({r['elapsed_sec']:.0f}s)")
        vg_seeds.append(r)
    vg_summary = summarize(vg_seeds, "vg_jumps")
    print(f"  -> u = {vg_summary['u_mean']:.4f} ± {vg_summary['u_std']:.4f}\n")

    audit = {
        "n_seeds": len(SEEDS),
        "n_epochs": N_EPOCHS,
        "warmup_epochs": WARMUP,
        "diffusion_only_merton": diff_summary,
        "vg_jumps": vg_summary,
        "paper_claims": {
            "diffusion_only_paper_u": 0.7585,
            "diffusion_only_paper_error_pct": 1.1,
            "vg_paper_u": 0.4021,
            "vg_paper_reduction_pct": 46.4,
        },
    }
    with open(out_dir / "audit_results.json", "w") as f:
        json.dump(audit, f, indent=2)
    print(f"saved: {out_dir/'audit_results.json'}")


if __name__ == "__main__":
    main()
