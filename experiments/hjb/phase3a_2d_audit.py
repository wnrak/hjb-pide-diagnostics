"""Phase 3a §4.8 audit: 2D coupled portfolio with the corrected IS.

Three seeds × 500 epochs of TwoAssetLevyHJBSolver under the IS-corrected
sampler that delegates to VarianceGammaMeasure. Reports learned (u1, u2,
cash), reductions vs the correlated Merton baseline, and 5,000-path
wealth metrics under the same coupled VG model. Compared against the
constrained diffusion-simplex baseline for downside-risk comparison.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from levy_flows.hjb.solver_2d import TwoAssetLevyHJBSolver
from experiments.hjb.run_2d_portfolio import (
    constrained_diffusion_baseline,
    simulate_two_asset_terminal_wealth,
    wealth_metrics,
    json_ready,
)


CONFIG = dict(
    r=0.02, mu1=0.10, mu2=0.06, sigma1=0.25, sigma2=0.15,
    rho=0.3, gamma=2.0,
    vg_theta1=-0.10, vg_theta2=-0.05, vg_sigma=0.2, vg_nu=0.3,
)
TRUNCATION = (0.01, 0.99)
N_EPOCHS = 500


def run_seed(seed: int) -> dict:
    np.random.seed(seed); torch.manual_seed(seed)
    solver = TwoAssetLevyHJBSolver(
        **CONFIG,
        truncation_min=TRUNCATION[0], truncation_max=TRUNCATION[1],
        hidden_dim=128, n_layers=4, n_levy_samples=32,
    )
    t0 = time.time()
    history = solver.fit(n_epochs=N_EPOCHS, batch_size=256, verbose=False)
    elapsed = time.time() - t0
    u1, u2 = solver.get_policy()
    return {
        "seed": seed,
        "u1": float(u1), "u2": float(u2), "cash": float(1.0 - u1 - u2),
        "elapsed_sec": elapsed,
        "final_loss": float(history["loss"][-1]),
        "merton_u1": float(solver.merton_u1),
        "merton_u2": float(solver.merton_u2),
    }


def main() -> None:
    out_dir = Path(__file__).resolve().parents[2] / "results" / "phase3a_2d"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Phase 3a §4.8 audit: 2D coupled portfolio (post-IS-fix) ===")
    seeds = [42, 123, 999]

    runs = []
    for s in seeds:
        r = run_seed(s)
        print(f"  seed {s}: u1={r['u1']:.4f}, u2={r['u2']:.4f}, cash={r['cash']:.4f}  "
              f"({r['elapsed_sec']:.0f}s)")
        runs.append(r)

    u1_arr = np.array([r["u1"] for r in runs])
    u2_arr = np.array([r["u2"] for r in runs])
    print(f"\n  u1 = {u1_arr.mean():.4f} ± {u1_arr.std(ddof=1):.4f}")
    print(f"  u2 = {u2_arr.mean():.4f} ± {u2_arr.std(ddof=1):.4f}")

    merton_u1 = runs[0]["merton_u1"]
    merton_u2 = runs[0]["merton_u2"]
    print(f"\n  correlated Merton: u1*={merton_u1:.4f}, u2*={merton_u2:.4f}")
    print(f"  reduction vs Merton: u1 = {100*(merton_u1 - u1_arr.mean())/merton_u1:.1f}%, "
          f"u2 = {100*(merton_u2 - u2_arr.mean())/merton_u2:.1f}%")

    # Constrained diffusion baseline (deterministic, no seed dependence)
    print("\n  computing constrained diffusion baseline ...")
    diff_baseline = constrained_diffusion_baseline(
        r=CONFIG["r"], mu1=CONFIG["mu1"], mu2=CONFIG["mu2"],
        sigma1=CONFIG["sigma1"], sigma2=CONFIG["sigma2"],
        rho=CONFIG["rho"], gamma=CONFIG["gamma"],
    )
    print(f"  diffusion baseline: u1={diff_baseline['u1']:.4f}, "
          f"u2={diff_baseline['u2']:.4f}, cash={diff_baseline['cash']:.4f}")

    # Wealth-metrics comparison under common 2D VG model, using the per-seed
    # learned policies and the deterministic baseline.
    sim_cfg = {**CONFIG}
    sim_cfg.pop("gamma", None)  # simulate_two_asset_terminal_wealth doesn't take gamma
    baseline_wealth = simulate_two_asset_terminal_wealth(
        diff_baseline, sim_cfg, n_paths=5000, seed=43,
    )
    wm_baseline = wealth_metrics(baseline_wealth)

    learned_wealth_metrics = []
    for r in runs:
        weights = {"u1": r["u1"], "u2": r["u2"], "cash": r["cash"]}
        w_arr = simulate_two_asset_terminal_wealth(
            weights, sim_cfg, n_paths=5000, seed=43,
        )
        learned_wealth_metrics.append(wealth_metrics(w_arr))

    # Average wealth metrics across seeds (since the sims share the path seed,
    # the variation here is across the learned policies, not across MC paths).
    mean_metric = lambda key: float(np.mean([m[key] for m in learned_wealth_metrics]))
    std_metric = lambda key: float(np.std([m[key] for m in learned_wealth_metrics], ddof=1))
    learned_wm = {
        k: {"mean": mean_metric(k), "std": std_metric(k)}
        for k in ["mean", "std", "var_5", "cvar_5"]
    }

    print(f"\n  wealth metrics under common 2D VG model:")
    print(f"    diffusion baseline: E[W]={wm_baseline['mean']:.3f}, "
          f"VaR_5={wm_baseline['var_5']:.3f}, CVaR_5={wm_baseline['cvar_5']:.3f}")
    print(f"    learned (mean across seeds): "
          f"E[W]={learned_wm['mean']['mean']:.3f}, "
          f"VaR_5={learned_wm['var_5']['mean']:.3f}, "
          f"CVaR_5={learned_wm['cvar_5']['mean']:.3f}")

    summary = {
        "config": CONFIG,
        "truncation": list(TRUNCATION),
        "n_epochs": N_EPOCHS,
        "seeds": seeds,
        "merton_correlated": {"u1": merton_u1, "u2": merton_u2},
        "diffusion_baseline": diff_baseline,
        "learned_per_seed": runs,
        "learned_summary": {
            "u1_mean": float(u1_arr.mean()), "u1_std": float(u1_arr.std(ddof=1)),
            "u2_mean": float(u2_arr.mean()), "u2_std": float(u2_arr.std(ddof=1)),
        },
        "wealth_metrics_diffusion_baseline": wm_baseline,
        "wealth_metrics_learned": learned_wm,
    }
    out_path = out_dir / "phase3a_2d_audit.json"
    with open(out_path, "w") as f:
        json.dump(json_ready(summary), f, indent=2)
    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()
