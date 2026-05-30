#!/usr/bin/env python
"""Run HJB policies using fixed VG MLE parameters and compare fairly."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from levy_flows.hjb.levy_integral import CompoundPoissonMeasure, VarianceGammaMeasure
from levy_flows.hjb.problems import MertonPortfolioProblem
from levy_flows.hjb.solver import LevyHJBSolver


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MLE_RESULTS = ROOT / "results" / "hjb_sp500_mle" / "results.json"
DEFAULT_OUTPUT = ROOT / "results" / "hjb_sp500_mle_policy"


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_mle_results(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def simulate_daily_vg_wealth(
    policy_u: float,
    vg_params: dict,
    n_days: int = 252,
    n_paths: int = 5000,
    risk_free_rate: float = 0.02,
    seed: int = 123,
) -> np.ndarray:
    """Simulate terminal wealth under i.i.d. daily VG returns.

    The fitted parameters are daily-return parameters. We compare policies under the
    same fitted return law to avoid unfair cross-model comparisons.
    """
    rng = np.random.default_rng(seed)

    mu = vg_params["mu"]
    sigma = vg_params["sigma"]
    theta = vg_params["theta"]
    nu = vg_params["nu"]

    wealth = np.ones(n_paths, dtype=float)
    rf_daily = risk_free_rate / 252.0

    for _ in range(n_days):
        g = rng.gamma(shape=1.0 / nu, scale=nu, size=n_paths)
        z = rng.standard_normal(n_paths)
        risky_return = mu + theta * g + sigma * np.sqrt(g) * z
        portfolio_return = rf_daily + policy_u * risky_return
        wealth *= np.maximum(1.0 + portfolio_return, 1e-8)

    return wealth


def wealth_metrics(wealth: np.ndarray) -> dict:
    var5 = float(np.quantile(wealth, 0.05))
    cvar5 = float(np.mean(wealth[wealth <= var5]))
    return {
        "mean": float(np.mean(wealth)),
        "std": float(np.std(wealth, ddof=1)),
        "var_5": var5,
        "cvar_5": cvar5,
    }


def json_ready(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {k: json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HJB with MLE-calibrated VG parameters.")
    parser.add_argument("--mle-results", type=Path, default=DEFAULT_MLE_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-paths", type=int, default=5000)
    parser.add_argument("--trading-days", type=int, default=252)
    parser.add_argument("--diffusion-scale", type=float, default=0.75)
    args = parser.parse_args()

    set_seed(args.seed)
    mle_result = load_mle_results(args.mle_results)
    vg_fit = mle_result["vg_mle"]
    sample = mle_result["sample_moments"]

    risk_free_rate = 0.02
    # The fixed dataset contains total risky returns, so the annualized mean is the
    # risky asset drift itself, not an excess return that should be shifted by r again.
    mu_annual = max(sample["mean_annualized"], risk_free_rate + 1e-4)
    # Avoid double-counting all observed volatility as diffusion when adding a jump
    # measure calibrated from the same data. Use a conservative diffusion share and
    # let the annualized Lévy generator carry tail risk.
    sigma_annual = max(sample["vol_annualized"] * args.diffusion_scale, 1e-4)

    problem = MertonPortfolioProblem(
        r=risk_free_rate,
        mu=mu_annual,
        sigma=sigma_annual,
        gamma=2.0,
        terminal_time=1.0,
    )
    merton_ratio = (problem.mu - problem.r) / (problem.gamma * problem.sigma**2)

    t_test = torch.zeros(256)
    x_test = torch.ones(256, 1)

    no_jump = CompoundPoissonMeasure(intensity=0.0, jump_mean=0.0, jump_std=0.01)
    diffusion_solver = LevyHJBSolver(
        problem=problem,
        levy_measure=no_jump,
        hidden_dim=128,
        n_layers=4,
        n_levy_samples=32,
    )
    diffusion_solver.fit(
        n_epochs=args.epochs,
        batch_size=256,
        warmup_epochs=min(100, args.epochs // 3),
        use_foc=True,
        verbose=False,
    )
    u_diffusion = float(diffusion_solver.policy(t_test, x_test).mean().item())

    vg_measure = VarianceGammaMeasure(
        sigma=max(vg_fit["sigma"], 1e-4),
        theta=vg_fit["theta"],
        nu=max(vg_fit["nu"], 1e-3),
        truncation_min=0.005,
        truncation_max=0.2,
        intensity_scale=float(args.trading_days),
    )
    vg_solver = LevyHJBSolver(
        problem=problem,
        levy_measure=vg_measure,
        hidden_dim=128,
        n_layers=4,
        n_levy_samples=64,
    )
    vg_solver.fit(
        n_epochs=args.epochs,
        batch_size=256,
        warmup_epochs=min(100, args.epochs // 3),
        use_foc=True,
        foc_frequency=2,
        verbose=False,
    )
    u_vg = float(vg_solver.policy(t_test, x_test).mean().item())

    diff_wealth = simulate_daily_vg_wealth(
        u_diffusion, vg_fit, n_paths=args.n_paths, risk_free_rate=risk_free_rate, seed=args.seed + 1
    )
    vg_wealth = simulate_daily_vg_wealth(
        u_vg, vg_fit, n_paths=args.n_paths, risk_free_rate=risk_free_rate, seed=args.seed + 1
    )

    result = {
        "mle_results_path": str(args.mle_results),
        "problem": {
            "r": risk_free_rate,
            "mu_annual": mu_annual,
            "sigma_annual": sigma_annual,
            "gamma": 2.0,
            "merton_ratio": merton_ratio,
            "trading_days": args.trading_days,
            "diffusion_scale": args.diffusion_scale,
        },
        "controls": {
            "diffusion_only": u_diffusion,
            "vg_mle": u_vg,
            "allocation_reduction_pct": 100.0 * (u_diffusion - u_vg) / max(abs(u_diffusion), 1e-8),
        },
        "vg_measure": {
            "sigma_daily": vg_fit["sigma"],
            "theta_daily": vg_fit["theta"],
            "nu_daily": vg_fit["nu"],
            "truncation_min": 0.005,
            "truncation_max": 0.2,
            "intensity_scale": float(args.trading_days),
            "effective_annual_intensity": vg_measure.intensity(),
            "expected_jump_effect": vg_measure.expected_jump_effect(),
        },
        "wealth_metrics_under_common_vg_model": {
            "diffusion_policy": wealth_metrics(diff_wealth),
            "vg_policy": wealth_metrics(vg_wealth),
        },
    }

    args.output.mkdir(parents=True, exist_ok=True)
    with open(args.output / "results.json", "w") as f:
        json.dump(json_ready(result), f, indent=2)

    print(json.dumps(json_ready(result), indent=2))


if __name__ == "__main__":
    main()
