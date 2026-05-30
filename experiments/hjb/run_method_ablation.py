#!/usr/bin/env python
"""Ablate the HJB-PIDE method around the control-dependent Lévy term."""

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from levy_flows.hjb.levy_integral import VarianceGammaMeasure
from levy_flows.hjb.problems import MertonPortfolioProblem
from levy_flows.hjb.solver import LevyHJBSolver


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MLE_RESULTS = ROOT / "results" / "hjb_sp500_mle" / "results.json"
DEFAULT_OUTPUT = ROOT / "results" / "hjb_method_ablation"


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


class UnclippedVarianceGammaMeasure(VarianceGammaMeasure):
    """VG measure with the same proposal, but without clipping weights."""

    def sample(self, n_samples: int, device: torch.device):
        n_pos = n_samples // 2
        n_neg = n_samples - n_pos

        rate_pos = self.M * 0.5
        gamma_pos = torch.distributions.Gamma(
            torch.tensor(1.0, device=device, dtype=torch.float32),
            torch.tensor(rate_pos, device=device, dtype=torch.float32),
        )
        z_pos = gamma_pos.sample((n_pos,)) + self.truncation_min
        z_pos = torch.clamp(z_pos, self.truncation_min, self.truncation_max)

        rate_neg = self.G * 0.5
        gamma_neg = torch.distributions.Gamma(
            torch.tensor(1.0, device=device, dtype=torch.float32),
            torch.tensor(rate_neg, device=device, dtype=torch.float32),
        )
        z_neg_abs = gamma_neg.sample((n_neg,)) + self.truncation_min
        z_neg_abs = torch.clamp(z_neg_abs, self.truncation_min, self.truncation_max)
        z_neg = -z_neg_abs

        z = torch.cat([z_pos, z_neg], dim=0)
        z_abs = torch.abs(z)

        nu_z = torch.zeros_like(z)
        pos_mask = z > 0
        neg_mask = z < 0
        nu_z[pos_mask] = (
            self.intensity_scale * self.C / z_abs[pos_mask] * torch.exp(-self.M * z_abs[pos_mask])
        )
        nu_z[neg_mask] = (
            self.intensity_scale * self.C / z_abs[neg_mask] * torch.exp(-self.G * z_abs[neg_mask])
        )

        # Same 50/50 mixture proposal as VarianceGammaMeasure; include the
        # 0.5 mixture probability factor in q(z).
        q_z = torch.zeros_like(z)
        q_z[pos_mask] = 0.5 * rate_pos * torch.exp(-rate_pos * (z_abs[pos_mask] - self.truncation_min))
        q_z[neg_mask] = 0.5 * rate_neg * torch.exp(-rate_neg * (z_abs[neg_mask] - self.truncation_min))

        raw_weights = nu_z / (q_z + 1e-10)
        return z.unsqueeze(-1), raw_weights


class EqualWeightMeasure:
    """Wrap a measure but discard importance weighting."""

    def __init__(self, base_measure: VarianceGammaMeasure):
        self.base_measure = base_measure

    def sample(self, n_samples: int, device: torch.device):
        z, weights = self.base_measure.sample(n_samples, device)
        return z, torch.ones_like(weights)

    def density(self, z: torch.Tensor) -> torch.Tensor:
        return self.base_measure.density(z)

    def intensity(self, truncation_min: float = None, truncation_max: float = None) -> float:
        return self.base_measure.intensity(truncation_min, truncation_max)

    def total_mass(self) -> float:
        return self.base_measure.total_mass()

    def expected_jump_effect(self):
        return self.base_measure.expected_jump_effect()


def simulate_daily_vg_wealth(
    policy_u: float,
    vg_params: dict,
    n_days: int = 252,
    n_paths: int = 5000,
    risk_free_rate: float = 0.02,
    seed: int = 123,
) -> np.ndarray:
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


def loss_summary(history: Dict[str, list]) -> dict:
    tail = history["interior"][-25:] if len(history["interior"]) >= 25 else history["interior"]
    total_tail = history["total"][-25:] if len(history["total"]) >= 25 else history["total"]
    return {
        "final_total_loss": float(history["total"][-1]),
        "final_interior_loss": float(history["interior"][-1]),
        "final_terminal_loss": float(history["terminal"][-1]),
        "final_optimality_loss": float(history["optimality"][-1]),
        "interior_loss_std_last_25": float(np.std(tail, ddof=1)) if len(tail) > 1 else 0.0,
        "total_loss_std_last_25": float(np.std(total_tail, ddof=1)) if len(total_tail) > 1 else 0.0,
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


def aggregate_variants(raw_results: List[dict]) -> List[dict]:
    grouped: Dict[str, List[dict]] = {}
    for result in raw_results:
        grouped.setdefault(result["variant"], []).append(result)

    summary = []
    for variant, runs in grouped.items():
        def values(path: List[str]) -> List[float]:
            out = []
            for run in runs:
                value = run
                for key in path:
                    value = value[key]
                out.append(float(value))
            return out

        row = {
            "variant": variant,
            "n_seeds": len(runs),
            "control_mean": float(np.mean(values(["control"]))),
            "control_std": float(np.std(values(["control"]), ddof=1)) if len(runs) > 1 else 0.0,
            "runtime_sec_mean": float(np.mean(values(["runtime_sec"]))),
            "runtime_sec_std": float(np.std(values(["runtime_sec"]), ddof=1)) if len(runs) > 1 else 0.0,
            "final_interior_loss_mean": float(np.mean(values(["losses", "final_interior_loss"]))),
            "final_interior_loss_std": float(np.std(values(["losses", "final_interior_loss"]), ddof=1)) if len(runs) > 1 else 0.0,
            "final_total_loss_mean": float(np.mean(values(["losses", "final_total_loss"]))),
            "final_total_loss_std": float(np.std(values(["losses", "final_total_loss"]), ddof=1)) if len(runs) > 1 else 0.0,
            "interior_loss_std_last_25_mean": float(np.mean(values(["losses", "interior_loss_std_last_25"]))),
            "interior_loss_std_last_25_std": float(np.std(values(["losses", "interior_loss_std_last_25"]), ddof=1)) if len(runs) > 1 else 0.0,
            "var_5_mean": float(np.mean(values(["wealth_metrics_under_common_vg_model", "var_5"]))),
            "var_5_std": float(np.std(values(["wealth_metrics_under_common_vg_model", "var_5"]), ddof=1)) if len(runs) > 1 else 0.0,
            "cvar_5_mean": float(np.mean(values(["wealth_metrics_under_common_vg_model", "cvar_5"]))),
            "cvar_5_std": float(np.std(values(["wealth_metrics_under_common_vg_model", "cvar_5"]), ddof=1)) if len(runs) > 1 else 0.0,
        }
        if "fixed_control" in runs[0]:
            row["fixed_control"] = float(runs[0]["fixed_control"])
        summary.append(row)

    preferred_order = {
        "full_method": 0,
        "no_importance_weighting": 1,
        "no_compensation": 2,
        "no_weight_clipping": 3,
        "fixed_merton_control": 4,
    }
    summary.sort(key=lambda item: preferred_order.get(item["variant"], 999))
    return summary


def build_problem(sample: dict, diffusion_scale: float) -> MertonPortfolioProblem:
    risk_free_rate = 0.02
    mu_annual = max(sample["mean_annualized"], risk_free_rate + 1e-4)
    sigma_annual = max(sample["vol_annualized"] * diffusion_scale, 1e-4)
    return MertonPortfolioProblem(
        r=risk_free_rate,
        mu=mu_annual,
        sigma=sigma_annual,
        gamma=2.0,
        terminal_time=1.0,
    )


def make_vg_measure(vg_fit: dict, trading_days: int, unclipped: bool = False):
    cls = UnclippedVarianceGammaMeasure if unclipped else VarianceGammaMeasure
    return cls(
        sigma=max(vg_fit["sigma"], 1e-4),
        theta=vg_fit["theta"],
        nu=max(vg_fit["nu"], 1e-3),
        truncation_min=0.005,
        truncation_max=0.2,
        intensity_scale=float(trading_days),
    )


def run_variant(
    name: str,
    build_measure: Callable[[], object],
    problem: MertonPortfolioProblem,
    vg_fit: dict,
    epochs: int,
    seed: int,
    n_paths: int,
    use_foc: bool,
    fixed_control: float = None,
    disable_compensation: bool = False,
) -> dict:
    set_seed(seed)
    solver = LevyHJBSolver(
        problem=problem,
        levy_measure=build_measure(),
        hidden_dim=128,
        n_layers=4,
        n_levy_samples=64,
    )

    if disable_compensation:
        solver.levy_integral.compensator_cutoff = 0.0

    if fixed_control is not None:
        control_value = float(fixed_control)

        def constant_policy(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            return torch.full((x.shape[0], 1), control_value, device=x.device)

        solver.policy = constant_policy  # type: ignore[method-assign]

    start = time.perf_counter()
    history = solver.fit(
        n_epochs=epochs,
        batch_size=256,
        warmup_epochs=min(100, epochs // 3),
        use_foc=use_foc,
        foc_frequency=2,
        lambda_optimality=0.0 if fixed_control is not None else 1.0,
        verbose=False,
    )
    runtime_sec = time.perf_counter() - start

    t_test = torch.zeros(256)
    x_test = torch.ones(256, 1)
    u_value = float(solver.policy(t_test, x_test).mean().item())

    common_wealth = simulate_daily_vg_wealth(
        u_value,
        vg_fit,
        n_paths=n_paths,
        risk_free_rate=problem.r,
        seed=seed + 1,
    )

    result = {
        "variant": name,
        "runtime_sec": runtime_sec,
        "control": u_value,
        "wealth_metrics_under_common_vg_model": wealth_metrics(common_wealth),
        "losses": loss_summary(history),
    }

    if math.isfinite(fixed_control) if fixed_control is not None else False:
        result["fixed_control"] = float(fixed_control)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run method ablations for the HJB-PIDE solver.")
    parser.add_argument("--mle-results", type=Path, default=DEFAULT_MLE_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--n-paths", type=int, default=4000)
    parser.add_argument("--trading-days", type=int, default=252)
    parser.add_argument("--diffusion-scale", type=float, default=0.75)
    args = parser.parse_args()

    mle = load_json(args.mle_results)
    vg_fit = mle["vg_mle"]
    sample = mle["sample_moments"]
    problem = build_problem(sample, args.diffusion_scale)
    merton_ratio = (problem.mu - problem.r) / (problem.gamma * problem.sigma**2)

    variants = [
        (
            "full_method",
            lambda: make_vg_measure(vg_fit, args.trading_days, unclipped=False),
            {"use_foc": True, "fixed_control": None, "disable_compensation": False},
        ),
        (
            "no_importance_weighting",
            lambda: EqualWeightMeasure(make_vg_measure(vg_fit, args.trading_days, unclipped=False)),
            {"use_foc": True, "fixed_control": None, "disable_compensation": False},
        ),
        (
            "no_compensation",
            lambda: make_vg_measure(vg_fit, args.trading_days, unclipped=False),
            {"use_foc": True, "fixed_control": None, "disable_compensation": True},
        ),
        (
            "no_weight_clipping",
            lambda: make_vg_measure(vg_fit, args.trading_days, unclipped=True),
            {"use_foc": True, "fixed_control": None, "disable_compensation": False},
        ),
        (
            "fixed_merton_control",
            lambda: make_vg_measure(vg_fit, args.trading_days, unclipped=False),
            {"use_foc": False, "fixed_control": merton_ratio, "disable_compensation": False},
        ),
    ]

    base_seed = args.seed
    seeds = args.seeds if args.seeds is not None else [base_seed]

    results = {
        "mle_results_path": str(args.mle_results),
        "problem": {
            "r": problem.r,
            "mu_annual": problem.mu,
            "sigma_annual": problem.sigma,
            "gamma": problem.gamma,
            "merton_ratio": merton_ratio,
            "trading_days": args.trading_days,
            "diffusion_scale": args.diffusion_scale,
        },
        "seeds": seeds,
        "raw_runs": [],
    }

    for seed_idx, seed in enumerate(seeds):
        for variant_idx, (name, measure_builder, kwargs) in enumerate(variants):
            variant_seed = seed + 100 * variant_idx + 1000 * seed_idx
            result = run_variant(
                name=name,
                build_measure=measure_builder,
                problem=problem,
                vg_fit=vg_fit,
                epochs=args.epochs,
                seed=variant_seed,
                n_paths=args.n_paths,
                use_foc=kwargs["use_foc"],
                fixed_control=kwargs["fixed_control"],
                disable_compensation=kwargs["disable_compensation"],
            )
            result["seed"] = seed
            results["raw_runs"].append(result)
            print(
                f"seed={seed} {name}: u={result['control']:.4f}, "
                f"interior={result['losses']['final_interior_loss']:.6f}, "
                f"runtime={result['runtime_sec']:.1f}s"
            )

    results["variants"] = aggregate_variants(results["raw_runs"])

    args.output.mkdir(parents=True, exist_ok=True)
    with open(args.output / "results.json", "w") as f:
        json.dump(json_ready(results), f, indent=2)

    print(json.dumps(json_ready(results), indent=2))


if __name__ == "__main__":
    main()
