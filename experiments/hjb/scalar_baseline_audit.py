"""Run the scalar/vector CRRA-VG baseline on the paper benchmarks.

Produces results/scalar_baseline/scalar_baseline_audit.json with:
- §4.5 1D VG benchmark u*_scalar (compared to FD v1, FD v2, neural)
- §4.8 2D coupled VG benchmark u*_scalar = (u_1, u_2) (compared to neural)
- diffusion-only Merton sanity check

The scalar baseline is the third reference for the audit story: no
solver machinery, just deterministic quadrature against the Lévy
density and a scalar/2D bounded optimization.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from levy_flows.hjb.scalar_baseline import (
    scalar_baseline_1d,
    scalar_baseline_2d,
    vg_density,
)


def main() -> None:
    out_dir = Path(__file__).resolve().parents[2] / "results" / "scalar_baseline"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {}

    # 1. Diffusion-only Merton
    print("=== diffusion-only Merton ===")
    res_diff = scalar_baseline_1d(
        r=0.02, mu=0.08, sigma=0.2, gamma=2.0,
        levy_density=lambda z: np.zeros_like(z),
        z_min=0.01, z_max=0.99,
    )
    print(f"  u*_scalar = {res_diff.u_star:.6f}, F* = {res_diff.F_at_u_star:.6f}")
    print(f"  analytical Merton = {0.06/(2*0.04):.6f}")
    summary["diffusion_only_merton"] = {
        "u_star": res_diff.u_star,
        "F_at_u_star": res_diff.F_at_u_star,
        "analytical": 0.06 / (2 * 0.04),
    }

    # 2. §4.5 1D VG benchmark
    print("\n=== §4.5 1D VG benchmark (γ=2, μ=0.08, σ=0.2, r=0.02; VG σ=0.2, θ=-0.1, ν=0.3) ===")
    rho_density_1d = lambda z: vg_density(z, sigma_vg=0.2, theta=-0.1, nu=0.3)
    t0 = time.time()
    res_vg_1d = scalar_baseline_1d(
        r=0.02, mu=0.08, sigma=0.2, gamma=2.0,
        levy_density=rho_density_1d,
        z_min=0.01, z_max=0.99,
        n_quad_per_side=4001,
        n_u_fine=2001,
    )
    elapsed_1d = time.time() - t0
    print(f"  u*_scalar  = {res_vg_1d.u_star:.6f}  (F* = {res_vg_1d.F_at_u_star:.6f})  [{elapsed_1d:.2f}s]")
    print(f"  reference numbers from prior audits:")
    print(f"    FD v1 (refined, V-interp):        0.347")
    print(f"    FD v2 (independent, post-fix):    0.344")
    print(f"    neural (5 seeds, post-IS-fix):    0.344 ± 0.006")
    gap_v1 = abs(res_vg_1d.u_star - 0.347) / 0.347 * 100
    gap_v2 = abs(res_vg_1d.u_star - 0.344) / 0.344 * 100
    gap_n = abs(res_vg_1d.u_star - 0.344) / 0.344 * 100
    print(f"  scalar vs FD v1:    {gap_v1:.2f}%")
    print(f"  scalar vs FD v2:    {gap_v2:.2f}%")
    print(f"  scalar vs neural:   {gap_n:.2f}%")
    summary["vg_1d_benchmark"] = {
        "u_star_scalar": res_vg_1d.u_star,
        "F_at_u_star": res_vg_1d.F_at_u_star,
        "reduction_vs_merton_pct": 100.0 * (0.75 - res_vg_1d.u_star) / 0.75,
        "elapsed_sec": elapsed_1d,
        "reference_fd_v1": 0.347,
        "reference_fd_v2": 0.344,
        "reference_neural_mean": 0.344,
        "reference_neural_std": 0.006,
        "scalar_vs_fd_v1_pct": gap_v1,
        "scalar_vs_fd_v2_pct": gap_v2,
        "scalar_vs_neural_pct": gap_n,
    }

    # 3. §4.8 2D coupled VG benchmark
    print("\n=== §4.8 2D coupled VG benchmark ===")
    print("  γ=2, μ=(0.10,0.06), σ=(0.25,0.15), ρ=0.3, r=0.02")
    print("  VG: σ=0.2, θ=(-0.10,-0.05), ν=0.3, |z|∈[0.01,0.99]")
    levy_1 = lambda z: vg_density(z, sigma_vg=0.2, theta=-0.10, nu=0.3)
    levy_2 = lambda z: vg_density(z, sigma_vg=0.2, theta=-0.05, nu=0.3)
    t0 = time.time()
    res_vg_2d = scalar_baseline_2d(
        r=0.02,
        mu=np.array([0.10, 0.06]),
        sigma=np.array([0.25, 0.15]),
        rho=0.3,
        gamma=2.0,
        levy_density_1=levy_1,
        levy_density_2=levy_2,
        z_min=0.01, z_max=0.99,
        n_quad_per_side=2001,
    )
    elapsed_2d = time.time() - t0
    u1, u2 = float(res_vg_2d.u_star[0]), float(res_vg_2d.u_star[1])
    print(f"  u*_scalar  = ({u1:.6f}, {u2:.6f}), cash = {res_vg_2d.cash_star:.6f}")
    print(f"  G* = {res_vg_2d.G_at_u_star:.6f}  [{elapsed_2d:.2f}s]")
    print(f"  neural (3 seeds, post-fix): u_1 = 0.348 ± 0.005, u_2 = 0.258 ± 0.005")
    gap_u1 = abs(u1 - 0.348) / 0.348 * 100
    gap_u2 = abs(u2 - 0.258) / 0.258 * 100
    print(f"  scalar vs neural u_1: {gap_u1:.2f}%")
    print(f"  scalar vs neural u_2: {gap_u2:.2f}%")
    summary["vg_2d_benchmark"] = {
        "u1_star_scalar": u1,
        "u2_star_scalar": u2,
        "cash_scalar": res_vg_2d.cash_star,
        "G_at_u_star": res_vg_2d.G_at_u_star,
        "reduction_u1_vs_merton_pct": 100.0 * (0.527 - u1) / 0.527,
        "reduction_u2_vs_merton_pct": 100.0 * (0.625 - u2) / 0.625,
        "elapsed_sec": elapsed_2d,
        "reference_neural_u1_mean": 0.348,
        "reference_neural_u2_mean": 0.258,
        "scalar_vs_neural_u1_pct": gap_u1,
        "scalar_vs_neural_u2_pct": gap_u2,
    }

    out_path = out_dir / "scalar_baseline_audit.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()
