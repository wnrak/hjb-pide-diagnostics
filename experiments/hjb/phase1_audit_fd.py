"""§4.5 audit: FD-PIDE comparison at matched truncation.

The previous paper ran neural at z_max = 2.0 vs FD at z_max = 0.5 — different
truncations on each side, so the "8% agreement" was apples-to-oranges. This
script runs FD-PIDE at z_max = 0.99 (matching the audited neural's bankruptcy-
safe truncation), at multiple grid resolutions to give an honest grid-
refinement table.

Outputs JSON to results/phase1_audit/fd_pide_audit.json.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from levy_flows.hjb.fd_pide_solver import FDPIDESolver, vg_levy_density


# Audited neural results from results/phase1_audit/audit_results.json
NEURAL_DIFFUSION_U = 0.7619404673576355
NEURAL_DIFFUSION_STD = 0.010854256320208597
NEURAL_VG_U = 0.47543853521347046
NEURAL_VG_STD = 0.008191493310449036


def run_fd(
    levy_density,
    z_min: float,
    z_max: float,
    n_x: int,
    n_t: int,
    n_z: int,
    label: str,
    n_u_grid: int = 20,
    use_v_interpolation: bool = False,
) -> dict:
    fd = FDPIDESolver(
        r=0.02, mu=0.08, sigma=0.2, gamma=2.0, T=1.0,
        levy_density=levy_density,
        z_min=z_min, z_max=z_max, n_z=n_z,
        n_x=n_x, n_t=n_t,
        n_u_grid=n_u_grid, u_min=0.0, u_max=1.0,
        use_v_interpolation=use_v_interpolation,
    )
    t0 = time.time()
    V, u_field, _ = fd.solve(verbose=False)
    elapsed = time.time() - t0
    u0 = fd.get_control_at_t0(1.0)
    return {
        "label": label,
        "n_x": n_x, "n_t": n_t, "n_z": n_z,
        "z_min": z_min, "z_max": z_max,
        "n_u_grid": n_u_grid,
        "use_v_interpolation": use_v_interpolation,
        "u_at_0_1": u0,
        "elapsed_sec": elapsed,
    }


def main() -> None:
    out_dir = Path(__file__).resolve().parents[2] / "results" / "phase1_audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== §4.5 FD-PIDE audit at matched truncation z_max = 0.99 ===\n")

    # Diffusion-only grid refinement
    print("--- Diffusion-only FD grid refinement ---")
    diff_grids = []
    for (nx, nt) in [(50, 25), (100, 50), (200, 100), (400, 200)]:
        r = run_fd(None, -0.99, 0.99, nx, nt, 50, f"diff_{nx}x{nt}")
        print(f"  {nx}x{nt}: u(0,1) = {r['u_at_0_1']:.4f}  ({r['elapsed_sec']:.1f}s)")
        diff_grids.append(r)
    fine_diff_u = diff_grids[-1]["u_at_0_1"]
    print(f"  finest: u = {fine_diff_u:.4f}, Merton = 0.75, FD error = {abs(fine_diff_u - 0.75)/0.75*100:.2f}%\n")

    # VG grid refinement at matched truncation
    print("--- VG FD grid refinement (z_max = 0.99 to match neural) ---")
    vg_density = lambda z: vg_levy_density(z, sigma=0.2, theta=-0.1, nu=0.3)
    vg_grids = []
    for (nx, nt, nz) in [(50, 25, 25), (100, 50, 50), (200, 100, 100), (400, 200, 200)]:
        r = run_fd(vg_density, -0.99, 0.99, nx, nt, nz, f"vg_{nx}x{nt}x{nz}")
        print(f"  {nx}x{nt}x{nz}: u(0,1) = {r['u_at_0_1']:.4f}  ({r['elapsed_sec']:.1f}s)")
        vg_grids.append(r)
    fine_vg_u = vg_grids[-1]["u_at_0_1"]
    print(f"  finest: u = {fine_vg_u:.4f}, reduction vs Merton = {(0.75 - fine_vg_u)/0.75*100:.1f}%\n")

    # Comparisons against neural (audited 5-seed mean) at the legacy FD setup
    diff_gap = abs(fine_diff_u - NEURAL_DIFFUSION_U) / NEURAL_DIFFUSION_U * 100
    vg_gap = abs(fine_vg_u - NEURAL_VG_U) / NEURAL_VG_U * 100
    print("--- Neural vs FD (matched truncation, default FD u-grid = 20, Taylor V) ---")
    print(f"  Diffusion: neural = {NEURAL_DIFFUSION_U:.4f} ± {NEURAL_DIFFUSION_STD:.4f}, FD = {fine_diff_u:.4f}, |gap| = {diff_gap:.2f}%")
    print(f"  VG:        neural = {NEURAL_VG_U:.4f} ± {NEURAL_VG_STD:.4f}, FD = {fine_vg_u:.4f}, |gap| = {vg_gap:.2f}%\n")

    # FD u-grid quantization: the default FD uses linspace(0, 1, 20), which
    # quantizes the optimum to 0.05 increments. Refine to expose what's
    # method-level FD bias vs u-grid quantization.
    print("--- FD u-grid refinement (200x100x100 spatial, vary n_u_grid; also test V interpolation) ---")
    fd_u_grid_sweep = []
    base_kwargs = dict(levy_density=vg_density, z_min=-0.99, z_max=0.99,
                        n_x=200, n_t=100, n_z=100)
    for (n_u, use_interp) in [
        (20, False), (50, False), (100, False), (200, False),
        (200, True),
    ]:
        r = run_fd(label=f"vg_nu={n_u}_{'interp' if use_interp else 'taylor'}",
                   n_u_grid=n_u, use_v_interpolation=use_interp,
                   **base_kwargs)
        gap = abs(r["u_at_0_1"] - NEURAL_VG_U) / NEURAL_VG_U * 100
        print(f"  n_u={n_u}, V={'interp' if use_interp else 'taylor':<6}: u(0,1) = {r['u_at_0_1']:.4f}, gap vs neural = {gap:.2f}%  ({r['elapsed_sec']:.1f}s)")
        r["gap_vs_neural_pct"] = gap
        fd_u_grid_sweep.append(r)

    refined_vg_u = fd_u_grid_sweep[-1]["u_at_0_1"]
    refined_gap = abs(refined_vg_u - NEURAL_VG_U) / NEURAL_VG_U * 100
    print(f"\n  refined FD (n_u=200, V interpolated): u = {refined_vg_u:.4f}, |gap vs neural| = {refined_gap:.2f}%")

    audit = {
        "neural_diffusion_u": NEURAL_DIFFUSION_U,
        "neural_diffusion_std": NEURAL_DIFFUSION_STD,
        "neural_vg_u": NEURAL_VG_U,
        "neural_vg_std": NEURAL_VG_STD,
        "fd_diffusion_grid_refinement": diff_grids,
        "fd_vg_grid_refinement": vg_grids,
        "fd_u_grid_sweep": fd_u_grid_sweep,
        "agreement_diffusion_pct_legacy_fd": diff_gap,
        "agreement_vg_pct_legacy_fd": vg_gap,
        "agreement_vg_pct_refined_fd": refined_gap,
        "matched_truncation_z_max": 0.99,
    }
    with open(out_dir / "fd_pide_audit.json", "w") as f:
        json.dump(audit, f, indent=2)
    print(f"\nsaved: {out_dir/'fd_pide_audit.json'}")


if __name__ == "__main__":
    main()
