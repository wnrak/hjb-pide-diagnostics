# Anonymized supplement — code, data, and result JSONs

This archive accompanies the submission *A Per-Component Diagnostic Protocol
for Neural HJB-PIDE Solvers under Control-Dependent Lévy Jumps*. It contains
every piece of code and data needed to reproduce the numeric claims of the
paper.

## Anonymity

This is the review archive. No author or institution names appear in any
file metadata, comments, or URLs. The public repository URL will replace this
archive at acceptance.

## Layout

```
supplement/
├── SUPPLEMENT.md                # this file
├── CONSISTENCY_LOG.md            # term-by-term sweep + rerunnable shell command
├── README.md                     # short pointer file (see paper §Reproducibility)
├── levy_flows/hjb/               # solver source
│   ├── solver.py                  # residual-trained neural 1D
│   ├── solver_2d.py               # residual-trained neural 2D
│   ├── fd_pide_solver.py          # FD v1 (log-x, implicit Euler, trapezoidal)
│   ├── fd_pide_v2.py              # FD v2 (linear-x, Crank-Nicolson, Simpson, Brent)
│   ├── scalar_baseline.py         # homogeneity-reduction quadrature solver
│   ├── levy_integral.py           # corrected truncated VG IS sampler
│   ├── problems.py
│   └── networks.py
├── data/
│   ├── README.md                    # S&P 500 bring-your-own-data instructions
│   ├── prepare_sp500_returns.py     # builds Date,Returns from a user-supplied Date,Close CSV
│   ├── sp500_schema_example.csv     # SYNTHETIC rows only — documents input format
│   └── mle_parameters.json          # fitted VG params + std errors used in the paper
├── experiments/hjb/               # experiment driver scripts
│   ├── phase1_audit.py             # §4.1 + §4.2
│   ├── phase1_audit_fd.py          # §4.5 FD spatial/u-grid sweep
│   ├── phase3a_audit.py            # §4.5 per-component diagnostic + FD v2
│   ├── phase3a_2d_audit.py         # §4.8 post-fix 2D
│   ├── phase2_interior_benchmark.py # §4.7 interior-optimum re-ablation
│   ├── run_method_ablation.py      # §4.7 saturating-benchmark ablation
│   ├── run_hjb_mle_policy.py       # App. S&P MLE policy
│   ├── run_sp500_mle.py            # App. S&P calibration (needs user-prepared returns)
│   ├── scalar_baseline_audit.py    # §4.5 scalar baseline
│   └── surface_eval_post_fix.py    # §4.5 surface vs V_ref
└── results/                        # all JSONs backing the paper's numbers
    ├── phase1_audit/{audit_results.json, fd_pide_audit.json, ...}
    ├── phase2_vg/phase2_vg_adjudication.json
    ├── phase2_interior/phase2_interior.json
    ├── phase3a/phase3a_audit.json
    ├── phase3a_2d/phase3a_2d_audit.json
    ├── post_fix_audit/{ablation/, sp500_mle_policy/}
    ├── scalar_baseline/scalar_baseline_audit.json
    ├── surface_eval_post_fix/surface_eval_post_fix.json
    └── hjb_sp500_mle/results.json
```

## Numeric-claim → source-JSON map

Every numeric claim in the paper traces to a result file here.

| Claim location in paper | Number | Source JSON |
|---|---|---|
| §4.1 diffusion-only Merton | $u = 0.762 \pm 0.011$ (5 seeds) | `results/phase1_audit/audit_results.json:diffusion_only_merton` |
| §4.2 VG (post-fix) | $u = 0.344 \pm 0.006$, $-54\%$ | `results/phase1_audit/audit_results.json:vg_jumps` |
| §4.5 FD v1 sweep | $u_{FD} \in [0.347, 0.352]$ | `results/phase1_audit/fd_pide_audit.json:fd_vg_grid_refinement` |
| §4.5 FD v2 finest | $u_{FD,v2} = 0.344$ | `results/phase3a/phase3a_audit.json:fd_v2_grid_refinement` |
| §4.5 per-component, neural compensator = $0.502 \times$ FD | constant ratio | `results/phase3a/phase3a_audit.json:breakdown_*` |
| §4.5 scalar baseline $u^\star = 0.3521$ | quadrature | `results/scalar_baseline/scalar_baseline_audit.json:vg_1d_benchmark` |
| §4.5 surface metrics ($\|V - V_{\text{ref}}\|$, etc.) | 4 surface norms | `results/surface_eval_post_fix/surface_eval_post_fix.json:metrics` |
| App. S&P MLE policy (post-fix) | $u_{VG} = 0.949$, $+0.8$ pp VaR | `results/post_fix_audit/sp500_mle_policy/results.json` |
| App. S&P VG MLE fit | $(\hat\sigma,\hat\theta,\hat\nu)$, std errors | `data/mle_parameters.json` (raw data not redistributed) |
| §4.7 saturating ablation (post-fix) | full $0.921$, no comp $0.410$ | `results/post_fix_audit/ablation/results.json` |
| §4.7 interior re-ablation | full $0.277$, FD ref $0.276$, no comp $0.006$ | `results/phase2_interior/phase2_interior.json` |
| §4.8 2D coupled (post-fix) | $u_1 = 0.348 \pm 0.005$, $u_2 = 0.258 \pm 0.005$ | `results/phase3a_2d/phase3a_2d_audit.json` |

## Re-running every headline number from scratch

Total wall-clock on the reference hardware (Apple M1, single CPU): about
4–5 hours.

The synthetic-jump benchmarks (steps 2-6, 9-10) are fully self-contained. The
S&P calibration steps (1, 7) require you to first supply your own lawfully
obtained S&P 500 daily close series — this repository does not redistribute
S&P 500 data (see `data/README.md`). Without it, the fitted parameters used in
the paper are available in `data/mle_parameters.json`.

```bash
# (optional) clean
rm -rf results/*

# 0.  [S&P only] prepare returns from your own daily close series (not shipped)
python data/prepare_sp500_returns.py \
    --input data/sp500_user.csv \
    --output data/sp500_returns_local.csv

# 1.  [S&P] VG MLE calibration on the prepared returns
python experiments/hjb/run_sp500_mle.py \
    --returns data/sp500_returns_local.csv

# 2.  §4.1 + §4.2 5-seed audit
python experiments/hjb/phase1_audit.py

# 3.  §4.5 FD-PIDE sweep
python experiments/hjb/phase1_audit_fd.py

# 4.  §4.5 per-component diagnostic + independent FD v2
python experiments/hjb/phase3a_audit.py

# 5.  §4.5 scalar baseline (homogeneity reduction)
python experiments/hjb/scalar_baseline_audit.py

# 6.  §4.5 surface metrics vs semi-analytic V_ref
python experiments/hjb/surface_eval_post_fix.py

# 7.  [S&P] MLE policy comparison
python experiments/hjb/run_hjb_mle_policy.py \
    --epochs 300 --output results/post_fix_audit/sp500_mle_policy

# 8.  §4.7 saturating-benchmark ablation
python experiments/hjb/run_method_ablation.py \
    --seeds 42 123 999 --epochs 200 \
    --output results/post_fix_audit/ablation

# 9.  §4.7 interior-optimum re-ablation
python experiments/hjb/phase2_interior_benchmark.py

# 10. §4.8 2D coupled portfolio
python experiments/hjb/phase3a_2d_audit.py
```

## Hardware/software protocol

- Python 3.11
- PyTorch 2.9.1 (CPU)
- SciPy ≥ 1.10, NumPy ≥ 1.24, pandas ≥ 2.0
- Apple M1 (CPU only); no CUDA/MPS dependencies
- Single-process; PyTorch's default CPU thread pool

## Frozen seeds

- 5-seed audits (§4.1, §4.2, §4.7 full method): `{42, 123, 999, 7, 2024}`
- 3-seed audits (§4.7 saturating ablation, §4.8 2D, per-component diagnostic): `{42, 123, 999}`
- Single-seed runs (S&P appendix, surface eval, scalar baseline): `42`
- Set inside each script via `numpy.random.seed` and `torch.manual_seed`.

## S&P 500 data

This repository does not redistribute S&P 500 index levels or derived return
series. The empirical calibration script expects a user-supplied CSV containing
daily dates and close levels obtained from a lawful data source. The included
schema example is synthetic and is provided only to document the expected file
format.

The paper's reported S&P 500 calibration can be reproduced by placing a lawfully
obtained daily close series at `supplement/data/sp500_user.csv` and running:

    python supplement/data/prepare_sp500_returns.py \
        --input supplement/data/sp500_user.csv \
        --output supplement/data/sp500_returns_local.csv
    python supplement/experiments/hjb/run_sp500_mle.py \
        --returns supplement/data/sp500_returns_local.csv

For exact auditability without redistributing third-party market data, the
fitted MLE parameters used in the paper are stored in
`supplement/data/mle_parameters.json`.
