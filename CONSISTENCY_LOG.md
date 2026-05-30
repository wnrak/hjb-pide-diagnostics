# Consistency log

What was checked, what was found, what was fixed. Run this sweep before any
"frozen" claim.

## Reviewer-flagged bugs (2026-04-28 pass)

| # | Issue | Where | Status |
|---|---|---|---|
| 3 | "first-order optimality conditions" prose bullet contradicted the corrected formulation | `hjb_pide_levy.tex:103` | fixed |
| 4 | additive `V(t,x+uz)` survived in the PINN-difficulty paragraph | `hjb_pide_levy.tex:97` | fixed |
| 5 | `eq:proposal` not normalized; mismatch between paper density and code's clamp-and-go sampler | `hjb_pide_levy.tex:250–258` | fixed |
| 8 | `figures/vg_calibration.pdf` shipped with old "Empirical (VG-simulated)" label; QQ panel still used `vg_samples` not real returns | `paper/generate_calibration_figures.py:154–157` plus shipped PDF | fixed (script edited + figure regenerated + synced into submission/) |
| 9 | Conclusion + §4.7 prose said removing compensator drops `u` from 0.99 to 0.72 (~28pp); table showed 0.921 → 0.410 (~51pp) | `hjb_pide_levy.tex:547`, `hjb_pide_levy.tex:732` | fixed |

## Mechanical sweep (2026-04-28)

For each term/regex below, I greps both `hjb_pide_levy.tex` and
`appendix_old_vs_audited.tex`, then verified each hit was either correct
or explicitly contextualized as historical. Run via the consistency
sweep at the bottom of this file.

| Sweep | Regex | Hits | Verdict |
|---|---|---|---|
| A. additive form leftovers | `x\+uz`, `V\(t, x\+`, `x \+ uz` | 0 (after fix 4) | clean |
| B. FOC leftovers | `first-order optimality cond`, `FOC loss`, `partial_u.*= 0` | 3 in main, all explicitly framed as the wrong-and-replaced approach | clean |
| C. self-normalized wording | `self-normalized`, `self normalize` | 2 in main + 1 in appendix, all describing the OLD 2D bug as a historical artifact | clean |
| D. stale headline numbers (0.475, 0.999, 1.317, 1.468) | exact-string grep | 0.475 only in §4.5 head-to-head Phase-1 history row + adjacent prose; 1.468 only in §4.4 explanatory paragraph noting "earlier drafts reported" | clean |
| E. compensation-shift "0.72" / "28-point" | `0\.72\b`, `~?28[- ]point` | 0.72 only as VaR_5 row in §4.3a sensitivity table and as "skewness = -0.72"; "28%" only as a row reduction column in §4.3a; no 28-point compensator-shift mentions | clean (after fix 9) |
| F. directional only / not supported / degenerate | exact strings | 1 hit in §4.8 prose explicitly saying "removing the previous draft's `directional only' caveat" — current/correct framing | clean |
| G. rerun pending / TBD / outside the scope | exact strings | 0 | clean |
| H. "Empirical (VG-simulated)" literal | exact string | 0 (only as a comment in `generate_calibration_figures.py` explaining the historical bug) | clean |

## Numeric claims → source JSON map

This is the audit map: every numeric claim in the paper text traces to
a result file in `results/`. Anyone re-checking the paper should be
able to verify each claim by `cat`ing the corresponding JSON.

| Section | Claim | JSON file |
|---|---|---|
| §4.1 (Diffusion-only Merton) | $u = 0.762 \pm 0.011$ | `results/phase1_audit/audit_results.json:diffusion_only_merton.u_mean,u_std` |
| §4.2 (VG, post-fix) | $u = 0.344 \pm 0.006$, $-54\%$ | `results/phase1_audit/audit_results.json:vg_jumps.u_mean,u_std` (re-run after IS fix; same script, same JSON path) |
| §4.5 (FD v1 spatial sweep) | $u_{FD} \in [0.347, 0.352]$ | `results/phase1_audit/fd_pide_audit.json:fd_vg_grid_refinement[*].u_at_0_1` |
| §4.5 (FD v2 finest) | $u_{FD,v2} = 0.344$ | `results/phase3a/phase3a_audit.json:fd_v2_grid_refinement[-1].u_at_0_1` |
| §4.5 (per-component diagnostic, neural compensator = 0.502 × FD) | constant ratio across u-grid | `results/phase3a/phase3a_audit.json:breakdown_FDv1`, `breakdown_FDv2`, `breakdown_neural` |
| §4.4 (S&P MLE policy, post-fix) | $u_{VG} = 0.949$, $+0.8$ pp VaR, $+1.0$ pp CVaR | `results/post_fix_audit/sp500_mle_policy/results.json:controls,wealth_metrics_under_common_vg_model` |
| §4.7 (saturating ablation, post-fix) | full $0.921 \pm 0.071$, no comp $0.410 \pm 0.067$ | `results/post_fix_audit/ablation/results.json:variants[*]` |
| §4.7 Phase 2.2 (interior, post-fix) | full $0.277 \pm 0.004$, FD ref $0.276$, no comp $0.006 \pm 0.002$ | `results/phase2_interior/phase2_interior.json:full_method_audit,fd_reference,ablation` |
| §4.8 (2D coupled, post-fix) | $u_1 = 0.348 \pm 0.005$, $u_2 = 0.258 \pm 0.005$ | `results/phase3a_2d/phase3a_2d_audit.json:learned_summary` |

## Consistency-sweep command (rerun this before declaring frozen)

```bash
cd paper/

echo "=== additive form ==="; grep -nE "x\+uz|x\+u z|V\(t, x\+|x \+ uz" hjb_pide_levy.tex appendix_old_vs_audited.tex || echo "  clean"
echo "=== FOC residual loss leftovers ==="; grep -nE "first-order optimality cond|FOC loss" hjb_pide_levy.tex appendix_old_vs_audited.tex || echo "  clean"
echo "=== self-normalized in current method ==="; grep -nE "self-normalized" hjb_pide_levy.tex | grep -v "previous\|previously\|earlier draft\|2D solver originally\|legacy\|old"
echo "=== rerun pending / TBD ==="; grep -nE "rerun pending|^.*TBD|outside the scope of this paper" hjb_pide_levy.tex appendix_old_vs_audited.tex || echo "  clean"
echo "=== VG-simulated literal ==="; grep -rn "VG-simulated" hjb_pide_levy.tex appendix_old_vs_audited.tex || echo "  clean"
echo "=== shipped figure md5 vs source-script output ==="; md5 -q figures/vg_calibration.pdf submission/figures/vg_calibration.pdf
echo "=== PDF metadata ==="; pdfinfo submission/hjb_pide_levy.pdf | head -5
```

## Build state (2026-04-28)

- 27 pages, 484 KB
- 0 undefined citations / undefined references
- 6 overfull `\hbox` warnings, max 30 pt (typesetting nits, not page margin)
- PDF metadata populated (title, author, subject, keywords)
- `paper/hjb_pide_levy.tex` and `paper/submission/hjb_pide_levy.tex` byte-identical
- `paper/figures/vg_calibration.pdf` and `paper/submission/figures/vg_calibration.pdf` byte-identical
