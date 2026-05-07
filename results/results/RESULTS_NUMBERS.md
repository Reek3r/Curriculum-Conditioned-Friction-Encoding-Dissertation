# Headline numbers from the probe pipeline

Generated from `probe_results.csv` and `ood_metrics.csv` — 48 probe rows, 600 ood rows.

## Block 1 — Linear probe R² at 100% checkpoint (Layer 2)

| Agent | mean R² | std (n=3) | min | max |
|---|---|---|---|---|
| A | 0.5334 | 0.0313 | 0.5088 | 0.5686 |
| B | 0.3697 | 0.0662 | 0.3266 | 0.4459 |
| C | 0.3766 | 0.0163 | 0.3626 | 0.3945 |

- Random-weight floor R² (n=3): mean = **0.0099**, std = 0.0000
- Raw-input ceiling R² (n=9): mean = **0.3967**, std = 0.0848

- **ΔR²(C − B) at 100% = +0.0069** (MDE = 0.08)
- **ΔR²(C − A) at 100% = -0.1568**

- Pre-registered pattern → **P2: representational convergence / plasticity loss** (B ≈ C at 100%)

## Block 2 — Linear probe R² across milestones (mean ± std, n=3 seeds)

| Agent | 50% | 75% | 100% |
|---|---|---|---|
| A | 0.498 ± 0.049 | 0.524 ± 0.036 | 0.533 ± 0.031 |
| B | 0.368 ± 0.090 | 0.365 ± 0.063 | 0.370 ± 0.066 |
| C | 0.342 ± 0.105 | 0.369 ± 0.075 | 0.377 ± 0.016 |

## Block 3 — Linear vs MLP at 100% (Layer 2)

| Agent | linear R² | MLP R² | ΔR² (MLP − linear) |
|---|---|---|---|
| A | 0.533 | 0.667 | +0.134 (MDE=±0.08) |
| B | 0.370 | 0.541 | +0.171 (MDE=±0.08) |
| C | 0.377 | 0.533 | +0.156 (MDE=±0.08) |

## Block 4 — OOD completion rates at 100% (mean across seeds)

| µ | Agent A | Agent B | Agent C |
|---|---|---|---|
| 0.200 | 0.16 | 0.88 | 0.73 |
| 0.242 | 0.05 | 0.88 | 0.70 |
| 0.284 | 0.01 | 0.90 | 0.72 |
| 0.326 | 0.01 | 0.87 | 0.72 |
| 0.368 | 0.06 | 0.92 | 0.74 |
| 0.411 | 0.15 | 0.91 | 0.77 |
| 0.453 | 0.33 | 0.91 | 0.85 |
| 0.495 | 0.54 | 0.92 | 0.86 |
| 0.537 | 0.69 | 0.91 | 0.84 |
| 0.579 | 0.82 | 0.91 | 0.83 |
| 0.621 | 0.85 | 0.88 | 0.84 |
| 0.663 | 0.86 | 0.88 | 0.83 |
| 0.705 | 0.89 | 0.86 | 0.83 |
| 0.747 | 0.91 | 0.85 | 0.78 |
| 0.789 | 0.91 | 0.84 | 0.77 |
| 0.832 | 0.90 | 0.84 | 0.76 |
| 0.874 | 0.93 | 0.82 | 0.74 |
| 0.916 | 0.91 | 0.80 | 0.74 |
| 0.958 | 0.92 | 0.81 | 0.73 |
| 1.000 | 0.90 | 0.79 | 0.73 |

## Block 5 — Methodology vigilance

- Production excludes the broken Agent C run (`AGENT C FULL` / W&B `quug7tn8`); see OPERATING_RULES.md hard rule 4. Bit-exact A=C in that broken run is preserved in `/workspace/viva_data/` as evidence for §4.2 only.
- Two seeds (A s2 = `sweet-tree-106`, B s2 = `magic-smoke-105`) trained at num_envs=256 vs others' 512 — see §3.9 paragraph in handoff.

