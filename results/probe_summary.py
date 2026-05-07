"""probe_summary.py — produce a text summary of headline numbers from the CSVs.

Reads:
    /workspace/probes/results/probe_results.csv
    /workspace/probes/results/ood_metrics.csv

Writes (overwrites each run):
    /workspace/probes/results/RESULTS_NUMBERS.md

Pure pandas — no Isaac Sim. Fast.
"""

import argparse
import functools
import os

print = functools.partial(print, flush=True)

import numpy as np
import pandas as pd


AGENTS = ("A", "B", "C")
MILESTONES = (50, 75, 100)
MDE = 0.08


def fmt(x):
    if pd.isna(x):
        return "—"
    return f"{x:.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes_csv", default="/workspace/probes/results/probe_results.csv")
    ap.add_argument("--ood_csv", default="/workspace/probes/results/ood_metrics.csv")
    ap.add_argument("--out", default="/workspace/probes/results/RESULTS_NUMBERS.md")
    args = ap.parse_args()

    probes = pd.read_csv(args.probes_csv)
    print(f"[summary] probes: {len(probes)} rows")
    ood = pd.read_csv(args.ood_csv) if os.path.exists(args.ood_csv) else pd.DataFrame()

    lines = ["# Headline numbers from the probe pipeline", ""]
    lines.append(f"Generated from `{os.path.basename(args.probes_csv)}` and "
                 f"`{os.path.basename(args.ood_csv)}` — {len(probes)} probe rows, {len(ood)} ood rows.")
    lines.append("")

    # ── Block 1: linear-probe at 100% ─────────────────────────────────────
    lines.append("## Block 1 — Linear probe R² at 100% checkpoint (Layer 2)")
    lines.append("")
    lines.append("| Agent | mean R² | std (n=3) | min | max |")
    lines.append("|---|---|---|---|---|")
    means_100 = {}
    for ag in AGENTS:
        sub = probes[(probes["agent"] == ag) & (probes["milestone"] == 100) & (probes["probe_type"] == "linear_l2")]["r2"]
        if len(sub) == 0:
            lines.append(f"| {ag} | — | — | — | — |")
            continue
        means_100[ag] = sub.mean()
        lines.append(f"| {ag} | {fmt(sub.mean())} | {fmt(sub.std(ddof=1)) if len(sub) > 1 else '—'} | "
                     f"{fmt(sub.min())} | {fmt(sub.max())} |")
    lines.append("")

    # Floor + ceiling
    floor = probes[probes["probe_type"] == "random_l2"]["r2"]
    ceiling = probes[probes["probe_type"] == "raw_input"]["r2"]
    lines.append(f"- Random-weight floor R² (n={len(floor)}): mean = **{fmt(floor.mean())}**" +
                 (f", std = {fmt(floor.std(ddof=1))}" if len(floor) > 1 else ""))
    lines.append(f"- Raw-input ceiling R² (n={len(ceiling)}): mean = **{fmt(ceiling.mean())}**" +
                 (f", std = {fmt(ceiling.std(ddof=1))}" if len(ceiling) > 1 else ""))
    lines.append("")

    # ΔR² and pattern
    if "B" in means_100 and "C" in means_100:
        d_cb = means_100["C"] - means_100["B"]
        lines.append(f"- **ΔR²(C − B) at 100% = {d_cb:+.4f}** (MDE = {MDE})")
    if "A" in means_100 and "C" in means_100:
        d_ca = means_100["C"] - means_100["A"]
        lines.append(f"- **ΔR²(C − A) at 100% = {d_ca:+.4f}**")
    lines.append("")

    # Pattern detection
    if "B" in means_100 and "C" in means_100:
        d_cb = means_100["C"] - means_100["B"]
        f0 = float(floor.mean()) if len(floor) > 0 else 0.0
        all_close_to_floor = all(abs(means_100[ag] - f0) < MDE for ag in means_100)
        if d_cb >= MDE:
            pat = "**P1: curriculum trace detected** (C significantly above B at 100%)"
        elif abs(d_cb) < MDE and not all_close_to_floor:
            pat = "**P2: representational convergence / plasticity loss** (B ≈ C at 100%)"
        elif all_close_to_floor:
            pat = "**P4: observability gap** (all agents close to random-weight floor)"
        else:
            pat = "*(P3 primacy bias check: examine 50%/75% trajectory in Block 2)*"
        lines.append(f"- Pre-registered pattern → {pat}")
    lines.append("")

    # ── Block 2: temporal trajectory ──────────────────────────────────────
    lines.append("## Block 2 — Linear probe R² across milestones (mean ± std, n=3 seeds)")
    lines.append("")
    lines.append("| Agent | 50% | 75% | 100% |")
    lines.append("|---|---|---|---|")
    for ag in AGENTS:
        cells = []
        for ms in MILESTONES:
            sub = probes[(probes["agent"] == ag) & (probes["milestone"] == ms) & (probes["probe_type"] == "linear_l2")]["r2"]
            if len(sub) == 0:
                cells.append("—")
            elif len(sub) == 1:
                cells.append(f"{sub.values[0]:.3f} (n=1)")
            else:
                cells.append(f"{sub.mean():.3f} ± {sub.std(ddof=1):.3f}")
        lines.append(f"| {ag} | {cells[0]} | {cells[1]} | {cells[2]} |")
    lines.append("")

    # ── Block 3: linear vs MLP at 100% ────────────────────────────────────
    lines.append("## Block 3 — Linear vs MLP at 100% (Layer 2)")
    lines.append("")
    lines.append("| Agent | linear R² | MLP R² | ΔR² (MLP − linear) |")
    lines.append("|---|---|---|---|")
    for ag in AGENTS:
        lin = probes[(probes["agent"] == ag) & (probes["milestone"] == 100) & (probes["probe_type"] == "linear_l2")].set_index("seed")["r2"]
        mlp = probes[(probes["agent"] == ag) & (probes["milestone"] == 100) & (probes["probe_type"] == "mlp_l2")].set_index("seed")["r2"]
        common = lin.index.intersection(mlp.index)
        if len(common) == 0:
            lines.append(f"| {ag} | — | — | — |")
            continue
        lin_m = lin.loc[common].mean()
        mlp_m = mlp.loc[common].mean()
        delta = (mlp.loc[common] - lin.loc[common]).mean()
        lines.append(f"| {ag} | {lin_m:.3f} | {mlp_m:.3f} | {delta:+.3f}" +
                     (f" (MDE=±{MDE})" if abs(delta) >= MDE else "") + " |")
    lines.append("")

    # ── Block 4: OOD behavioural ──────────────────────────────────────────
    if not ood.empty:
        lines.append("## Block 4 — OOD completion rates at 100% (mean across seeds)")
        lines.append("")
        lines.append("| µ | Agent A | Agent B | Agent C |")
        lines.append("|---|---|---|---|")
        sub = ood[ood["milestone"] == 100]
        for mu in sorted(sub["mu"].unique()):
            cells = []
            for ag in AGENTS:
                vals = sub[(sub["agent"] == ag) & (np.isclose(sub["mu"], mu))]["completion_rate"]
                cells.append(f"{vals.mean():.2f}" if len(vals) > 0 else "—")
            lines.append(f"| {mu:.3f} | {cells[0]} | {cells[1]} | {cells[2]} |")
        lines.append("")

    # ── Block 5: methodology vigilance ────────────────────────────────────
    lines.append("## Block 5 — Methodology vigilance")
    lines.append("")
    lines.append("- Production excludes the broken Agent C run (`AGENT C FULL` / W&B `quug7tn8`); "
                 "see OPERATING_RULES.md hard rule 4. Bit-exact A=C in that broken run is preserved in "
                 "`/workspace/viva_data/` as evidence for §4.2 only.")
    lines.append("- Two seeds (A s2 = `sweet-tree-106`, B s2 = `magic-smoke-105`) trained at "
                 "num_envs=256 vs others' 512 — see §3.9 paragraph in handoff.")
    lines.append("")

    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[summary] wrote {args.out} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
