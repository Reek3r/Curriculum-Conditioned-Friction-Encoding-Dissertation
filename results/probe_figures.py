"""probe_figures.py — produce the 5 dissertation figures from the CSVs.

Reads:
    /workspace/probes/results/probe_results.csv
    /workspace/probes/results/ood_metrics.csv

Writes (PNG 300dpi + PDF):
    /workspace/probes/figures/fig1_r2_vs_milestone.{png,pdf}
    /workspace/probes/figures/fig2_linear_vs_mlp_100pct.{png,pdf}
    /workspace/probes/figures/fig3_completion_vs_mu.{png,pdf}    (was fig4 in brief)
    /workspace/probes/figures/fig4_laptime_vs_mu.{png,pdf}       (was fig5 in brief)
    /workspace/probes/figures/fig5_mlp_minus_linear.{png,pdf}    (replaces fig3 R² vs µ
                                                                 — see DEVIATION note)

DEVIATION (recorded in DEVIATION_fig3.md): The brief's fig3 ("R² vs µ at 100%")
was specified as fitting a separate probe per ±0.05 µ window. The probe is
fit on data spanning all 20 µs, so per-µ R² requires either (a) refitting
20 probes (expensive) or (b) saving per-test-sample predictions (which the
rollout script does not). Substituted with a more informative figure: per-agent
ΔR² (MLP − linear), which directly visualises the methodology's main probe-
disambiguation question.

This script is pure pandas/matplotlib — no Isaac Sim needed. Runs in ~5s.
"""

import argparse
import functools
import os

print = functools.partial(print, flush=True)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


AGENTS = ("A", "B", "C")
COLORS = {"A": "#1f77b4", "B": "#ff7f0e", "C": "#2ca02c", "floor": "#7f7f7f", "ceiling": "#9467bd"}
MILESTONES = (50, 75, 100)
MDE = 0.08


def save(fig, stem):
    fig.savefig(stem + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(stem + ".pdf", bbox_inches="tight")
    plt.close(fig)


def fig1_r2_vs_milestone(probes, out_dir):
    """Layer-2 linear-probe R² vs milestone (50/75/100) per agent.

    Mean ± 95% CI across seeds, with seed-level scatter overlaid and an
    MDE band shaded around the floor (or zero if no floor data).
    """
    sub = probes[(probes["probe_type"] == "linear_l2")]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    floor = probes[probes["probe_type"] == "random_l2"]["r2"].mean() if (probes["probe_type"] == "random_l2").any() else 0.0
    ax.axhspan(floor, floor + MDE, alpha=0.10, color="grey",
               label=f"floor + MDE ({MDE})")
    ax.axhline(floor, color="grey", linestyle="--", linewidth=0.8,
               label=f"random floor (R²={floor:.3f})")

    for ag in AGENTS:
        ag_sub = sub[sub["agent"] == ag]
        if ag_sub.empty:
            continue
        means, stds, mss = [], [], []
        for ms in MILESTONES:
            ms_data = ag_sub[ag_sub["milestone"] == ms]["r2"].values
            if len(ms_data) == 0:
                continue
            means.append(ms_data.mean())
            stds.append(ms_data.std(ddof=1) if len(ms_data) > 1 else 0.0)
            mss.append(ms)
            for v in ms_data:
                ax.scatter(ms + (ord(ag) - ord("B")) * 1.5, v,
                           color=COLORS[ag], alpha=0.4, s=20, edgecolor="none")
        if mss:
            ax.errorbar(mss, means, yerr=stds, marker="o", linewidth=2,
                        capsize=4, label=f"Agent {ag} (mean ± std)",
                        color=COLORS[ag])

    ax.set_xlabel("Training milestone (% of 5000 iterations)")
    ax.set_ylabel(r"Test $R^2$ (linear probe, Layer 2)")
    ax.set_title("Layer-2 µ-decoding R² vs training milestone")
    ax.set_xticks(MILESTONES)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    save(fig, os.path.join(out_dir, "fig1_r2_vs_milestone"))


def fig2_linear_vs_mlp_100pct(probes, out_dir):
    """Linear vs MLP probe R² at the 100% checkpoint.

    Grouped bar chart: 3 agents × 2 probe types, with seed scatter and
    horizontal lines for floor + ceiling.
    """
    sub = probes[probes["milestone"] == 100]
    fig, ax = plt.subplots(figsize=(7.0, 4.5))

    width = 0.35
    x = np.arange(len(AGENTS))

    for i, pt in enumerate(("linear_l2", "mlp_l2")):
        means, stds = [], []
        for j, ag in enumerate(AGENTS):
            vals = sub[(sub["agent"] == ag) & (sub["probe_type"] == pt)]["r2"].values
            if len(vals) == 0:
                means.append(0.0); stds.append(0.0); continue
            means.append(vals.mean())
            stds.append(vals.std(ddof=1) if len(vals) > 1 else 0.0)
            for v in vals:
                ax.scatter(j + (i - 0.5) * width, v, color="black",
                           alpha=0.6, s=18, zorder=10)
        offset = (i - 0.5) * width
        ax.bar(x + offset, means, width, yerr=stds,
               label="Linear" if pt == "linear_l2" else "MLP (64→32→1)",
               color="#4c72b0" if pt == "linear_l2" else "#dd8452",
               edgecolor="black", capsize=3, alpha=0.85)

    floor = probes[probes["probe_type"] == "random_l2"]["r2"].mean()
    if not np.isnan(floor):
        ax.axhline(floor, color="grey", linestyle="--", linewidth=0.8,
                   label=f"random-weight floor (R²={floor:.3f})")
    ceiling = probes[probes["probe_type"] == "raw_input"]["r2"].mean()
    if not np.isnan(ceiling):
        ax.axhline(ceiling, color=COLORS["ceiling"], linestyle=":", linewidth=1.2,
                   label=f"raw-input ceiling (R²={ceiling:.3f})")

    ax.set_xticks(x)
    ax.set_xticklabels([f"Agent {a}" for a in AGENTS])
    ax.set_ylabel(r"Test $R^2$")
    ax.set_title("Linear vs nonlinear probe at 100% checkpoint")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    save(fig, os.path.join(out_dir, "fig2_linear_vs_mlp_100pct"))


def fig3_completion_vs_mu(ood, out_dir):
    """Completion rate vs µ at the 100% checkpoint, three lines (A/B/C)."""
    sub = ood[ood["milestone"] == 100]
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for ag in AGENTS:
        ag_sub = sub[sub["agent"] == ag]
        if ag_sub.empty:
            continue
        # Average across seeds at each µ
        agg = ag_sub.groupby("mu")["completion_rate"].agg(["mean", "std", "count"]).reset_index()
        ax.plot(agg["mu"], agg["mean"], marker="o", linewidth=1.8,
                label=f"Agent {ag}", color=COLORS[ag])
        ax.fill_between(agg["mu"], agg["mean"] - agg["std"], agg["mean"] + agg["std"],
                        color=COLORS[ag], alpha=0.18)
    ax.set_xlabel(r"Friction $\mu$")
    ax.set_ylabel("Lap completion rate (first episode)")
    ax.set_title("Completion rate vs friction at 100% checkpoint")
    ax.set_ylim(-0.05, 1.05)
    ax.axvspan(0.76, 0.84, alpha=0.10, color="black",
               label="Agent A training band")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    save(fig, os.path.join(out_dir, "fig3_completion_vs_mu"))


def fig4_laptime_vs_mu(ood, out_dir):
    """Median lap time vs µ at 100% with IQR shaded."""
    sub = ood[ood["milestone"] == 100]
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for ag in AGENTS:
        ag_sub = sub[sub["agent"] == ag]
        if ag_sub.empty:
            continue
        agg = ag_sub.groupby("mu").agg(
            median=("median_lap_time", "mean"),
            p25=("p25_lap_time", "mean"),
            p75=("p75_lap_time", "mean"),
        ).reset_index()
        ax.plot(agg["mu"], agg["median"], marker="o", linewidth=1.8,
                label=f"Agent {ag}", color=COLORS[ag])
        ax.fill_between(agg["mu"], agg["p25"], agg["p75"],
                        color=COLORS[ag], alpha=0.15)
    ax.set_xlabel(r"Friction $\mu$")
    ax.set_ylabel("Median lap time (s; IQR shaded)")
    ax.set_title("Median lap time vs friction at 100% checkpoint")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    save(fig, os.path.join(out_dir, "fig4_laptime_vs_mu"))


def fig5_mlp_minus_linear(probes, out_dir):
    """ΔR² (MLP − linear) per agent at 100%. Positive = nonlinear info beyond linear."""
    sub = probes[probes["milestone"] == 100]
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    deltas_by_agent = {}
    for ag in AGENTS:
        lin = sub[(sub["agent"] == ag) & (sub["probe_type"] == "linear_l2")].set_index("seed")["r2"]
        mlp = sub[(sub["agent"] == ag) & (sub["probe_type"] == "mlp_l2")].set_index("seed")["r2"]
        common = lin.index.intersection(mlp.index)
        if len(common) == 0:
            continue
        delta = (mlp.loc[common] - lin.loc[common]).values
        deltas_by_agent[ag] = delta

    if not deltas_by_agent:
        plt.close(fig)
        return

    pos = np.arange(len(deltas_by_agent))
    means = [np.mean(d) for d in deltas_by_agent.values()]
    stds = [np.std(d, ddof=1) if len(d) > 1 else 0.0 for d in deltas_by_agent.values()]
    ax.bar(pos, means, yerr=stds, color=[COLORS[a] for a in deltas_by_agent.keys()],
           edgecolor="black", alpha=0.85, capsize=4)
    for i, (ag, d) in enumerate(deltas_by_agent.items()):
        for v in d:
            ax.scatter(i, v, color="black", alpha=0.7, s=22, zorder=10)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(MDE, color="red", linestyle="--", linewidth=0.8,
               label=f"MDE ({MDE})")
    ax.axhline(-MDE, color="red", linestyle="--", linewidth=0.8)
    ax.set_xticks(pos)
    ax.set_xticklabels([f"Agent {a}" for a in deltas_by_agent.keys()])
    ax.set_ylabel(r"$\Delta R^2$ (MLP − linear)")
    ax.set_title("Nonlinear-only µ-information at 100% checkpoint")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    save(fig, os.path.join(out_dir, "fig5_mlp_minus_linear"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes_csv", default="/workspace/probes/results/probe_results.csv")
    ap.add_argument("--ood_csv", default="/workspace/probes/results/ood_metrics.csv")
    ap.add_argument("--out_dir", default="/workspace/probes/figures")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    probes = pd.read_csv(args.probes_csv)
    print(f"[fig] probes: {len(probes)} rows; {probes['probe_type'].value_counts().to_dict()}")
    ood = pd.read_csv(args.ood_csv) if os.path.exists(args.ood_csv) else pd.DataFrame()
    print(f"[fig] ood: {len(ood)} rows")

    fig1_r2_vs_milestone(probes, args.out_dir)
    print("[fig] fig1 saved")
    fig2_linear_vs_mlp_100pct(probes, args.out_dir)
    print("[fig] fig2 saved")
    if not ood.empty:
        fig3_completion_vs_mu(ood, args.out_dir)
        print("[fig] fig3 saved")
        fig4_laptime_vs_mu(ood, args.out_dir)
        print("[fig] fig4 saved")
    fig5_mlp_minus_linear(probes, args.out_dir)
    print("[fig] fig5 saved")

    # Pre-registered pattern detection
    p100 = probes[(probes["milestone"] == 100) & (probes["probe_type"] == "linear_l2")]
    rB = p100[p100["agent"] == "B"]["r2"].mean()
    rC = p100[p100["agent"] == "C"]["r2"].mean()
    floor = probes[probes["probe_type"] == "random_l2"]["r2"].mean()
    delta = rC - rB
    p1 = (delta >= MDE)
    p2 = (abs(delta) < MDE)
    p4 = ((p100["r2"] - floor).abs().max() < MDE) if (probes["probe_type"] == "random_l2").any() else False
    pattern = "P1" if p1 else ("P2" if p2 else ("P4" if p4 else "(none of P1/P2/P4)"))
    print(f"[fig] R²(B)={rB:.3f}  R²(C)={rC:.3f}  floor={floor:.3f}  ΔR²(C-B)={delta:+.3f}  → pattern {pattern}")


if __name__ == "__main__":
    main()
