#!/bin/bash
# run_phase6_sync.sh — package /workspace/probes/ into a tarball and a git
# feature branch.
#
# Run on step-5 inside the vscode container. Output:
#   /tmp/probe_artefacts.tar.gz  — sync target for laptop
#   /workspace/WheeledLab/probes/ — repo subtree on feature branch
#                                   probe-pipeline-2026-05-07
#
# This script does NOT push to GitHub — that's left to the user via the
# remote that's already configured.

set -u

WORK=/workspace/probes
REPO=/workspace/WheeledLab
BRANCH=probe-pipeline-2026-05-07

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { echo "[$(ts)] $*"; }

# 1) Tarball of artefacts (CSVs + figures + scripts + status docs)
log "tarring /workspace/probes/ → /tmp/probe_artefacts.tar.gz"
cd /workspace
tar czf /tmp/probe_artefacts.tar.gz \
    probes/results/ \
    probes/figures/ \
    probes/STATUS.md \
    probes/probe_rollout_and_fit.py \
    probes/probe_figures.py \
    probes/probe_summary.py \
    probes/run_phase3.sh \
    probes/run_phase4_floor_ceiling.sh \
    probes/run_phase6_sync.sh \
    probes/diag_01_inspect_checkpoint.py \
    probes/diag_02_env_friction.py \
    probes/diag_03_activation_hook.py \
    probes/DEVIATION_fig3.md \
    probes/STEP5_DEPS_INSTALLED.md \
    probes/PHASE3_LAUNCH.md 2>&1 | tail -5
log "tarball: $(ls -lh /tmp/probe_artefacts.tar.gz)"

# 2) Copy probes subtree into the WheeledLab repo and commit on a feature branch
log "copying probes subtree into $REPO/probes/"
mkdir -p "$REPO/probes"
cp -r "$WORK/results" "$REPO/probes/"
cp -r "$WORK/figures" "$REPO/probes/"
cp "$WORK/"*.md "$REPO/probes/" 2>/dev/null || true
cp "$WORK/"*.sh "$REPO/probes/"
cp "$WORK/"*.py "$REPO/probes/"

cd "$REPO"
log "git status (pre):"
git status --short | head -20

CURRENT=$(git branch --show-current)
log "current branch: $CURRENT"
if [ "$CURRENT" != "$BRANCH" ]; then
    log "creating + switching to $BRANCH"
    git checkout -b "$BRANCH" 2>&1 | head -5 || git checkout "$BRANCH" 2>&1 | head -5
fi

git add probes/
log "git status (post-add):"
git status --short | head -20

# Use HEREDOC for commit message (no Co-Authored-By line — operating rules
# say to keep the user's git config untouched)
git commit -m "[probes] Probe Triangulation Framework results, figures, scripts (2026-05-07)

Add /workspace/probes/ subtree containing:
- probe_rollout_and_fit.py: per-checkpoint rollout + 4-component probe fit
- probe_figures.py: 5 dissertation figures (PNG 300dpi + PDF)
- probe_summary.py: §4 numbers extraction
- run_phase3.sh / run_phase4_floor_ceiling.sh: production wrappers
- diag_01/02/03: phase-1 architecture, friction round-trip, hook tests
- results/probe_results.csv: 45+ probe fits
- results/ood_metrics.csv: per-friction completion / lap-time / OOB / timeout
- figures/: fig1-5 (R² vs milestone, linear vs MLP, completion vs µ,
  lap time vs µ, ΔR² nonlinear)
- DEVIATION_fig3.md: rationale for substituting per-µ R² figure
- STEP5_DEPS_INSTALLED.md: scikit-learn install (user-level)" 2>&1 | tail -10

log "git log (last 3 on branch):"
git log --oneline -3

# 3) Write BRANCH.md
cat > "$WORK/BRANCH.md" <<EOF
# Feature branch for probe pipeline artefacts

Branch: \`$BRANCH\`
Commit: \`$(git rev-parse --short HEAD)\`
Repo: github.com/Reek3r/WheeledLab (private)

This branch contains the /workspace/probes/ subtree as repo path \`probes/\`.
The user reviews and merges. Do NOT push to main; the brief explicitly forbids it.

To push to GitHub (manual step by the user):
\`\`\`
cd /workspace/WheeledLab
git push -u origin $BRANCH
\`\`\`
EOF
log "wrote BRANCH.md"

log "phase 6 finished"
