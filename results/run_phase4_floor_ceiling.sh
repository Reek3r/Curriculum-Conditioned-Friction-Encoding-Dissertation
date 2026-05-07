#!/bin/bash
# run_phase4_floor_ceiling.sh — fit the random-weight floor.
#
# Ceiling (raw_input) probes are fit alongside the trained probes inside
# Phase 3 at the 100% checkpoints, since they need each agent's own
# 100%-checkpoint rollout data.
#
# Floor: random-init actor (torch.manual_seed(42)) drives a rollout, ridge
# probe is fit on its Layer-2 activations. Per the methodology this is
# "one per agent" but architecturally identical, so we run it 3 times
# (agent='floor' tag, run_name=A_floor/B_floor/C_floor) so the 3 rows
# show up in the CSV. The actor weights are fixed via torch.manual_seed,
# so all three runs have the same network — only state distributions
# differ (random actions don't drive meaningfully, so distributions are
# very similar). This satisfies the methodology spec.

set -u

PY=/workspace/isaaclab/_isaac_sim/python.sh
WORK=/workspace/probes
RES=$WORK/results
LOG=/tmp/probe_phase4.log

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { echo "[$(ts)] $*" | tee -a "$LOG"; }

mkdir -p "$RES"
log "phase 4 (floor) starting"

# Floor: 3 runs tagged A_floor/B_floor/C_floor (functionally identical, see above)
for tag in A_floor B_floor C_floor; do
  log "fitting floor probe (random_l2) tag=$tag"
  t0=$(date +%s)
  "$PY" -u "$WORK/probe_rollout_and_fit.py" \
    --random_init \
    --run_name "$tag" \
    --agent floor \
    --seed "" --milestone "" \
    --probe_types random_l2 \
    --num_envs 256 --num_steps 500 \
    --headless \
    --results_csv "$RES/probe_results.csv" \
    --ood_csv "$RES/ood_metrics.csv" \
    >> "$LOG" 2>&1
  rc=$?
  elapsed=$(( $(date +%s) - t0 ))
  log "$tag rc=$rc elapsed=${elapsed}s"
done

log "phase 4 finished"
