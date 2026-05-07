#!/bin/bash
# run_phase3.sh — production rollout loop over the 27 milestone checkpoints.
#
# Invoke from inside the vscode container at /workspace/probes/, idempotent:
#     ./run_phase3.sh                       # Tier 1+2 (linear at 50/75/100, MLP at 100)
#     ./run_phase3.sh tier1                 # Tier 1 only (100% checkpoints, linear+MLP)
#
# The script writes per-checkpoint status to /tmp/probe_phase3.log AND updates
# /workspace/probes/STATUS.md every checkpoint. Each probe_rollout_and_fit.py
# invocation is independent, so a crash mid-loop just resumes (idempotent CSV
# dedup re-runs only missing rows).

set -u

PY=/workspace/isaaclab/_isaac_sim/python.sh
WORK=/workspace/probes
CKPT=$WORK/checkpoints
RES=$WORK/results
STATUS=$WORK/STATUS.md
LOG=/tmp/probe_phase3.log

# (run_dir, agent, seed)
RUNS=(
  "forgotten-transport-91 A 0"
  "lively-planet-95       A 1"
  "sweet-tree-106         A 2"
  "ancient-pilot-90       B 0"
  "firm-terrain-96        B 1"
  "magic-smoke-105        B 2"
  "pious-star-94          C 0"
  "flowing-pond-99        C 1"
  "glamorous-sunset-100   C 2"
)

if [ "${1:-tier12}" = "tier1" ]; then
  MILESTONES=(100)
else
  MILESTONES=(50 75 100)
fi

TOTAL=$(( ${#RUNS[@]} * ${#MILESTONES[@]} ))
N=0

mkdir -p "$RES"

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() {
  echo "[$(ts)] $*" | tee -a "$LOG"
}

write_status() {
  local n=$1 total=$2 last=$3
  cat > "$STATUS" <<EOF
$(ts) — phase 3 — $n/$total checkpoints done
last: $last
log: $LOG
csv: $RES/probe_results.csv
ood: $RES/ood_metrics.csv
EOF
}

log "phase 3 starting; $TOTAL checkpoints; ${#RUNS[@]} runs × ${#MILESTONES[@]} milestones"
write_status 0 "$TOTAL" "(none yet)"

for triplet in "${RUNS[@]}"; do
  read -r run agent seed <<< "$triplet"
  for ms in "${MILESTONES[@]}"; do
    case "$ms" in
      50)  ITER="iter2500_50pct" ;;
      75)  ITER="iter3750_75pct" ;;
      100) ITER="iter4999_100pct" ;;
      *)   log "skip: unknown milestone $ms"; continue ;;
    esac
    PT="$CKPT/$run/model_milestone_${ITER}.pt"
    if [ ! -f "$PT" ]; then
      log "MISSING $run $ms% — $PT not found, skipping"
      continue
    fi

    # Tier 1+2: linear at all milestones, MLP + raw-input ceiling at 100% only
    PROBES="linear_l2"
    if [ "$ms" = "100" ]; then
      PROBES="linear_l2,mlp_l2,raw_input"
    fi

    N=$((N + 1))
    log "[$N/$TOTAL] $run ${ms}% ($agent s$seed) probes=$PROBES — start"
    t0=$(date +%s)
    "$PY" -u "$WORK/probe_rollout_and_fit.py" \
      --checkpoint "$PT" \
      --run_name "$run" \
      --agent "$agent" \
      --seed "$seed" \
      --milestone "$ms" \
      --probe_types "$PROBES" \
      --num_envs 256 \
      --num_steps 500 \
      --headless \
      --results_csv "$RES/probe_results.csv" \
      --ood_csv "$RES/ood_metrics.csv" \
      >> "$LOG" 2>&1
    rc=$?
    elapsed=$(( $(date +%s) - t0 ))
    log "[$N/$TOTAL] $run ${ms}% — rc=$rc elapsed=${elapsed}s"
    write_status "$N" "$TOTAL" "$run ${ms}% rc=$rc (${elapsed}s)"
  done
done

log "phase 3 finished"
write_status "$N" "$TOTAL" "ALL DONE"
