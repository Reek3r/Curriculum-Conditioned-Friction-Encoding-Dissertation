#!/bin/bash
# Production training launcher.
# Usage: AGENT=A SEED=1 bash launch_production.sh
set -eu

: "${AGENT:?Set AGENT=A|B|C}"
: "${SEED:?Set SEED=0|1|2}"

case "$AGENT" in A|B|C) ;; *) echo "AGENT must be A, B, or C"; exit 1 ;; esac

LOG_DIR=/workspace/production_logs
mkdir -p "$LOG_DIR"
TAG="agent${AGENT}_seed${SEED}"
LOG="$LOG_DIR/prod_${TAG}.log"
PIDFILE="$LOG_DIR/prod_${TAG}.pid"

if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "ERROR: $TAG already running with PID $(cat "$PIDFILE")"
    exit 2
fi

cd /workspace/WheeledLab/source/wheeledlab_rl

TRAIN_SEED=${SEED} nohup /workspace/isaaclab/_isaac_sim/python.sh scripts/train_rl.py \
    -r "AGENT_${AGENT}_CONFIG" \
    --headless \
    train.log.video=false \
    train.log.run_name="prod_${TAG}" \
    env_setup.num_envs=512 \
    train.num_iterations=5000 \
    > "$LOG" 2>&1 &

PID=$!
echo "$PID" > "$PIDFILE"
echo "Launched ${TAG} as PID $PID"
echo "Log: $LOG"
echo "Monitor: tail -f $LOG"
