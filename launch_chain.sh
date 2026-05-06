#!/bin/bash
# Chained launcher: runs SEEDS sequentially for AGENT.
# Usage: AGENT=A SEEDS="1 2" bash launch_chain.sh
set -eu

: "${AGENT:?Set AGENT=A|B|C}"
: "${SEEDS:?Set SEEDS=\"1 2\" or similar}"

LOG_DIR=/workspace/production_logs
mkdir -p "$LOG_DIR"
CHAIN_LOG="$LOG_DIR/chain_${AGENT}_$(date +%s).log"

(
    for SEED in $SEEDS; do
        echo "[$(date)] Starting agent${AGENT} seed${SEED}..."
        AGENT="$AGENT" SEED="$SEED" bash /workspace/WheeledLab/launch_production.sh
        PID=$(cat "$LOG_DIR/prod_agent${AGENT}_seed${SEED}.pid")
        echo "[$(date)] Waiting for PID $PID to finish..."
        while kill -0 "$PID" 2>/dev/null; do sleep 60; done
        echo "[$(date)] agent${AGENT} seed${SEED} complete."
        sleep 30
    done
    echo "[$(date)] CHAIN COMPLETE for AGENT=${AGENT}, SEEDS=${SEEDS}"
) > "$CHAIN_LOG" 2>&1 &

CHAIN_PID=$!
echo "Chain launched as PID $CHAIN_PID"
echo "Chain log: $CHAIN_LOG"
echo "Monitor: tail -f $CHAIN_LOG"
