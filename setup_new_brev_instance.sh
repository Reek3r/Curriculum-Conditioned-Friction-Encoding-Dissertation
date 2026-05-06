#!/bin/bash
# One-shot setup for fresh Brev L40S instance running the IsaacLab Launchable.
# Run from inside `docker exec -it vscode bash`.
set -e

echo "=== STEP 1: Sanity check ==="
test -d /workspace/isaaclab || { echo "FAIL: /workspace/isaaclab missing — wrong Launchable?"; exit 1; }

echo ""
echo "=== STEP 2: Clone or update repo ==="
if [ ! -d /workspace/WheeledLab ]; then
    cd /workspace
    git clone https://github.com/Reek3r/WheeledLab.git
fi
cd /workspace/WheeledLab
git fetch --all --tags
git reset --hard origin/main
git log -1 --oneline

echo ""
echo "=== STEP 3: Run setup_brev (re-applies sim5 patch idempotently) ==="
bash setup_brev.sh

echo ""
echo "=== STEP 4: W&B credentials ==="
cat > ~/.netrc << 'NETRC_EOF'
machine api.wandb.ai
  login user
  password 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
NETRC_EOF
chmod 600 ~/.netrc

echo ""
echo "=== STEP 5: Verify patches landed ==="
echo -n "Sim 5.0 termination patch: "
grep -q "env.reset_buf.float()" /workspace/isaaclab/source/isaaclab/isaaclab/envs/mdp/rewards.py \
    && echo "OK" || { echo "FAIL: missing"; exit 2; }

echo -n "ADR bucket-regen fix:      "
grep -q "Regenerated buckets" /workspace/WheeledLab/source/wheeledlab_tasks/wheeledlab_tasks/drifting/mushr_drift_env.py \
    && echo "OK" || { echo "FAIL: missing"; exit 3; }

echo -n "ADR_THRESHOLD: "
grep "ADR_THRESHOLD\s*=" /workspace/WheeledLab/source/wheeledlab_tasks/wheeledlab_tasks/drifting/mushr_drift_env.py | head -1 | sed 's/^[[:space:]]*//'
echo -n "ADR_WINDOW:    "
grep "ADR_WINDOW\s*=" /workspace/WheeledLab/source/wheeledlab_tasks/wheeledlab_tasks/drifting/mushr_drift_env.py | head -1 | sed 's/^[[:space:]]*//'
echo -n "bonus_per_lap: "
grep -A1 'bonus_per_lap"' /workspace/WheeledLab/source/wheeledlab_tasks/wheeledlab_tasks/drifting/mushr_drift_env_cfg.py | tail -1 | sed 's/^[[:space:]]*//'

echo ""
echo "=== STEP 6: Production logs dir ==="
mkdir -p /workspace/production_logs

echo ""
echo "=== STEP 7: 1-iter smoke test (~90s) ==="
cd /workspace/WheeledLab/source/wheeledlab_rl
TRAIN_SEED=999 timeout 180 /workspace/isaaclab/_isaac_sim/python.sh scripts/train_rl.py \
    -r AGENT_A_CONFIG --headless train.log.video=false \
    train.log.no_wandb=true train.log.run_name=setup_smoke \
    env_setup.num_envs=4 train.num_iterations=1 2>&1 | tail -3

echo ""
echo "=========================================="
echo "INSTANCE READY — production launch examples"
echo "=========================================="
echo "Single run:  AGENT=A SEED=1 bash /workspace/WheeledLab/launch_production.sh"
echo "Two seeds:   AGENT=A SEEDS=\"1 2\" bash /workspace/WheeledLab/launch_chain.sh"
