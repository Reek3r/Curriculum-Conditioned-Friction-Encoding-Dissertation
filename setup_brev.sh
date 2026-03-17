#!/bin/bash
# WheeledLab Brev Cloud Setup Script
# Run once after launching a new Brev Isaac Lab instance

set -e

echo "=== Installing WheeledLab packages ==="
cd /home/user/WheeledLab/source
pip install -e wheeledlab --no-deps
pip install -e wheeledlab_assets --no-deps
pip install -e wheeledlab_tasks --no-deps
pip install -e wheeledlab_rl --no-deps

echo "=== Installing W&B ==="
pip install wandb

echo "=== Setup complete. Next steps: ==="
echo "1. Run: wandb login"
echo "2. Run verification: cd wheeledlab_rl && python scripts/train_rl.py -r RSS_DRIFT_CONFIG --headless train.log.video=false env_setup.num_envs=64 train.num_iterations=50"
