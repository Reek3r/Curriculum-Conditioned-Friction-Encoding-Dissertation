#!/bin/bash
# WheeledLab Brev Cloud Setup Script
# Run once after launching a new Brev Isaac Lab instance

set -e

echo "=== Installing WheeledLab packages ==="
cd "$(dirname "$0")/source"
/workspace/isaaclab/_isaac_sim/python.sh -m pip install -e wheeledlab --no-deps
/workspace/isaaclab/_isaac_sim/python.sh -m pip install -e wheeledlab_assets --no-deps
/workspace/isaaclab/_isaac_sim/python.sh -m pip install -e wheeledlab_tasks --no-deps
/workspace/isaaclab/_isaac_sim/python.sh -m pip install -e wheeledlab_rl --no-deps

echo "=== Installing W&B ==="
/workspace/isaaclab/_isaac_sim/python.sh -m pip install wandb
/workspace/isaaclab/_isaac_sim/python.sh -m pip install av

echo "=== Setup complete. Next steps: ==="
echo "1. Run: wandb login"
echo "2. Run verification: cd wheeledlab_rl && python scripts/train_rl.py -r RSS_DRIFT_CONFIG --headless train.log.video=false env_setup.num_envs=64 train.num_iterations=50"

# === Sim 5.0 termination divergence patch (Fix B) ===
# Gates is_terminated_term reward to fire only on actual reset step.
# Restores Sim 5.1-equivalent behaviour. See dissertation §3.9 / Appendix F.2.
echo "=== Applying Sim 5.0 termination patch ==="
PATCH_FILE=/workspace/isaaclab/source/isaaclab/isaaclab/envs/mdp/rewards.py
PATCH_MARKER='env.reset_buf.float()).float()'
PATCH_OLD='return (reset_buf * (~env.termination_manager.time_outs)).float()'
if grep -qF "$PATCH_MARKER" "$PATCH_FILE"; then
    echo "Fix B already present — skipping"
elif grep -qF "$PATCH_OLD" "$PATCH_FILE"; then
    sed -i 's|return (reset_buf \* (~env\.termination_manager\.time_outs))\.float()|return (reset_buf * (~env.termination_manager.time_outs) * env.reset_buf.float()).float()|' "$PATCH_FILE"
    grep -qF "$PATCH_MARKER" "$PATCH_FILE" && echo "Fix B applied" || { echo "ERROR: sed reported success but marker not found post-patch"; exit 1; }
else
    echo "ERROR: Fix B target string not found in $PATCH_FILE — manual intervention required"
    exit 1
fi

# === vecenv_wrapper.py routing patch ===
# Routes get_observations through env subclass (for k=5 history buffer in MushrDriftEnv).
# Without this, the wrapper falls back to the 14-dim observation_manager output and
# breaks the 70-dim policy network. See dissertation §3.2 / Methodology.
echo "=== Applying vecenv_wrapper get_observations routing patch ==="
WRAPPER_FILE=/workspace/isaaclab/source/isaaclab_rl/isaaclab_rl/rsl_rl/vecenv_wrapper.py
WRAPPER_MARKER='if hasattr(self.unwrapped, "get_observations"):'
WRAPPER_OLD='        if hasattr(self.unwrapped, "observation_manager"):'
WRAPPER_NEW='        if hasattr(self.unwrapped, "get_observations"):\
            obs_dict = self.unwrapped.get_observations()\
        elif hasattr(self.unwrapped, "observation_manager"):'

if grep -qF "$WRAPPER_MARKER" "$WRAPPER_FILE"; then
    echo "Wrapper patch already present — skipping"
elif grep -qF '        if hasattr(self.unwrapped, "observation_manager"):' "$WRAPPER_FILE"; then
    sed -i 's|        if hasattr(self.unwrapped, "observation_manager"):|        if hasattr(self.unwrapped, "get_observations"):\n            obs_dict = self.unwrapped.get_observations()\n        elif hasattr(self.unwrapped, "observation_manager"):|' "$WRAPPER_FILE"
    grep -qF "$WRAPPER_MARKER" "$WRAPPER_FILE" && echo "Wrapper patch applied" || { echo "ERROR: sed reported success but marker not found post-patch"; exit 1; }
else
    echo "ERROR: Wrapper patch target string not found in $WRAPPER_FILE — manual intervention required"
    exit 1
fi
