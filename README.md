# Curriculum-Conditioned Friction Encoding in Wheeled Robot RL Policies

**Author:** Muhammad Rooman Waheed (11094142)  
**Module:** EEEN30330 — Individual 3rd Year Project  
**Institution:** Department of Electrical and Electronic Engineering, University of Manchester  
**Supervisor:** Shanika Matharage
**Date:** May 2026

---

## 1. Introduction

This repository contains the simulation, training, and analysis code for a dissertation investigating whether domain randomisation (DR) curriculum structure influences the degree to which a reinforcement learning (RL) policy's internal representations encode surface friction coefficient (µ). Three agents — trained under narrow fixed, wide uniform, and success-gated adaptive DR curricula — share an identical policy architecture but differ strictly in the order and breadth of friction conditions they encounter during training. Post-hoc linear probes applied to frozen hidden-layer activations measure whether these curriculum differences produce measurably different internal representations of the ground-truth friction parameter.

The codebase is a fork of [WheeledLab](https://github.com/UWRobotLearning/WheeledLab) (Han et al., 2025), an open-source wheeled-robot learning framework built on NVIDIA Isaac Lab. All modifications made for this dissertation are confined to the files listed in Section 6 below. The upstream WheeledLab platform provides the MuSHR rover asset, the base environment class, and the rsl_rl PPO training infrastructure. This project adds the three-agent curriculum configurations, the adaptive domain randomisation (ADR) environment subclass with success-gated expansion logic, custom reward shaping, temporal observation stacking, milestone checkpointing, and Weights & Biases integration for ADR boundary tracking.

---

## 2. Contextual Overview

The software pipeline has three stages, executed sequentially:

```
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 1: TRAINING                                              │
│                                                                 │
│  Isaac Lab (PhysX)  ──►  PPO (rsl_rl)  ──►  Checkpoints        │
│  3 agents × 3 seeds × 5 milestones = 45 saved models           │
│  Logging: Weights & Biases (per-episode µ, rewards, ADR bounds) │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 2: ACTIVATION EXTRACTION                                 │
│                                                                 │
│  Load frozen checkpoint  ──►  Roll out across 20-point µ grid   │
│  ──►  Record hidden-layer activations via forward hooks         │
│  ──►  Save (activations, µ_ground_truth) pairs                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 3: PROBE ANALYSIS (CPU-only)                             │
│                                                                 │
│  Ridge regression (sklearn) on activations → µ prediction       │
│  4-component Probe Triangulation Framework:                     │
│    1. Random-weight baseline                                    │
│    2. Linear encoder probe (primary)                            │
│    3. Raw-input ceiling probe                                   │
│    4. MLP probe at final checkpoint                             │
│  Output: R² values per agent × checkpoint × layer               │
└─────────────────────────────────────────────────────────────────┘
```

**Agent configurations:**

| Agent | Curriculum | µ Range | Environment Class |
|-------|-----------|---------|-------------------|
| A — Narrow Fixed DR | Fixed uniform | µ ∈ [0.76, 0.84] | `MushrDriftEnv` |
| B — Wide Uniform DR | Fixed uniform | µ ∈ [0.20, 1.00] | `MushrDriftEnv` |
| C — Standard ADR | Success-gated adaptive | Starts [0.76, 0.84], expands to [0.20, 1.00] | `MushrDriftADREnv` |

All agents use the same policy architecture (70→64→64→2 ELU MLP with k=5 temporal observation stacking) and are trained with PPO using identical hyperparameters.

---

## 3. Installation Instructions

### Prerequisites

- NVIDIA GPU with CUDA support (tested on RTX 3070 locally, L40S / A6000 on cloud)
- [NVIDIA Isaac Sim 5.0.0](https://developer.nvidia.com/isaac-sim) with Isaac Lab
- Python 3.10 (via Isaac Lab's bundled Python environment)
- Conda (for local development) or a Brev/cloud Isaac Lab Launchable

### Step-by-step setup

**1. Install Isaac Lab** following the [official instructions](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation.html).

**2. Clone this repository** into your workspace:

```bash
cd /path/to/your/workspace
git clone https://github.com/Reek3r/WheeledLab.git
cd WheeledLab
```

**3. Install the four WheeledLab packages:**

```bash
# On Brev/cloud (no sudo):
pip install --user -e source/wheeledlab
pip install --user -e source/wheeledlab_assets
pip install --user -e source/wheeledlab_tasks
pip install --user -e source/wheeledlab_rl

# Locally with conda (activate your Isaac Lab env first):
conda activate env_isaaclab
pip install -e source/wheeledlab
pip install -e source/wheeledlab_assets
pip install -e source/wheeledlab_tasks
pip install -e source/wheeledlab_rl
```

**4. Patch the Isaac Lab vecenv wrapper** (required — this fix does not live in this repository):

In your Isaac Lab installation, open:
```
<isaaclab_root>/source/isaaclab_rl/isaaclab_rl/rsl_rl/vecenv_wrapper.py
```

Locate the `get_observations` method and ensure the first check is:
```python
if hasattr(self.unwrapped, "get_observations"):
    return self.unwrapped.get_observations()
```

This patch routes observation calls correctly through the WheeledLab environment wrapper. Without it, training will fail with an attribute error. **This patch does not survive a fresh Isaac Lab reinstall** — it must be reapplied.

**5. (Optional) Configure Weights & Biases:**

```bash
export WANDB_API_KEY=<your_key>
```

### Dependencies

| Dependency | Version | Purpose |
|-----------|---------|---------|
| Isaac Sim | 5.0.0 | Physics simulation (PhysX backend) |
| Isaac Lab | 0.47.1 | RL environment framework |
| gymnasium | 1.0.0 | Environment API (shadows Isaac Lab's 1.2.0) |
| rsl_rl | bundled | PPO implementation |
| scikit-learn | ≥1.3 | Ridge regression for linear probes |
| PyTorch | bundled | Network, forward hooks for activation extraction |
| wandb | ≥0.16 | Experiment tracking |

---

## 4. How to Run the Software

### Training a single agent

**On Brev / cloud** (use Isaac Lab's bundled Python):
```bash
/workspace/isaaclab/_isaac_sim/python.sh \
  source/wheeledlab_rl/scripts/train_rl.py \
  -r AGENT_A_CONFIG \
  --headless \
  train.log.video=false \
  env_setup.num_envs=1024 \
  train.num_iterations=5000
```

**Locally** (with conda environment activated):
```bash
python source/wheeledlab_rl/scripts/train_rl.py \
  -r AGENT_A_CONFIG \
  --headless \
  train.log.video=false \
  env_setup.num_envs=32 \
  train.num_iterations=5000
```

Replace `AGENT_A_CONFIG` with `AGENT_B_CONFIG` or `AGENT_C_CONFIG` for the other agents.

### Running all 9 training runs (3 agents × 3 seeds)

A launch script is provided for use inside a `tmux` session to survive disconnects:

```bash
tmux new -s training
bash run_all_training.sh
```

Detach with `Ctrl+B` then `D`. Reattach with `tmux attach -t training`.

### Running the probe analysis

After training and activation extraction are complete:

```bash
python analysis/run_probes.py --checkpoint_dir <path_to_checkpoints> --output_dir <path_to_results>
```

This script executes all four probe components (random-weight baseline, linear encoder probe, raw-input ceiling, MLP probe) and outputs R² values for every cell in the agent × checkpoint × layer grid.

---

## 5. Technical Details

### Observation space

Each timestep the policy receives a 14-dimensional proprioceptive observation: base linear velocity (3), base angular velocity (3), projected gravity (3), goal-relative position (2), wheel velocities (2), and track progress (1). A temporal history buffer stacks the most recent k=5 observations, producing a 70-dimensional input vector. On episode reset, the buffer is zero-padded.

### Policy architecture

A two-hidden-layer MLP with 64 units per layer and ELU activations. Input dimension is 70 (14 × k=5). Output dimension is 2 (linear and angular velocity commands). Separate actor and critic networks share the same architecture but not weights.

### Training algorithm

Proximal Policy Optimisation (PPO) with: clipping parameter ε=0.2, discount factor γ=0.997, GAE parameter λ=0.95, learning rate 1×10⁻³ with KL-adaptive schedule (target KL=0.01). Training runs for 5000 iterations. Checkpoints are saved at 0%, 25%, 50%, 75%, and 100% of total iterations via a custom `MilestoneOnPolicyRunner`.

### Reward function

```
r_t = w1 · ω_progress  −  w2 · ‖a_t − a_{t−1}‖₂  +  w5 · b_lap
```

where ω_progress is angular progress around the track centreline, the action-smoothness penalty discourages jerk, and b_lap is a one-time bonus awarded on lap completion. Side-slip penalty weight is deliberately set to 0.0 — suppressing lateral sliding penalties preserves the friction-dependent dynamics that make µ observable from state transitions.

### ADR gate logic (Agent C only)

The `MushrDriftADREnv` subclass implements success-gated curriculum expansion. An episode is counted as successful if **both**: (1) `track_progress_completed ≥ 1.0` (full lap), and (2) `episode_length_s ≤ T_max = 5.5s`. Over a rolling window of 200 episodes, if the success rate exceeds 80%, the friction range expands symmetrically by ±0.02 per gate pass, from the initial [0.76, 0.84] toward the target [0.20, 1.00].

### Track geometry

A high-lateral-force cornering circuit with centreline radius r_min = 0.8m, inner termination radius 0.3m, and total track length 8.227m. The track is designed so that completing a lap within the time gate requires the policy to operate near the friction limit, ensuring that µ-dependent dynamics are exercised during training.

### Friction parameter handling

Ground-truth µ is read from the PhysX material tensor at indices `mats[:, 2:10, 0].mean(dim=1)` (indices 0–1 are the ground plane, not wheels). Friction randomisation uses event mode `"reset"` (re-sampled every episode reset), not `"startup"`.

---

## 6. Repository Structure and Modified Files

This repository is a fork of [UWRobotLearning/WheeledLab](https://github.com/UWRobotLearning/WheeledLab). The upstream code provides the simulation platform, MuSHR rover asset, and base training infrastructure. **All files not listed below are unmodified from the upstream repository.**

### Files modified or added for this dissertation

| File | Purpose |
|------|---------|
| `source/wheeledlab_tasks/wheeledlab_tasks/drifting/mushr_drift_env_cfg.py` | Reward function, event configurations, termination conditions, and all three agent-specific environment configs (`MushrDriftAgentACfg`, `MushrDriftAgentBCfg`, `MushrDriftAgentCCfg`) |
| `source/wheeledlab_tasks/wheeledlab_tasks/drifting/mushr_drift_env.py` | `MushrDriftADREnv` subclass implementing the ADR success gate, friction-range expansion logic, and per-episode µ logging |
| `source/wheeledlab_tasks/wheeledlab_tasks/drifting/__init__.py` | Updated exports for the three agent configs |
| `source/wheeledlab_tasks/wheeledlab_tasks/__init__.py` | Gymnasium registrations for `Isaac-MushrDriftRL-AgentA-v0`, `AgentB-v0`, `AgentC-v0` |
| `source/wheeledlab_rl/wheeledlab_rl/configs/runs/rss_cfgs.py` | Hydra run configurations for `AGENT_A_CONFIG`, `AGENT_B_CONFIG`, `AGENT_C_CONFIG` |
| `source/wheeledlab_rl/wheeledlab_rl/configs/runs/__init__.py` | Hydra registration calls for the three agent configs |
| `source/wheeledlab_rl/wheeledlab_rl/utils/modified_rsl_rl_runner.py` | `MilestoneOnPolicyRunner` with milestone checkpointing at 0/25/50/75/100% and W&B ADR boundary logging |
| `run_all_training.sh` | Bash script for launching all 9 training runs sequentially |

### External patch (not in this repository)

The Isaac Lab file `vecenv_wrapper.py` requires a one-line patch to route `get_observations()` calls through the WheeledLab environment wrapper. See Installation Instructions (Step 4) for details. This patch must be applied to the Isaac Lab installation, not this repository.

### Third-party code acknowledgement

The WheeledLab framework (Han et al., 2025) is used under its original licence. The upstream repository provides the MuSHR rover URDF asset, the base `MushrDriftEnv` environment class, the rsl_rl PPO training loop, and the Hydra configuration system. All upstream code is unmodified except for the files listed above.

---

## 7. Known Issues and Future Improvements

### Known issues

1. **vecenv_wrapper.py patch is fragile.** The patch to Isaac Lab's vecenv wrapper does not persist across fresh Isaac Lab installations. It must be manually reapplied after any reinstall. A post-install script or monkey-patch at import time would be more robust.

2. **Seed control via Hydra.** The environment config hardcodes `seed: int = 42`. Hydra override syntax `+env_setup.seed=N` was required to vary seeds across runs; standard override syntax failed with a `ConfigCompositionException`.

3. **Agent C does not run on local RTX 3070.** An OpenGL driver conflict with the AMD integrated GPU causes crashes. This is a driver-level issue, not a code bug — Agent C runs correctly on cloud GPUs (L40S, A6000).

4. **gymnasium version mismatch.** Isaac Lab 0.47.1 expects gymnasium 1.2.0; WheeledLab pins 1.0.0. Installing 1.0.0 with `--user` shadows the system package. Training runs produce a deprecation warning but function correctly.

5. **`bonus_per_lap` magnitude.** The lap completion bonus (provisional value 5000) was flagged during development as approximately 3× higher than the accumulated ω_progress reward over a typical episode. Recalibration may be needed for future work.

### Future improvements

1. **Automated probe pipeline.** Currently, activation extraction and probe fitting are separate manual steps. A single end-to-end script that takes a training run directory and produces the full R² grid would improve reproducibility.

2. **Extended history buffer.** The k=5 temporal window was validated as sufficient via k-ablation diagnostic, but k=10 may capture longer-timescale friction dynamics. The observation dimension increase (70→140) would need to be evaluated against the 64-unit hidden layer capacity.

3. **Sim-to-real transfer.** This dissertation evaluates representations within a single simulator. Transferring the trained policies to a physical MuSHR rover would test whether the friction representations identified by probing correspond to real-world physics adaptation. This was outside the scope of the current project.

4. **Nonlinear curriculum structures.** Only uniform and adaptive curricula were tested. Curricula with explicit easy-to-hard staging, reverse curricula, or multi-parameter co-randomisation remain unexplored.

---

## References

- Han, Z., et al. (2025). *WheeledLab: An Open-Source Framework for Wheeled Robot Learning.* GitHub: [UWRobotLearning/WheeledLab](https://github.com/UWRobotLearning/WheeledLab).
- Schulman, J., et al. (2017). *Proximal Policy Optimization Algorithms.* arXiv:1707.06347.
- Tobin, J., et al. (2017). *Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World.* IROS 2017.
- NVIDIA (2025). *Isaac Lab.* [https://isaac-sim.github.io/IsaacLab](https://isaac-sim.github.io/IsaacLab).

---

## Licence

This repository is a fork of WheeledLab and is distributed under the same licence as the upstream project. See [LICENSE](LICENSE) for details.
