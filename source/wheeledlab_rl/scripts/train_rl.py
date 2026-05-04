###################################
###### BEGIN ISAACLAB SPINUP ######
###################################

from wheeledlab_rl.startup import startup
import argparse
parser = argparse.ArgumentParser(description="Train an RL Agent in WheeledLab.")
parser.add_argument('-r', "--run-config-name", type=str, default="RSS_DRIFT_CONFIG", help="Run in headless mode.")
simulation_app, args_cli = startup(parser=parser)

#######################
###### END SETUP ######
#######################

import gymnasium as gym
import os
import random
import numpy as np
import torch

from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from wheeledlab_rl.configs import RunConfig
from wheeledlab_rl import WHEELEDLAB_RL_LOGS_DIR

from wheeledlab_rl.utils import (
    OnPolicyRunner as ModifiedRslRunner,
    CustomRecordVideo,
    hydra_run_config,
    ClipAction,
)

def _resolve_seed(default_seed: int) -> int:
    """Resolve seed from TRAIN_SEED env var, falling back to config default.

    Centralises seed source so the top-of-main helper and the existing
    env.seed() block (later in main) read from the same value. Workaround for
    Hydra `train.seed=N` overrides being silently ignored on this codebase.
    """
    override = os.environ.get("TRAIN_SEED")
    if override is None:
        return int(default_seed)
    try:
        return int(override)
    except ValueError as exc:
        raise ValueError(
            f"TRAIN_SEED env var must be an integer, got {override!r}"
        ) from exc


def _set_global_seed(seed: int) -> None:
    """Seed all RNGs touched during training: Python, NumPy, torch, CUDA, cuDNN.

    Called at the top of main() before any env/runner/model construction so that
    network initialization, action sampling, and replay-time augmentation all
    derive from the same seed. cuDNN deterministic flags add ~5-10% wall-time
    overhead but are required for n=3 seed-level variance to be reproducible.
    See §3.9 / Appendix A.3 for derivation of weight-L2 evidence at iter 0.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@hydra_run_config(run_config_name=args_cli.run_config_name)
def main(run_cfg: RunConfig): # TODO: Add SB3 config support

    #################
    #### LOGGING ####
    #################

    ##### Aliasing Configs #####
    env_cfg = run_cfg.env
    agent_cfg = run_cfg.agent
    train_cfg = run_cfg.train
    log_cfg = train_cfg.log
    env_setup = run_cfg.env_setup

    #################
    ##### SEED ######
    #################
    # Resolve seed from TRAIN_SEED env var (Hydra train.seed=N is silently
    # ignored on this codebase; see hydra.py override mechanism). Write the
    # resolved value back to agent_cfg.seed so the existing env.seed() call
    # later in main() and the rsl_rl runner construction observe the same
    # value. Seed BEFORE env/runner/model construction so network init is
    # deterministic across re-runs of the same TRAIN_SEED.
    _resolved_seed = _resolve_seed(agent_cfg.seed)
    agent_cfg.seed = _resolved_seed
    env_cfg.seed = _resolved_seed
    _set_global_seed(_resolved_seed)
    print(f"[SEED] Global seed set to {_resolved_seed} (source: "
          f"{'TRAIN_SEED env var' if os.environ.get('TRAIN_SEED') else 'agent_cfg default'})")

    if not log_cfg.no_wandb:
        import wandb
        run = wandb.init(
            project=log_cfg.wandb_project,
        )
        log_cfg.run_name = wandb.run.name

    if not os.path.exists(log_cfg.model_save_path):
        os.makedirs(log_cfg.model_save_path)

    ## UPDATE CONFIGS WANDB ##
    if not log_cfg.no_wandb:
        wandb.config.update(run_cfg.to_dict())

    # Save the config file
    if not log_cfg.no_log:
        dump_yaml(os.path.join(log_cfg.run_log_dir, "run_config.yaml"), run_cfg)

    ############################
    #### CREATE ENVIRONMENT ####
    ############################

    env = gym.make(env_setup.task_name, cfg=env_cfg, render_mode="rgb_array" if log_cfg.video else None)

    ####### INSTANTIATE ENV #######
    env.action_space.low = -1.
    env.action_space.high = 1.
    env = ClipAction(env)

    # Wrap the environment in recorder
    if log_cfg.video:
        video_kwargs = {
            "video_folder": os.path.join(log_cfg.run_log_dir, "videos"),
            "step_trigger": lambda step: step % log_cfg.video_interval == 0,
            "video_length": log_cfg.video_length,
            "disable_logger": True,
            "enable_wandb": not log_cfg.no_wandb,
            "video_resolution": log_cfg.video_resolution,
            "video_crf": log_cfg.video_crf,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = CustomRecordVideo(env, **video_kwargs)

    # TODO: add back support for SB3
    env = RslRlVecEnvWrapper(env)

    try:
        em = env.unwrapped.event_manager
        print(f"[EVENT MANAGER] {dir(em)}")
        for attr in ['active_terms', 'terms', '_terms']:
            if hasattr(em, attr):
                print(f"[EVENT TERMS via {attr}] {getattr(em, attr)}")
                break
    except Exception as e:
        print(f"[EVENT MANAGER ERROR] {e}")

    try:
        robot = env.unwrapped.scene["robot"]
        wheel_indices = robot.find_bodies(".*wheel_link")
        print(f"[WHEEL BODIES] {wheel_indices}")
    except Exception as e:
        print(f"[WHEEL BODIES ERROR] {e}")

    try:
        robot = env.unwrapped.scene["robot"]
        mats = robot.root_physx_view.get_material_properties()
        print(f"[MATS SHAPE] {mats.shape}")
        print(f"[MATS ENV0 ALL SHAPES] {mats[0, :, :].cpu().tolist()}")
    except Exception as e:
        print(f"[MATS ERROR] {e}")
    # --- DIAGNOSTIC: read per-env friction from PhysX ---
    def log_mu_diagnostic(env_wrapped, iteration_label):
        if log_cfg.no_wandb:
            return
        try:
            raw_env = env_wrapped.unwrapped
            robot = raw_env.scene["robot"]
            mats = robot.root_physx_view.get_material_properties()
            # mats shape: (num_envs, num_shapes, 3) — static_friction, dynamic_friction, restitution
            wheel_frictions = mats[:, 2:10, 0]
            static_frictions = wheel_frictions.mean(dim=1).cpu().tolist()
            wandb.log({
                "debug/mu_per_env": wandb.Histogram(static_frictions),
                **{f"debug/mu_env_{i}": v for i, v in enumerate(static_frictions[:8])}
            }, step=iteration_label)
            print(f"[MU DIAGNOSTIC] iter={iteration_label} frictions={static_frictions[:4]}")
        except Exception as e:
            print(f"[MU DIAGNOSTIC ERROR] {e}")
    
    runner = ModifiedRslRunner(env, agent_cfg.to_dict(), log_cfg, device=train_cfg.device)

    ##### LOAD EXISTING RUN? #####

    if train_cfg.load_run is not None:
        chkpt = "model_"
        if train_cfg.load_run_checkpoint > 0:
            chkpt = f"{chkpt}{train_cfg.load_run_checkpoint}"
        resume_path = get_checkpoint_path(WHEELEDLAB_RL_LOGS_DIR, run_dir=train_cfg.load_run,
                                        other_dirs=["models"], checkpoint=f"{chkpt}.*")
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        runner.load(resume_path)

    # --- MU DIAGNOSTIC ---
    def read_frictions(env_chain):
        _raw = env_chain
        while hasattr(_raw, 'env'):
            _raw = _raw.env
        mats = _raw.unwrapped.scene["robot"].root_physx_view.get_material_properties()
        # Shapes 0-1 are ground plane. Shapes 2-9 are wheel collision shapes.
        # Take mean static friction across wheel shapes per env.
        wheel_frictions = mats[:, 2:10, 0]  # (num_envs, 8)
        return wheel_frictions.mean(dim=1).cpu().tolist()  # (num_envs,)

    try:
        print(f"[MU PRE-TRAIN] {read_frictions(env)}")
    except Exception as e:
        print(f"[MU PRE-TRAIN ERROR] {e}")

    #################
    ##### TRAIN #####
    #################

    _seed_override = os.environ.get("TRAIN_SEED")
    if _seed_override is not None:
        agent_cfg.seed = int(_seed_override)
    env.seed(agent_cfg.seed)
    print(f"Setting seed: {agent_cfg.seed}")
    env.common_step_counter = train_cfg.set_env_step
    runner.learn(num_learning_iterations=train_cfg.num_iterations)

    try:
        print(f"[MU POST-TRAIN] {read_frictions(env)}")
    except Exception as e:
        print(f"[MU POST-TRAIN ERROR] {e}")

    if not log_cfg.no_wandb:
        run.finish()
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()