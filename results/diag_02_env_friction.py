"""diag_02_env_friction.py

Phase 1 diagnostic: confirm we can set per-env friction to a single µ across all
envs by mutating `friction_term.func.material_buckets` and resetting, and that
the round-trip via `robot.root_physx_view.get_material_properties()[:, 2:10, 0]`
matches the target to 4 decimal places.

Per TROUBLESHOOTING_GUIDE.md §1, §3:
    - read static_µ from mats[:, 2:10, 0].mean(dim=1) (wheel bodies are 2..9)
    - mutating term_cfg.params alone is NOT enough; you must rebuild and assign
      `friction_term.func.material_buckets` (the cached bucket tensor) and then
      env.reset() to fire the reset-mode event that resamples per-env material
      from the bucket pool.

Per METHODOLOGY_LOCKED_SPECS.md the rollouts use the base task
`Isaac-MushrDriftRL-v0`. Agent C's ADR env auto-mutates buckets in its own
`_reset_idx`, so we deliberately do NOT use that task for probe rollouts.

Usage:
    python.sh diag_02_env_friction.py --mu 0.42 --num_envs 8

Exits 0 on PASS, 1 on FAIL.
"""

import argparse
import functools
import sys

# Force stdout/stderr to be line-buffered so prints survive Isaac Sim's
# simulation_app.close() teardown (which can leave buffered stdout unflushed).
print = functools.partial(print, flush=True)  # noqa: A001 — intentional shadow
RESULT_FILE = "/tmp/diag02_result.txt"


def write_result(line):
    try:
        with open(RESULT_FILE, "a") as f:
            f.write(line.rstrip() + "\n")
    except Exception:
        pass


# --- Isaac Sim app launch must precede any isaaclab.* import that touches sim
from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--mu", type=float, default=0.42, help="target friction value")
ap.add_argument("--num_envs", type=int, default=8, help="number of envs (small for diag)")
ap.add_argument("--task", type=str, default="Isaac-MushrDriftRL-AgentB-v0",
                help="gym task id; default uses Agent B cfg (mode='reset' friction, "
                     "MushrDriftEnv entry point — same env class as A; avoids the "
                     "MushrDriftADREnv auto-mutation in the AgentC task)")
ap.add_argument("--num_buckets", type=int, default=20)
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
if not getattr(args, "headless", False):
    args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import gymnasium as gym
import wheeledlab_tasks  # triggers gym.register
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


def build_single_mu_buckets(mu, num_buckets):
    """Construct a (num_buckets, 3) bucket tensor that samples (mu, mu, 0.0).

    Mirrors MushrDriftADREnv._reset_idx logic when lo == hi: rand * 0 + mu = mu.
    The third column is restitution, kept at 0.0.
    """
    ranges = torch.tensor([(mu, mu), (mu, mu), (0.0, 0.0)], device="cpu")
    rng = torch.rand((num_buckets, 3), device="cpu")
    new_buckets = (ranges[:, 1] - ranges[:, 0]) * rng + ranges[:, 0]
    new_buckets[:, 1] = torch.min(new_buckets[:, 0], new_buckets[:, 1])
    return new_buckets


def main():
    print(f"[diag_02] task={args.task} num_envs={args.num_envs} target_mu={args.mu}")
    env_cfg = parse_env_cfg(args.task, num_envs=args.num_envs)
    env = gym.make(args.task, cfg=env_cfg)

    base = env.unwrapped

    em = base.event_manager
    try:
        friction_term_cfg = em.get_term_cfg("change_wheel_friction")
    except (KeyError, ValueError, AttributeError) as e:
        print(f"[diag_02] FAIL: 'change_wheel_friction' not findable via "
              f"em.get_term_cfg: {type(e).__name__}: {e}")
        sys.exit(1)
    print(f"[diag_02] friction term mode: {getattr(friction_term_cfg, 'mode', '?')}")
    print(f"[diag_02] friction term func type: {type(friction_term_cfg.func).__name__}")
    print(f"[diag_02] friction term params (pre): {friction_term_cfg.params}")

    # Mutate params (defensive — methodology says params alone insufficient, but harmless)
    friction_term_cfg.params["static_friction_range"] = (args.mu, args.mu)
    friction_term_cfg.params["dynamic_friction_range"] = (args.mu, args.mu)

    # Rebuild buckets and assign on the instance
    if not hasattr(friction_term_cfg.func, "material_buckets"):
        print("[diag_02] FAIL: friction_term.func has no .material_buckets attribute "
              "— ManagerTermBase shape is unexpected")
        sys.exit(1)

    new_buckets = build_single_mu_buckets(args.mu, args.num_buckets)
    friction_term_cfg.func.material_buckets = new_buckets
    print(f"[diag_02] new buckets shape: {tuple(new_buckets.shape)}")
    print(f"[diag_02] new buckets sample (first 3): {new_buckets[:3].tolist()}")

    # Apply via reset
    obs, _ = env.reset()

    # Step once with zero actions to ensure the sim has flushed materials
    if hasattr(base, "single_action_space"):
        action_dim = int(base.single_action_space.shape[0])
    else:
        action_dim = 2
    actions = torch.zeros((args.num_envs, action_dim), device=base.device)
    env.step(actions)

    # Read material properties back via PhysX view
    robot = base.scene["robot"]
    mats = robot.root_physx_view.get_material_properties()
    print(f"[diag_02] mats shape: {tuple(mats.shape)}")  # (num_envs, num_collision_bodies, 3)

    body_names = robot.body_names if hasattr(robot, "body_names") else None
    if body_names:
        print(f"[diag_02] articulation body_names ({len(body_names)}): {body_names}")
        name_wheel_idx = [i for i, n in enumerate(body_names) if "wheel_link" in n and "steer" not in n]
        print(f"[diag_02] articulation wheel_link indices (asset_cfg space): {name_wheel_idx}")

    # Print all 10 PhysX-body static-mu rows for env 0 so we can see which bodies
    # actually got the bucket value applied (the wheels) vs which kept defaults.
    print(f"[diag_02] mats[0, :, 0] (static µ per body, env 0): {mats[0, :, 0].tolist()}")
    print(f"[diag_02] mats[0, :, 1] (dynamic µ per body, env 0): {mats[0, :, 1].tolist()}")
    print(f"[diag_02] mats[0, :, 2] (restitution per body, env 0): {mats[0, :, 2].tolist()}")

    target = float(args.mu)

    # Empirically identify which PhysX body rows received the bucket value.
    # The friction event applies the same bucket sample to all wheels in an env,
    # so the wheel rows in env 0 should equal target_µ to <1e-4.
    physx_close = (mats[0, :, 0] - target).abs() < 1e-3
    physx_wheel_idx = [int(i) for i, c in enumerate(physx_close.tolist()) if c]
    print(f"[diag_02] PhysX body indices with µ ≈ {target:.4f} in env 0: {physx_wheel_idx}")

    failures = []
    if len(physx_wheel_idx) == 0:
        failures.append("no PhysX bodies received the target µ — bucket override did not apply")
    else:
        # Round-trip across all envs using the empirically-located wheel rows
        mu_per_env = mats[:, physx_wheel_idx, 0].float().mean(dim=1)
        print(f"[diag_02] mu_per_env (PhysX wheel rows): {mu_per_env.tolist()}")
        max_diff = float((mu_per_env - target).abs().max().item())
        print(f"[diag_02] max |Δµ| across {args.num_envs} envs: {max_diff:.6f}")
        if max_diff > 1e-4:
            failures.append(f"round-trip Δµ={max_diff:.6f} > 1e-4 across envs")

    # Also test the documented recipe (slice 2:10) so the discrepancy with the
    # Troubleshooting Guide is recorded for the rollout-script writer.
    if mats.shape[1] >= 10:
        mu_per_env_2_10 = mats[:, 2:10, 0].float().mean(dim=1)
        max_diff_2_10 = float((mu_per_env_2_10 - target).abs().max().item())
        print(f"[diag_02] DIAGNOSTIC: max |Δµ| using documented slice 2:10: {max_diff_2_10:.6f}")

    if failures:
        verdict = "[diag_02] FAIL: " + " | ".join(failures)
        print(verdict)
        write_result(verdict)
        env.close()
        simulation_app.close()
        sys.exit(1)
    verdict = (f"[diag_02] PASS: friction round-trip matches µ={target} to 4 dp "
               f"across all envs (PhysX wheel idx={physx_wheel_idx})")
    print(verdict)
    write_result(verdict)
    env.close()
    simulation_app.close()
    sys.exit(0)


if __name__ == "__main__":
    main()
