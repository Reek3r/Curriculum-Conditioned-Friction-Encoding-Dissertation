"""probe_rollout_and_fit.py — single-checkpoint rollout + probe fit.

Per METHODOLOGY_LOCKED_SPECS.md:
    - 20 friction values uniformly in [0.200, 1.000]
    - 256 parallel envs per friction (one rollout per friction)
    - 500 policy steps per rollout, deterministic mean action
    - Layer-2 (actor[3] ELU output) activation probe site
    - 70/30 env-level train/test split, fixed seed 42
    - RidgeCV on log-spaced α grid
    - MLPRegressor 64->32->1 ReLU, only at 100% checkpoint
    - Idempotent CSV writes (dedup on agent/seed/run/milestone/probe_type)

Output rows (one per probe fit):
    agent, seed, run_name, milestone, probe_type, r2, mse,
    n_train_samples, n_test_samples, alpha, notes

Side-output rows (ood_metrics.csv):
    one row per µ — completion_rate, median_lap_time, IQR, oob_rate,
    timeout_rate, n_episodes
"""

import argparse
import csv
import functools
import os
import sys
import time

# Force stdout to flush per-print so output survives Isaac Sim teardown.
print = functools.partial(print, flush=True)  # noqa: A001

# Run-name -> (agent, seed); authoritative mapping confirmed by user 7 May 2026.
RUN_MAPPING = {
    "forgotten-transport-91": ("A", 0),
    "lively-planet-95":       ("A", 1),
    "sweet-tree-106":         ("A", 2),
    "ancient-pilot-90":       ("B", 0),
    "firm-terrain-96":        ("B", 1),
    "magic-smoke-105":        ("B", 2),
    "pious-star-94":          ("C", 0),
    "flowing-pond-99":        ("C", 1),
    "glamorous-sunset-100":   ("C", 2),
}

# ─── App launch must precede isaaclab.* imports ────────────────────────────
from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", default="", help="Path to model_milestone_*.pt (omit for --random_init)")
ap.add_argument("--run_name", required=True, help="Filesystem run dir name (e.g. pious-star-94)")
ap.add_argument("--agent", required=True, help="A/B/C, or 'floor'/'ceiling' for non-trained variants")
ap.add_argument("--seed", default="", help="0/1/2 (or empty for floor/ceiling)")
ap.add_argument("--milestone", default="", help="50/75/100 (or empty for floor/ceiling)")
ap.add_argument("--random_init", action="store_true",
                help="Use a torch.manual_seed(42) random-init actor instead of loading a checkpoint")
ap.add_argument("--task", default="Isaac-MushrDriftRL-v0",
                help="Gym task id used for rollouts (uniform across all agents)")
ap.add_argument("--num_envs", type=int, default=256)
ap.add_argument("--num_steps", type=int, default=500)
ap.add_argument("--num_buckets", type=int, default=20)
ap.add_argument("--mu_grid_n", type=int, default=20)
ap.add_argument("--mu_min", type=float, default=0.200)
ap.add_argument("--mu_max", type=float, default=1.000)
ap.add_argument("--results_csv", default="/workspace/probes/results/probe_results.csv")
ap.add_argument("--ood_csv", default="/workspace/probes/results/ood_metrics.csv")
ap.add_argument("--probe_types", default="linear_l2",
                help="Comma-separated subset of {linear_l2,mlp_l2,random_l2,raw_input}; "
                     "mlp_l2 only valid for milestone=100; random_l2 implies --random_init")
ap.add_argument("--mlp_max_train", type=int, default=200000,
                help="Maximum training samples for MLP probe; subsample if more "
                     "available (sklearn MLPRegressor on CPU; full 1M+ samples "
                     "is gratuitously slow without improving R²)")
ap.add_argument("--smoke_one_mu", type=float, default=None,
                help="If set, run only this single µ (smoke-test mode)")
ap.add_argument("--force", action="store_true",
                help="Overwrite existing CSV rows for this (agent,seed,run,milestone,probe_type)")
ap.add_argument("--seed_torch", type=int, default=42)
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
if not getattr(args, "headless", False):
    args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ─── Heavy imports after app launch ────────────────────────────────────────
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import wheeledlab_tasks  # registers gym tasks

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from sklearn.linear_model import RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error


ALPHA_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3, 1e4)


def build_actor():
    return nn.Sequential(
        nn.Linear(70, 64),
        nn.ELU(),
        nn.Linear(64, 64),
        nn.ELU(),
        nn.Linear(64, 2),
    )


def load_actor(ckpt_path, device):
    actor = build_actor().to(device)
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
    sd = ckpt.get("model_state_dict", ckpt)
    actor_sd = {k.replace("actor.", "", 1): v for k, v in sd.items() if k.startswith("actor.")}
    actor.load_state_dict(actor_sd, strict=True)
    actor.eval()
    return actor


def build_single_mu_buckets(mu, num_buckets):
    ranges = torch.tensor([(mu, mu), (mu, mu), (0.0, 0.0)], device="cpu")
    rng = torch.rand((num_buckets, 3), device="cpu")
    new_buckets = (ranges[:, 1] - ranges[:, 0]) * rng + ranges[:, 0]
    new_buckets[:, 1] = torch.min(new_buckets[:, 0], new_buckets[:, 1])
    return new_buckets


def set_friction(env, target_mu, num_buckets):
    em = env.unwrapped.event_manager
    friction_term_cfg = em.get_term_cfg("change_wheel_friction")
    friction_term_cfg.params["static_friction_range"] = (target_mu, target_mu)
    friction_term_cfg.params["dynamic_friction_range"] = (target_mu, target_mu)
    friction_term_cfg.func.material_buckets = build_single_mu_buckets(target_mu, num_buckets)


def detect_wheel_phys_indices(env, target_mu_for_test=0.42, num_buckets=20):
    """One-time empirical lookup: which axis-1 indices of mats received the bucket value.

    For the MuSHR rover this should return [2,3,4,5,6,7,8,9] (matches the
    documented `mats[:, 2:10, 0]` recipe in TROUBLESHOOTING_GUIDE.md §1).
    """
    set_friction(env, target_mu_for_test, num_buckets)
    env.reset()
    # env.reset() doesn't run get_observations(); for the friction round-trip
    # check we don't need stacked obs anyway.
    robot = env.unwrapped.scene["robot"]
    mats = robot.root_physx_view.get_material_properties()
    close = (mats[0, :, 0] - target_mu_for_test).abs() < 1e-3
    return [int(i) for i, c in enumerate(close.tolist()) if c]


def read_mu_per_env(env, wheel_phys_idx):
    robot = env.unwrapped.scene["robot"]
    mats = robot.root_physx_view.get_material_properties()
    return mats[:, wheel_phys_idx, 0].float().mean(dim=1).cpu().numpy()


TWO_PI = float(2 * np.pi)


def rollout_one_mu(env, actor, hook_module, target_mu, num_steps, num_buckets, wheel_phys_idx, device):
    """Run one rollout at fixed µ. Per-env first-episode metrics only.

    Returns dict with:
        obs_buf      : (num_envs, T, 70) — observation seen at each step
        act_buf      : (num_envs, T, 64) — Layer 2 activation at each step
        mu           : (num_envs,) — actual µ read back per env
        in_ep        : (num_envs, T) bool — True iff this env was still in its
                       first episode at step t (false from termination onward)
        first_lap_t  : (num_envs,) — seconds to first lap (inf if not reached
                       within first episode)
        completed    : (num_envs,) — env reached lap_progress>=2π in first ep
        oob          : (num_envs,) — env terminated (terminated flag, non-truncate)
        timeout      : (num_envs,) — env truncated (time-out, non-terminate)
    """
    set_friction(env, target_mu, num_buckets)
    env.reset()
    base = env.unwrapped
    n = base.num_envs
    # env.reset() returns 14-dim raw obs; the env's get_observations() rolls
    # the history buffer and produces the 70-dim stacked view we want.
    obs_dict = base.get_observations()

    obs_buf = torch.zeros((n, num_steps, 70), device=device)
    act_buf = torch.zeros((n, num_steps, 64), device=device)
    in_ep = torch.zeros((n, num_steps), dtype=torch.bool, device=device)
    has_terminated = torch.zeros(n, dtype=torch.bool, device=device)
    oob_flag = torch.zeros(n, dtype=torch.bool, device=device)
    timeout_flag = torch.zeros(n, dtype=torch.bool, device=device)
    completed_first_ep = torch.zeros(n, dtype=torch.bool, device=device)
    first_lap_t = torch.full((n,), float("inf"), device=device)

    captured = []
    def hook(_m, _i, out):
        captured.append(out.detach())

    handle = hook_module.register_forward_hook(hook)
    POLICY_DT = 0.02
    try:
        for t in range(num_steps):
            still_first_ep = ~has_terminated
            in_ep[:, t] = still_first_ep

            # Pre-step: capture obs + activation
            obs = obs_dict["policy"].to(device)  # (n, 70)
            obs_buf[:, t, :] = obs
            with torch.no_grad():
                action_mean = actor(obs)
            assert len(captured) >= 1, "hook did not fire"
            act_buf[:, t, :] = captured[-1]
            captured.clear()

            # Pre-step: lap-completion check based on lap_progress accumulated
            # through end of previous step (env's step() increments BEFORE super())
            laps_pre = base.lap_progress_buf / TWO_PI
            new_compl_in = still_first_ep & ~completed_first_ep & (laps_pre >= 1.0)
            if new_compl_in.any():
                t_sec = base.episode_length_buf.float() * POLICY_DT
                first_lap_t = torch.where(new_compl_in, t_sec, first_lap_t)
            completed_first_ep |= new_compl_in

            obs_dict, _, terminated, truncated, info = env.step(action_mean)
            new_term = terminated.bool()
            new_trunc = truncated.bool()

            # First-time terminal classification (only credit on first occurrence)
            first_term = new_term & ~has_terminated
            first_trunc = new_trunc & ~has_terminated & ~new_term
            oob_flag |= first_term
            timeout_flag |= first_trunc

            # Cross-and-terminate-same-step catch: env may have crossed lap AND
            # terminated in this single step; lap_progress was reset, but
            # last_ep_lap_progress retains the pre-reset value.
            just_terminated = (new_term | new_trunc) & ~has_terminated
            if just_terminated.any() and hasattr(base, "last_ep_lap_progress"):
                last_laps = base.last_ep_lap_progress
                last_len = base.last_ep_length_s if hasattr(base, "last_ep_length_s") \
                           else torch.zeros_like(last_laps)
                cross_at_term = just_terminated & ~completed_first_ep & (last_laps >= 1.0)
                if cross_at_term.any():
                    first_lap_t = torch.where(cross_at_term, last_len, first_lap_t)
                completed_first_ep |= cross_at_term

            has_terminated |= new_term | new_trunc
    finally:
        handle.remove()

    # End-of-rollout: for envs that never terminated, check final lap progress
    end_still_first_ep = ~has_terminated
    end_laps = base.lap_progress_buf / TWO_PI
    end_compl = end_still_first_ep & ~completed_first_ep & (end_laps >= 1.0)
    if end_compl.any():
        end_t = base.episode_length_buf.float() * POLICY_DT
        first_lap_t = torch.where(end_compl, end_t, first_lap_t)
    completed_first_ep |= end_compl

    mu_per_env = torch.tensor(read_mu_per_env(env, wheel_phys_idx), device=device)

    return {
        "obs": obs_buf,
        "act": act_buf,
        "mu": mu_per_env,
        "in_ep": in_ep,
        "first_lap_t": first_lap_t,
        "completed": completed_first_ep,
        "oob": oob_flag,
        "timeout": timeout_flag,
    }


def fit_ridge(X_train, y_train, X_test, y_test):
    model = RidgeCV(alphas=ALPHA_GRID, cv=5)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return r2_score(y_test, pred), mean_squared_error(y_test, pred), float(model.alpha_)


def fit_mlp(X_train, y_train, X_test, y_test):
    model = MLPRegressor(
        hidden_layer_sizes=(32,),
        activation="relu",
        solver="adam",
        learning_rate_init=1e-3,
        max_iter=100,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=10,
        random_state=42,
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return r2_score(y_test, pred), mean_squared_error(y_test, pred)


PROBE_HEADER = [
    "agent", "seed", "run_name", "milestone", "probe_type",
    "r2", "mse", "n_train_samples", "n_test_samples", "alpha", "notes",
]
OOD_HEADER = [
    "agent", "seed", "run_name", "milestone", "mu", "n_episodes",
    "completion_rate", "median_lap_time", "p25_lap_time", "p75_lap_time",
    "oob_rate", "timeout_rate",
]


def existing_probe_keys(path):
    if not os.path.exists(path):
        return set()
    keys = set()
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            keys.add((r["agent"], r["seed"], r["run_name"], r["milestone"], r["probe_type"]))
    return keys


def append_csv_row(path, header, row, force_keys=None):
    new_file = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if new_file:
            w.writeheader()
        w.writerow(row)


def main():
    print(f"[probe] checkpoint={args.checkpoint}")
    print(f"[probe] tag agent={args.agent} seed={args.seed} run={args.run_name} milestone={args.milestone}")
    np.random.seed(args.seed_torch)
    torch.manual_seed(args.seed_torch)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    probe_types = [p.strip() for p in args.probe_types.split(",") if p.strip()]
    milestone_int = int(args.milestone) if args.milestone else None
    if "mlp_l2" in probe_types and milestone_int != 100:
        print("[probe] WARN: mlp_l2 requested but milestone!=100; dropping mlp_l2")
        probe_types = [p for p in probe_types if p != "mlp_l2"]

    # Idempotency check
    existing = existing_probe_keys(args.results_csv)
    todo = []
    for pt in probe_types:
        key = (args.agent, str(args.seed), args.run_name, str(args.milestone), pt)
        if key in existing and not args.force:
            print(f"[probe] SKIP {pt}: already in CSV")
        else:
            todo.append(pt)
    if not todo:
        print("[probe] nothing to do; exiting")
        simulation_app.close()
        return

    # Build env
    print(f"[probe] building env task={args.task} num_envs={args.num_envs}")
    env_cfg = parse_env_cfg(args.task, num_envs=args.num_envs)
    env = gym.make(args.task, cfg=env_cfg)

    # Detect wheel rows in PhysX mats once
    print("[probe] detecting wheel PhysX indices...")
    wheel_phys_idx = detect_wheel_phys_indices(env, target_mu_for_test=0.42, num_buckets=args.num_buckets)
    print(f"[probe] wheel PhysX indices: {wheel_phys_idx}")
    if not wheel_phys_idx:
        print("[probe] FAIL: no wheel rows located in mats")
        simulation_app.close()
        sys.exit(1)

    # Load actor + register hook on actor[3]
    if args.random_init:
        print("[probe] using random-init actor (torch.manual_seed(42))")
        torch.manual_seed(42)
        actor = build_actor().to(device).eval()
    else:
        if not args.checkpoint:
            print("[probe] FAIL: --checkpoint required unless --random_init")
            simulation_app.close()
            sys.exit(1)
        print(f"[probe] loading actor from {args.checkpoint}")
        actor = load_actor(args.checkpoint, device)
    hook_module = actor[3]

    # ─── Friction grid loop ────────────────────────────────────────────────
    if args.smoke_one_mu is not None:
        mu_grid = np.array([args.smoke_one_mu])
    else:
        mu_grid = np.linspace(args.mu_min, args.mu_max, args.mu_grid_n)
    print(f"[probe] mu_grid: {mu_grid.tolist()}")

    # 70/30 env-level split, fixed
    n = args.num_envs
    rng = np.random.default_rng(args.seed_torch)
    perm = rng.permutation(n)
    n_train = int(round(0.7 * n))
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]
    print(f"[probe] env split: {len(train_idx)} train / {len(test_idx)} test")

    all_act = []   # list of (n_envs, T, 64) tensors per µ
    all_obs = []
    all_mu = []
    all_in_ep = []
    ood_rows = []

    t_start = time.time()
    for i, mu in enumerate(mu_grid):
        t_mu = time.time()
        out = rollout_one_mu(env, actor, hook_module, float(mu),
                             args.num_steps, args.num_buckets, wheel_phys_idx, device)
        elapsed = time.time() - t_mu

        # OOD metrics for this µ
        in_ep_mask = out["in_ep"].any(dim=1)  # any in-episode timestep
        n_ep = int(in_ep_mask.sum().item())
        comp_rate = float(out["completed"].float().mean().item())
        flt = out["first_lap_t"].cpu().numpy()
        flt_finite = flt[np.isfinite(flt)]
        if flt_finite.size > 0:
            median_lt = float(np.median(flt_finite))
            p25 = float(np.percentile(flt_finite, 25))
            p75 = float(np.percentile(flt_finite, 75))
        else:
            median_lt = p25 = p75 = float("nan")
        oob_rate = float(out["oob"].float().mean().item())
        timeout_rate = float(out["timeout"].float().mean().item())
        ood_rows.append({
            "agent": args.agent, "seed": args.seed, "run_name": args.run_name,
            "milestone": args.milestone, "mu": float(mu), "n_episodes": n_ep,
            "completion_rate": comp_rate, "median_lap_time": median_lt,
            "p25_lap_time": p25, "p75_lap_time": p75,
            "oob_rate": oob_rate, "timeout_rate": timeout_rate,
        })

        all_act.append(out["act"])
        all_obs.append(out["obs"])
        all_mu.append(out["mu"])
        all_in_ep.append(out["in_ep"])
        print(f"[probe] µ={mu:.3f}  comp={comp_rate:.2f}  oob={oob_rate:.2f}  "
              f"timeout={timeout_rate:.2f}  ({elapsed:.1f}s)")

    elapsed_rollout = time.time() - t_start
    print(f"[probe] all rollouts done in {elapsed_rollout:.1f}s "
          f"({elapsed_rollout / max(len(mu_grid),1):.1f}s/µ)")

    # ─── Build flat (n_samples, F) dataset using only in-episode timesteps ─
    # Each env at each µ contributes up to num_steps samples.
    # Train/test split is at env-level for this checkpoint.
    def gather(env_ids):
        Xs_act, Xs_obs, ys = [], [], []
        for i, mu in enumerate(mu_grid):
            act = all_act[i][env_ids]            # (k, T, 64)
            obs = all_obs[i][env_ids]            # (k, T, 70)
            mu_per_env = all_mu[i][env_ids]      # (k,)
            in_ep_mask = all_in_ep[i][env_ids]   # (k, T)
            # broadcast mu across timesteps
            mu_full = mu_per_env.unsqueeze(1).expand(-1, args.num_steps)  # (k, T)
            mask_flat = in_ep_mask.reshape(-1)
            Xs_act.append(act.reshape(-1, 64)[mask_flat])
            Xs_obs.append(obs.reshape(-1, 70)[mask_flat])
            ys.append(mu_full.reshape(-1)[mask_flat])
        return torch.cat(Xs_act).cpu().numpy(), torch.cat(Xs_obs).cpu().numpy(), torch.cat(ys).cpu().numpy()

    print("[probe] gathering flat datasets...")
    X_act_train, X_obs_train, y_train = gather(train_idx)
    X_act_test, X_obs_test, y_test = gather(test_idx)
    print(f"[probe] train samples: {len(y_train)}, test samples: {len(y_test)}")

    # ─── Fit probes ────────────────────────────────────────────────────────
    notes = ""
    if "linear_l2" in todo:
        t = time.time()
        r2, mse, alpha = fit_ridge(X_act_train, y_train, X_act_test, y_test)
        print(f"[probe] linear_l2 R²={r2:.4f}  MSE={mse:.5f}  α={alpha:.3g}  ({time.time()-t:.1f}s)")
        row_notes = "r2_out_of_range" if not (-1.0 <= r2 <= 1.0) else ""
        append_csv_row(args.results_csv, PROBE_HEADER, {
            "agent": args.agent, "seed": args.seed, "run_name": args.run_name,
            "milestone": args.milestone, "probe_type": "linear_l2",
            "r2": r2, "mse": mse,
            "n_train_samples": len(y_train), "n_test_samples": len(y_test),
            "alpha": alpha, "notes": row_notes,
        })

    if "mlp_l2" in todo:
        t = time.time()
        if len(y_train) > args.mlp_max_train:
            sub_idx = rng.choice(len(y_train), size=args.mlp_max_train, replace=False)
            X_mlp_train = X_act_train[sub_idx]
            y_mlp_train = y_train[sub_idx]
            print(f"[probe] mlp subsample: {args.mlp_max_train}/{len(y_train)} train samples")
        else:
            X_mlp_train = X_act_train
            y_mlp_train = y_train
        r2, mse = fit_mlp(X_mlp_train, y_mlp_train, X_act_test, y_test)
        print(f"[probe] mlp_l2 R²={r2:.4f}  MSE={mse:.5f}  ({time.time()-t:.1f}s)")
        row_notes = "r2_out_of_range" if not (-1.0 <= r2 <= 1.0) else ""
        append_csv_row(args.results_csv, PROBE_HEADER, {
            "agent": args.agent, "seed": args.seed, "run_name": args.run_name,
            "milestone": args.milestone, "probe_type": "mlp_l2",
            "r2": r2, "mse": mse,
            "n_train_samples": len(y_train), "n_test_samples": len(y_test),
            "alpha": "", "notes": row_notes,
        })

    if "random_l2" in todo:
        # Same fit recipe as linear_l2; the difference is that the actor is
        # random-init (caller supplies --random_init). We tag it under
        # probe_type='random_l2' so figures can find it.
        t = time.time()
        r2, mse, alpha = fit_ridge(X_act_train, y_train, X_act_test, y_test)
        print(f"[probe] random_l2 R²={r2:.4f}  MSE={mse:.5f}  α={alpha:.3g}  ({time.time()-t:.1f}s)")
        row_notes = "r2_out_of_range" if not (-1.0 <= r2 <= 1.0) else ""
        append_csv_row(args.results_csv, PROBE_HEADER, {
            "agent": args.agent, "seed": args.seed, "run_name": args.run_name,
            "milestone": args.milestone, "probe_type": "random_l2",
            "r2": r2, "mse": mse,
            "n_train_samples": len(y_train), "n_test_samples": len(y_test),
            "alpha": alpha, "notes": row_notes,
        })

    if "raw_input" in todo:
        # Probe on raw 70-dim observations (history-stacked). Independent of
        # encoder; establishes the linearly-decodable µ-information ceiling
        # that the encoder either improves on or compresses away.
        t = time.time()
        r2, mse, alpha = fit_ridge(X_obs_train, y_train, X_obs_test, y_test)
        print(f"[probe] raw_input R²={r2:.4f}  MSE={mse:.5f}  α={alpha:.3g}  ({time.time()-t:.1f}s)")
        row_notes = "r2_out_of_range" if not (-1.0 <= r2 <= 1.0) else ""
        append_csv_row(args.results_csv, PROBE_HEADER, {
            "agent": args.agent, "seed": args.seed, "run_name": args.run_name,
            "milestone": args.milestone, "probe_type": "raw_input",
            "r2": r2, "mse": mse,
            "n_train_samples": len(y_train), "n_test_samples": len(y_test),
            "alpha": alpha, "notes": row_notes,
        })

    # ─── Append OOD rows ───────────────────────────────────────────────────
    for r in ood_rows:
        append_csv_row(args.ood_csv, OOD_HEADER, r)
    print(f"[probe] wrote {len(ood_rows)} ood rows to {args.ood_csv}")

    env.close()
    simulation_app.close()
    print("[probe] DONE")


if __name__ == "__main__":
    main()
