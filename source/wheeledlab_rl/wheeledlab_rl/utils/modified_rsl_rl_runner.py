import warnings
import rsl_rl
from rsl_rl import runners
import os

try:
    from tqdm import TqdmExperimentalWarning

    # Remove experimental warning
    warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
    from tqdm.rich import tqdm
except ImportError:
    # Rich not installed, we only throw an error
    # if the progress bar is used
    tqdm = None

import time
import torch
from collections import deque


class OnPolicyRunner(runners.OnPolicyRunner):
    """Override for logging purposes"""

    def __init__(self, env, agent_cfg, log_cfg, device="cpu"):
        super().__init__(env, agent_cfg, log_cfg.run_log_dir, device)
        self.no_log = log_cfg.no_log
        self.no_wandb = log_cfg.no_wandb
        self.logger_type = None
        if not self.no_wandb:
            self.logger_type = "wandb"

    def _milestone_iters(self, start_iter: int, num_learning_iterations: int) -> dict:
        """Return {absolute_iteration: pct_label} for 0/25/50/75/100% milestones."""
        tot_iter = start_iter + num_learning_iterations
        milestones = {}
        for pct in [0, 25, 50, 75, 100]:
            if pct == 0:
                it = start_iter
            elif pct == 100:
                it = tot_iter - 1
            else:
                it = start_iter + round(num_learning_iterations * pct / 100)
                it = max(start_iter, min(tot_iter - 1, it))
            milestones[it] = pct
        return milestones

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # initialize writer
        if not self.no_log and self.logger_type == "wandb":
            from rsl_rl.utils.wandb_utils import WandbSummaryWriter

            self.writer = WandbSummaryWriter(
                log_dir=self.log_dir, flush_secs=10, cfg=self.cfg
            )
            self.writer.log_config(
                self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg
            )

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
        obs = self.env.get_observations()
        critic_obs = obs.get("critic", obs.get("policy", obs))
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        self.train_mode()  # switch to train mode (for dropout for example)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.device
        )
        cur_episode_length = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.device
        )

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations

        # --- Milestone setup ----------------------------------------------- #
        milestone_iters = self._milestone_iters(start_iter, num_learning_iterations)
        os.makedirs(os.path.join(self.log_dir, "models"), exist_ok=True)
        print(
            "[MilestoneRunner] Milestones: "
            + ", ".join(f"{pct}%→iter{it}" for it, pct in sorted(milestone_iters.items()))
        )
        # ------------------------------------------------------------------- #

        for it in tqdm(range(start_iter, tot_iter)):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs)
                    obs, rewards, dones, infos = self.env.step(actions)
                    if i == 0:
                        print(f"[INFOS KEYS STEP] {list(infos.keys())}")
                        if "episode" in infos:
                            print(f"[EPISODE KEYS] {list(infos['episode'].keys())}")
                    if actions.isnan().any():
                        raise ValueError("NaN in actions")
                    critic_obs = obs
                    obs, critic_obs, rewards, dones = (
                        obs.to(self.device),
                        critic_obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                    )
                    self.alg.process_env_step(obs, rewards, dones, infos)

                    if not self.no_log:
                        new_ids = (dones > 0).nonzero(as_tuple=False)

                        if new_ids.numel() > 0:
                            try:
                                _raw = self.env
                                while hasattr(_raw, 'env'):
                                    _raw = _raw.env
                                mats = _raw.unwrapped.scene["robot"].root_physx_view.get_material_properties()
                                wheel_mu = mats[:, 2:10, 0].mean(dim=1).cpu()
                                ep_mu = wheel_mu[new_ids[:, 0].cpu()].tolist()
                                if "episode" not in infos:
                                    infos["episode"] = {}
                                infos["episode"]["mu"] = sum(ep_mu) / len(ep_mu)
                            except Exception as _e:
                                print(f"[MU INJECT ERROR] {_e}")

                            # Step 3+4: log progress, episode length, radial distance
                            try:
                                _env = self.env
                                while hasattr(_env, 'env'):
                                    _env = _env.env
                                done_ids = new_ids[:, 0].cpu()
                                # track_progress_completed: laps finished this episode
                                prog = _env.last_ep_lap_progress[done_ids].tolist()
                                # episode_length_s: seconds elapsed this episode
                                ep_s = _env.last_ep_length_s[done_ids].tolist()
                                # min_radial_distance: minimum distance to circuit centre
                                min_rad = _env.last_ep_min_radial[done_ids].tolist()
                                if "episode" not in infos:
                                    infos["episode"] = {}
                                infos["episode"]["track_progress_completed"] = (
                                    sum(prog) / len(prog)
                                )
                                infos["episode"]["episode_length_s"] = (
                                    sum(ep_s) / len(ep_s)
                                )
                                infos["episode"]["min_radial_distance"] = (
                                    sum(min_rad) / len(min_rad)
                                )
                            except Exception as _e:
                                print(f"[PROGRESS LOG ERROR] {_e}")

                        ep_info = {}
                        if "log" in infos:
                            ep_info.update(infos["log"])
                        if "episode" in infos:
                            ep_info.update(infos["episode"])
                        if ep_info:
                            ep_infos.append(ep_info)

                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        rewbuffer.extend(
                            cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist()
                        )
                        lenbuffer.extend(
                            cur_episode_length[new_ids][:, 0].cpu().numpy().tolist()
                        )
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                start = stop
                self.alg.compute_returns(critic_obs)

            loss_dict = self.alg.update()
            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it
            if not self.no_log:
                if not self.no_wandb:
                    self.log(locals())
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, "models", f"model_{it}.pt"))

            # --- Milestone save -------------------------------------------- #
            if it in milestone_iters:
                pct = milestone_iters[it]
                path = os.path.join(
                    self.log_dir, "models", f"model_milestone_iter{it}_{pct}pct.pt"
                )
                self.save(path)
                print(f"[MilestoneRunner] ✓ {pct}% checkpoint → {path}")
            # ---------------------------------------------------------------- #

            ep_infos.clear()