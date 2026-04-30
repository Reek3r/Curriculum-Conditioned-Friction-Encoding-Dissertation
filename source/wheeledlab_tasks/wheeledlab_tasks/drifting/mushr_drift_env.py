"""
Custom ManagerBasedRLEnv subclass for MuSHR drift task.
Adds per-environment lap progress accumulator with clean reset semantics,
episode-end snapshots for W&B logging, and k=5 observation history buffer.
"""
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

TWO_PI = 2.0 * torch.pi
POLICY_DT = 0.02  # sim_dt=0.005 x decimation=4

# Track centre is at world origin (0, 0). CORNER_IN_RADIUS = 0.3 m.
# If min_radial_distance < CORNER_IN_RADIUS the T_max physics is wrong.
CORNER_IN_RADIUS = 0.3

# History buffer constants
HISTORY_LEN = 5   # k: number of frames stacked
OBS_DIM = 14      # single-frame observation dimension
STACK_DIM = HISTORY_LEN * OBS_DIM  # 70: MLP input dimension


class MushrDriftEnv(ManagerBasedRLEnv):
    """ManagerBasedRLEnv with lap progress accumulator, episode snapshots,
    and k=5 observation history buffer.

    Buffers available for logging after each step():
        last_ep_lap_progress  (num_envs,): laps completed at episode end
        last_ep_length_s      (num_envs,): episode duration in seconds
        last_ep_min_radial    (num_envs,): min distance to circuit centre (m)

    Observation contract:
        Raw obs from observation_manager: (num_envs, 14)
        Returned obs["policy"] after stacking: (num_envs, 70)
        Zero-padded at episode start for first k-1 steps.
        rsl_rl MLP input layer is sized from get_observations() at runner
        init, which returns (num_envs, 70), so no manual cfg change needed.
    """

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode=render_mode, **kwargs)

        # ── Live accumulators ──────────────────────────────────────────────
        self.lap_progress_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float32
        )
        self.prev_lap_progress_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float32
        )
        self.min_radial_buf = torch.full(
            (self.num_envs,), fill_value=float('inf'),
            device=self.device, dtype=torch.float32
        )

        # ── Snapshots: stable after step() returns, readable by runner ─────
        self.last_ep_lap_progress = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float32
        )
        self.last_ep_length_s = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float32
        )
        self.last_ep_min_radial = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float32
        )

        # ── History buffer ─────────────────────────────────────────────────
        # Shape: (num_envs, k, obs_dim). Zero-initialised = correct zero-pad
        # for the first k-1 steps of each episode.
        self.obs_history_buf = torch.zeros(
            (self.num_envs, HISTORY_LEN, OBS_DIM),
            device=self.device, dtype=torch.float32
        )

    # ── Observation overrides ──────────────────────────────────────────────

    def get_observations(self) -> dict:
        """Return stacked observations (num_envs, 70).

        Called by rsl_rl at runner init to size the MLP, and again at the
        start of learn(). Must return the same shape as step() to ensure the
        network input dimension is consistent.

        At runner init the history buffer is all zeros (correct: episode start
        zero-padding). The raw obs is inserted into slot[-1] so the buffer
        advances correctly even when called outside step().
        """
        obs_dict = self.observation_manager.compute()
        raw_obs = obs_dict["policy"]  # (num_envs, 14)

        # Validate OBS_DIM on first call — catches config mismatches early
        if raw_obs.shape[1] != OBS_DIM:
            raise ValueError(
                f"[MushrDriftEnv] Expected OBS_DIM={OBS_DIM} but observation "
                f"manager returned shape {raw_obs.shape}. Update OBS_DIM."
            )

        # Advance ring buffer and insert new obs
        self.obs_history_buf = torch.roll(self.obs_history_buf, shifts=-1, dims=1)
        self.obs_history_buf[:, -1, :] = raw_obs

        obs_dict["policy"] = self.obs_history_buf.reshape(self.num_envs, STACK_DIM)
        return obs_dict

    def _get_observations(self) -> dict:
        """Fallback path used by DirectRLEnv-style callers. Same override."""
        return self.get_observations()

    # ── Step ───────────────────────────────────────────────────────────────

    def step(self, action):
        # Snapshot and increment lap progress BEFORE super().step() so that
        # lap_bonus (called inside reward_manager.compute()) reads consistent
        # pre/post-step buffer values.
        self.prev_lap_progress_buf = self.lap_progress_buf.clone()

        asset: RigidObject = self.scene[SceneEntityCfg("robot").name]
        omega = asset.data.root_link_ang_vel_w[..., 2]
        delta = (omega * POLICY_DT).clamp(min=0.0)
        self.lap_progress_buf += delta

        if hasattr(self, 'first_lap_time'):
            # Must run BEFORE super().step() at line 135 so first_lap_time is finite
            # when MushrDriftADREnv._reset_idx reads it on episodes where lap-cross and
            # termination coincide. Do not move below super().step().
            just_crossed = (self.lap_progress_buf / TWO_PI >= 1.0) & \
                           (self.first_lap_time == float('inf'))
            if just_crossed.any():
                self.first_lap_time[just_crossed] = \
                    self.episode_length_buf[just_crossed].float() * POLICY_DT

        # Update running minimum radial distance from circuit centre (x=0, y=0)
        pos = asset.data.root_link_pos_w
        radial = torch.sqrt(pos[:, 0] ** 2 + pos[:, 1] ** 2)
        self.min_radial_buf = torch.minimum(self.min_radial_buf, radial)

        # Call super — observation_manager.compute() runs here, reward fires here
        obs_dict, reward, terminated, truncated, info = super().step(action)

        # Stack observations: advance ring buffer, insert new raw obs
        raw_obs = obs_dict["policy"]  # (num_envs, 14) from observation_manager
        self.obs_history_buf = torch.roll(self.obs_history_buf, shifts=-1, dims=1)
        self.obs_history_buf[:, -1, :] = raw_obs
        obs_dict["policy"] = self.obs_history_buf.reshape(self.num_envs, STACK_DIM)

        return obs_dict, reward, terminated, truncated, info

    # ── Reset ──────────────────────────────────────────────────────────────

    def _reset_idx(self, env_ids):
        # Snapshot BEFORE super() zeroes episode_length_buf and before we
        # zero lap_progress_buf. Runner reads these after step() returns.
        self.last_ep_lap_progress[env_ids] = (
            self.lap_progress_buf[env_ids] / TWO_PI
        )
        self.last_ep_length_s[env_ids] = (
            self.episode_length_buf[env_ids].float() * POLICY_DT
        )
        self.last_ep_min_radial[env_ids] = self.min_radial_buf[env_ids]

        # Now reset
        super()._reset_idx(env_ids)
        self.lap_progress_buf[env_ids] = 0.0
        self.prev_lap_progress_buf[env_ids] = 0.0
        self.min_radial_buf[env_ids] = float('inf')

        # Zero-pad history buffer for reset envs
        self.obs_history_buf[env_ids] = 0.0

class MushrDriftADREnv(MushrDriftEnv):
    """MushrDriftEnv with Automatic Domain Randomisation for friction.

    Starts with narrow µ range identical to Agent A (0.76, 0.84).
    Expands symmetrically toward Agent B's range (0.2, 1.0) when
    success rate over ADR_WINDOW episodes exceeds ADR_THRESHOLD.

    Success gate:
        (laps_completed >= 1.0) AND (episode_time_s <= T_MAX)

    Expansion is one-directional (no contraction).
    """

    T_MAX         = 5.5
    ADR_STEP      = 0.02
    ADR_WINDOW    = 200 #TEMP Revert to 200
    ADR_THRESHOLD = 0.8 #TEMP Revert to 0.8
    ADR_START_LO  = 0.76
    ADR_START_HI  = 0.84
    ADR_TARGET_LO = 0.2
    ADR_TARGET_HI = 1.0

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        self.adr_mu_lo = self.ADR_START_LO
        self.adr_mu_hi = self.ADR_START_HI
        self._adr_successes = 0
        self._adr_total = 0
        # Track time of first lap completion per episode
        self.first_lap_time = torch.full(
            (self.num_envs,), float('inf'),
            device=self.device, dtype=torch.float32
        )

    def _reset_idx(self, env_ids):
        # ── Score gate BEFORE reset zeroes buffers ───────────────────
        laps    = self.lap_progress_buf[env_ids] / TWO_PI
        success = (laps >= 1.0) & (self.first_lap_time[env_ids] <= self.T_MAX)

        # TEMPORARY DIAGNOSTIC — re-enable for debugging only
        # if len(env_ids) > 0:
        #     n_resets = len(env_ids)
        #     completed_mask = laps >= 1.0
        #     n_lap1 = int(completed_mask.sum().item())
        #     flt = self.first_lap_time[env_ids]
        #     finite_mask = completed_mask & (flt < float('inf'))
        #     finite_times = flt[finite_mask]
        #     n_passed = int(((laps >= 1.0) & (flt <= self.T_MAX)).sum().item())
        # 
        #     if finite_times.numel() > 0:
        #         p50 = finite_times.median().item()
        #         p90 = finite_times.quantile(0.9).item()
        #         p50_str = f"{p50:.2f}s"
        #         p90_str = f"{p90:.2f}s"
        #     else:
        #         p50_str = "—"
        #         p90_str = "—"
        # 
        #     print(f"[ADR-DIAG] resets={n_resets} | "
        #           f"lap>=1: {n_lap1}/{n_resets} ({100*n_lap1/max(n_resets,1):.0f}%) | "
        #           f"flt p50={p50_str} p90={p90_str} | "
        #           f"passed: {n_passed} | "
        #           f"window: {self._adr_successes}/{self._adr_total} | "
        #           f"mu=[{self.adr_mu_lo:.3f},{self.adr_mu_hi:.3f}]")

        self._adr_successes += int(success.sum().item())
        self._adr_total     += len(env_ids)

        # ── Evaluate expansion when window fills ─────────────────────
        if self._adr_total >= self.ADR_WINDOW:
            rate = self._adr_successes / self._adr_total
            if rate >= self.ADR_THRESHOLD:
                self.adr_mu_lo = max(self.ADR_TARGET_LO,
                                     self.adr_mu_lo - self.ADR_STEP)
                self.adr_mu_hi = min(self.ADR_TARGET_HI,
                                     self.adr_mu_hi + self.ADR_STEP)
                print(f"[ADR] Expanded mu -> ({self.adr_mu_lo:.3f}, {self.adr_mu_hi:.3f})")
            self._adr_successes = 0
            self._adr_total     = 0

        # ── Patch friction range in event manager BEFORE reset fires ─
        friction_term = self.event_manager.get_term_cfg("change_wheel_friction")
        friction_term.params["static_friction_range"] = (self.adr_mu_lo, self.adr_mu_hi)
        friction_term.params["dynamic_friction_range"] = (self.adr_mu_lo, self.adr_mu_hi)

        # ── Delegate to MushrDriftEnv._reset_idx (snapshots + reset) ─
        super()._reset_idx(env_ids)
        self.first_lap_time[env_ids] = float('inf')