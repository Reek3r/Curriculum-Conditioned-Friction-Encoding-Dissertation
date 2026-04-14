"""
Custom ManagerBasedRLEnv subclass for MuSHR drift task.
Adds per-environment lap progress accumulator with clean reset semantics,
and episode-end snapshots for W&B logging.
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


class MushrDriftEnv(ManagerBasedRLEnv):
    """ManagerBasedRLEnv with lap progress accumulator and episode snapshots.

    Buffers available for logging after each step():
        last_ep_lap_progress  (num_envs,): laps completed at episode end
        last_ep_length_s      (num_envs,): episode duration in seconds
        last_ep_min_radial    (num_envs,): min distance to circuit centre (m)
    """

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        # Live accumulators
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
        # Snapshots: stable after step() returns, readable by runner
        self.last_ep_lap_progress = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float32
        )
        self.last_ep_length_s = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float32
        )
        self.last_ep_min_radial = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float32
        )

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

    def step(self, action):
        # Snapshot and increment lap progress BEFORE super().step() so that
        # lap_bonus (called inside reward_manager.compute()) reads consistent
        # pre/post-step buffer values.
        self.prev_lap_progress_buf = self.lap_progress_buf.clone()
        asset: RigidObject = self.scene[SceneEntityCfg("robot").name]
        omega = asset.data.root_link_ang_vel_w[..., 2]
        delta = (omega * POLICY_DT).clamp(min=0.0)
        self.lap_progress_buf += delta
        # Update running minimum radial distance from circuit centre (x=0, y=0)
        pos = asset.data.root_link_pos_w
        radial = torch.sqrt(pos[:, 0] ** 2 + pos[:, 1] ** 2)
        self.min_radial_buf = torch.minimum(self.min_radial_buf, radial)
        return super().step(action)