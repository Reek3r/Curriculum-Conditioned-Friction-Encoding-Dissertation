"""
Custom ManagerBasedRLEnv subclass for MuSHR drift task.
Adds per-environment lap progress accumulator with clean reset semantics.
"""

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

TWO_PI = 2.0 * torch.pi
POLICY_DT = 0.02  # sim_dt=0.005 x decimation=4


class MushrDriftEnv(ManagerBasedRLEnv):
    """ManagerBasedRLEnv with lap progress accumulator.

    Adds:
        lap_progress_buf (torch.Tensor): shape (num_envs,). Cumulative angular
            progress in radians this episode, clamped at zero (reverse driving
            does not reduce progress). Divide by 2*pi for lap fraction.
    """

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        self.lap_progress_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float32
        )
        self.prev_lap_progress_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float32
        )

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        self.lap_progress_buf[env_ids] = 0.0
        self.prev_lap_progress_buf[env_ids] = 0.0

    def step(self, action):
        # Snapshot BEFORE increment so lap_bonus reads consistent pre/post values
        self.prev_lap_progress_buf = self.lap_progress_buf.clone()
        asset: RigidObject = self.scene[SceneEntityCfg("robot").name]
        omega = asset.data.root_link_ang_vel_w[..., 2]
        delta = (omega * POLICY_DT).clamp(min=0.0)
        self.lap_progress_buf += delta
        return super().step(action)