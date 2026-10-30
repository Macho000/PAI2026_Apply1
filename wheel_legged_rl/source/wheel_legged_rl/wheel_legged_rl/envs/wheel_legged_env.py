"""``ManagerBasedRLEnv`` + zero-weight reward pruning + per-step diagnostics for W&B.

Diagnostics are written to ``extras["log"]`` under ``Diagnostics/<name>``; rsl_rl averages every key of
``extras["log"]`` over the rollout and forwards it to TensorBoard / W&B.
"""

from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedRLEnv

from ..evaluation.diagnostics import RollingRatio, cost_of_transport, slip_ratio
from ..rewards.rewards import contact_mask, mechanical_power_from_tensors, wheel_slip_from_tensors
from ..terrains.layout import column_terrain_names
from .base_env_cfg import WheelLeggedEnvCfg


def prune_zero_weight_rewards(cfg: WheelLeggedEnvCfg) -> list[str]:
    """Disable reward terms with weight 0 (they would still be computed every step). Returns the removed names."""
    removed = []
    for name in list(vars(cfg.rewards)):
        term = getattr(cfg.rewards, name)
        if term is not None and hasattr(term, "weight") and term.weight == 0.0:
            setattr(cfg.rewards, name, None)
            removed.append(name)
    return removed


class WheelLeggedEnv(ManagerBasedRLEnv):
    cfg: WheelLeggedEnvCfg

    def __init__(self, cfg: WheelLeggedEnvCfg, render_mode: str | None = None, **kwargs):
        self.disabled_rewards = prune_zero_weight_rewards(cfg)
        super().__init__(cfg, render_mode, **kwargs)
        robot = self.scene["robot"]
        wl = self.cfg.wl
        self._robot = robot
        self._leg_ids, _ = robot.find_joints(wl.leg_joint_names, preserve_order=False)
        self._wheel_ids, _ = robot.find_joints(wl.wheel_joint_names, preserve_order=False)
        self._wheel_body_ids, _ = robot.find_bodies(wl.wheel_body_names, preserve_order=True)
        self._contact = self.scene.sensors["contact_forces"]
        self._wheel_sensor_ids, _ = self._contact.find_bodies(wl.wheel_body_names, preserve_order=True)
        self._wheel_axis = torch.tensor(wl.wheel_axis, device=self.device, dtype=torch.float)
        self._mass = float(robot.root_physx_view.get_masses().sum(dim=1).mean())  # mean over the randomised robots
        self._fall_rate = RollingRatio(window=int(self.max_episode_length))
        self._init_terrain_groups()
        # privileged critic terms cache values of the startup randomisation, which ran after they were built
        for terms in self.observation_manager._group_obs_class_term_cfgs.values():
            for term_cfg in terms:
                if hasattr(term_cfg.func, "refresh"):
                    term_cfg.func.refresh()
        print(f"[wheel_legged_rl] disabled zero-weight rewards: {self.disabled_rewards}")
        print(f"[wheel_legged_rl] sim joint order: {robot.data.joint_names}")

    def _init_terrain_groups(self) -> None:
        """Sub-terrain type of every robot (fixed: the curriculum only changes the row) for per-terrain diagnostics."""
        self._terrain_names: list[str] = []
        terrain = self.scene.terrain
        generator = self.cfg.scene.terrain.terrain_generator
        if generator is None or not generator.curriculum or getattr(terrain, "terrain_types", None) is None:
            return
        columns = column_terrain_names(
            {name: sub.proportion for name, sub in generator.sub_terrains.items()}, generator.num_cols
        )
        self._terrain_names = list(dict.fromkeys(columns))
        col_to_group = torch.tensor([self._terrain_names.index(c) for c in columns], device=self.device)
        self._terrain_group = col_to_group[terrain.terrain_types]
        self._terrain_group_size = torch.bincount(self._terrain_group, minlength=len(self._terrain_names)).clamp(min=1)

    def desynchronize_timers(self) -> None:
        """Spread the command-resampling and interval-event (push) timers uniformly over their periods.

        Call at the start of training, with ``init_at_random_ep_len``. Otherwise every robot changes its command and
        gets its first push at the same moment (10 s in), so the first ~20 PPO iterations -- e.g. a critic warm-up --
        see neither, and the value error jumps when they all happen at once.
        """
        for term in self.command_manager._terms.values():
            term.time_left.uniform_(0.0, term.cfg.resampling_time_range[1])
        events = self.event_manager
        interval = events._mode_term_cfgs.get("interval", [])
        for cfg, time_left in zip(interval, events._interval_term_time_left, strict=True):
            if not cfg.is_global_time:
                time_left.uniform_(0.0, cfg.interval_range_s[1])

    def step(self, action: torch.Tensor):
        obs, rew, terminated, truncated, extras = super().step(action)
        log = dict(extras.get("log", {}))  # new dict each step: rsl_rl keeps references to earlier ones
        log.update(self._diagnostics(terminated, truncated))
        extras["log"] = log
        return obs, rew, terminated, truncated, extras

    @torch.no_grad()
    def _diagnostics(self, terminated: torch.Tensor, truncated: torch.Tensor) -> dict[str, float | torch.Tensor]:
        """0-d tensors are passed through (rsl_rl accepts them) to avoid a GPU sync per metric."""
        data = self._robot.data
        ended = int((terminated | truncated).sum())
        self._fall_rate.update(int(terminated.sum()), ended)

        torque, vel = data.applied_torque, data.joint_vel
        p_leg = mechanical_power_from_tensors(torque[:, self._leg_ids], vel[:, self._leg_ids])
        p_wheel = mechanical_power_from_tensors(torque[:, self._wheel_ids], vel[:, self._wheel_ids])
        speed = data.root_lin_vel_b[:, :2].norm(dim=-1)
        cot, epm = cost_of_transport(p_leg + p_wheel, self._mass, speed)

        ids = self._wheel_body_ids
        slip = wheel_slip_from_tensors(
            data.body_link_lin_vel_w[:, ids],
            data.body_link_ang_vel_w[:, ids],
            data.body_link_quat_w[:, ids],
            self._wheel_axis,
            self.cfg.wl.wheel_radius,
        )
        rolling_speed = data.joint_vel[:, self._wheel_ids] * self.cfg.wl.wheel_radius
        contact = contact_mask(self._contact.data.net_forces_w_history[:, :, self._wheel_sensor_ids], 1.0)

        am = self.action_manager
        action_rate = (am.action - am.prev_action).square().sum(dim=-1).mean()

        out = {
            "Diagnostics/fall_rate": self._fall_rate.value,
            "Diagnostics/mech_power_leg": p_leg.mean(),
            "Diagnostics/mech_power_wheel": p_wheel.mean(),
            "Diagnostics/cost_of_transport": cot,
            "Diagnostics/energy_per_meter": epm,
            "Diagnostics/wheel_slip_ratio": slip_ratio(slip, rolling_speed, contact),
            "Diagnostics/wheel_contact_ratio": contact.float().mean(),
            "Diagnostics/action_rate": action_rate,
            "Diagnostics/base_speed": speed.mean(),
        }
        terrain = self.scene.terrain
        if getattr(self.cfg.curriculum, "terrain_levels", None) is not None and terrain.terrain_levels is not None:
            out["Diagnostics/terrain_level_mean"] = terrain.terrain_levels.float().mean()
        if self._terrain_names:
            cmd = self.command_manager.get_command("base_velocity")[:, :2]
            vel_error = (data.root_lin_vel_b[:, :2] - cmd).norm(dim=-1)
            k = len(self._terrain_names)
            group, size = self._terrain_group, self._terrain_group_size
            err = torch.zeros(k, device=self.device).index_add_(0, group, vel_error) / size
            level = torch.zeros(k, device=self.device).index_add_(0, group, terrain.terrain_levels.float()) / size
            for i, name in enumerate(self._terrain_names):
                out[f"Diagnostics/{name}/vel_error"] = err[i]
                out[f"Diagnostics/{name}/terrain_level"] = level[i]
        return out
