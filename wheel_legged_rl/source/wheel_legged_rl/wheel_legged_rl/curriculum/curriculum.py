"""In-phase curricula: command range growth per axis (``command_levels``) and terrain rows
(``terrain_levels_tracking``)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from isaaclab.managers import CurriculumTermCfg, ManagerTermBase

from .levels import AXES, expand_range, should_expand, stand_gate_passed, terrain_level_moves

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class command_levels(ManagerTermBase):
    """Grow ``ranges.<axis>`` toward ``limit_ranges.<axis>`` when tracking is good.

    Evaluated at most once per ``max_episode_length`` env steps, on the envs being reset (their episodic sums are
    still available because curricula run before the reward manager resets).

    With ``stand_gate`` the range is held at its initial value (a stand-still-only range) until standing still has
    been learnt; only then does it grow. The gate is judged on every episode that ended since the previous check:

    * ``min_time_out_ratio``: fraction of those episodes that ended by time-out rather than by a fall;
    * ``max_cost``: ``{reward_term: limit}`` on the raw per-step value of a (penalty) reward term, i.e. its episodic
      sum / weight / episode duration -- e.g. ``stand_still`` [rad] and ``base_height`` [m^2];
    * ``consecutive``: number of consecutive passing checks required.

    Once passed, the gate stays open. The stage is not stored in checkpoints: a resumed run starts gated again (a
    policy that already stands passes after ``consecutive`` checks).
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._last_update = 0
        gate = cfg.params.get("stand_gate")
        self._gate_open = gate is None
        self._gate_passes = 0
        self._cost_names = list((gate or {}).get("max_cost", {}))
        inactive = set(self._cost_names) - set(env.reward_manager.active_terms)
        if inactive:
            raise ValueError(f"stand_gate.max_cost uses disabled reward terms {sorted(inactive)} (give them a weight)")
        self._reset_stats()
        self._log: dict[str, float] = {"stand_stage": float(not self._gate_open)}

    def _reset_stats(self) -> None:
        zero = torch.zeros((), device=self._env.device)
        self._num_episodes = zero.clone()
        self._num_time_outs = zero.clone()
        self._cost_sums = {name: zero.clone() for name in self._cost_names}

    def _accumulate(self, env: ManagerBasedRLEnv, env_ids: Sequence[int]) -> None:
        """Add the episodes ending now (skips the initial reset, whose episodes have length 0)."""
        length = env.episode_length_buf[env_ids]
        ended = length > 0
        self._num_episodes += ended.sum()
        self._num_time_outs += (env.termination_manager.time_outs[env_ids] & ended).sum()
        for name in self._cost_names:
            weight = env.reward_manager.get_term_cfg(name).weight
            raw = env.reward_manager._episode_sums[name][env_ids] / (weight * env.step_dt * length.clamp(min=1))
            self._cost_sums[name] += (raw * ended).sum()

    def _gate_check(self, stand_gate: dict) -> bool:
        n = max(float(self._num_episodes), 1.0)
        time_out_ratio = float(self._num_time_outs) / n
        costs = {name: float(total) / n for name, total in self._cost_sums.items()}
        self._reset_stats()
        self._log.update(time_out_ratio=time_out_ratio, **{f"{k}_cost": v for k, v in costs.items()})
        passed = stand_gate_passed(
            time_out_ratio, costs, stand_gate.get("min_time_out_ratio", 0.0), stand_gate.get("max_cost", {})
        )
        self._gate_passes = self._gate_passes + 1 if passed else 0
        return self._gate_passes >= stand_gate.get("consecutive", 1)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        axis: str,
        reward_term_name: str,
        command_name: str = "base_velocity",
        threshold: float = 0.8,
        delta: float = 0.1,
        stand_gate: dict | None = None,
    ) -> dict[str, float]:
        if axis not in AXES:
            raise ValueError(f"axis must be one of {AXES}, got {axis}")
        cmd_cfg = env.command_manager.get_term(command_name).cfg
        current = tuple(getattr(cmd_cfg.ranges, axis))
        if not self._gate_open:
            self._accumulate(env, env_ids)
        if env.common_step_counter - self._last_update >= env.max_episode_length and len(env_ids) > 0:
            self._last_update = env.common_step_counter
            if not self._gate_open:
                self._gate_open = self._gate_check(stand_gate)
                self._log["stand_stage"] = float(not self._gate_open)
            else:
                episode_sum = env.reward_manager._episode_sums[reward_term_name][env_ids]
                per_second = float(torch.mean(episode_sum)) / env.max_episode_length_s
                weight = env.reward_manager.get_term_cfg(reward_term_name).weight
                if should_expand(per_second, weight, threshold):
                    current = expand_range(current, tuple(getattr(cmd_cfg.limit_ranges, axis)), delta)
                    setattr(cmd_cfg.ranges, axis, current)
        return {"range": current[1] - current[0], **self._log}


def terrain_levels_tracking(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy",
    promote_distance: float | None = None,
    demote_tracking: float = 0.5,
) -> torch.Tensor:
    """Move the robots whose episode ended to a harder / easier terrain row (``levels.terrain_level_moves``).

    ``distance`` is the planar distance from the tile origin; ``promote_distance`` defaults to half the tile size
    (the robot left its tile). The tracking ratio is the episodic sum of ``reward_term_name`` / (weight x episode
    duration), 1 = perfect tracking. Returns the mean terrain level.
    """
    terrain = env.scene.terrain
    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    if promote_distance is None:
        promote_distance = terrain.cfg.terrain_generator.size[0] / 2
    robot = env.scene["robot"]
    distance = torch.norm(robot.data.root_pos_w[ids, :2] - env.scene.env_origins[ids, :2], dim=1)
    fell = env.termination_manager.terminated[ids]
    length = env.episode_length_buf[ids]
    weight = env.reward_manager.get_term_cfg(reward_term_name).weight
    tracking = env.reward_manager._episode_sums[reward_term_name][ids] / (weight * length.clamp(min=1) * env.step_dt)
    move_up, move_down = terrain_level_moves(fell, distance, tracking, promote_distance, demote_tracking)
    ended = length > 0  # the initial reset (robots still at their spawn grid) moves nobody
    move_up, move_down = move_up & ended, move_down & ended
    terrain.update_env_origins(ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())
