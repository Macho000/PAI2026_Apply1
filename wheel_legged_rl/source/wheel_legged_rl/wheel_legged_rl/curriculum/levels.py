"""Pure helpers for command curricula (unit-tested without Isaac Sim)."""

from __future__ import annotations

from collections.abc import Mapping

AXES = ("lin_vel_x", "lin_vel_y", "ang_vel_z")


def expand_range(current: tuple[float, float], limit: tuple[float, float], delta: float) -> tuple[float, float]:
    """Widen ``current`` by ``delta`` on both sides without leaving ``limit``."""
    return (max(limit[0], current[0] - delta), min(limit[1], current[1] + delta))


def should_expand(mean_episode_reward_per_s: float, reward_weight: float, threshold: float) -> bool:
    """Tracking reward (per second of episode) above ``threshold`` x its maximum (= weight)."""
    return reward_weight > 0 and mean_episode_reward_per_s > threshold * reward_weight


def stand_gate_passed(
    time_out_ratio: float,
    mean_costs: Mapping[str, float],
    min_time_out_ratio: float,
    max_costs: Mapping[str, float],
) -> bool:
    """Stand-still stage done: few falls and every raw per-step cost at or below its limit."""
    missing = set(max_costs) - set(mean_costs)
    if missing:
        raise KeyError(f"stand gate: no cost for {sorted(missing)}")
    return time_out_ratio >= min_time_out_ratio and all(mean_costs[k] <= v for k, v in max_costs.items())


def terrain_level_moves(fell, distance, tracking_ratio, promote_distance: float, demote_tracking: float):
    """Terrain-level change of the robots whose episode just ended (torch tensors or numpy arrays, elementwise).

    * up: survived (time-out) and left the tile, i.e. ``distance`` from the tile origin > ``promote_distance``;
    * down: fell, or tracked the velocity command poorly (episodic tracking reward / its maximum <
      ``demote_tracking``) -- e.g. stuck in front of a step. Never both.

    Unlike Isaac Lab's ``terrain_levels_vel`` a short distance alone does not demote: with yaw-rate commands the
    robot drives arcs and may end near its origin although it followed the command.
    """
    up = ~fell & (distance > promote_distance)
    down = (fell | (tracking_ratio < demote_tracking)) & ~up
    return up, down
