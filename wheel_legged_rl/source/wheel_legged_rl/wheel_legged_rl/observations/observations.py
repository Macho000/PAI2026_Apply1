"""Observation terms that are not in ``isaaclab.envs.mdp``.

``terrain_height_scan`` is the *perception interface*: a robot-centred, yaw-aligned height map (base height minus
terrain height minus ``offset``). In simulation it is filled from a RayCaster; on the robot from a LiDAR/depth
elevation map. Corruptions that a real perception stack produces (dropout, bias) are applied here so the policy
does not learn to rely on perfect terrain knowledge.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ..rewards.rewards import contact_mask

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.managers import SceneEntityCfg


def corrupt_height_map(
    heights: torch.Tensor, dropout_prob: float, bias_range: tuple[float, float], dropout_value: float = 0.0
) -> torch.Tensor:
    """Per-env constant bias + per-cell dropout (cells replaced by ``dropout_value``)."""
    if bias_range[0] != 0.0 or bias_range[1] != 0.0:
        bias = torch.empty(heights.shape[0], 1, device=heights.device).uniform_(*bias_range)
        heights = heights + bias
    if dropout_prob > 0.0:
        mask = torch.rand_like(heights) < dropout_prob
        heights = torch.where(mask, torch.full_like(heights, dropout_value), heights)
    return heights


def terrain_height_scan(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    offset: float = 0.5,
    dropout_prob: float = 0.0,
    bias_range: tuple[float, float] = (0.0, 0.0),
) -> torch.Tensor:
    """(num_envs, num_rays) height map: ``sensor_z - hit_z - offset`` (clip in the term config)."""
    sensor = env.scene.sensors[sensor_cfg.name]
    heights = sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2] - offset
    # rays that miss (e.g. off the terrain border) return inf; treat as "far below"
    heights = torch.nan_to_num(heights, nan=1.0, posinf=1.0, neginf=-1.0)
    return corrupt_height_map(heights, dropout_prob, bias_range)


def wheel_contact(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg, threshold: float = 1.0) -> torch.Tensor:
    """(num_envs, num_wheels) float in {0, 1}. Privileged (critic only)."""
    sensor = env.scene.sensors[sensor_cfg.name]
    return contact_mask(sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids], threshold).float()
