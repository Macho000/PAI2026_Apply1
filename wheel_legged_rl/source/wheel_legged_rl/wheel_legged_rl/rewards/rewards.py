"""Wheel-leg specific reward terms.

Generic terms (velocity tracking, orientation, action rate, joint limits, undesired contacts, ...) are taken from
``isaaclab.envs.mdp`` directly in ``envs/base_env_cfg.py``. This module only adds what the wheel changes:

* ``wheel_slip`` replaces ``feet_slide`` (a rolling wheel *should* have horizontal contact-point velocity at its
  centre; slip is the mismatch between that velocity and omega * r);
* ``mechanical_power`` can be applied to legs and wheels separately (wheels spin continuously);
* ``stand_still`` penalises leg deviation *and* wheel speed when the command is zero.

The math lives in the ``*_from_tensors`` helpers so it can be unit-tested without Isaac Sim.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg

# ---------------------------------------------------------------------------------------------------------------------
# Pure tensor math
# ---------------------------------------------------------------------------------------------------------------------


def quat_rotate(q_wxyz: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vectors ``v`` (..., 3) by unit quaternions ``q`` (..., 4) in (w, x, y, z) order."""
    w = q_wxyz[..., :1]
    xyz = q_wxyz[..., 1:]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)


def wheel_slip_from_tensors(
    wheel_lin_vel_w: torch.Tensor,
    wheel_ang_vel_w: torch.Tensor,
    wheel_quat_w: torch.Tensor,
    wheel_axis_local: torch.Tensor,
    wheel_radius: float,
    ground_normal_w: torch.Tensor | None = None,
) -> torch.Tensor:
    """Longitudinal slip speed [m/s] of each wheel, shape (num_envs, num_wheels).

    With spin axis ``a`` (world), ground normal ``n`` and rolling direction ``f = normalize(a x n)``, the contact
    point moves with ``v_c - omega r f`` where ``omega = w . a``. Zero slip means ``v_c . f = omega r``. Lateral
    motion along ``a`` is not counted (that is side-slip, penalised by velocity tracking).
    """
    axis_w = quat_rotate(wheel_quat_w, wheel_axis_local.expand_as(wheel_lin_vel_w))
    if ground_normal_w is None:
        ground_normal_w = torch.zeros_like(axis_w)
        ground_normal_w[..., 2] = 1.0
    forward = torch.cross(axis_w, ground_normal_w, dim=-1)
    forward = forward / forward.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    spin = (wheel_ang_vel_w * axis_w).sum(dim=-1)
    return (wheel_lin_vel_w * forward).sum(dim=-1) - spin * wheel_radius


def mechanical_power_from_tensors(torque: torch.Tensor, joint_vel: torch.Tensor) -> torch.Tensor:
    """Sum over joints of |tau * omega| [W] (positive and negative work both cost energy in a motor)."""
    return (torque * joint_vel).abs().sum(dim=-1)


def stand_still_from_tensors(
    command: torch.Tensor,
    leg_pos: torch.Tensor,
    leg_default: torch.Tensor,
    wheel_vel: torch.Tensor,
    command_threshold: float,
    wheel_vel_scale: float,
) -> torch.Tensor:
    """L1 leg deviation + scaled L1 wheel speed, active only when ||command|| < threshold."""
    cost = (leg_pos - leg_default).abs().sum(dim=-1) + wheel_vel_scale * wheel_vel.abs().sum(dim=-1)
    return cost * (command.norm(dim=-1) < command_threshold)


def contact_mask(net_forces_w_history: torch.Tensor, threshold: float) -> torch.Tensor:
    """(num_envs, num_bodies) bool: max contact force over the history exceeds ``threshold``."""
    return net_forces_w_history.norm(dim=-1).max(dim=1)[0] > threshold


# ---------------------------------------------------------------------------------------------------------------------
# Reward terms (Isaac Lab signature)
# ---------------------------------------------------------------------------------------------------------------------


def mechanical_power(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    ids = asset_cfg.joint_ids
    return mechanical_power_from_tensors(asset.data.applied_torque[:, ids], asset.data.joint_vel[:, ids])


def wheel_slip(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    wheel_radius: float,
    wheel_axis: tuple[float, float, float] = (0.0, 1.0, 0.0),
    contact_threshold: float = 1.0,
) -> torch.Tensor:
    """Sum of squared longitudinal slip over wheels in contact. ``asset_cfg.body_names`` = wheel bodies."""
    asset = env.scene[asset_cfg.name]
    sensor = env.scene.sensors[sensor_cfg.name]
    ids = asset_cfg.body_ids
    axis = torch.tensor(wheel_axis, device=env.device, dtype=torch.float)
    slip = wheel_slip_from_tensors(
        asset.data.body_link_lin_vel_w[:, ids],
        asset.data.body_link_ang_vel_w[:, ids],
        asset.data.body_link_quat_w[:, ids],
        axis,
        wheel_radius,
    )
    in_contact = contact_mask(sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids], contact_threshold)
    return (slip.square() * in_contact).sum(dim=-1)


def stand_still(
    env: ManagerBasedRLEnv,
    command_name: str,
    leg_cfg: SceneEntityCfg,
    wheel_cfg: SceneEntityCfg,
    command_threshold: float = 0.1,
    wheel_vel_scale: float = 0.05,
) -> torch.Tensor:
    asset = env.scene[leg_cfg.name]
    return stand_still_from_tensors(
        env.command_manager.get_command(command_name),
        asset.data.joint_pos[:, leg_cfg.joint_ids],
        asset.data.default_joint_pos[:, leg_cfg.joint_ids],
        asset.data.joint_vel[:, wheel_cfg.joint_ids],
        command_threshold,
        wheel_vel_scale,
    )
