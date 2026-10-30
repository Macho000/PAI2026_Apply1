"""Domain-randomisation events not covered by ``isaaclab.envs.mdp``.

* ``randomize_motor_strength`` -- per-motor torque-constant error of the QDD actuators (``QDDActuator.strength``);
* ``randomize_joint_friction`` -- per-motor Coulomb / viscous friction of the QDD actuators, as absolute values
  (the nominal friction of the Go2-W spec is 0, so a scale would do nothing);
* ``sync_actuator_delay``      -- one command delay per robot shared by every actuator group (the real robot sends
  all motor commands in one packet; ``DelayedPDActuator`` alone samples an independent delay per group);
* ``randomize_rigid_body_inertia`` -- inertia error on top of the mass randomisation (uniform scale per body, which
  keeps the tensor physically valid).

Kp / Kd errors use Isaac Lab's ``randomize_actuator_gains`` and CoM offsets ``randomize_rigid_body_com`` directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.actuators import DelayedPDActuator
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg

from ..actuators.selection import actuator_joints

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedEnv


def _sample_actuator_tensor(
    env: ManagerBasedEnv, env_ids: torch.Tensor | None, asset_cfg: SceneEntityCfg, attr: str, value_range
) -> None:
    """``actuator.<attr>[env, joint] ~ U(value_range)`` for the joints selected by ``asset_cfg``."""
    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)
    lo, hi = value_range
    for actuator, local, _ in actuator_joints(asset, asset_cfg.joint_ids, attr):
        tensor = getattr(actuator, attr)
        tensor[env_ids[:, None], local] = torch.empty(len(env_ids), len(local), device=asset.device).uniform_(lo, hi)


def randomize_motor_strength(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    strength_range: tuple[float, float],
) -> None:
    """Sample ``QDDActuator.strength`` uniformly in ``strength_range``, independently per env and motor."""
    _sample_actuator_tensor(env, env_ids, asset_cfg, "strength", strength_range)


def randomize_joint_friction(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    static_range: tuple[float, float],
    viscous_range: tuple[float, float],
) -> None:
    """Sample the actuator friction (absolute values) independently per env and motor: Coulomb [N*m] in
    ``static_range``, viscous [N*m*s/rad] in ``viscous_range``. Use one term per actuator group for group ranges."""
    _sample_actuator_tensor(env, env_ids, asset_cfg, "friction_static", static_range)
    _sample_actuator_tensor(env, env_ids, asset_cfg, "friction_viscous", viscous_range)


class sync_actuator_delay(ManagerTermBase):
    """Give every ``DelayedPDActuator`` of the asset the same random delay per env (use with ``mode="reset"``).

    The range is the actuators' own ``min_delay``..``max_delay`` (physics steps; set in the phase YAML
    ``actuators`` section), which must be equal for all groups. The actuators sample their own delays in
    ``Articulation.reset``; reset events run afterwards and overwrite them.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self._actuators = [a for a in asset.actuators.values() if isinstance(a, DelayedPDActuator)]
        ranges = {(a.cfg.min_delay, a.cfg.max_delay) for a in self._actuators}
        if len(ranges) > 1:
            raise ValueError(f"sync_actuator_delay: actuator groups have different delay ranges {sorted(ranges)}")
        self._range = ranges.pop() if ranges else (0, 0)

    def __call__(self, env: ManagerBasedEnv, env_ids: torch.Tensor | None, asset_cfg: SceneEntityCfg) -> None:
        lo, hi = self._range
        if hi == 0 or not self._actuators:
            return
        n = env.scene.num_envs if env_ids is None else len(env_ids)
        lags = torch.randint(lo, hi + 1, (n,), dtype=torch.int, device=env.device)
        for actuator in self._actuators:
            for buffer in (
                actuator.positions_delay_buffer,
                actuator.velocities_delay_buffer,
                actuator.efforts_delay_buffer,
            ):
                buffer.set_time_lag(lags, env_ids)


class randomize_rigid_body_inertia(ManagerTermBase):
    """Scale the inertia tensors of the bodies by a uniform factor per env and body (``mode="startup"`` only).

    Applied to the current inertia, so it composes with ``randomize_rigid_body_mass(recompute_inertia=True)``
    declared before it (mass-proportional scaling, then this error on top).
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        if cfg.mode != "startup":
            raise ValueError("randomize_rigid_body_inertia multiplies the current inertia: use mode='startup'")

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        scale_range: tuple[float, float],
    ) -> None:
        asset: Articulation = env.scene[asset_cfg.name]
        env_ids = torch.arange(env.scene.num_envs) if env_ids is None else env_ids.cpu()
        if asset_cfg.body_ids == slice(None):
            body_ids = torch.arange(asset.num_bodies)
        else:
            body_ids = torch.tensor(asset_cfg.body_ids)
        inertias = asset.root_physx_view.get_inertias()  # (num_envs, num_bodies, 9), CPU
        scale = torch.empty(len(env_ids), len(body_ids), 1).uniform_(*scale_range)
        inertias[env_ids[:, None], body_ids] *= scale
        asset.root_physx_view.set_inertias(inertias, env_ids)
