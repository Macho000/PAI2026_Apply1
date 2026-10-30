"""Privileged observations for the critic: the current values of the domain randomisation (never used by the actor).

Every term is normalised as ``(value - center) / half_width`` so that the nominal model maps to about 0 and the
randomisation range to about [-1, 1] (the critic has no observation normaliser, and the new input columns start
from zero weights when a checkpoint of an earlier phase is expanded). Defaults match the Phase 3 ranges.

Terms set by startup events (friction, mass, CoM, inertia) are read from PhysX (CPU) once and cached: startup
events run after the observation manager is built (which also calls ``reset()`` on its terms), so ``WheelLeggedEnv``
calls :meth:`_StartupParams.refresh` when its construction is complete. Per-episode terms (gains, strength, friction,
delay) are read from the actuators at every step (GPU tensors).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.actuators import DelayedPDActuator
from isaaclab.managers import ManagerTermBase, ObservationTermCfg, SceneEntityCfg

from ..actuators.selection import actuator_joints

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedEnv


# friction 0.3-1.2; added mass -1..+3 kg, CoM +-3/2/2 cm, inertia (mass ratio 0.86-1.43) x (0.8-1.2) = 0.7-1.7
_FRICTION_CENTER, _FRICTION_HALF_WIDTH = 0.75, 0.45
_INERTIAL_CENTER = (1.0, 0.0, 0.0, 0.0, 1.2)
_INERTIAL_HALF_WIDTH = (2.0, 0.03, 0.02, 0.02, 0.5)


def _body_ids(asset, asset_cfg: SceneEntityCfg) -> list[int]:
    return list(range(asset.num_bodies)) if asset_cfg.body_ids == slice(None) else list(asset_cfg.body_ids)


class _StartupParams(ManagerTermBase):
    """Caches a tensor computed from PhysX. :meth:`refresh` must be called after the startup events."""

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self._value = self._read(cfg.params)

    def refresh(self) -> None:
        self._value = self._read(self.cfg.params)

    def _read(self, params: dict) -> torch.Tensor:
        raise NotImplementedError


class wheel_friction(_StartupParams):
    """Static and dynamic friction of the wheel collision shapes: ``[static x W, dynamic x W]``.

    ``randomize_rigid_body_material`` samples every shape independently, so the wheels of one robot differ. The
    ground material is 1.0 with ``multiply``, so these are the effective contact coefficients.
    """

    def _read(self, params: dict) -> torch.Tensor:
        asset = self.asset
        shapes_per_body = [
            asset._physics_sim_view.create_rigid_body_view(path).max_shapes
            for path in asset.root_physx_view.link_paths[0]
        ]
        start = [sum(shapes_per_body[:b]) for b in range(len(shapes_per_body))]
        material = asset.root_physx_view.get_material_properties()  # (num_envs, num_shapes, 3), CPU
        static, dynamic = [], []
        for b in _body_ids(asset, params["asset_cfg"]):
            shapes = slice(start[b], start[b] + shapes_per_body[b])
            static.append(material[:, shapes, 0].mean(dim=1))
            dynamic.append(material[:, shapes, 1].mean(dim=1))
        value = torch.stack(static + dynamic, dim=-1).to(self._env.device)
        return (value - params.get("center", _FRICTION_CENTER)) / params.get("half_width", _FRICTION_HALF_WIDTH)

    def __call__(
        self, env, asset_cfg, center: float = _FRICTION_CENTER, half_width: float = _FRICTION_HALF_WIDTH
    ) -> torch.Tensor:
        return self._value


class base_inertial_params(_StartupParams):
    """``[added mass, CoM offset x/y/z, inertia scale]`` of one body (the base), relative to the URDF values.

    The inertia scale is ``trace(I) / trace(I_urdf)``: the mass-proportional part and the separate inertia error.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self._body = _body_ids(asset, cfg.params["asset_cfg"])[0]
        self._default_com = asset.root_physx_view.get_coms()[:, self._body, :3].clone()  # before the startup events
        super().__init__(cfg, env)

    def _read(self, params: dict) -> torch.Tensor:
        view, data, b = self.asset.root_physx_view, self.asset.data, self._body
        added_mass = view.get_masses()[:, b] - data.default_mass[:, b].cpu()
        com_offset = view.get_coms()[:, b, :3] - self._default_com
        diag = [0, 4, 8]
        inertia_scale = view.get_inertias()[:, b, diag].sum(-1) / data.default_inertia[:, b, diag].cpu().sum(-1)
        value = torch.cat([added_mass[:, None], com_offset, inertia_scale[:, None]], dim=-1).to(self._env.device)
        center = torch.tensor(params.get("center", _INERTIAL_CENTER), device=self._env.device)
        half_width = torch.tensor(params.get("half_width", _INERTIAL_HALF_WIDTH), device=self._env.device)
        return (value - center) / half_width

    def __call__(
        self,
        env,
        asset_cfg,
        center: tuple[float, ...] = _INERTIAL_CENTER,
        half_width: tuple[float, ...] = _INERTIAL_HALF_WIDTH,
    ) -> torch.Tensor:
        return self._value


def _actuator_values(asset, asset_cfg: SceneEntityCfg, attr: str, nominal_attr: str | None = None) -> torch.Tensor:
    """``actuator.<attr>`` (divided by the articulation's ``data.<nominal_attr>`` if given) in selection order."""
    joint_ids = asset_cfg.joint_ids
    n = asset.num_joints if isinstance(joint_ids, slice) else len(joint_ids)
    out = torch.zeros(asset.num_instances, n, device=asset.device)
    for actuator, local, cols in actuator_joints(asset, joint_ids, attr):
        value = getattr(actuator, attr)[:, local]
        if nominal_attr is not None:
            joints = actuator.joint_indices
            nominal = getattr(asset.data, nominal_attr)[:, joints][:, local]
            value = value / nominal
        out[:, cols] = value
    return out


def actuator_gain_scale(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg, gain: str, center: float = 1.0, half_width: float = 0.2
) -> torch.Tensor:
    """Kp (``gain="stiffness"``) or Kd (``"damping"``) relative to the nominal gain. Select joints with a non-zero
    nominal gain (the wheels have Kp = 0)."""
    asset: Articulation = env.scene[asset_cfg.name]
    scale = _actuator_values(asset, asset_cfg, gain, f"default_joint_{gain}")
    return (scale - center) / half_width


def motor_strength(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg, center: float = 1.0, half_width: float = 0.2
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return (_actuator_values(asset, asset_cfg, "strength") - center) / half_width


def joint_friction(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg,
    static_center: float = 0.1,
    static_half_width: float = 0.1,
    viscous_center: float = 0.01,
    viscous_half_width: float = 0.01,
) -> torch.Tensor:
    """``[Coulomb x J, viscous x J]`` friction of the actuators."""
    asset: Articulation = env.scene[asset_cfg.name]
    static = (_actuator_values(asset, asset_cfg, "friction_static") - static_center) / static_half_width
    viscous = (_actuator_values(asset, asset_cfg, "friction_viscous") - viscous_center) / viscous_half_width
    return torch.cat([static, viscous], dim=-1)


def command_delay(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Command delay [physics steps] of the robot, mapped from ``min_delay..max_delay`` to [-1, 1] (0 if no delay).

    All groups share one delay (``events.sync_actuator_delay``), so the first delayed actuator is representative.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    actuator = next((a for a in asset.actuators.values() if isinstance(a, DelayedPDActuator)), None)
    if actuator is None or actuator.cfg.max_delay == 0:
        return torch.zeros(asset.num_instances, 1, device=asset.device)
    lo, hi = actuator.cfg.min_delay, actuator.cfg.max_delay
    lag = actuator.positions_delay_buffer.time_lags.float()
    half = max((hi - lo) / 2.0, 0.5)
    return ((lag - (lo + hi) / 2.0) / half).unsqueeze(-1)
