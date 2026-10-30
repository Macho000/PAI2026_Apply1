"""Explicit (Python-side) QDD actuator for Isaac Lab and its leg (position-PD) configuration.

The torque is computed here, not by the PhysX joint drive (whose gains are zero for explicit actuators), so the
control law can be swapped 1:1 with the one running on the motor driver:

    tau = s * clip_TN(Kp (q* - q) + Kd (dq* - dq) + tau_ff) - friction(dq)

with a per-episode random command delay (``min_delay``..``max_delay`` physics steps) inherited from
:class:`isaaclab.actuators.DelayedPDActuator`. ``s`` is the motor strength (per env and joint, 1 by default), set by
the ``randomize_motor_strength`` event: it scales the torque constant, i.e. the commanded torque and the torque axis
of the T-N curve together (``s * clip_TN(tau) == clip_{s TN}(s tau)``); the speed axis is left unchanged.
"""

from __future__ import annotations

from dataclasses import MISSING

import torch
from isaaclab.actuators import DelayedPDActuator, DelayedPDActuatorCfg
from isaaclab.utils import configclass
from isaaclab.utils.types import ArticulationActions

from .qdd_model import joint_friction, tn_torque_limit


class QDDActuator(DelayedPDActuator):
    """Delayed PD actuator with a T-N torque envelope and joint friction."""

    cfg: QDDActuatorCfg

    def __init__(self, cfg: QDDActuatorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self._joint_vel = torch.zeros_like(self.computed_effort)
        self._y1 = self._parse_joint_parameter(cfg.y1, None)
        self._y2 = self._parse_joint_parameter(cfg.y2, None)
        self._x1 = self._parse_joint_parameter(cfg.x1, None)
        self._x2 = self._parse_joint_parameter(cfg.x2, None)
        self.friction_static = self._parse_joint_parameter(cfg.model_friction_static, 0.0)
        """Coulomb friction [N*m] (num_envs, num_joints); ``events.randomize_joint_friction`` overwrites it."""
        self.friction_viscous = self._parse_joint_parameter(cfg.model_friction_viscous, 0.0)
        """Viscous friction [N*m*s/rad] (num_envs, num_joints)."""
        self._va = self._parse_joint_parameter(cfg.model_friction_activation_vel, 0.01)
        self.strength = torch.ones_like(self.computed_effort)
        """Motor strength (num_envs, num_joints); written by ``events.randomize_motor_strength``."""

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        self._joint_vel[:] = joint_vel
        control_action = super().compute(control_action, joint_pos, joint_vel)  # delay + PD + _clip_effort
        self.applied_effort = self.applied_effort - joint_friction(
            joint_vel, self.friction_static, self.friction_viscous, self._va
        )
        control_action.joint_positions = None
        control_action.joint_velocities = None
        control_action.joint_efforts = self.applied_effort
        return control_action

    def _clip_effort(self, effort: torch.Tensor) -> torch.Tensor:
        return self.strength * tn_torque_limit(effort, self._joint_vel, self._y1, self._y2, self._x1, self._x2)


@configclass
class QDDActuatorCfg(DelayedPDActuatorCfg):
    """T-N curve parameters (see :class:`~wheel_legged_rl.actuators.qdd_model.TNCurve`) plus friction."""

    class_type: type = QDDActuator

    y1: float | dict[str, float] = MISSING
    y2: float | dict[str, float] = MISSING
    x1: float | dict[str, float] = MISSING
    x2: float | dict[str, float] = MISSING
    model_friction_static: float | dict[str, float] = 0.0
    """Coulomb friction [N*m] applied by the model (not the PhysX joint friction)."""
    model_friction_viscous: float | dict[str, float] = 0.0
    model_friction_activation_vel: float | dict[str, float] = 0.01


class LegActuator(QDDActuator):
    """Position-PD leg joint."""


@configclass
class LegActuatorCfg(QDDActuatorCfg):
    class_type: type = LegActuator
