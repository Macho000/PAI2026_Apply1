"""Quasi-direct-drive (QDD) actuator model. Isaac-independent (torch only for the batched limit function).

Two ways to describe an actuator:

* :class:`TNCurve` -- the four numbers Unitree publishes for its motors (the "T-N curve")::

        |torque| ^
            Y2 --|                      (torque opposing velocity, i.e. braking)
            Y1 --|-----------.          (torque along velocity, i.e. motoring)
                 |           |\\
                 |           | \\
                 +-----------+--+---> |velocity|
                            X1  X2

* :class:`QDDParams` -- physical motor/gearbox parameters, from which a :class:`TNCurve`, the reflected rotor
  inertia (armature) and the joint friction are derived.

The limit is applied exactly as ``unitree_rl_lab``'s ``UnitreeActuator`` does, so a policy trained here sees the
same torque envelope as one trained there: below ``X1`` the limit is ``Y1`` (same direction) or ``Y2`` (opposite
direction); above ``X1`` it falls linearly to zero at ``X2``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch


@dataclass(frozen=True)
class TNCurve:
    """Output-shaft torque-speed envelope. Units: N*m and rad/s."""

    y1: float
    """Peak torque when torque and velocity have the same sign."""
    y2: float
    """Peak torque when torque and velocity have opposite signs (usually > y1: friction helps braking)."""
    x1: float
    """Speed up to which the peak torque is available (knee point)."""
    x2: float
    """No-load speed; the torque limit reaches zero here."""

    def __post_init__(self):
        if not (0 < self.y1 and 0 < self.y2 and 0 <= self.x1 < self.x2):
            raise ValueError(f"invalid T-N curve {self}")

    @property
    def peak_torque(self) -> float:
        return max(self.y1, self.y2)

    def geared(self, ratio: float) -> TNCurve:
        """The curve seen through an additional ideal reduction ``ratio`` (torque x ratio, speed / ratio)."""
        return TNCurve(self.y1 * ratio, self.y2 * ratio, self.x1 / ratio, self.x2 / ratio)

    def limit(self, velocity: float, same_direction: bool = True) -> float:
        """Scalar torque limit (for reports and tests); see :func:`tn_torque_limit` for the batched version."""
        peak = self.y1 if same_direction else self.y2
        v = abs(velocity)
        if v < self.x1:
            return peak
        return max(0.0, peak * (1.0 - (v - self.x1) / (self.x2 - self.x1)))


@dataclass(frozen=True)
class QDDParams:
    """Physical parameters of a motor + single-stage (or lumped) reduction."""

    kt: float
    """Motor torque constant [N*m/A] (motor side)."""
    gear_ratio: float
    """Reduction ratio N (output speed = motor speed / N)."""
    i_max: float
    """Peak phase current [A]."""
    motor_no_load_speed: float
    """Motor no-load speed at the bus voltage [rad/s] (motor side)."""
    motor_knee_speed: float
    """Motor speed up to which the peak current can be driven [rad/s] (motor side)."""
    rotor_inertia: float = 0.0
    """Rotor inertia [kg*m^2] (motor side)."""
    gear_inertia: float = 0.0
    """Additional inertia of the gear train, already reflected to the output [kg*m^2]."""
    gear_efficiency: float = 1.0
    """Forward efficiency eta. Motoring torque = kt*i*N*eta; braking torque = kt*i*N/eta."""
    coulomb_friction: float = 0.0
    """Static (Coulomb) friction at the output [N*m]."""
    viscous_friction: float = 0.0
    """Viscous friction at the output [N*m*s/rad]."""

    def tn_curve(self) -> TNCurve:
        stall = self.kt * self.i_max * self.gear_ratio
        return TNCurve(
            y1=stall * self.gear_efficiency,
            y2=stall / self.gear_efficiency,
            x1=self.motor_knee_speed / self.gear_ratio,
            x2=self.motor_no_load_speed / self.gear_ratio,
        )

    @property
    def armature(self) -> float:
        """Reflected inertia at the output: J_rotor * N^2 + J_gear."""
        return self.rotor_inertia * self.gear_ratio**2 + self.gear_inertia

    def with_extra_reduction(self, ratio: float) -> QDDParams:
        """Same motor behind an additional ``ratio`` reduction (e.g. a linkage-driven knee)."""
        return replace(
            self,
            gear_ratio=self.gear_ratio * ratio,
            gear_inertia=self.gear_inertia * ratio**2,
            coulomb_friction=self.coulomb_friction * ratio,
            viscous_friction=self.viscous_friction * ratio**2,
        )


def tn_torque_limit(
    effort: torch.Tensor,
    joint_vel: torch.Tensor,
    y1: torch.Tensor,
    y2: torch.Tensor,
    x1: torch.Tensor,
    x2: torch.Tensor,
) -> torch.Tensor:
    """Clip ``effort`` to the T-N envelope. All tensors broadcast to (num_envs, num_joints)."""
    same_direction = (joint_vel * effort) > 0
    peak = torch.where(same_direction, y1, y2)
    speed = joint_vel.abs()
    derated = (peak * (1.0 - (speed - x1) / (x2 - x1))).clamp(min=0.0)
    max_effort = torch.where(speed < x1, peak, derated)
    return torch.clamp(effort, -max_effort, max_effort)


def joint_friction(
    joint_vel: torch.Tensor, static: torch.Tensor, viscous: torch.Tensor, activation_vel: torch.Tensor
) -> torch.Tensor:
    """Friction torque opposing motion: Fs*tanh(v/va) + Fd*v (smooth Coulomb + viscous, as in unitree_rl_lab)."""
    return static * torch.tanh(joint_vel / activation_vel) + viscous * joint_vel


# Unitree Go2 hip/thigh motor ("Go2HV") as published in unitree_rl_lab/assets/robots/unitree_actuators.py.
GO2HV_CURVE = TNCurve(y1=20.2, y2=23.4, x1=13.5, x2=30.0)
