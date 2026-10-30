"""Robot-agnostic description of a four-wheel-leg robot (pure data, no Isaac imports).

Everything robot-specific lives in one :class:`RobotSpec` instance (see ``robots/go2w.py``). The environment
(``envs/base_env_cfg.py``), the asset builder (``robots/builder.py``) and the deploy exporter only read this object,
so porting to a new robot means writing a new spec -- not touching RL logic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..actuators.qdd_model import QDDParams, TNCurve


@dataclass(frozen=True)
class ActuatorSpec:
    """Common actuator description (one group of joints sharing the same motor)."""

    joint_names_expr: tuple[str, ...]
    curve: TNCurve | None = None
    """T-N envelope. Derived from :attr:`qdd` when omitted."""
    qdd: QDDParams | None = None
    """Physical parameters; used for the curve / armature / friction when those are not given explicitly."""
    damping: float = 0.0
    """Kd [N*m*s/rad]."""
    min_delay: int = 0
    """Minimum command delay in physics steps (sampled uniformly per episode)."""
    max_delay: int = 0
    armature: float | None = None
    friction_static: float | None = None
    friction_viscous: float | None = None
    friction_activation_vel: float = 0.01
    velocity_limit: float | None = None
    """Hard joint speed limit written to PhysX [rad/s]. Defaults to the curve's no-load speed."""

    @property
    def stiffness(self) -> float:  # overridden by LegActuatorSpec
        return 0.0

    def resolved_curve(self) -> TNCurve:
        if self.curve is not None:
            return self.curve
        if self.qdd is not None:
            return self.qdd.tn_curve()
        raise ValueError(f"actuator {self.joint_names_expr}: give either curve or qdd")

    def resolved_armature(self) -> float:
        if self.armature is not None:
            return self.armature
        return self.qdd.armature if self.qdd is not None else 0.0

    def resolved_friction(self) -> tuple[float, float]:
        fs = self.friction_static
        fd = self.friction_viscous
        if fs is None:
            fs = self.qdd.coulomb_friction if self.qdd is not None else 0.0
        if fd is None:
            fd = self.qdd.viscous_friction if self.qdd is not None else 0.0
        return fs, fd

    def resolved_velocity_limit(self) -> float:
        return self.velocity_limit if self.velocity_limit is not None else self.resolved_curve().x2


@dataclass(frozen=True)
class LegActuatorSpec(ActuatorSpec):
    """Position-controlled joint: tau = Kp (q* - q) + Kd (0 - dq)."""

    kp: float = 0.0

    @property
    def stiffness(self) -> float:
        return self.kp


@dataclass(frozen=True)
class WheelActuatorSpec(ActuatorSpec):
    """Velocity-controlled joint: tau = Kd (w* - w). Kp is always zero (continuous joint)."""


@dataclass(frozen=True)
class RobotSpec:
    name: str

    # --- asset -----------------------------------------------------------------------------------------------------
    urdf_path: str | None = None
    usd_path: str | None = None
    cylinder_wheels: bool = False
    """Replace wheel collision meshes with fitted cylinders when importing the URDF."""
    merge_fixed_joints: bool = True
    self_collisions: bool = False
    solver_position_iterations: int = 4
    solver_velocity_iterations: int = 0

    # --- topology ----------------------------------------------------------------------------------------------------
    base_body_name: str = "base"
    leg_joint_names: tuple[str, ...] = ()
    """Exactly the leg joints (explicit names, not regexes)."""
    wheel_joint_names: tuple[str, ...] = ()
    wheel_body_names: tuple[str, ...] = ()
    """Wheel bodies, same order as :attr:`wheel_joint_names`."""
    sdk_joint_order: tuple[str, ...] = ()
    """Joint order of the real robot's low-level interface (motor index i = sdk_joint_order[i])."""

    # --- geometry / stance -------------------------------------------------------------------------------------------
    wheel_radius: float = 0.0
    wheel_axis: tuple[float, float, float] = (0.0, 1.0, 0.0)
    """Spin axis of every wheel in its own link frame (URDF joint axis). Positive speed must roll forward."""
    default_joint_pos: dict[str, float] = field(default_factory=dict)
    """Regex -> position [rad]. Wheels should be 0."""
    init_base_height: float = 0.0
    """Spawn height of the base [m]."""
    nominal_base_height: float = 0.0
    """Base height at the default pose on flat ground [m] (from the URDF audit)."""
    soft_joint_pos_limit_factor: float = 0.9

    # --- actuation ---------------------------------------------------------------------------------------------------
    actuators: dict[str, ActuatorSpec] = field(default_factory=dict)

    # --- contacts ----------------------------------------------------------------------------------------------------
    termination_contact_bodies: tuple[str, ...] = ()
    """Regexes; contact on these ends the episode."""
    undesired_contact_bodies: tuple[str, ...] = ()
    """Regexes; contact on these is penalised (reward ``undesired_contacts``)."""
    mass_randomization_body: str | None = None
    """Body whose mass is randomised (payload). Defaults to the base."""

    # -----------------------------------------------------------------------------------------------------------------

    @property
    def all_joint_names(self) -> tuple[str, ...]:
        return self.leg_joint_names + self.wheel_joint_names

    @property
    def num_legs_joints(self) -> int:
        return len(self.leg_joint_names)

    @property
    def num_wheels(self) -> int:
        return len(self.wheel_joint_names)

    @property
    def action_dim(self) -> int:
        return self.num_legs_joints + self.num_wheels

    def default_position(self, joint_name: str) -> float:
        for pattern, value in self.default_joint_pos.items():
            if re.fullmatch(pattern, joint_name):
                return value
        return 0.0

    def actuator_for(self, joint_name: str) -> ActuatorSpec:
        matches = [a for a in self.actuators.values() if any(re.fullmatch(p, joint_name) for p in a.joint_names_expr)]
        if len(matches) != 1:
            raise ValueError(f"joint '{joint_name}' matches {len(matches)} actuator groups (expected 1)")
        return matches[0]

    def validate(self) -> None:
        """Consistency checks that do not need the asset. Raises ``ValueError`` on the first problem."""
        if not self.leg_joint_names or not self.wheel_joint_names:
            raise ValueError("leg_joint_names and wheel_joint_names must be non-empty")
        if set(self.leg_joint_names) & set(self.wheel_joint_names):
            raise ValueError("a joint cannot be both leg and wheel")
        if len(self.wheel_body_names) != len(self.wheel_joint_names):
            raise ValueError("wheel_body_names must match wheel_joint_names one-to-one")
        if self.sdk_joint_order and sorted(self.sdk_joint_order) != sorted(self.all_joint_names):
            raise ValueError("sdk_joint_order must be a permutation of leg + wheel joints")
        if self.wheel_radius <= 0:
            raise ValueError("wheel_radius must be positive")
        for name in self.all_joint_names:
            act = self.actuator_for(name)
            if name in self.wheel_joint_names and not isinstance(act, WheelActuatorSpec):
                raise ValueError(f"wheel joint '{name}' must use a WheelActuatorSpec")
            if name in self.leg_joint_names and not isinstance(act, LegActuatorSpec):
                raise ValueError(f"leg joint '{name}' must use a LegActuatorSpec")
        for name in self.wheel_joint_names:
            if self.default_position(name) != 0.0:
                raise ValueError(f"wheel joint '{name}' must default to 0 (continuous joint)")
        if self.init_base_height < self.nominal_base_height:
            raise ValueError("init_base_height below the nominal stance height would spawn the wheels in the ground")
