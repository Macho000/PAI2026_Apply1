"""Velocity-controlled wheel: tau = clip_TN(Kd (w* - w)) - friction(w). Kp is forced to zero."""

from __future__ import annotations

from isaaclab.utils import configclass

from .leg_actuator import QDDActuator, QDDActuatorCfg


class WheelActuator(QDDActuator):
    """QDD actuator used as a velocity servo (continuous joint, position target ignored)."""

    cfg: WheelActuatorCfg

    def __init__(self, cfg: WheelActuatorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        if bool((self.stiffness != 0).any()):
            raise ValueError(f"WheelActuator requires zero stiffness, got {self.stiffness[0].tolist()}")


@configclass
class WheelActuatorCfg(QDDActuatorCfg):
    class_type: type = WheelActuator
    stiffness: float = 0.0
