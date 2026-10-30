"""Body-frame velocity command [vx, vy, wz] with curriculum limits.

``ranges`` is the current sampling range, grown by ``curriculum.command_levels`` up to ``limit_ranges``. The deploy
exporter writes ``limit_ranges`` (the joystick clamp on the robot), as ``unitree_rl_lab`` does.
"""

from __future__ import annotations

from isaaclab.envs.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.utils import configclass


@configclass
class WheelLeggedVelocityCommandCfg(UniformVelocityCommandCfg):
    class_type: type = UniformVelocityCommand

    asset_name: str = "robot"
    heading_command: bool = False
    rel_heading_envs: float = 0.0
    resampling_time_range: tuple[float, float] = (10.0, 10.0)

    ranges: UniformVelocityCommandCfg.Ranges = UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(0.0, 0.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0)
    )
    limit_ranges: UniformVelocityCommandCfg.Ranges = UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(0.0, 0.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0)
    )
    """Upper bound of the command curriculum and of the joystick on the robot."""
