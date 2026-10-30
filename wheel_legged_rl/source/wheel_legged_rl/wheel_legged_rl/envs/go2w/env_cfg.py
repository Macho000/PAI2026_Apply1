"""Go2-W task configurations: the robot-agnostic env + GO2W_SPEC + a phase YAML."""

from __future__ import annotations

from pathlib import Path

from isaaclab.utils import configclass

from ...robots.go2w import GO2W_SPEC
from ..base_env_cfg import WheelLeggedEnvCfg, apply_phase_config, configure_for_robot, finalize

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "go2w"


@configclass
class Go2WPhase1FlatEnvCfg(WheelLeggedEnvCfg):
    """Phase 1: flat ground, stand still / drive forward / drive backward."""

    def __post_init__(self):
        super().__post_init__()
        configure_for_robot(self, GO2W_SPEC)
        apply_phase_config(self, CONFIG_DIR / "phase1_flat.yaml")


def make_play(cfg: WheelLeggedEnvCfg) -> None:
    """Small, noise-free, un-pushed variant for visual checks, with the full command range from the start."""
    cfg.scene.num_envs = 32
    cfg.observations.policy.enable_corruption = False
    cfg.events.push_robot = None
    cfg.events.reset_base.params["pose_range"] = {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (0.0, 0.0)}
    cmd = cfg.commands.base_velocity
    cmd.ranges = cmd.limit_ranges.copy()
    for name in ("lin_vel_x_levels", "lin_vel_y_levels", "ang_vel_z_levels"):
        setattr(cfg.curriculum, name, None)
    generator = cfg.scene.terrain.terrain_generator
    if generator is not None:
        generator.num_rows = min(generator.num_rows, 5)
        generator.num_cols = min(generator.num_cols, 5)
    finalize(cfg)


@configclass
class Go2WPhase1FlatEnvCfg_PLAY(Go2WPhase1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        make_play(self)


@configclass
class Go2WPhase2Flat3AxisEnvCfg(WheelLeggedEnvCfg):
    """Phase 2: flat ground, 3-axis commands (vx, vy, yaw rate). Fine-tuned from Phase 1."""

    def __post_init__(self):
        super().__post_init__()
        configure_for_robot(self, GO2W_SPEC)
        apply_phase_config(self, CONFIG_DIR / "phase2_flat3axis.yaml")


@configclass
class Go2WPhase2Flat3AxisEnvCfg_PLAY(Go2WPhase2Flat3AxisEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        make_play(self)


@configclass
class Go2WPhase3FlatCoordEnvCfg(WheelLeggedEnvCfg):
    """Phase 3: flat ground, 3-axis + high-speed commands, full domain randomisation. Fine-tuned from Phase 2."""

    def __post_init__(self):
        super().__post_init__()
        configure_for_robot(self, GO2W_SPEC)
        apply_phase_config(self, CONFIG_DIR / "phase3_flatcoord.yaml")


@configclass
class Go2WPhase3FlatCoordEnvCfg_PLAY(Go2WPhase3FlatCoordEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        make_play(self)


@configclass
class Go2WPhase4StepsEnvCfg(WheelLeggedEnvCfg):
    """Phase 4: flat ground + small steps (0 -> 0.15 m) with a terrain-level curriculum. Fine-tuned from Phase 3."""

    def __post_init__(self):
        super().__post_init__()
        configure_for_robot(self, GO2W_SPEC)
        apply_phase_config(self, CONFIG_DIR / "phase4_steps.yaml")


@configclass
class Go2WPhase4StepsEnvCfg_PLAY(Go2WPhase4StepsEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        make_play(self)
