"""Robot-agnostic manager-based environment for four-wheel-leg robots.

``WheelLeggedEnvCfg`` declares every term with robot-independent defaults. A robot task (``envs/<robot>/env_cfg.py``)
calls :func:`configure_for_robot` with its :class:`RobotSpec` and then :func:`apply_phase_config` with a phase YAML.

Interface fixed across all phases (so weights transfer from one phase to the next):

    policy obs (240) = base_ang_vel 3 | projected_gravity 3 | velocity_commands 3 | joint_pos_rel(legs) 12
                     | joint_vel_rel(legs+wheels) 16 | last_action 16 | terrain_height_scan 187
    critic obs (263) = policy terms without noise | base_lin_vel 3 | joint_effort 16 | wheel_contact 4
               (+90 from Phase 3: the domain randomisation values, appended last -- see CriticObsCfg)
    action (16)      = JointPositionAction(legs) 12 | JointVelocityAction(wheels) 4

Term names deliberately match the observation / action registry of ``unitree_rl_lab/deploy`` (C++).
"""

from __future__ import annotations

import math
import re
from dataclasses import MISSING
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs import mdp as ilmdp
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from ..commands.velocity_commands import WheelLeggedVelocityCommandCfg
from ..curriculum.curriculum import command_levels, terrain_levels_tracking
from ..events import events as wl_ev
from ..observations import observations as wl_obs
from ..observations import privileged as wl_priv
from ..rewards import rewards as wl_rew
from ..robots.builder import build_articulation_cfg
from ..robots.robot_config import RobotSpec
from ..terrains.terrain_config import apply_terrain_section
from ..utils.config_loader import apply_overrides, apply_reward_section, load_yaml, resolve_config_path

# Placeholders replaced by configure_for_robot(); keeping them explicit makes a forgotten call fail loudly.
_LEGS = SceneEntityCfg("robot", joint_names=MISSING)
_WHEELS = SceneEntityCfg("robot", joint_names=MISSING)
_ALL = SceneEntityCfg("robot", joint_names=MISSING)

HEIGHT_SCAN_SIZE = (1.6, 1.0)
HEIGHT_SCAN_RESOLUTION = 0.1  # 17 x 11 = 187 rays


##
# Scene
##


@configclass
class WheelLeggedSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",  # replaced by the phase YAML ``terrain`` section
        max_init_terrain_level=0,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )
    robot: ArticulationCfg = MISSING
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=HEIGHT_SCAN_RESOLUTION, size=list(HEIGHT_SCAN_SIZE)),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=750.0, color=(0.9, 0.9, 0.9)),
    )


##
# MDP
##


@configclass
class CommandsCfg:
    base_velocity = WheelLeggedVelocityCommandCfg(debug_vis=True)


@configclass
class ActionsCfg:
    # Order matters: the ActionManager concatenates terms in declaration order -> [legs 12 | wheels 4].
    JointPositionAction = ilmdp.JointPositionActionCfg(
        asset_name="robot", joint_names=MISSING, scale=0.25, use_default_offset=True
    )
    # scale 5: |a| ~ 1 -> 5 rad/s, +-1 m/s needs |a| ~ 2; with 10 the noise alone saturated the +-30.1 rad/s clip
    JointVelocityAction = ilmdp.JointVelocityActionCfg(
        asset_name="robot", joint_names=MISSING, scale=5.0, use_default_offset=True, clip=MISSING
    )


@configclass
class PolicyObsCfg(ObsGroup):
    """Only signals available on the robot. Order and names are the deploy interface -- do not reorder."""

    base_ang_vel = ObsTerm(func=ilmdp.base_ang_vel, scale=0.25, noise=Unoise(n_min=-0.2, n_max=0.2))
    projected_gravity = ObsTerm(func=ilmdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
    velocity_commands = ObsTerm(func=ilmdp.generated_commands, params={"command_name": "base_velocity"})
    joint_pos_rel = ObsTerm(
        func=ilmdp.joint_pos_rel, params={"asset_cfg": _LEGS}, noise=Unoise(n_min=-0.01, n_max=0.01)
    )
    joint_vel_rel = ObsTerm(
        func=ilmdp.joint_vel_rel, params={"asset_cfg": _ALL}, scale=0.05, noise=Unoise(n_min=-1.5, n_max=1.5)
    )
    last_action = ObsTerm(func=ilmdp.last_action)
    terrain_height_scan = ObsTerm(
        func=wl_obs.terrain_height_scan,
        params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": 0.5},
        noise=Unoise(n_min=-0.02, n_max=0.02),
        clip=(-1.0, 1.0),
    )

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True
        self.history_length = 1


@configclass
class CriticObsCfg(PolicyObsCfg):
    """Policy terms without noise (Isaac Lab applies noise only when ``enable_corruption``) + privileged state."""

    base_lin_vel = ObsTerm(func=ilmdp.base_lin_vel)
    joint_effort = ObsTerm(func=ilmdp.joint_effort, params={"asset_cfg": _ALL}, scale=0.05)
    wheel_contact = ObsTerm(
        func=wl_obs.wheel_contact, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=MISSING)}
    )
    # Domain randomisation values (observations/privileged.py), normalised to ~[-1, 1]. They come after every
    # earlier critic term, so a critic of an earlier phase is expanded by zero input columns (agents/checkpoint.py).
    # Phases 1-2 disable them (``null``); dims for Go2-W in brackets.
    wheel_friction = ObsTerm(  # [8] static x 4 wheels, dynamic x 4
        func=wl_priv.wheel_friction, params={"asset_cfg": SceneEntityCfg("robot", body_names=MISSING)}
    )
    base_inertial = ObsTerm(  # [5] added mass, CoM offset x/y/z, inertia scale
        func=wl_priv.base_inertial_params, params={"asset_cfg": SceneEntityCfg("robot", body_names=MISSING)}
    )
    kp_scale = ObsTerm(func=wl_priv.actuator_gain_scale, params={"asset_cfg": _LEGS, "gain": "stiffness"})  # [12]
    kd_scale = ObsTerm(func=wl_priv.actuator_gain_scale, params={"asset_cfg": _ALL, "gain": "damping"})  # [16]
    motor_strength = ObsTerm(func=wl_priv.motor_strength, params={"asset_cfg": _ALL})  # [16]
    joint_friction = ObsTerm(func=wl_priv.joint_friction, params={"asset_cfg": _ALL})  # [32] Coulomb, viscous
    command_delay = ObsTerm(func=wl_priv.command_delay, params={"asset_cfg": SceneEntityCfg("robot")})  # [1]

    def __post_init__(self):
        super().__post_init__()
        self.enable_corruption = False


@configclass
class ObservationsCfg:
    policy: PolicyObsCfg = PolicyObsCfg()
    critic: CriticObsCfg = CriticObsCfg()


@configclass
class RewardsCfg:
    """All available terms. Weights come only from the phase YAML (unlisted terms are disabled)."""

    track_lin_vel_xy = RewTerm(
        func=ilmdp.track_lin_vel_xy_exp, weight=0.0, params={"command_name": "base_velocity", "std": 0.5}
    )
    track_ang_vel_z = RewTerm(
        func=ilmdp.track_ang_vel_z_exp, weight=0.0, params={"command_name": "base_velocity", "std": 0.5}
    )
    flat_orientation = RewTerm(func=ilmdp.flat_orientation_l2, weight=0.0)
    lin_vel_z = RewTerm(func=ilmdp.lin_vel_z_l2, weight=0.0)
    ang_vel_xy = RewTerm(func=ilmdp.ang_vel_xy_l2, weight=0.0)
    termination = RewTerm(func=ilmdp.is_terminated, weight=0.0)
    stand_still = RewTerm(
        func=wl_rew.stand_still,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "leg_cfg": _LEGS,
            "wheel_cfg": _WHEELS,
            "command_threshold": 0.1,
            "wheel_vel_scale": 0.05,
        },
    )
    action_rate = RewTerm(func=ilmdp.action_rate_l2, weight=0.0)
    mech_power_legs = RewTerm(func=wl_rew.mechanical_power, weight=0.0, params={"asset_cfg": _LEGS})
    mech_power_wheels = RewTerm(func=wl_rew.mechanical_power, weight=0.0, params={"asset_cfg": _WHEELS})
    joint_torques_legs = RewTerm(func=ilmdp.joint_torques_l2, weight=0.0, params={"asset_cfg": _LEGS})
    joint_pos_limits_legs = RewTerm(func=ilmdp.joint_pos_limits, weight=0.0, params={"asset_cfg": _LEGS})
    wheel_slip = RewTerm(
        func=wl_rew.wheel_slip,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=MISSING),
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "wheel_radius": MISSING,
            "wheel_axis": MISSING,
            "contact_threshold": 1.0,
        },
    )
    base_height = RewTerm(
        func=ilmdp.base_height_l2,
        weight=0.0,
        params={"target_height": MISSING, "sensor_cfg": SceneEntityCfg("height_scanner")},
    )
    undesired_contacts = RewTerm(
        func=ilmdp.undesired_contacts,
        weight=0.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=MISSING), "threshold": 1.0},
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=ilmdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=ilmdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=MISSING), "threshold": 1.0},
    )
    bad_orientation = DoneTerm(func=ilmdp.bad_orientation, params={"limit_angle": 0.8})


@configclass
class EventCfg:
    """Default = the full domain randomisation of the plan. Phase YAMLs narrow a term or disable it (``null``).
    Startup terms run in declaration order: the mass terms rescale the inertia, ``base_inertia`` then adds its error.
    """

    # startup
    physics_material = EventTerm(
        func=ilmdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.2),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )
    add_base_mass = EventTerm(
        func=ilmdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
            "recompute_inertia": True,  # payload spread like the base
        },
    )
    link_mass = EventTerm(  # legs and wheels (CAD error); every body except the base
        func=ilmdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "mass_distribution_params": (0.9, 1.1),
            "operation": "scale",
            "recompute_inertia": True,
        },
    )
    base_inertia = EventTerm(
        func=wl_ev.randomize_rigid_body_inertia,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names=MISSING), "scale_range": (0.8, 1.2)},
    )
    base_com = EventTerm(
        func=ilmdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "com_range": {"x": (-0.03, 0.03), "y": (-0.02, 0.02), "z": (-0.02, 0.02)},
        },
    )
    # reset
    actuator_gains = EventTerm(  # Kp / Kd error per motor (the wheels have Kp = 0: only Kd changes)
        func=ilmdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )
    motor_strength = EventTerm(  # torque constant error per motor: scales the torque axis of the T-N curve
        func=wl_ev.randomize_motor_strength,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot"), "strength_range": (0.8, 1.2)},
    )
    joint_friction = EventTerm(  # absolute values per motor (ESTIMATE ranges; the nominal model has no friction)
        func=wl_ev.randomize_joint_friction,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot"), "static_range": (0.0, 0.2), "viscous_range": (0.0, 0.02)},
    )
    # one command delay per robot for every actuator group; the range is the actuators' min_delay..max_delay
    # (phase YAML ``actuators`` section, 0 by default = no delay)
    actuator_delay = EventTerm(
        func=wl_ev.sync_actuator_delay, mode="reset", params={"asset_cfg": SceneEntityCfg("robot")}
    )
    reset_base = EventTerm(
        func=ilmdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-math.pi, math.pi)},
            "velocity_range": {
                "x": (-0.2, 0.2),
                "y": (-0.2, 0.2),
                "z": (-0.2, 0.2),
                "roll": (-0.2, 0.2),
                "pitch": (-0.2, 0.2),
                "yaw": (-0.2, 0.2),
            },
        },
    )
    reset_robot_joints = EventTerm(
        func=ilmdp.reset_joints_by_offset,
        mode="reset",
        params={"position_range": (-0.1, 0.1), "velocity_range": (-1.0, 1.0)},
    )
    # interval
    push_robot = EventTerm(
        func=ilmdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class CurriculumCfg:
    lin_vel_x_levels = CurrTerm(
        func=command_levels, params={"axis": "lin_vel_x", "reward_term_name": "track_lin_vel_xy", "stand_gate": None}
    )
    lin_vel_y_levels = CurrTerm(
        func=command_levels, params={"axis": "lin_vel_y", "reward_term_name": "track_lin_vel_xy", "stand_gate": None}
    )
    ang_vel_z_levels = CurrTerm(
        func=command_levels, params={"axis": "ang_vel_z", "reward_term_name": "track_ang_vel_z", "stand_gate": None}
    )
    # rows of the terrain grid (generator terrains only; finalize() drops it on a plane). See levels.terrain_level_moves
    terrain_levels = CurrTerm(
        func=terrain_levels_tracking,
        params={"reward_term_name": "track_lin_vel_xy", "promote_distance": None, "demote_tracking": 0.5},
    )


@configclass
class WheelLeggedInfoCfg:
    """Plain-typed robot facts needed at runtime (diagnostics, deploy export). Filled from the RobotSpec."""

    robot_name: str = MISSING
    phase_config: str = MISSING
    leg_joint_names: list[str] = MISSING
    wheel_joint_names: list[str] = MISSING
    wheel_body_names: list[str] = MISSING
    sdk_joint_order: list[str] = MISSING
    wheel_radius: float = MISSING
    wheel_axis: tuple[float, float, float] = MISSING
    base_body_name: str = MISSING


@configclass
class WheelLeggedEnvCfg(ManagerBasedRLEnvCfg):
    scene: WheelLeggedSceneCfg = WheelLeggedSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    wl: WheelLeggedInfoCfg = WheelLeggedInfoCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15


##
# Assembly helpers
##


def configure_for_robot(cfg: WheelLeggedEnvCfg, spec: RobotSpec) -> None:
    """Fill every robot-dependent field of ``cfg`` from ``spec``."""
    legs, wheels = list(spec.leg_joint_names), list(spec.wheel_joint_names)
    wheel_bodies = list(spec.wheel_body_names)
    cfg.scene.robot = build_articulation_cfg(spec)
    cfg.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + spec.base_body_name

    # entity selectors shared by several terms (fresh objects: Isaac Lab resolves ids in place)
    def leg_sel():
        return SceneEntityCfg("robot", joint_names=legs, preserve_order=False)

    def wheel_sel():
        return SceneEntityCfg("robot", joint_names=wheels, preserve_order=False)

    def all_sel():
        return SceneEntityCfg("robot", joint_names=legs + wheels, preserve_order=False)

    # actions
    cfg.actions.JointPositionAction.joint_names = legs
    cfg.actions.JointVelocityAction.joint_names = wheels
    wheel_limit = min(spec.actuator_for(w).resolved_velocity_limit() for w in wheels)
    cfg.actions.JointVelocityAction.clip = {".*": (-wheel_limit, wheel_limit)}

    # observations
    for group in (cfg.observations.policy, cfg.observations.critic):
        group.joint_pos_rel.params["asset_cfg"] = leg_sel()
        group.joint_vel_rel.params["asset_cfg"] = all_sel()
    cfg.observations.critic.joint_effort.params["asset_cfg"] = all_sel()
    critic = cfg.observations.critic
    critic.wheel_contact.params["sensor_cfg"] = SceneEntityCfg("contact_forces", body_names=wheel_bodies)
    critic.wheel_friction.params["asset_cfg"] = SceneEntityCfg("robot", body_names=wheel_bodies, preserve_order=True)
    critic.base_inertial.params["asset_cfg"] = SceneEntityCfg(
        "robot", body_names=spec.mass_randomization_body or spec.base_body_name
    )
    critic.kp_scale.params["asset_cfg"] = leg_sel()
    for term in (critic.kd_scale, critic.motor_strength, critic.joint_friction):
        term.params["asset_cfg"] = all_sel()

    # rewards
    r = cfg.rewards
    r.stand_still.params["leg_cfg"] = leg_sel()
    r.stand_still.params["wheel_cfg"] = wheel_sel()
    r.mech_power_legs.params["asset_cfg"] = leg_sel()
    r.mech_power_wheels.params["asset_cfg"] = wheel_sel()
    r.joint_torques_legs.params["asset_cfg"] = leg_sel()
    r.joint_pos_limits_legs.params["asset_cfg"] = leg_sel()
    r.wheel_slip.params.update(
        sensor_cfg=SceneEntityCfg("contact_forces", body_names=wheel_bodies),
        asset_cfg=SceneEntityCfg("robot", body_names=wheel_bodies),
        wheel_radius=spec.wheel_radius,
        wheel_axis=tuple(spec.wheel_axis),
    )
    r.base_height.params["target_height"] = spec.nominal_base_height
    r.undesired_contacts.params["sensor_cfg"] = SceneEntityCfg(
        "contact_forces", body_names=list(spec.undesired_contact_bodies)
    )

    # terminations / events
    cfg.terminations.base_contact.params["sensor_cfg"] = SceneEntityCfg(
        "contact_forces", body_names=list(spec.termination_contact_bodies)
    )
    mass_body = spec.mass_randomization_body or spec.base_body_name
    ev = cfg.events
    ev.add_base_mass.params["asset_cfg"] = SceneEntityCfg("robot", body_names=mass_body)
    ev.link_mass.params["asset_cfg"] = SceneEntityCfg("robot", body_names=f"(?!{re.escape(mass_body)}$).*")
    ev.base_inertia.params["asset_cfg"] = SceneEntityCfg("robot", body_names=mass_body)
    ev.base_com.params["asset_cfg"] = SceneEntityCfg("robot", body_names=mass_body)

    # runtime info
    cfg.wl.robot_name = spec.name
    cfg.wl.leg_joint_names = legs
    cfg.wl.wheel_joint_names = wheels
    cfg.wl.wheel_body_names = wheel_bodies
    cfg.wl.sdk_joint_order = list(spec.sdk_joint_order)
    cfg.wl.wheel_radius = spec.wheel_radius
    cfg.wl.wheel_axis = tuple(spec.wheel_axis)
    cfg.wl.base_body_name = spec.base_body_name


# YAML sections handled specially; everything else is a plain attribute override on the env cfg.
_SPECIAL_SECTIONS = ("rewards", "terrain", "actuators", "evaluation")


def apply_phase_config(cfg: WheelLeggedEnvCfg, path: str | Path) -> dict:
    """Apply a phase YAML (see ``configs/<robot>/phase*.yaml``). Returns the parsed YAML."""
    path = resolve_config_path(path)
    data = load_yaml(path)
    cfg.wl.phase_config = str(Path(path).resolve())

    apply_reward_section(cfg.rewards, data.get("rewards"))
    apply_terrain_section(cfg.scene.terrain, data.get("terrain"))
    for group, overrides in (data.get("actuators") or {}).items():
        if group not in cfg.scene.robot.actuators:
            raise KeyError(f"unknown actuator group '{group}' (have {sorted(cfg.scene.robot.actuators)})")
        apply_overrides(cfg.scene.robot.actuators[group], overrides, f"actuators.{group}")
    apply_overrides(cfg, {k: v for k, v in data.items() if k not in _SPECIAL_SECTIONS})

    finalize(cfg)
    return data


def finalize(cfg: WheelLeggedEnvCfg) -> None:
    """Derived settings that depend on the final values (call after every override)."""
    cfg.sim.render_interval = cfg.decimation
    cfg.sim.physics_material = cfg.scene.terrain.physics_material
    if cfg.scene.height_scanner is not None:
        cfg.scene.height_scanner.update_period = cfg.decimation * cfg.sim.dt
    if cfg.scene.contact_forces is not None:
        cfg.scene.contact_forces.update_period = cfg.sim.dt
    generator = cfg.scene.terrain.terrain_generator
    if generator is not None:
        generator.curriculum = getattr(cfg.curriculum, "terrain_levels", None) is not None
    elif getattr(cfg.curriculum, "terrain_levels", None) is not None:
        cfg.curriculum.terrain_levels = None  # plane terrain has no levels
