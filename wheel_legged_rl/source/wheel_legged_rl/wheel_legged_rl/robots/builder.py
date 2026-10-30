"""RobotSpec -> Isaac Lab ``ArticulationCfg``. The only place where the robot description meets Isaac Lab."""

from __future__ import annotations

import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.sim.converters import UrdfConverterCfg

from ..actuators.leg_actuator import LegActuatorCfg
from ..actuators.wheel_actuator import WheelActuatorCfg
from ..utils.urdf_tools import prepare_urdf
from .robot_config import ActuatorSpec, RobotSpec, WheelActuatorSpec

CACHE_DIR = Path(os.environ.get("WL_CACHE_DIR", Path.home() / ".cache" / "wheel_legged_rl"))


def actuator_cfg(spec: ActuatorSpec) -> LegActuatorCfg | WheelActuatorCfg:
    curve = spec.resolved_curve()
    fs, fd = spec.resolved_friction()
    common = dict(
        joint_names_expr=list(spec.joint_names_expr),
        damping=spec.damping,
        armature=spec.resolved_armature(),
        friction=0.0,  # friction is modelled by the actuator, not by the PhysX joint
        effort_limit=curve.peak_torque,
        velocity_limit=spec.resolved_velocity_limit(),
        velocity_limit_sim=spec.resolved_velocity_limit(),
        min_delay=spec.min_delay,
        max_delay=spec.max_delay,
        y1=curve.y1,
        y2=curve.y2,
        x1=curve.x1,
        x2=curve.x2,
        model_friction_static=fs,
        model_friction_viscous=fd,
        model_friction_activation_vel=spec.friction_activation_vel,
    )
    if isinstance(spec, WheelActuatorSpec):
        return WheelActuatorCfg(stiffness=0.0, **common)
    return LegActuatorCfg(stiffness=spec.stiffness, **common)


def build_articulation_cfg(
    spec: RobotSpec, prim_path: str = "{ENV_REGEX_NS}/Robot", fix_base: bool = False
) -> ArticulationCfg:
    spec.validate()
    rigid_props = sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False,
        retain_accelerations=False,
        linear_damping=0.0,
        angular_damping=0.0,
        max_linear_velocity=1000.0,
        max_angular_velocity=1000.0,
        max_depenetration_velocity=1.0,
    )
    articulation_props = sim_utils.ArticulationRootPropertiesCfg(
        enabled_self_collisions=spec.self_collisions,
        solver_position_iteration_count=spec.solver_position_iterations,
        solver_velocity_iteration_count=spec.solver_velocity_iterations,
        fix_root_link=fix_base if spec.usd_path else None,
    )
    if spec.usd_path:
        spawn = sim_utils.UsdFileCfg(
            usd_path=spec.usd_path,
            activate_contact_sensors=True,
            rigid_props=rigid_props,
            articulation_props=articulation_props,
        )
    else:
        if not spec.urdf_path or not Path(spec.urdf_path).is_file():
            raise FileNotFoundError(
                f"URDF for '{spec.name}' not found at {spec.urdf_path}. "
                "Run scripts/fetch_go2w_description.sh or set the path in the RobotSpec / environment."
            )
        urdf = prepare_urdf(
            spec.urdf_path,
            CACHE_DIR / "urdf",
            wheel_joints=list(spec.wheel_joint_names),
            cylinder_wheels=spec.cylinder_wheels,
        )
        spawn = sim_utils.UrdfFileCfg(
            asset_path=str(urdf),
            # separate dirs: a fixed- and a free-base import of the same URDF must not overwrite each other
            usd_dir=str(CACHE_DIR / "usd" / spec.name / ("fixed_base" if fix_base else "floating_base")),
            fix_base=fix_base,
            merge_fixed_joints=spec.merge_fixed_joints,
            self_collision=spec.self_collisions,
            activate_contact_sensors=True,
            # explicit actuators compute the torque in Python; the PhysX drive only needs to exist
            joint_drive=UrdfConverterCfg.JointDriveCfg(
                drive_type="force",
                target_type="none",
                gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
            ),
            rigid_props=rigid_props,
            articulation_props=articulation_props,
        )
    return ArticulationCfg(
        prim_path=prim_path,
        spawn=spawn,
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, spec.init_base_height),
            joint_pos=dict(spec.default_joint_pos),
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=spec.soft_joint_pos_limit_factor,
        actuators={name: actuator_cfg(a) for name, a in spec.actuators.items()},
    )
