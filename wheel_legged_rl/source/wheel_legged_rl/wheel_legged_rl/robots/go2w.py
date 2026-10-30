"""Unitree Go2-W (four legs x (hip, thigh, calf) + four driven wheels).

Values marked URDF were read from ``unitree_ros/robots/go2w_description/urdf/go2w_description.urdf`` with
``scripts/urdf_audit.py`` (see ``docs/URDF_AUDIT_go2w_generated.md``). Values marked ESTIMATE are not published by
Unitree and must be confirmed on hardware before sim-to-real.
"""

from __future__ import annotations

import os
from pathlib import Path

from ..actuators.qdd_model import GO2HV_CURVE
from .robot_config import LegActuatorSpec, RobotSpec, WheelActuatorSpec

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
"""``wheel_legged_rl/`` (the directory holding ``source/``, ``scripts/`` and ``third_party/``)."""

_DESCRIPTION = _PROJECT_ROOT / "third_party" / "unitree_ros" / "robots" / "go2w_description"
GO2W_URDF_PATH = os.environ.get("WL_GO2W_URDF", str(_DESCRIPTION / "urdf" / "go2w_description.urdf"))
"""Fetched by ``scripts/fetch_go2w_description.sh``; override with the ``WL_GO2W_URDF`` environment variable."""

_LEGS = ("FL", "FR", "RL", "RR")

GO2W_SPEC = RobotSpec(
    name="go2w",
    urdf_path=GO2W_URDF_PATH,
    # The URDF wheels collide through a tessellated mesh that the importer turns into a convex hull (a polygon).
    # Fitted cylinders (r = 0.0860, w = 0.0518, offset +-0.048 along y) roll smoothly. Confirm with joint_test.py roll.
    cylinder_wheels=True,
    merge_fixed_joints=True,
    base_body_name="base",
    leg_joint_names=tuple(f"{leg}_{j}_joint" for leg in _LEGS for j in ("hip", "thigh", "calf")),
    wheel_joint_names=tuple(f"{leg}_foot_joint" for leg in _LEGS),  # URDF: continuous, axis +y
    wheel_body_names=tuple(f"{leg}_foot" for leg in _LEGS),
    # unitree_rl_lab UNITREE_GO2W_CFG.joint_sdk_names (motor index order of the Go2-W low-level API)
    sdk_joint_order=(
        "FR_hip_joint",
        "FR_thigh_joint",
        "FR_calf_joint",
        "FL_hip_joint",
        "FL_thigh_joint",
        "FL_calf_joint",
        "RR_hip_joint",
        "RR_thigh_joint",
        "RR_calf_joint",
        "RL_hip_joint",
        "RL_thigh_joint",
        "RL_calf_joint",
        "FR_foot_joint",
        "FL_foot_joint",
        "RR_foot_joint",
        "RL_foot_joint",
    ),  # fmt: skip
    wheel_radius=0.0860,  # URDF (collision mesh left/right_wheel.dae)
    wheel_axis=(0.0, 1.0, 0.0),  # URDF: all four wheel joints use axis +y
    default_joint_pos={
        ".*_hip_joint": 0.0,
        ".*_thigh_joint": 0.8,
        ".*_calf_joint": -1.5,
        ".*_foot_joint": 0.0,
    },
    nominal_base_height=0.408,  # URDF forward kinematics at the default pose (0.4076 m)
    init_base_height=0.45,
    soft_joint_pos_limit_factor=0.9,
    actuators={
        # URDF effort 23.7 / velocity 30.1. Curve = unitree_rl_lab UnitreeActuatorCfg_Go2HV. Gains = UNITREE_GO2W_CFG.
        "hip_thigh": LegActuatorSpec(
            joint_names_expr=(".*_hip_joint", ".*_thigh_joint"),
            curve=GO2HV_CURVE,
            kp=25.0,
            damping=0.5,
            velocity_limit=30.1,
        ),
        # URDF effort 35.55 / velocity 20.07 (Go2 URDF: 45.43 / 15.70 -> different knee reduction).
        # ESTIMATE: Go2HV behind an extra 1.5:1 reduction -> Y1 30.3, Y2 35.1, X1 9.0, X2 20.0.
        "calf": LegActuatorSpec(
            joint_names_expr=(".*_calf_joint",),
            curve=GO2HV_CURVE.geared(1.5),
            kp=25.0,
            damping=0.5,
            velocity_limit=20.07,
        ),
        # URDF effort 23.7 / velocity 30.1. ESTIMATE: same curve as Go2HV. Kd = UNITREE_GO2W_CFG (0.5).
        # ESTIMATE armature (reflected rotor inertia). Required for the explicit velocity loop: the wheel alone has
        # I_yy = 0.0012 kg m^2, so Kd*dt/I = 0.5*0.005/0.0012 = 2.1 > 2 is unstable at the 200 Hz physics step
        # (joint_test showed +21 / -17 rad/s for a +5 rad/s command). With 0.005: 0.37. Checked by urdf_audit.py.
        "wheel": WheelActuatorSpec(
            joint_names_expr=(".*_foot_joint",),
            curve=GO2HV_CURVE,
            damping=0.5,
            armature=0.005,
            velocity_limit=30.1,
        ),
    },
    # Isaac Sim 5.1 keeps Head_upper / Head_lower (fixed, with mass) as separate bodies: include them explicitly
    termination_contact_bodies=("base", "Head_.*"),
    undesired_contact_bodies=("base", "Head_.*", ".*_hip", ".*_thigh"),
    mass_randomization_body="base",
)
