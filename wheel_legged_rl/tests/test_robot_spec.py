import dataclasses

import pytest

from wheel_legged_rl.robots.go2w import GO2W_SPEC
from wheel_legged_rl.robots.robot_config import LegActuatorSpec, WheelActuatorSpec


def test_go2w_spec_valid():
    GO2W_SPEC.validate()
    assert GO2W_SPEC.action_dim == 16
    assert isinstance(GO2W_SPEC.actuator_for("FL_calf_joint"), LegActuatorSpec)
    assert isinstance(GO2W_SPEC.actuator_for("RR_foot_joint"), WheelActuatorSpec)
    assert GO2W_SPEC.actuator_for("RR_foot_joint").stiffness == 0.0
    assert GO2W_SPEC.actuator_for("FL_hip_joint").stiffness == 25.0
    assert GO2W_SPEC.default_position("RL_thigh_joint") == 0.8
    assert GO2W_SPEC.default_position("FR_calf_joint") == -1.5


def test_go2w_sdk_order_is_unitree():
    assert GO2W_SPEC.sdk_joint_order[:3] == ("FR_hip_joint", "FR_thigh_joint", "FR_calf_joint")
    assert GO2W_SPEC.sdk_joint_order[12:] == ("FR_foot_joint", "FL_foot_joint", "RR_foot_joint", "RL_foot_joint")


@pytest.mark.parametrize(
    "change, message",
    [
        ({"wheel_body_names": ("FL_foot",)}, "one-to-one"),
        ({"sdk_joint_order": ("FL_hip_joint",)}, "permutation"),
        ({"default_joint_pos": {".*_foot_joint": 0.3, ".*_thigh_joint": 0.8, ".*_calf_joint": -1.5}}, "default to 0"),
        ({"init_base_height": 0.3}, "spawn"),
        ({"wheel_radius": 0.0}, "positive"),
    ],
)
def test_invalid_specs(change, message):
    with pytest.raises(ValueError, match=message):
        dataclasses.replace(GO2W_SPEC, **change).validate()


def test_wheel_must_use_wheel_actuator():
    acts = dict(GO2W_SPEC.actuators)
    acts["wheel"] = LegActuatorSpec(joint_names_expr=(".*_foot_joint",), curve=acts["wheel"].curve, kp=1.0)
    with pytest.raises(ValueError, match="WheelActuatorSpec"):
        dataclasses.replace(GO2W_SPEC, actuators=acts).validate()
