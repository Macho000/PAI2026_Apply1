"""deploy.yaml builder + PolicyInterface, with the Go2-W spec and a PhysX-like (breadth-first) joint order."""

import numpy as np
import pytest
import yaml

from wheel_legged_rl.deployment.export_deploy_cfg import (
    build_deploy_cfg,
    load_deploy_cfg,
    sim_to_sdk_map,
    validate_deploy_cfg,
    write_deploy_cfg,
)
from wheel_legged_rl.deployment.policy_interface import PolicyInterface, RobotState
from wheel_legged_rl.robots.go2w import GO2W_SPEC

LEGS = ("FL", "FR", "RL", "RR")
# Isaac Sim orders articulation joints breadth-first: all hips, all thighs, all calves, all wheels.
SIM = [f"{leg}_{j}_joint" for j in ("hip", "thigh", "calf", "foot") for leg in LEGS]
N_SCAN = 187


def spec_deploy_cfg(spec=GO2W_SPEC, sim=SIM, history=1):
    """What collect_from_env() produces for the Phase 1 env (values from base_env_cfg / phase1 yaml)."""
    leg_ids = [i for i, n in enumerate(sim) if n in spec.leg_joint_names]
    wheel_ids = [i for i, n in enumerate(sim) if n in spec.wheel_joint_names]
    default = [spec.default_position(n) for n in sim]
    wheel_lim = spec.actuator_for(spec.wheel_joint_names[0]).resolved_velocity_limit()
    obs = [
        dict(name="base_ang_vel", dim=3, scale=[0.25] * 3, clip=None, params={}),
        dict(name="projected_gravity", dim=3, scale=[1.0] * 3, clip=None, params={}),
        dict(name="velocity_commands", dim=3, scale=[1.0] * 3, clip=None, params={"command_name": "base_velocity"}),
        dict(name="joint_pos_rel", dim=12, scale=[1.0] * 12, clip=None, params={"asset_cfg": {"joint_ids": leg_ids}}),
        dict(
            name="joint_vel_rel",
            dim=16,
            scale=[0.05] * 16,
            clip=None,
            params={"asset_cfg": {"joint_ids": list(range(16))}},
        ),
        dict(name="last_action", dim=16, scale=[1.0] * 16, clip=None, params={}),
        dict(name="terrain_height_scan", dim=N_SCAN, scale=[1.0] * N_SCAN, clip=[-1.0, 1.0], params={"offset": 0.5}),
    ]
    for o in obs:
        o["history_length"] = history
    return build_deploy_cfg(
        sim_joint_names=sim,
        sdk_joint_order=spec.sdk_joint_order,
        stiffness_sim=[spec.actuator_for(n).stiffness for n in sim],
        damping_sim=[spec.actuator_for(n).damping for n in sim],
        default_joint_pos_sim=default,
        step_dt=0.02,
        command_ranges={"lin_vel_x": [-1.0, 1.0], "lin_vel_y": [0.0, 0.0], "ang_vel_z": [0.0, 0.0]},
        actions=[
            dict(
                name="JointPositionAction",
                joint_ids=leg_ids,
                scale=[0.25] * 12,
                offset=[default[i] for i in leg_ids],
                clip=None,
            ),
            dict(
                name="JointVelocityAction",
                joint_ids=wheel_ids,
                scale=[10.0] * 4,
                offset=[0.0] * 4,
                clip=[[-wheel_lim, wheel_lim]] * 4,
            ),
        ],
        observations=obs,
        robot={"name": "go2w", "wheel_radius": spec.wheel_radius},
    )


def test_joint_ids_map_semantics():
    m = sim_to_sdk_map(SIM, GO2W_SPEC.sdk_joint_order)
    # C++: joint_pos_sim[i] = motor[joint_ids_map[i]]  -> FL_hip (sim 0) is SDK motor 3, FR_hip (sim 1) is motor 0
    assert m[0] == 3 and m[1] == 0 and m[12] == 13 and m[13] == 12
    assert [GO2W_SPEC.sdk_joint_order[k] for k in m] == SIM


def test_deploy_cfg_layout():
    cfg = spec_deploy_cfg()
    validate_deploy_cfg(cfg)
    assert cfg["policy"]["obs_dim"] == 240 and cfg["policy"]["action_dim"] == 16
    assert cfg["policy"]["cpp_unsupported"] == ["terrain_height_scan"]
    # Kp in SDK order: legs 25, wheels 0 (SDK 12..15)
    assert cfg["stiffness"] == [25.0] * 12 + [0.0] * 4
    assert cfg["damping"] == [0.5] * 16
    # legs = first 12 sim joints, wheels = last 4 (checked by joint_test.py on the real import)
    assert cfg["actions"]["JointPositionAction"]["joint_ids"] == list(range(12))
    assert cfg["actions"]["JointVelocityAction"]["joint_ids"] == [12, 13, 14, 15]
    assert cfg["actions"]["JointVelocityAction"]["clip"] == [[-30.1, 30.1]] * 4
    assert list(cfg["observations"])[0] == "base_ang_vel"
    assert all("params" in t for t in cfg["observations"].values())  # C++ single-group detection needs params


def test_yaml_roundtrip(tmp_path):
    cfg = spec_deploy_cfg()
    path = write_deploy_cfg(cfg, tmp_path)
    assert path == tmp_path / "params" / "deploy.yaml"
    loaded = load_deploy_cfg(path)
    assert loaded == yaml.safe_load(yaml.safe_dump(cfg))


def test_validate_rejects_overlapping_actions():
    cfg = spec_deploy_cfg()
    cfg["actions"]["JointVelocityAction"]["joint_ids"] = [11, 12, 13, 14]
    with pytest.raises(ValueError, match="already driven"):
        validate_deploy_cfg(cfg)


def _state(rng, height=True):
    return RobotState(
        base_ang_vel=rng.normal(size=3),
        projected_gravity=np.array([0.0, 0.0, -1.0]),
        command=np.array([2.0, 0.3, 0.0]),  # outside the phase 1 range -> clamped
        joint_pos=rng.normal(size=16),
        joint_vel=rng.normal(size=16) * 5,
        height_map=rng.normal(size=N_SCAN) * 2 if height else None,
    )


def test_policy_interface_observation():
    cfg = spec_deploy_cfg()
    iface = PolicyInterface(cfg, lambda obs: np.zeros(16))
    rng = np.random.default_rng(0)
    s = _state(rng)
    obs = iface.build_observation(s)
    assert obs.shape == (240,)
    m = np.asarray(cfg["joint_ids_map"])
    q_sim, dq_sim = s.joint_pos[m], s.joint_vel[m]
    np.testing.assert_allclose(obs[0:3], 0.25 * s.base_ang_vel)
    np.testing.assert_allclose(obs[6:9], [1.0, 0.0, 0.0])  # clamped to limit ranges
    np.testing.assert_allclose(obs[9:21], q_sim[:12] - np.asarray(cfg["default_joint_pos"])[:12])
    np.testing.assert_allclose(obs[21:37], 0.05 * dq_sim)
    np.testing.assert_allclose(obs[37:53], 0.0)
    np.testing.assert_allclose(obs[53:], np.clip(s.height_map, -1, 1))
    with pytest.raises(ValueError, match="height_map"):
        iface.build_observation(_state(rng, height=False))


def test_policy_interface_actions_to_sdk():
    cfg = spec_deploy_cfg()
    action = np.linspace(-4, 4, 16)
    iface = PolicyInterface(cfg, lambda obs: action)
    cmd = iface.step(_state(np.random.default_rng(1)))
    m = np.asarray(cfg["joint_ids_map"])
    default = np.asarray(cfg["default_joint_pos"])
    # legs: q* = default + 0.25 a, in SDK order
    for sim_i in range(12):
        assert cmd.q_target[m[sim_i]] == pytest.approx(default[sim_i] + 0.25 * action[sim_i])
        assert cmd.dq_target[m[sim_i]] == 0.0
    # wheels: w* = clip(10 a, +-30.1)
    for k, sim_i in enumerate(range(12, 16)):
        assert cmd.dq_target[m[sim_i]] == pytest.approx(np.clip(10 * action[12 + k], -30.1, 30.1))
        assert cmd.kp[m[sim_i]] == 0.0
    np.testing.assert_allclose(iface.last_action, action)
    # last_action appears in the next observation
    obs = iface.build_observation(_state(np.random.default_rng(2)))
    np.testing.assert_allclose(obs[37:53], action)


def test_history_stacking_oldest_first():
    cfg = spec_deploy_cfg(history=3)
    assert cfg["policy"]["obs_dim"] == 720
    iface = PolicyInterface(cfg, lambda obs: np.zeros(16))
    rng = np.random.default_rng(3)
    s1, s2 = _state(rng), _state(rng)
    o1 = iface.build_observation(s1)
    np.testing.assert_allclose(o1[0:3], o1[3:6])  # history filled with the first sample after reset
    o2 = iface.build_observation(s2)
    np.testing.assert_allclose(o2[3:6], 0.25 * s1.base_ang_vel)
    np.testing.assert_allclose(o2[6:9], 0.25 * s2.base_ang_vel)
