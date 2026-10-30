import ast
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from wheel_legged_rl.utils.config_loader import (
    apply_overrides,
    apply_reward_section,
    deep_merge,
    load_yaml,
    resolve_config_path,
    reward_term_names,
)

PKG = Path(__file__).resolve().parents[1] / "source" / "wheel_legged_rl" / "wheel_legged_rl"
PHASE1 = PKG / "configs" / "go2w" / "phase1_flat.yaml"
PHASE2 = PKG / "configs" / "go2w" / "phase2_flat3axis.yaml"
PHASE3 = PKG / "configs" / "go2w" / "phase3_flatcoord.yaml"
PHASE4 = PKG / "configs" / "go2w" / "phase4_steps.yaml"


@dataclass
class Noise:
    n_min: float = -1.0
    n_max: float = 1.0


@dataclass
class Term:
    func: object = print
    weight: float = 0.0
    params: dict = field(default_factory=dict)


@dataclass
class Rewards:
    track: Term = field(default_factory=lambda: Term(params={"std": 0.5, "command_name": "base_velocity"}))
    slip: Term = field(default_factory=Term)
    power: Term = field(default_factory=Term)


@dataclass
class Ranges:
    lin_vel_x: tuple = (0.0, 0.0)


@dataclass
class Cmd:
    ranges: Ranges = field(default_factory=Ranges)
    rel_standing_envs: float = 0.0


@dataclass
class Cfg:
    episode_length_s: float = 20.0
    cmd: Cmd = field(default_factory=Cmd)
    noise: Noise = field(default_factory=Noise)
    params: dict = field(default_factory=lambda: {"velocity_range": {"x": (0.0, 0.0)}})
    term: Term | None = field(default_factory=Term)


def test_apply_overrides_nested_and_types():
    cfg = Cfg()
    apply_overrides(
        cfg,
        {
            "episode_length_s": 10,
            "cmd": {"ranges": {"lin_vel_x": [-1, 1]}, "rel_standing_envs": 0.2},
            "noise": {"n_min": -0.02},
            "params": {"velocity_range": {"x": [-0.5, 0.5], "y": [-0.1, 0.1]}},
            "term": None,
        },
    )
    assert cfg.episode_length_s == 10.0 and isinstance(cfg.episode_length_s, float)
    assert cfg.cmd.ranges.lin_vel_x == (-1, 1)
    assert cfg.cmd.rel_standing_envs == 0.2
    assert cfg.noise.n_min == -0.02 and cfg.noise.n_max == 1.0
    assert cfg.params["velocity_range"] == {"x": (-0.5, 0.5), "y": (-0.1, 0.1)}  # new keys: lists -> tuples
    assert cfg.term is None


def test_unknown_key_raises():
    with pytest.raises(KeyError, match="cmd.typo"):
        apply_overrides(Cfg(), {"cmd": {"typo": 1}})


def test_reward_section_unlisted_disabled():
    r = Rewards()
    r.power.weight = -1.0  # a "source default" that must not survive
    apply_reward_section(r, {"track": {"weight": 1.5, "params": {"std": 0.25}}, "slip": -0.1})
    assert r.track.weight == 1.5 and r.track.params["std"] == 0.25
    assert r.slip.weight == -0.1
    assert r.power.weight == 0.0
    apply_reward_section(r, {"slip": None})
    assert r.slip.weight == 0.0 and r.track.weight == 0.0


@pytest.mark.parametrize(
    "section, error",
    [
        ({"nope": 1.0}, "unknown reward"),
        ({"track": {"params": {"bad": 1}}}, "unknown param"),
        ({"track": {"wieght": 1}}, "unexpected keys"),
        ({"track": "x"}, None),
    ],
)
def test_reward_section_errors(section, error):
    with pytest.raises((KeyError, TypeError), match=error):
        apply_reward_section(Rewards(), section)


def test_reward_term_names():
    assert sorted(reward_term_names(Rewards())) == ["power", "slip", "track"]


def test_yaml_base_merge(tmp_path):
    (tmp_path / "base.yaml").write_text("a: 1\nb: {c: 2, d: 3}\n")
    (tmp_path / "child.yaml").write_text("_base_: base.yaml\nb: {d: 4}\ne: 5\n")
    assert load_yaml(tmp_path / "child.yaml") == {"a": 1, "b": {"c": 2, "d": 4}, "e": 5}
    assert deep_merge({"x": {"y": 1}}, {"x": {"z": 2}}) == {"x": {"y": 1, "z": 2}}


def test_env_var_overrides_config_path(monkeypatch, tmp_path):
    monkeypatch.delenv("WL_ENV_CONFIG", raising=False)
    assert resolve_config_path("a.yaml") == Path("a.yaml")
    monkeypatch.setenv("WL_ENV_CONFIG", str(tmp_path / "b.yaml"))
    assert resolve_config_path("a.yaml") == tmp_path / "b.yaml"


# ------------------------------------------------------------------ phase YAML vs env config (static, no Isaac)


def _class_fields(module: Path, class_name: str) -> dict[str, ast.AST]:
    tree = ast.parse(module.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    out = {}
    for stmt in cls.body:
        if isinstance(stmt, ast.Assign):
            out[stmt.targets[0].id] = stmt.value
        elif isinstance(stmt, ast.AnnAssign):
            out[stmt.target.id] = stmt.value
    return out


def test_phase1_yaml_matches_env_cfg():
    base = PKG / "envs" / "base_env_cfg.py"
    data = load_yaml(PHASE1)
    rewards = _class_fields(base, "RewardsCfg")
    assert set(data["rewards"]) <= set(rewards), set(data["rewards"]) - set(rewards)
    assert set(data["curriculum"]) <= set(_class_fields(base, "CurriculumCfg"))
    assert set(data["events"]) <= set(_class_fields(base, "EventCfg"))
    assert set(data["terminations"]) <= set(_class_fields(base, "TerminationsCfg"))
    assert set(data["observations"]["policy"]) <= set(_class_fields(base, "PolicyObsCfg"))
    # plan section 9: phase 1 weights
    w = {k: (v["weight"] if isinstance(v, dict) else v) for k, v in data["rewards"].items()}
    assert w == {
        "track_lin_vel_xy": 1.5,
        "track_ang_vel_z": 0.75,
        "flat_orientation": -2.5,
        "lin_vel_z": -2.0,
        "ang_vel_xy": -0.05,
        "termination": -10.0,
        "stand_still": -0.5,
        "base_height": -30.0,
        "action_rate": -0.01,
    }
    # stand stage first: every initial command is below 0.1 km/h and counts as a stand-still command
    cmd = data["commands"]["base_velocity"]
    stand_threshold = data["rewards"]["stand_still"]["params"]["command_threshold"]
    assert max(abs(v) for v in cmd["ranges"]["lin_vel_x"]) < stand_threshold <= 0.1 / 3.6
    gate = data["curriculum"]["lin_vel_x_levels"]["params"]["stand_gate"]
    assert set(gate["max_cost"]) <= {k for k, v in w.items() if v != 0.0}
    # plan section 12: phase 1 command limits
    lim = data["commands"]["base_velocity"]["limit_ranges"]
    assert lim == {"lin_vel_x": [-1.0, 1.0], "lin_vel_y": [0.0, 0.0], "ang_vel_z": [0.0, 0.0]}
    assert data["commands"]["base_velocity"]["rel_standing_envs"] == 0.2
    assert list(data["terrain"]["sub_terrains"]) == ["flat"]


def test_phase2_yaml_matches_env_cfg():
    base = PKG / "envs" / "base_env_cfg.py"
    p1, data = load_yaml(PHASE1), load_yaml(PHASE2)
    rewards = _class_fields(base, "RewardsCfg")
    assert set(data["rewards"]) <= set(rewards), set(data["rewards"]) - set(rewards)
    assert set(data["curriculum"]) <= set(_class_fields(base, "CurriculumCfg"))
    assert set(data["events"]) <= set(_class_fields(base, "EventCfg"))
    # plan section 12: phase 1 terms + action_rate, joint limits, collisions, stand_still
    w = {k: (v["weight"] if isinstance(v, dict) else v) for k, v in data["rewards"].items()}
    w1 = {k: (v["weight"] if isinstance(v, dict) else v) for k, v in p1["rewards"].items()}
    assert set(w) == set(w1) | {"joint_pos_limits_legs", "undesired_contacts"}
    assert all(w[k] < 0 for k in ("action_rate", "joint_pos_limits_legs", "undesired_contacts", "stand_still"))
    # plan section 12: phase 2 command limits; vx keeps the phase 1 distribution
    cmd = data["commands"]["base_velocity"]
    assert cmd["limit_ranges"] == {"lin_vel_x": [-1.0, 1.0], "lin_vel_y": [-0.5, 0.5], "ang_vel_z": [-1.5, 1.5]}
    assert cmd["ranges"]["lin_vel_x"] == p1["commands"]["base_velocity"]["limit_ranges"]["lin_vel_x"]
    for axis in ("lin_vel_y", "ang_vel_z"):
        lo, hi = cmd["ranges"][axis]
        assert cmd["limit_ranges"][axis][0] <= lo < hi <= cmd["limit_ranges"][axis][1]
        assert data["curriculum"][f"{axis}_levels"]["params"]["delta"] > 0
    assert data["curriculum"]["lin_vel_x_levels"] is None  # no stand gate: fine-tuned from a standing policy
    # observation / action interface unchanged -> phase 1 weights load
    assert data["observations"] == p1["observations"]
    assert data["terrain"] == p1["terrain"]
    for scen in data["evaluation"]["scenarios"].values():
        assert all(len(seg["command"]) == 3 for seg in scen)
    assert set(data["evaluation"]["criteria"]) >= {"vel_rmse", "yaw_rate_rmse", "fall_rate", "standstill_speed"}


# terms added for phase 3; phases 1-2 switch them off explicitly so that their behaviour does not change
MODEL_DR_EVENTS = ("link_mass", "base_inertia", "base_com", "actuator_gains", "motor_strength", "joint_friction")
# privileged critic terms (the randomised values); appended after the phase 1-2 critic terms, off in phases 1-2
PRIVILEGED_CRITIC = (
    "wheel_friction",
    "base_inertial",
    "kp_scale",
    "kd_scale",
    "motor_strength",
    "joint_friction",
    "command_delay",
)


def test_phase1_2_disable_model_randomisation():
    events = set(_class_fields(PKG / "envs" / "base_env_cfg.py", "EventCfg"))
    assert set(MODEL_DR_EVENTS) | {"actuator_delay"} <= events
    for path in (PHASE1, PHASE2):
        data = load_yaml(path)
        assert all(data["events"][k] is None for k in MODEL_DR_EVENTS), path
        assert "actuators" not in data  # no command delay
        assert data["observations"]["critic"] == {k: None for k in PRIVILEGED_CRITIC}, path


def test_privileged_critic_terms_come_last():
    fields = list(_class_fields(PKG / "envs" / "base_env_cfg.py", "CriticObsCfg"))
    # critic = policy terms (inherited, first) + these; the new ones last -> zero-column expansion of the checkpoint
    assert fields == ["base_lin_vel", "joint_effort", "wheel_contact", *PRIVILEGED_CRITIC]


def test_phase3_yaml_matches_env_cfg():
    base = PKG / "envs" / "base_env_cfg.py"
    p2, data = load_yaml(PHASE2), load_yaml(PHASE3)
    assert set(data["rewards"]) <= set(_class_fields(base, "RewardsCfg"))
    assert set(data["curriculum"]) <= set(_class_fields(base, "CurriculumCfg"))
    assert set(data["events"]) <= set(_class_fields(base, "EventCfg"))
    # plan section 9: phase 2 terms + energy (legs / wheels), leg torques, wheel slip
    w = {k: (v["weight"] if isinstance(v, dict) else v) for k, v in data["rewards"].items()}
    w2 = {k: (v["weight"] if isinstance(v, dict) else v) for k, v in p2["rewards"].items()}
    assert set(w) == set(w2) | {"mech_power_legs", "mech_power_wheels", "joint_torques_legs", "wheel_slip"}
    assert all(w[k] == w2[k] for k in w2)
    assert all(w[k] < 0 for k in set(w) - set(w2))
    # plan section 12: + high speed; vy / yaw rate keep the phase 2 distribution
    cmd = data["commands"]["base_velocity"]
    cmd2 = p2["commands"]["base_velocity"]["limit_ranges"]
    assert cmd["ranges"] == {**cmd2}
    assert cmd["limit_ranges"]["lin_vel_x"] == [-2.0, 2.0]
    assert {k: v for k, v in cmd["limit_ranges"].items() if k != "lin_vel_x"} == {
        k: v for k, v in cmd2.items() if k != "lin_vel_x"
    }
    assert data["curriculum"]["lin_vel_x_levels"]["params"]["delta"] > 0
    assert "stand_gate" not in data["curriculum"]["lin_vel_x_levels"]["params"]
    # plan section 12: friction randomisation; the sim-to-real terms are all on
    ev = data["events"]
    assert ev["physics_material"]["params"]["static_friction_range"] == [0.3, 1.2]
    assert all(ev[k] is not None for k in MODEL_DR_EVENTS) and "actuator_delay" not in ev
    delays = {(a["min_delay"], a["max_delay"]) for a in data["actuators"].values()}
    assert delays == {(0, 4)}  # one range for every group (events.sync_actuator_delay); 5 ms physics steps
    assert set(data["actuators"]) == {"hip_thigh", "calf", "wheel"}
    # actor observation unchanged -> phase 2 actor loads; the critic gets every privileged term
    assert data["observations"]["policy"] == p2["observations"]["policy"]
    assert "critic" not in data["observations"]
    assert data["episode_length_s"] == p2["episode_length_s"]
    # evaluation: every earlier scenario is kept (forgetting check) + attitude criterion
    assert set(p2["evaluation"]["scenarios"]) < set(data["evaluation"]["scenarios"])
    assert set(data["evaluation"]["criteria"]) == set(p2["evaluation"]["criteria"]) | {"roll_pitch_rms"}


def test_phase4_yaml_matches_env_cfg():
    from wheel_legged_rl.terrains.layout import column_terrain_names

    base = PKG / "envs" / "base_env_cfg.py"
    p3, data = load_yaml(PHASE3), load_yaml(PHASE4)
    assert set(data["curriculum"]) <= set(_class_fields(base, "CurriculumCfg"))
    # plan section 12: same commands / rewards / randomisation as phase 3; only the terrain (+ its curriculum) changes
    for key in ("rewards", "events", "actuators", "observations", "terminations", "episode_length_s"):
        assert data[key] == p3[key], key
    cmd, cmd3 = data["commands"]["base_velocity"], p3["commands"]["base_velocity"]
    assert cmd["limit_ranges"] == cmd3["limit_ranges"] and cmd["ranges"] == cmd3["limit_ranges"]
    assert data["curriculum"]["lin_vel_x_levels"] is None
    params = data["curriculum"]["terrain_levels"]["params"]
    assert set(params) == {"reward_term_name", "promote_distance", "demote_tracking"}
    assert params["reward_term_name"] in data["rewards"]
    # terrain: small steps 0 -> 0.15 m over the rows; flat >= 20 % against forgetting
    terrain = data["terrain"]
    assert terrain["generator"]["curriculum"] is True
    assert terrain["max_init_terrain_level"] < terrain["generator"]["num_rows"]
    subs = terrain["sub_terrains"]
    total = sum(s["proportion"] for s in subs.values())
    assert subs["flat"]["type"] == "flat" and subs["flat"]["proportion"] / total >= 0.2
    for name, sub in subs.items():
        rng = sub.get("step_height_range") or [2 * h for h in sub.get("grid_height_range", [0.0, 0.0])]
        assert rng[0] == 0.0 and rng[1] <= 0.15 + 1e-9, name  # neighbouring grid cells differ by up to 2 h
    # evaluation: every phase 3 scenario on flat ground + step traversals on every evaluation step terrain
    ev = data["evaluation"]
    scen = ev["scenarios"]
    assert set(p3["evaluation"]["scenarios"]) < set(scen)
    for name, segments in p3["evaluation"]["scenarios"].items():
        assert scen[name] == {"terrain": "flat", "group": "flat", "segments": segments}, name
    steps = {k: v for k, v in scen.items() if v["group"] == "steps"}
    assert steps and all(v["goal_distance"] > 3.0 for v in steps.values())  # past the last curb (3 m out)
    eval_subs = ev["terrain"]["sub_terrains"]
    assert {v["terrain"] for v in scen.values()} == set(eval_subs)
    gen = ev["terrain"]["generator"]
    assert gen["num_rows"] == 1 and gen["difficulty_range"] == [1.0, 1.0]
    cols = column_terrain_names({k: v["proportion"] for k, v in eval_subs.items()}, gen["num_cols"])
    assert set(cols) == set(eval_subs) and cols.count("flat") >= 3  # flat scenarios never leave flat ground
    assert ev["criteria"]["steps/success_rate"] == {"min": 0.8}
    assert {f"flat/{k}" for k in p3["evaluation"]["criteria"]} | {"steps/success_rate"} == set(ev["criteria"])
    for key in ev["criteria"]:
        assert key.split("/")[0] in {v["group"] for v in scen.values()}


def test_policy_obs_order_is_deploy_interface():
    fields = list(_class_fields(PKG / "envs" / "base_env_cfg.py", "PolicyObsCfg"))
    assert fields == [
        "base_ang_vel",
        "projected_gravity",
        "velocity_commands",
        "joint_pos_rel",
        "joint_vel_rel",
        "last_action",
        "terrain_height_scan",
    ]
    actions = list(_class_fields(PKG / "envs" / "base_env_cfg.py", "ActionsCfg"))
    assert actions == ["JointPositionAction", "JointVelocityAction"]
