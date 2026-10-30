"""``params/deploy.yaml`` -- everything a controller needs to run the exported policy outside Isaac Lab.

The layout is a superset of what ``unitree_rl_lab/deploy`` (C++, ``deploy/robots/go2w``) reads:

``joint_ids_map``      sim joint i  -> SDK motor index (``joint_pos_sim[i] = motor[joint_ids_map[i]].q``)
``step_dt``            policy period [s]
``stiffness/damping``  Kp / Kd per motor, **SDK order**
``default_joint_pos``  default positions, **sim order**
``commands``           ``base_velocity.ranges`` (joystick clamp)
``actions``            ordered terms: ``joint_ids`` (sim indices), ``scale``, ``offset``, ``clip``
``observations``       ordered terms: ``params``, ``scale``, ``clip``, ``history_length``

Extra sections ignored by the C++ controller but used by :mod:`.policy_interface` and humans:
``robot`` (names, orders, wheel radius), ``policy`` (dimensions), ``perception`` (height-map geometry).

Observation terms without a C++ implementation (``terrain_height_scan``) are listed in ``policy.cpp_unsupported``;
the C++ controller needs a matching ``REGISTER_OBSERVATION`` before it can run such a policy.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

CPP_OBSERVATIONS = {
    "base_ang_vel",
    "projected_gravity",
    "joint_pos",
    "joint_pos_rel",
    "joint_vel_rel",
    "last_action",
    "velocity_commands",
    "gait_phase",
}
CPP_ACTIONS = {"JointPositionAction", "JointVelocityAction"}


def _round(x: Any) -> Any:
    if isinstance(x, float):
        return float(f"{x:.6g}")
    if isinstance(x, (list, tuple)):
        return [_round(v) for v in x]
    if isinstance(x, dict):
        return {k: _round(v) for k, v in x.items()}
    return x


def sim_to_sdk_map(sim_joint_names: Sequence[str], sdk_joint_order: Sequence[str]) -> list[int]:
    """``map[i]`` = SDK motor index of sim joint ``i`` (unitree_rl_lab ``joint_ids_map`` semantics)."""
    if sorted(sim_joint_names) != sorted(sdk_joint_order):
        raise ValueError(f"sim joints {list(sim_joint_names)} and SDK joints {list(sdk_joint_order)} differ")
    return [list(sdk_joint_order).index(n) for n in sim_joint_names]


def build_deploy_cfg(
    *,
    sim_joint_names: Sequence[str],
    sdk_joint_order: Sequence[str],
    stiffness_sim: Sequence[float],
    damping_sim: Sequence[float],
    default_joint_pos_sim: Sequence[float],
    step_dt: float,
    command_ranges: dict[str, Sequence[float]],
    actions: Sequence[dict],
    observations: Sequence[dict],
    robot: dict | None = None,
    perception: dict | None = None,
) -> dict:
    """Assemble the deploy dict.

    ``actions`` items: ``{name, joint_ids, scale, offset, clip}`` (lists per action dim; ``clip`` = list of
    ``[lo, hi]`` or ``None``). ``observations`` items: ``{name, dim, scale, clip, history_length, params}``.
    """
    n = len(sim_joint_names)
    joint_ids_map = sim_to_sdk_map(sim_joint_names, sdk_joint_order)
    stiffness_sdk = [0.0] * n
    damping_sdk = [0.0] * n
    for sim_i, sdk_i in enumerate(joint_ids_map):
        stiffness_sdk[sdk_i] = float(stiffness_sim[sim_i])
        damping_sdk[sdk_i] = float(damping_sim[sim_i])

    act_out: dict[str, dict] = {}
    for a in actions:
        dim = len(a["joint_ids"])
        act_out[a["name"]] = {
            "joint_names": [sim_joint_names[i] for i in a["joint_ids"]],
            "joint_ids": list(a["joint_ids"]),
            "scale": list(a["scale"]),
            "offset": list(a["offset"]),
            "clip": [list(c) for c in a["clip"]] if a.get("clip") is not None else None,
        }
        for key in ("scale", "offset"):
            if (n_key := len(act_out[a["name"]][key])) != dim:
                raise ValueError(f"action {a['name']}: {key} has {n_key} entries, expected {dim}")

    obs_out: dict[str, dict] = {}
    for o in observations:
        obs_out[o["name"]] = {
            "params": o.get("params") or {},
            "clip": list(o["clip"]) if o.get("clip") is not None else None,
            "scale": list(o["scale"]),
            "history_length": int(o.get("history_length") or 1),
        }
        if len(o["scale"]) != o["dim"]:
            raise ValueError(f"observation {o['name']}: scale has {len(o['scale'])} entries, expected {o['dim']}")

    obs_dim = sum(o["dim"] * int(o.get("history_length") or 1) for o in observations)
    cfg = {
        "joint_ids_map": joint_ids_map,
        "step_dt": float(step_dt),
        "stiffness": stiffness_sdk,
        "damping": damping_sdk,
        "default_joint_pos": [float(v) for v in default_joint_pos_sim],
        "commands": {"base_velocity": {"ranges": {k: list(v) for k, v in command_ranges.items()}}},
        "actions": act_out,
        "observations": obs_out,
        "policy": {
            "obs_dim": obs_dim,
            "action_dim": sum(len(a["joint_ids"]) for a in actions),
            "obs_terms": [{"name": o["name"], "dim": o["dim"]} for o in observations],
            "cpp_unsupported": sorted(
                [o["name"] for o in observations if o["name"] not in CPP_OBSERVATIONS]
                + [a["name"] for a in actions if a["name"] not in CPP_ACTIONS]
            ),
        },
        "robot": {"sim_joint_names": list(sim_joint_names), "sdk_joint_names": list(sdk_joint_order), **(robot or {})},
    }
    if perception:
        cfg["perception"] = perception
    validate_deploy_cfg(cfg)
    return _round(cfg)


def validate_deploy_cfg(cfg: dict) -> None:
    """Structural checks shared by the exporter and the policy interface. Raises ``ValueError``."""
    n = len(cfg["joint_ids_map"])
    if sorted(cfg["joint_ids_map"]) != list(range(n)):
        raise ValueError("joint_ids_map must be a permutation of range(num_joints)")
    for key in ("stiffness", "damping", "default_joint_pos"):
        if len(cfg[key]) != n:
            raise ValueError(f"{key} must have {n} entries")
    seen: set[int] = set()
    for name, term in cfg["actions"].items():
        ids = term["joint_ids"]
        if set(ids) & seen:
            raise ValueError(f"action {name} drives joints already driven by another term")
        seen |= set(ids)
        if term["clip"] is not None and len(term["clip"]) != len(ids):
            raise ValueError(f"action {name}: clip must have one [lo, hi] per joint")
    for name, term in cfg["observations"].items():
        if term["clip"] is not None and len(term["clip"]) != 2:
            raise ValueError(f"observation {name}: clip must be [lo, hi]")


def write_deploy_cfg(cfg: dict, log_dir: str | Path) -> Path:
    path = Path(log_dir) / "params" / "deploy.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, default_flow_style=None, sort_keys=False)
    return path


def load_deploy_cfg(path: str | Path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    validate_deploy_cfg(cfg)
    return cfg


# ---------------------------------------------------------------------------------------------------------------------
# Isaac Lab side
# ---------------------------------------------------------------------------------------------------------------------


def _per_dim(value: Any, dim: int, default: float) -> list[float]:
    """float | tensor (num_envs, dim) | tensor (dim,) | None -> list of ``dim`` floats."""
    if value is None:
        return [default] * dim
    if hasattr(value, "detach"):
        t = value.detach().cpu()
        if t.ndim == 2:
            t = t[0]
        if t.ndim == 0:
            return [float(t)] * dim
        return [float(v) for v in t.tolist()]
    return [float(value)] * dim


def _joint_ids(ids: Any, num_joints: int) -> list[int]:
    if isinstance(ids, slice):
        return list(range(num_joints))[ids]
    return [int(i) for i in (ids.tolist() if hasattr(ids, "tolist") else ids)]


def _obs_params(term_cfg: Any, num_joints: int) -> dict:
    """Keep what a controller needs: joint ids for joint terms, command name, sensor offset."""
    params: dict[str, Any] = {}
    for key, value in term_cfg.params.items():
        if hasattr(value, "joint_ids") and hasattr(value, "name"):  # SceneEntityCfg
            if value.joint_names is not None:
                params[key] = {"joint_ids": _joint_ids(value.joint_ids, num_joints)}
        elif isinstance(value, (int, float, str, bool)):
            params[key] = value
    return params


def collect_from_env(env) -> dict:
    """Read the live environment (after ``gym.make``) and build the deploy dict."""
    env = getattr(env, "unwrapped", env)
    robot = env.scene["robot"]
    wl = env.cfg.wl
    sim_names = list(robot.data.joint_names)
    n = len(sim_names)

    actions = []
    for name, term in zip(env.action_manager.active_terms, env.action_manager._terms.values(), strict=True):
        dim = term.action_dim
        clip = None
        if getattr(term.cfg, "clip", None) is not None:
            c = term._clip[0].detach().cpu().tolist()
            clip = [list(pair) for pair in c]
        offset = _per_dim(getattr(term, "_offset", 0.0), dim, 0.0)
        actions.append(
            {
                "name": name,
                "joint_ids": _joint_ids(term._joint_ids, n),
                "scale": _per_dim(term._scale, dim, 1.0),
                "offset": offset,
                "clip": clip,
            }
        )

    om = env.observation_manager
    observations = []
    for name, term_cfg, dims in zip(
        om.active_terms["policy"], om._group_obs_term_cfgs["policy"], om.group_obs_term_dim["policy"], strict=True
    ):
        history = max(1, int(term_cfg.history_length or 1))
        dim = int(dims[-1]) // history  # the manager reports the flattened (history * dim) size
        observations.append(
            {
                "name": name,
                "dim": dim,
                "scale": _per_dim(term_cfg.scale, dim, 1.0),
                "clip": list(term_cfg.clip) if term_cfg.clip is not None else None,
                "history_length": history,
                "params": _obs_params(term_cfg, n),
            }
        )

    cmd_cfg = env.cfg.commands.base_velocity
    ranges_cfg = getattr(cmd_cfg, "limit_ranges", None) or cmd_cfg.ranges
    command_ranges = {k: list(getattr(ranges_cfg, k)) for k in ("lin_vel_x", "lin_vel_y", "ang_vel_z")}

    perception = None
    scanner_cfg = getattr(env.cfg.scene, "height_scanner", None)
    if scanner_cfg is not None:
        pat = scanner_cfg.pattern_cfg
        perception = {
            "height_scan": {
                "frame": scanner_cfg.ray_alignment,
                "size": list(pat.size),
                "resolution": float(pat.resolution),
                "ordering": pat.ordering,
                "value": "base_z - hit_z - offset, clipped",
            }
        }

    return build_deploy_cfg(
        sim_joint_names=sim_names,
        sdk_joint_order=list(wl.sdk_joint_order) or sim_names,
        stiffness_sim=robot.data.default_joint_stiffness[0].tolist(),
        damping_sim=robot.data.default_joint_damping[0].tolist(),
        default_joint_pos_sim=robot.data.default_joint_pos[0].tolist(),
        step_dt=env.step_dt,
        command_ranges=command_ranges,
        actions=actions,
        observations=observations,
        robot={
            "name": wl.robot_name,
            "leg_joint_names": list(wl.leg_joint_names),
            "wheel_joint_names": list(wl.wheel_joint_names),
            "wheel_radius": float(wl.wheel_radius),
        },
        perception=perception,
    )


def export_deploy_cfg(env, log_dir: str | Path) -> Path:
    return write_deploy_cfg(collect_from_env(env), log_dir)
