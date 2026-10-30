"""Run an exported policy from ``deploy.yaml`` without Isaac Lab (numpy; ONNX Runtime optional).

The same code path is meant for Sim2Sim (e.g. MuJoCo) and the robot's ECU::

    iface = PolicyInterface.from_run_dir("logs/rsl_rl/go2w_wheel_leg/<run>")
    iface.reset()
    while running:
        cmd = iface.step(RobotState(...))        # sensors in SDK (motor) order
        send(cmd.q_target, cmd.dq_target, cmd.kp, cmd.kd)   # SDK order

Observation terms are rebuilt from the names, parameters, scales and clip ranges in ``deploy.yaml``, mirroring
the C++ registry in ``unitree_rl_lab/deploy`` (plus ``terrain_height_scan``).
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .export_deploy_cfg import load_deploy_cfg, validate_deploy_cfg


@dataclass
class RobotState:
    """One control-step snapshot. Joint arrays are in **SDK order** (as the motor driver reports them)."""

    base_ang_vel: np.ndarray  # (3,) gyro, body frame [rad/s]
    projected_gravity: np.ndarray  # (3,) gravity direction in body frame (unit vector)
    command: np.ndarray  # (3,) vx, vy, wz
    joint_pos: np.ndarray  # (N,) [rad]
    joint_vel: np.ndarray  # (N,) [rad/s]
    height_map: np.ndarray | None = None  # (num_rays,) base_z - terrain_z - offset, perception frame of deploy.yaml


@dataclass
class JointCommand:
    """Targets in **SDK order**. Position-controlled joints have ``dq_target = 0``; wheels have ``kp = 0``."""

    q_target: np.ndarray
    dq_target: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    action: np.ndarray  # raw policy output (sim order of the action terms)


class OnnxPolicy:
    def __init__(self, path: str | Path):
        import onnxruntime as ort  # optional dependency

        self._session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self._input = self._session.get_inputs()[0].name

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        out = self._session.run(None, {self._input: obs.astype(np.float32)[None]})[0]
        return np.asarray(out[0], dtype=np.float64)


class PolicyInterface:
    def __init__(self, deploy_cfg: dict, policy: Callable[[np.ndarray], np.ndarray]):
        validate_deploy_cfg(deploy_cfg)
        self.cfg = deploy_cfg
        self.policy = policy
        self.joint_ids_map = np.asarray(deploy_cfg["joint_ids_map"], dtype=int)  # sim i -> sdk index
        self.num_joints = len(self.joint_ids_map)
        self.default_joint_pos = np.asarray(deploy_cfg["default_joint_pos"], dtype=np.float64)  # sim order
        self.kp_sdk = np.asarray(deploy_cfg["stiffness"], dtype=np.float64)
        self.kd_sdk = np.asarray(deploy_cfg["damping"], dtype=np.float64)
        self.action_terms = list(deploy_cfg["actions"].items())
        self.action_dim = sum(len(t["joint_ids"]) for _, t in self.action_terms)
        self.obs_terms = list(deploy_cfg["observations"].items())
        self.ranges = deploy_cfg["commands"]["base_velocity"]["ranges"]
        self._obs_fns = {
            "base_ang_vel": lambda s, p: s.base_ang_vel,
            "projected_gravity": lambda s, p: s.projected_gravity,
            "velocity_commands": self._velocity_commands,
            "joint_pos": lambda s, p: self._select(self._sim(s.joint_pos), p),
            "joint_pos_rel": lambda s, p: self._select(self._sim(s.joint_pos) - self.default_joint_pos, p),
            "joint_vel_rel": lambda s, p: self._select(self._sim(s.joint_vel), p),
            "last_action": lambda s, p: self.last_action,
            "terrain_height_scan": self._height_scan,
        }
        unknown = [name for name, _ in self.obs_terms if name not in self._obs_fns]
        if unknown:
            raise ValueError(f"observation terms without a Python implementation: {unknown}")
        self.reset()

    # ------------------------------------------------------------------ construction

    @classmethod
    def from_run_dir(cls, run_dir: str | Path, policy_file: str = "exported/policy.onnx") -> PolicyInterface:
        run_dir = Path(run_dir)
        return cls(load_deploy_cfg(run_dir / "params" / "deploy.yaml"), OnnxPolicy(run_dir / policy_file))

    # ------------------------------------------------------------------ state

    def reset(self) -> None:
        self.last_action = np.zeros(self.action_dim)
        self._history: dict[str, deque] = {
            name: deque(maxlen=int(t.get("history_length") or 1)) for name, t in self.obs_terms
        }

    @property
    def obs_dim(self) -> int:
        return int(self.cfg["policy"]["obs_dim"])

    # ------------------------------------------------------------------ observation terms

    def _sim(self, sdk_values: np.ndarray) -> np.ndarray:
        """SDK order -> sim order."""
        return np.asarray(sdk_values, dtype=np.float64)[self.joint_ids_map]

    @staticmethod
    def _select(sim_values: np.ndarray, params: dict) -> np.ndarray:
        ids = (params or {}).get("asset_cfg", {}).get("joint_ids")
        return sim_values if ids is None else sim_values[np.asarray(ids, dtype=int)]

    def _velocity_commands(self, s: RobotState, params: dict) -> np.ndarray:
        lo = np.array([self.ranges[k][0] for k in ("lin_vel_x", "lin_vel_y", "ang_vel_z")])
        hi = np.array([self.ranges[k][1] for k in ("lin_vel_x", "lin_vel_y", "ang_vel_z")])
        return np.clip(np.asarray(s.command, dtype=np.float64), lo, hi)

    @staticmethod
    def _height_scan(s: RobotState, params: dict) -> np.ndarray:
        if s.height_map is None:
            raise ValueError("policy uses terrain_height_scan: RobotState.height_map is required")
        return np.asarray(s.height_map, dtype=np.float64)

    def build_observation(self, state: RobotState) -> np.ndarray:
        parts = []
        for name, term in self.obs_terms:
            value = np.asarray(self._obs_fns[name](state, term.get("params") or {}), dtype=np.float64)
            if term.get("clip") is not None:
                value = np.clip(value, term["clip"][0], term["clip"][1])  # Isaac Lab order: clip, then scale
            value = value * np.asarray(term["scale"], dtype=np.float64)
            hist = self._history[name]
            if not hist:  # first step after reset: fill the history with the current value
                hist.extend([value] * hist.maxlen)
            else:
                hist.append(value)
            parts.extend(hist)  # oldest -> newest
        obs = np.concatenate(parts)
        if obs.shape[0] != self.obs_dim:
            raise ValueError(f"observation has {obs.shape[0]} entries, deploy.yaml says {self.obs_dim}")
        return obs

    # ------------------------------------------------------------------ actions

    def process_action(self, action: np.ndarray) -> JointCommand:
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (self.action_dim,):
            raise ValueError(f"policy returned shape {action.shape}, expected ({self.action_dim},)")
        q_sim = self.default_joint_pos.copy()
        dq_sim = np.zeros(self.num_joints)
        idx = 0
        for name, term in self.action_terms:
            ids = np.asarray(term["joint_ids"], dtype=int)
            a = action[idx : idx + len(ids)]
            idx += len(ids)
            target = a * np.asarray(term["scale"]) + np.asarray(term["offset"])
            if term.get("clip") is not None:
                clip = np.asarray(term["clip"], dtype=np.float64)
                target = np.clip(target, clip[:, 0], clip[:, 1])
            if name == "JointPositionAction":
                q_sim[ids] = target
            elif name == "JointVelocityAction":
                dq_sim[ids] = target
            else:
                raise ValueError(f"unsupported action term {name}")
        q_sdk = np.zeros(self.num_joints)
        dq_sdk = np.zeros(self.num_joints)
        q_sdk[self.joint_ids_map] = q_sim
        dq_sdk[self.joint_ids_map] = dq_sim
        return JointCommand(
            q_target=q_sdk, dq_target=dq_sdk, kp=self.kp_sdk.copy(), kd=self.kd_sdk.copy(), action=action
        )

    def step(self, state: RobotState) -> JointCommand:
        obs = self.build_observation(state)
        action = np.asarray(self.policy(obs), dtype=np.float64)
        self.last_action = action
        return self.process_action(action)
