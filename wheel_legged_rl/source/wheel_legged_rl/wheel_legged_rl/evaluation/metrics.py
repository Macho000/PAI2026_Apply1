"""Evaluation metrics (numpy only, no Isaac).

``evaluate.py`` records an :class:`EpisodeLog` per (scenario, env) and calls :func:`summarize`. Everything here is
deterministic and unit-tested so that metric changes are deliberate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

GRAVITY = 9.81


@dataclass
class EpisodeLog:
    """Time series of one robot in one scenario. Arrays have shape (T, ...) with T = number of policy steps."""

    dt: float
    command: np.ndarray  # (T, 3) vx, vy, wz commanded (body frame)
    lin_vel_b: np.ndarray  # (T, 3) base linear velocity (body frame)
    ang_vel_b: np.ndarray  # (T, 3) base angular velocity (body frame)
    roll_pitch: np.ndarray  # (T, 2) [rad]
    power: np.ndarray  # (T,) total mechanical power |tau * omega| [W]
    fell: bool  # terminated by a failure condition (not a time-out)
    distance: float = 0.0  # planar distance travelled [m]
    wheel_slip: np.ndarray | None = None  # (T, W) longitudinal slip [m/s]
    wheel_rolling_speed: np.ndarray | None = None  # (T, W) omega * r [m/s]
    wheel_contact: np.ndarray | None = None  # (T, W) bool
    reached_goal: bool | None = None  # terrain tasks: crossed the course
    valid: np.ndarray | None = None  # (T,) bool; False during transients (e.g. right after a command change)
    extra: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.command)


def _mask(log: EpisodeLog, skip: int) -> np.ndarray:
    """Samples used by tracking metrics: after the first ``skip`` steps and where ``valid``."""
    m = np.arange(len(log)) >= skip
    if log.valid is not None:
        m &= log.valid.astype(bool)
    return m


def transient_mask(command: np.ndarray, settle_steps: int) -> np.ndarray:
    """False for ``settle_steps`` steps after every change of the command (and at the start)."""
    valid = np.ones(len(command), dtype=bool)
    changes = [0] + [t for t in range(1, len(command)) if not np.allclose(command[t], command[t - 1])]
    for t in changes:
        valid[t : t + settle_steps] = False
    return valid


def velocity_rmse(log: EpisodeLog, skip: int = 0) -> float:
    """RMSE of the planar velocity error ||v_xy - cmd_xy|| after the first ``skip`` steps."""
    m = _mask(log, skip)
    err = log.lin_vel_b[m, :2] - log.command[m, :2]
    return float(np.sqrt(np.mean(np.sum(err**2, axis=-1)))) if len(err) else float("nan")


def vx_rmse(log: EpisodeLog, skip: int = 0) -> float:
    m = _mask(log, skip)
    err = log.lin_vel_b[m, 0] - log.command[m, 0]
    return float(np.sqrt(np.mean(err**2))) if len(err) else float("nan")


def yaw_rate_rmse(log: EpisodeLog, skip: int = 0) -> float:
    m = _mask(log, skip)
    err = log.ang_vel_b[m, 2] - log.command[m, 2]
    return float(np.sqrt(np.mean(err**2))) if len(err) else float("nan")


def _signed_bias(actual: np.ndarray, command: np.ndarray, m: np.ndarray, command_threshold: float) -> float:
    """Mean of (actual - command) * sign(command) where |command| >= threshold: < 0 means slower than commanded."""
    m = m & (np.abs(command) >= command_threshold)
    if not m.any():
        return float("nan")
    return float(np.mean((actual[m] - command[m]) * np.sign(command[m])))


def vx_bias(log: EpisodeLog, skip: int = 0, command_threshold: float = 0.1) -> float:
    """Signed vx error along the commanded direction [m/s] (NaN if vx is never commanded)."""
    return _signed_bias(log.lin_vel_b[:, 0], log.command[:, 0], _mask(log, skip), command_threshold)


def yaw_rate_bias(log: EpisodeLog, skip: int = 0, command_threshold: float = 0.1) -> float:
    """Signed yaw-rate error along the commanded direction [rad/s] (NaN if wz is never commanded)."""
    return _signed_bias(log.ang_vel_b[:, 2], log.command[:, 2], _mask(log, skip), command_threshold)


def standstill_residual_speed(log: EpisodeLog, skip: int = 0, command_threshold: float = 1e-3) -> float:
    """Mean planar speed while the command is zero (NaN if the command never is)."""
    idle = _mask(log, skip) & (np.linalg.norm(log.command, axis=-1) < command_threshold)
    if not idle.any():
        return float("nan")
    return float(np.mean(np.linalg.norm(log.lin_vel_b[idle, :2], axis=-1)))


def roll_pitch_rms(log: EpisodeLog, skip: int = 0) -> float:
    rp = log.roll_pitch[_mask(log, skip)]
    return float(np.sqrt(np.mean(np.sum(rp**2, axis=-1)))) if len(rp) else float("nan")


def reached_goal(pos: np.ndarray, fell: bool, goal_distance: float) -> bool:
    """Traversal success: no fall and the last planar position ``pos[-1]`` at least ``goal_distance`` from
    ``pos[0]`` (``pos``: (T, 2) positions while alive)."""
    if fell or len(pos) == 0:
        return False
    return bool(np.linalg.norm(pos[-1] - pos[0]) >= goal_distance)


def energy_per_meter(log: EpisodeLog) -> float:
    """Total mechanical energy / planar distance [J/m] (NaN if the robot did not move 0.5 m)."""
    if log.distance < 0.5:
        return float("nan")
    return float(np.sum(log.power) * log.dt / log.distance)


def cost_of_transport(log: EpisodeLog, mass: float) -> float:
    epm = energy_per_meter(log)
    return epm / (mass * GRAVITY) if np.isfinite(epm) else float("nan")


def wheel_slip_ratio(log: EpisodeLog, min_speed: float = 0.1) -> float:
    """Mean |slip| / max(|omega r|, min_speed) over (step, wheel) samples in contact."""
    if log.wheel_slip is None or log.wheel_contact is None:
        return float("nan")
    contact = log.wheel_contact.astype(bool)
    if not contact.any():
        return float("nan")
    ratio = np.abs(log.wheel_slip) / np.maximum(np.abs(log.wheel_rolling_speed), min_speed)
    return float(np.mean(ratio[contact]))


def _nanmean(values) -> float:
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if len(arr) else float("nan")


def summarize(logs: list[EpisodeLog], mass: float, skip_s: float = 1.0) -> dict[str, float]:
    """Aggregate metrics over episodes. The first ``skip_s`` seconds (and samples with ``valid == False``) are
    excluded from tracking / attitude metrics."""
    if not logs:
        return {}
    skip = int(round(skip_s / logs[0].dt))
    n = len(logs)
    falls = sum(log.fell for log in logs)
    goals = [log.reached_goal for log in logs if log.reached_goal is not None]
    return {
        "num_episodes": n,
        "fall_rate": falls / n,
        "success_rate": (sum(goals) / len(goals)) if goals else (n - falls) / n,
        "vel_rmse": _nanmean(velocity_rmse(log, skip) for log in logs),
        "vx_rmse": _nanmean(vx_rmse(log, skip) for log in logs),
        "yaw_rate_rmse": _nanmean(yaw_rate_rmse(log, skip) for log in logs),
        "vx_bias": _nanmean(vx_bias(log, skip) for log in logs),
        "yaw_rate_bias": _nanmean(yaw_rate_bias(log, skip) for log in logs),
        "standstill_speed": _nanmean(standstill_residual_speed(log, skip) for log in logs),
        "roll_pitch_rms": _nanmean(roll_pitch_rms(log, skip) for log in logs),
        "energy_per_meter": _nanmean(energy_per_meter(log) for log in logs),
        "cost_of_transport": _nanmean(cost_of_transport(log, mass) for log in logs),
        "wheel_slip_ratio": _nanmean(wheel_slip_ratio(log) for log in logs),
    }


def with_groups(summary: dict[str, float], groups: dict[str, dict[str, float]]) -> dict[str, float]:
    """``summary`` plus every group's metrics as ``<group>/<metric>`` (for criteria and selection weights)."""
    out = dict(summary)
    for group, metrics in groups.items():
        out.update({f"{group}/{k}": v for k, v in metrics.items()})
    return out


def check_criteria(summary: dict[str, float], criteria: dict[str, dict[str, float]]) -> dict[str, bool]:
    """``criteria = {metric: {"max": x} | {"min": y}}`` -> pass/fail per metric (NaN fails)."""
    result = {}
    for metric, bound in criteria.items():
        value = summary.get(metric, float("nan"))
        ok = np.isfinite(value)
        if "max" in bound:
            ok = ok and value <= bound["max"]
        if "min" in bound:
            ok = ok and value >= bound["min"]
        result[metric] = bool(ok)
    return result


def score(summary: dict[str, float], weights: dict[str, float]) -> float:
    """Scalar used to pick the best checkpoint: sum(weight * metric), lower is better. NaN metrics count as +inf."""
    total = 0.0
    for metric, w in weights.items():
        v = summary.get(metric, float("nan"))
        if not np.isfinite(v):
            return float("inf")
        total += w * v
    return total
