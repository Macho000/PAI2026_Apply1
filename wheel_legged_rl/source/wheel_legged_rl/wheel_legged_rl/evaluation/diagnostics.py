"""Per-step training diagnostics (torch only; the env glue is in ``envs/wheel_legged_env.py``).

The tensor functions avoid data-dependent Python branches so they never force a GPU synchronisation.

Logged as ``Diagnostics/<name>`` through ``extras["log"]`` so rsl_rl forwards them to TensorBoard / W&B.
"""

from __future__ import annotations

from collections import deque

import torch

GRAVITY = 9.81


class RollingRatio:
    """sum(numerators) / sum(denominators) over the last ``window`` updates."""

    def __init__(self, window: int):
        self._num: deque[float] = deque(maxlen=window)
        self._den: deque[float] = deque(maxlen=window)

    def update(self, numerator: float, denominator: float) -> None:
        self._num.append(float(numerator))
        self._den.append(float(denominator))

    @property
    def value(self) -> float:
        den = sum(self._den)
        return sum(self._num) / den if den > 0 else 0.0


def slip_ratio(slip: torch.Tensor, rolling_speed: torch.Tensor, in_contact: torch.Tensor, min_speed: float = 0.1):
    """Mean |slip| / max(|omega r|, min_speed) over wheels in contact (0 if none)."""
    ratio = slip.abs() / rolling_speed.abs().clamp(min=min_speed)
    return (ratio * in_contact).sum() / in_contact.sum().clamp(min=1)


def cost_of_transport(power: torch.Tensor, mass: float, speed: torch.Tensor, min_speed: float = 0.2):
    """Mean P / (m g v) over envs moving faster than ``min_speed`` (0 if none), and energy per meter P / v."""
    moving = (speed > min_speed).float()
    per_meter = power / speed.clamp(min=min_speed)
    epm = (per_meter * moving).sum() / moving.sum().clamp(min=1)
    return epm / (mass * GRAVITY), epm
