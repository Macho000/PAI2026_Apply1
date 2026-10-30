import math

import pytest
import torch

from wheel_legged_rl.rewards.rewards import (
    contact_mask,
    mechanical_power_from_tensors,
    quat_rotate,
    stand_still_from_tensors,
    wheel_slip_from_tensors,
)

Y = torch.tensor([0.0, 1.0, 0.0])


def yaw_quat(yaw: float) -> torch.Tensor:
    return torch.tensor([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


def test_quat_rotate():
    v = quat_rotate(yaw_quat(math.pi / 2), torch.tensor([1.0, 0.0, 0.0]))
    assert v.tolist() == pytest.approx([0.0, 1.0, 0.0], abs=1e-6)


@pytest.mark.parametrize("yaw", [0.0, 0.7, -2.0, math.pi])
def test_pure_rolling_has_zero_slip(yaw):
    """Positive spin about +y rolls toward the wheel's +x (plan: omega y x (-r z) = -omega r x at the contact)."""
    r, omega = 0.086, 5.0
    q = yaw_quat(yaw)
    forward_w = quat_rotate(q, torch.tensor([1.0, 0.0, 0.0]))
    v = (omega * r * forward_w).reshape(1, 1, 3)
    w = (omega * quat_rotate(q, Y)).reshape(1, 1, 3)
    slip = wheel_slip_from_tensors(v, w, q.reshape(1, 1, 4), Y, r)
    assert float(slip) == pytest.approx(0.0, abs=1e-6)


def test_locked_wheel_slip_equals_speed_and_lateral_ignored():
    q = yaw_quat(0.0).reshape(1, 1, 4)
    locked = wheel_slip_from_tensors(torch.tensor([[[0.5, 0.0, 0.0]]]), torch.zeros(1, 1, 3), q, Y, 0.1)
    assert float(locked) == pytest.approx(0.5)
    spinning = wheel_slip_from_tensors(torch.zeros(1, 1, 3), torch.tensor([[[0.0, 10.0, 0.0]]]), q, Y, 0.1)
    assert float(spinning) == pytest.approx(-1.0)
    lateral = wheel_slip_from_tensors(torch.tensor([[[0.0, 0.7, 0.0]]]), torch.zeros(1, 1, 3), q, Y, 0.1)
    assert float(lateral) == pytest.approx(0.0, abs=1e-7)


def test_mechanical_power():
    tau = torch.tensor([[2.0, -3.0]])
    vel = torch.tensor([[1.0, 2.0]])
    assert float(mechanical_power_from_tensors(tau, vel)) == pytest.approx(8.0)


def test_stand_still_only_when_command_zero():
    legs = torch.tensor([[0.1, -0.2], [0.1, -0.2]])
    default = torch.zeros(2, 2)
    wheels = torch.tensor([[10.0], [10.0]])
    cmd = torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    out = stand_still_from_tensors(cmd, legs, default, wheels, 0.1, 0.05)
    assert out.tolist() == pytest.approx([0.3 + 0.5, 0.0])


def test_contact_mask_uses_history_max():
    hist = torch.zeros(1, 3, 2, 3)
    hist[0, 1, 0, 2] = 5.0
    assert contact_mask(hist, 1.0).tolist() == [[True, False]]
