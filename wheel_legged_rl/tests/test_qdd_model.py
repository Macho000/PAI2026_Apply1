import pytest
import torch

from wheel_legged_rl.actuators.qdd_model import GO2HV_CURVE, QDDParams, TNCurve, joint_friction, tn_torque_limit


def test_curve_scalar_limit():
    c = GO2HV_CURVE
    assert c.limit(0.0) == 20.2
    assert c.limit(0.0, same_direction=False) == 23.4
    assert c.limit(13.0) == 20.2
    assert c.limit((13.5 + 30.0) / 2) == pytest.approx(10.1)
    assert c.limit(30.0) == 0.0 and c.limit(40.0) == 0.0


def test_geared_curve_is_go2w_calf_estimate():
    g = GO2HV_CURVE.geared(1.5)
    assert (g.y1, g.y2, g.x1, g.x2) == pytest.approx((30.3, 35.1, 9.0, 20.0))


def test_invalid_curve():
    with pytest.raises(ValueError):
        TNCurve(10, 10, 20, 10)


def test_qdd_params_to_curve_and_armature():
    p = QDDParams(
        kt=0.1,
        gear_ratio=10,
        i_max=20,
        motor_no_load_speed=300,
        motor_knee_speed=150,
        rotor_inertia=1e-5,
        gear_efficiency=0.9,
    )
    c = p.tn_curve()
    assert c.y1 == pytest.approx(18.0) and c.y2 == pytest.approx(20 / 0.9)
    assert c.x1 == pytest.approx(15.0) and c.x2 == pytest.approx(30.0)
    assert p.armature == pytest.approx(1e-3)
    q = p.with_extra_reduction(2.0)
    assert q.tn_curve().y1 == pytest.approx(36.0) and q.tn_curve().x2 == pytest.approx(15.0)
    assert q.armature == pytest.approx(4e-3)


def test_batched_limit_matches_scalar():
    c = GO2HV_CURVE
    vel = torch.tensor([[0.0, 5.0, -20.0, 25.0, 35.0, -5.0]])
    effort = torch.tensor([[100.0, 100.0, -100.0, -100.0, 100.0, 100.0]])
    t = lambda v: torch.full_like(vel, v)  # noqa: E731
    out = tn_torque_limit(effort, vel, t(c.y1), t(c.y2), t(c.x1), t(c.x2))
    for i in range(vel.shape[1]):
        v, e = float(vel[0, i]), float(effort[0, i])
        same = v * e > 0
        assert abs(float(out[0, i])) == pytest.approx(c.limit(v, same), abs=1e-5)
        assert float(out[0, i]) * e >= 0  # never flips sign


def test_small_torque_not_clipped():
    ones = torch.ones(1, 1)
    out = tn_torque_limit(torch.tensor([[3.0]]), torch.tensor([[10.0]]), 20 * ones, 23 * ones, 13 * ones, 30 * ones)
    assert float(out) == 3.0


def test_friction_opposes_motion():
    v = torch.tensor([-2.0, 0.0, 2.0])
    f = joint_friction(v, torch.tensor(0.5), torch.tensor(0.1), torch.tensor(0.01))
    assert f[0] < 0 and f[1] == 0 and f[2] > 0
    assert float(f[2]) == pytest.approx(0.5 * torch.tanh(torch.tensor(200.0)).item() + 0.2)
