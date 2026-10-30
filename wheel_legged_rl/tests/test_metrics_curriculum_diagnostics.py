import math

import numpy as np
import pytest
import torch

from wheel_legged_rl.curriculum.levels import expand_range, should_expand, stand_gate_passed, terrain_level_moves
from wheel_legged_rl.evaluation import metrics as M
from wheel_legged_rl.evaluation.diagnostics import RollingRatio, cost_of_transport, slip_ratio


def make_log(T=100, dt=0.02, vx_cmd=1.0, vx=0.9, fell=False, power=50.0):
    cmd = np.zeros((T, 3))
    cmd[:, 0] = vx_cmd
    vel = np.zeros((T, 3))
    vel[:, 0] = vx
    return M.EpisodeLog(
        dt=dt,
        command=cmd,
        lin_vel_b=vel,
        ang_vel_b=np.zeros((T, 3)),
        roll_pitch=np.full((T, 2), 0.03),
        power=np.full(T, power),
        fell=fell,
        distance=abs(vx) * T * dt,
        wheel_slip=np.full((T, 4), 0.05),
        wheel_rolling_speed=np.full((T, 4), 0.5),
        wheel_contact=np.ones((T, 4), bool),
    )


def test_tracking_metrics():
    log = make_log()
    assert M.velocity_rmse(log) == pytest.approx(0.1)
    assert M.vx_rmse(log) == pytest.approx(0.1)
    assert M.yaw_rate_rmse(log) == 0.0
    assert M.roll_pitch_rms(log) == pytest.approx(math.sqrt(2) * 0.03)
    assert math.isnan(M.standstill_residual_speed(log))
    assert M.wheel_slip_ratio(log) == pytest.approx(0.1)


def test_signed_bias():
    assert M.vx_bias(make_log(vx_cmd=1.0, vx=0.9)) == pytest.approx(-0.1)  # too slow forwards
    assert M.vx_bias(make_log(vx_cmd=-1.0, vx=-0.9)) == pytest.approx(-0.1)  # too slow backwards
    assert M.vx_bias(make_log(vx_cmd=-1.0, vx=-1.2)) == pytest.approx(0.2)  # overshoot
    assert math.isnan(M.vx_bias(make_log(vx_cmd=0.0, vx=0.05)))  # standing: no direction
    log = make_log(vx_cmd=0.0, vx=0.0)
    log.command[:, 2] = -1.5
    log.ang_vel_b[:, 2] = -1.4
    assert M.yaw_rate_bias(log) == pytest.approx(-0.1)
    assert math.isnan(M.yaw_rate_bias(make_log()))


def test_energy_and_cot():
    log = make_log(T=100, dt=0.02, vx=1.0, power=50.0)  # 2 m in 2 s at 50 W -> 50 J/m
    assert M.energy_per_meter(log) == pytest.approx(50.0)
    assert M.cost_of_transport(log, mass=19.5) == pytest.approx(50.0 / (19.5 * 9.81))
    assert math.isnan(M.energy_per_meter(make_log(vx=0.0)))


def test_standstill_and_skip():
    log = make_log(vx_cmd=0.0, vx=0.02)
    log.lin_vel_b[:10, 0] = 1.0  # transient at the start
    assert M.standstill_residual_speed(log, skip=10) == pytest.approx(0.02)


def test_summarize_and_criteria():
    logs = [make_log(), make_log(fell=True), make_log(vx_cmd=0.0, vx=0.01)]
    s = M.summarize(logs, mass=19.5, skip_s=0.0)
    assert s["num_episodes"] == 3
    assert s["fall_rate"] == pytest.approx(1 / 3)
    assert s["success_rate"] == pytest.approx(2 / 3)
    assert s["standstill_speed"] == pytest.approx(0.01)
    res = M.check_criteria(s, {"fall_rate": {"max": 0.01}, "vx_rmse": {"max": 0.15}, "nope": {"min": 0}})
    assert res == {"fall_rate": False, "vx_rmse": True, "nope": False}
    assert M.score(s, {"vx_rmse": 1.0, "fall_rate": 10.0}) == pytest.approx(s["vx_rmse"] + 10 / 3)
    assert M.score(s, {"nope": 1.0}) == math.inf


def test_curriculum_levels():
    assert expand_range((-0.5, 0.5), (-1.0, 1.0), 0.1) == pytest.approx((-0.6, 0.6))
    assert expand_range((-0.95, 0.95), (-1.0, 1.0), 0.1) == (-1.0, 1.0)
    assert expand_range((0.0, 0.0), (0.0, 0.0), 0.1) == (0.0, 0.0)
    assert should_expand(1.3, 1.5, 0.8) and not should_expand(1.1, 1.5, 0.8)
    assert not should_expand(1.0, 0.0, 0.8)


def test_stand_gate():
    limits = {"stand_still": 0.5, "base_height": 0.001}
    good = {"stand_still": 0.2, "base_height": 0.0004}
    assert stand_gate_passed(0.97, good, 0.95, limits)
    assert not stand_gate_passed(0.90, good, 0.95, limits)  # too many falls
    assert not stand_gate_passed(0.97, {**good, "base_height": 0.004}, 0.95, limits)  # body too low
    assert not stand_gate_passed(0.97, {**good, "stand_still": 0.8}, 0.95, limits)
    assert stand_gate_passed(1.0, {}, 0.95, {})
    with pytest.raises(KeyError):
        stand_gate_passed(1.0, {"stand_still": 0.1}, 0.95, limits)


def test_rolling_ratio():
    r = RollingRatio(window=2)
    assert r.value == 0.0
    r.update(1, 10)
    r.update(0, 0)
    assert r.value == pytest.approx(0.1)
    r.update(3, 10)  # first sample drops out
    assert r.value == pytest.approx(0.3)


def test_torch_diagnostics():
    slip = torch.tensor([[0.1, 0.5]])
    speed = torch.tensor([[1.0, 0.0]])
    contact = torch.tensor([[True, False]])
    assert float(slip_ratio(slip, speed, contact)) == pytest.approx(0.1)
    assert float(slip_ratio(slip, speed, torch.zeros_like(contact))) == 0.0
    cot, epm = cost_of_transport(torch.tensor([100.0, 50.0]), 10.0, torch.tensor([2.0, 0.0]))
    assert float(epm) == pytest.approx(50.0) and float(cot) == pytest.approx(50.0 / 98.1)


def test_transient_mask_and_valid():
    cmd = np.zeros((10, 3))
    cmd[:5, 0] = 1.0
    valid = M.transient_mask(cmd, settle_steps=2)
    assert valid.tolist() == [False, False, True, True, True, False, False, True, True, True]
    log = make_log(T=10, vx_cmd=0.0, vx=0.0)
    log.command = cmd
    log.lin_vel_b[:, 0] = np.r_[np.full(5, 1.0), [0.8, 0.3, 0.0, 0.0, 0.0]]  # decelerates after the stop
    log.valid = valid
    assert M.standstill_residual_speed(log) == 0.0  # the braking transient is excluded
    assert M.vx_rmse(log) == 0.0


def test_empty_episode_counts_as_fall():
    empty = make_log(T=0)
    empty.fell = True
    s = M.summarize([make_log(), empty], mass=19.5, skip_s=0.0)
    assert s["fall_rate"] == 0.5 and s["vx_rmse"] == pytest.approx(0.1)


def test_reached_goal_and_groups():
    pos = np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 2.0]])  # |end - start| = 3.61
    assert M.reached_goal(pos, fell=False, goal_distance=3.5)
    assert not M.reached_goal(pos, fell=True, goal_distance=3.5)
    assert not M.reached_goal(pos, fell=False, goal_distance=4.0)
    assert not M.reached_goal(pos[:0], fell=False, goal_distance=0.0)
    ok, stuck, fallen = make_log(), make_log(), make_log(fell=True)
    ok.reached_goal, stuck.reached_goal, fallen.reached_goal = True, False, False
    s = M.summarize([ok, stuck, fallen, make_log()], mass=19.5, skip_s=0.0)
    assert s["success_rate"] == pytest.approx(1 / 3)  # only episodes with a goal count
    judged = M.with_groups({"fall_rate": 0.5}, {"flat": {"fall_rate": 0.0}, "steps": {"success_rate": 0.9}})
    assert judged == {"fall_rate": 0.5, "flat/fall_rate": 0.0, "steps/success_rate": 0.9}
    res = M.check_criteria(judged, {"flat/fall_rate": {"max": 0.01}, "steps/success_rate": {"min": 0.8}})
    assert res == {"flat/fall_rate": True, "steps/success_rate": True}
    assert M.score(judged, {"flat/fall_rate": 10.0, "steps/success_rate": -1.0}) == pytest.approx(-0.9)


def test_terrain_level_moves():
    fell = torch.tensor([False, False, True, False, False, True])
    distance = torch.tensor([5.0, 5.0, 5.0, 1.0, 1.0, 1.0])
    tracking = torch.tensor([0.9, 0.2, 0.9, 0.9, 0.2, 0.9])
    up, down = terrain_level_moves(fell, distance, tracking, promote_distance=4.0, demote_tracking=0.5)
    # left the tile: up (even with poor tracking); fell: down; arc near the origin but tracked well: stays
    assert up.tolist() == [True, True, False, False, False, False]
    assert down.tolist() == [False, False, True, False, True, True]
    up, down = terrain_level_moves(fell.numpy(), distance.numpy(), tracking.numpy(), 4.0, 0.5)  # numpy works too
    assert up.tolist() == [True, True, False, False, False, False]


def test_column_terrain_names():
    from wheel_legged_rl.terrains.layout import column_terrain_names

    phase4 = {"flat": 0.2, "random_grid": 0.2, "curbs_up": 0.3, "curbs_down": 0.3}
    cols = column_terrain_names(phase4, 20)
    assert [cols.count(k) for k in phase4] == [4, 4, 6, 6] and cols == sorted(cols, key=list(phase4).index)
    assert column_terrain_names(phase4, 5) == ["flat", "random_grid", "curbs_up", "curbs_up", "curbs_down"]
    assert column_terrain_names({"flat": 3, "a": 1, "b": 1}, 5) == ["flat"] * 3 + ["a", "b"]
    # Isaac Lab v2.3.0 TerrainGenerator._generate_curriculum_terrains, verbatim
    props = np.array(list(phase4.values()))
    props /= props.sum()
    isaac = [int(np.min(np.where(c / 20 + 0.001 < np.cumsum(props))[0])) for c in range(20)]
    assert cols == [list(phase4)[i] for i in isaac]
    with pytest.raises(ValueError):
        column_terrain_names({}, 4)
