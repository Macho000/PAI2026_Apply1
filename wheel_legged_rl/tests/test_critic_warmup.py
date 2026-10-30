import math

import pytest
import torch
from torch import nn

from wheel_legged_rl.agents.critic_warmup import CriticWarmup, critic_metrics, expand_state_dict, explained_variance


def _critic(num_obs: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(num_obs, 8), nn.ELU(), nn.Linear(8, 1))


def test_expanded_critic_keeps_the_value_function():
    torch.manual_seed(0)
    old, new = _critic(5), _critic(8)
    state, expanded = expand_state_dict(old.state_dict(), new.state_dict(), ["0.weight"])
    new.load_state_dict(state)
    assert expanded == ["0.weight"]
    obs, privileged = torch.randn(32, 5), torch.randn(32, 3)
    # V_new(obs, priv) == V_old(obs) (up to float summation order: the matmul is wider)
    torch.testing.assert_close(new(torch.cat([obs, privileged], dim=-1)), old(obs), rtol=0, atol=1e-6)
    assert torch.equal(new[0].weight[:, 5:], torch.zeros(8, 3))
    assert torch.equal(new[2].weight, old[2].weight) and torch.equal(new[0].bias, old[0].bias)


def test_equal_shapes_load_as_is():
    old, new = _critic(5), _critic(5)
    state, expanded = expand_state_dict(old.state_dict(), new.state_dict(), ["0.weight"])
    assert expanded == [] and all(torch.equal(state[k], v) for k, v in old.state_dict().items())


@pytest.mark.parametrize(
    "target, expandable, error",
    [
        (_critic(8), [], "cannot load '0.weight'"),  # growth only where allowed
        (_critic(3), ["0.weight"], "cannot load '0.weight'"),  # no shrinking
        (nn.Sequential(nn.Linear(5, 9), nn.ELU(), nn.Linear(9, 1)), ["0.weight"], "cannot load"),  # hidden size
        (nn.Sequential(nn.Linear(5, 8)), ["0.weight"], "keys differ"),
    ],
)
def test_expand_errors(target, expandable, error):
    with pytest.raises(ValueError, match=error):
        expand_state_dict(_critic(5).state_dict(), target.state_dict(), expandable)


def test_critic_metrics():
    returns = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert explained_variance(returns, returns) == pytest.approx(1.0)
    m = critic_metrics(returns - 1.0, returns)  # constant offset: EV blind, bias not
    assert m["ev"] == pytest.approx(1.0)
    assert m["value_bias"] == pytest.approx(1.0) and m["value_mse"] == pytest.approx(1.0)
    assert m["rel_bias"] == pytest.approx(1.0 / float(returns.std()))
    assert critic_metrics(torch.zeros(4), returns)["ev"] == pytest.approx(0.0)
    assert math.isnan(explained_variance(torch.zeros(3), torch.ones(3)))


def _run(warmup: CriticWarmup, metrics: list[tuple[float, float, float]]) -> list[str | None]:
    """Feed (ev_pre, ev_gap, rel_bias) per iteration, with an update after every non-final decision."""
    reasons = []
    for ev, gap, bias in metrics:
        reasons.append(warmup.step(ev, gap, bias))
        warmup.record_update()
    return reasons


def test_warmup_converges_on_a_plateau():
    w = CriticWarmup(min_iterations=2, max_iterations=20, patience=2, ev_plateau_delta=0.01, ev_min=0.5)
    reasons = _run(w, [(0.2, math.nan, 0.5), (0.6, 0.0, 0.2), (0.8, 0.0, 0.05), (0.805, 0.0, 0.05), (0.81, 0.0, 0.05)])
    assert reasons == [None, None, None, None, "converged"]
    assert w.updates == 4 and not w.active and w.reason == "converged"


def test_warmup_needs_small_bias_and_min_ev():
    w = CriticWarmup(min_iterations=0, max_iterations=5, patience=1)
    assert _run(w, [(0.9, 0.0, 0.5), (0.9, 0.0, 0.5), (0.9, 0.0, 0.5)]) == [None, None, None]  # biased
    w = CriticWarmup(min_iterations=0, max_iterations=3, patience=1, ev_min=0.5)
    assert _run(w, [(0.3, 0.0, 0.0), (0.3, 0.0, 0.0), (0.3, 0.0, 0.0), (0.3, 0.0, 0.0)])[-1] == "max_iterations"
    assert w.undertrained(0.3)


def test_warmup_stops_when_the_critic_memorises():
    w = CriticWarmup(min_iterations=2, max_ev_gap=0.05)
    assert _run(w, [(0.5, math.nan, 0.0), (0.7, 0.2, 0.0), (0.8, 0.2, 0.0)]) == [None, None, "ev_gap"]  # not before min
    assert w.updates == 2


def test_warmup_fixed_length_and_validation():
    w = CriticWarmup(min_iterations=3, max_iterations=3)
    assert _run(w, [(0.9, 0.5, 0.0)] * 4) == [None, None, None, "max_iterations"]
    assert w.step(0.9, 0.0, 0.0) is None and w.updates == 3  # inactive afterwards
    with pytest.raises(ValueError):
        CriticWarmup(min_iterations=5, max_iterations=2)
