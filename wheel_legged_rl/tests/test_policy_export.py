"""Checkpoint -> TorchScript / ONNX, compared against an rsl_rl-shaped actor (no rsl_rl import needed)."""

import numpy as np
import pytest
import torch
from torch import nn

from wheel_legged_rl.deployment.policy_export import (
    NORMALIZER_EPS,
    export_onnx,
    export_torchscript,
    load_policy,
)

OBS, ACT = 240, 16


def rsl_rl_like_checkpoint(path, normalize: bool):
    torch.manual_seed(0)
    actor = nn.Sequential(
        nn.Linear(OBS, 512), nn.ELU(), nn.Linear(512, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU(), nn.Linear(128, ACT)
    )
    state = {f"actor.{k}": v for k, v in actor.state_dict().items()}
    state["std"] = torch.ones(ACT)
    state["critic.0.weight"] = torch.zeros(1, OBS)  # ignored
    mean, std = torch.randn(1, OBS), torch.rand(1, OBS) + 0.5
    if normalize:
        state.update(
            {
                "actor_obs_normalizer._mean": mean,
                "actor_obs_normalizer._var": std**2,
                "actor_obs_normalizer._std": std,
                "actor_obs_normalizer.count": torch.tensor(10),
            }
        )
    torch.save({"model_state_dict": state, "optimizer_state_dict": {}, "iter": 100, "infos": None}, path)

    def reference(obs):
        x = (obs - mean) / (std + NORMALIZER_EPS) if normalize else obs
        return actor(x)

    return reference


@pytest.mark.parametrize("normalize", [False, True])
def test_load_and_export(tmp_path, normalize):
    ref = rsl_rl_like_checkpoint(tmp_path / "model_100.pt", normalize)
    policy = load_policy(tmp_path / "model_100.pt")
    assert (policy.obs_dim, policy.action_dim) == (OBS, ACT)
    obs = torch.randn(8, OBS)
    with torch.no_grad():
        torch.testing.assert_close(policy(obs), ref(obs))

    jit = torch.jit.load(str(export_torchscript(policy, tmp_path / "exported" / "policy.pt")))
    with torch.no_grad():
        torch.testing.assert_close(jit(obs), ref(obs))

    ort = pytest.importorskip("onnxruntime")
    onnx_path = export_onnx(policy, tmp_path / "exported" / "policy.onnx")
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    (inp,) = sess.get_inputs()
    assert inp.name == "obs" and inp.shape == [1, OBS]  # fixed shape: unitree OrtRunner multiplies the dims
    assert sess.get_outputs()[0].name == "actions"
    out = sess.run(None, {"obs": obs[:1].numpy()})[0]
    np.testing.assert_allclose(out, ref(obs[:1]).detach().numpy(), rtol=1e-4, atol=1e-5)


def test_onnx_policy_in_interface(tmp_path):
    pytest.importorskip("onnxruntime")
    from test_deploy import _state, spec_deploy_cfg

    from wheel_legged_rl.deployment.policy_interface import OnnxPolicy, PolicyInterface

    rsl_rl_like_checkpoint(tmp_path / "m.pt", normalize=True)
    policy = load_policy(tmp_path / "m.pt")
    export_onnx(policy, tmp_path / "policy.onnx")
    iface = PolicyInterface(spec_deploy_cfg(), OnnxPolicy(tmp_path / "policy.onnx"))
    state = _state(np.random.default_rng(0))
    cmd = iface.step(state)
    iface2 = PolicyInterface(
        spec_deploy_cfg(), lambda o: policy(torch.tensor(o, dtype=torch.float32)[None])[0].detach().numpy()
    )
    cmd2 = iface2.step(state)
    np.testing.assert_allclose(cmd.q_target, cmd2.q_target, atol=1e-5)
    np.testing.assert_allclose(cmd.dq_target, cmd2.dq_target, atol=1e-4)


def test_bad_checkpoint(tmp_path):
    torch.save({"model_state_dict": {"foo": torch.zeros(1)}}, tmp_path / "bad.pt")
    with pytest.raises(KeyError, match="actor"):
        load_policy(tmp_path / "bad.pt")
