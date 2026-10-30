"""rsl_rl checkpoint -> TorchScript / ONNX, without Isaac Lab or rsl_rl.

The actor of ``rsl_rl.modules.ActorCritic`` (rsl-rl-lib 3.0.1) is an ``MLP`` (``nn.Sequential`` of Linear /
activation) stored under ``actor.<i>.weight|bias``, optionally preceded by an ``EmpiricalNormalization``
(``actor_obs_normalizer._mean/_std``, output ``(x - mean) / (std + eps)``). Both are rebuilt here from the
state dict. The ONNX file has one fixed-shape input ``obs`` [1, obs_dim] and one output ``actions``
[1, action_dim] -- the contract of Isaac Lab's exporter and of ``unitree_rl_lab/deploy`` (``OrtRunner``).
"""

from __future__ import annotations

import re
from pathlib import Path

import torch
from torch import nn

ACTIVATIONS = {
    "elu": nn.ELU,
    "selu": nn.SELU,
    "relu": nn.ReLU,
    "lrelu": nn.LeakyReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "identity": nn.Identity,
}
NORMALIZER_EPS = 1e-2  # rsl_rl EmpiricalNormalization default


class Normalizer(nn.Module):
    def __init__(self, mean: torch.Tensor, std: torch.Tensor, eps: float = NORMALIZER_EPS):
        super().__init__()
        self.register_buffer("mean", mean.reshape(1, -1).clone())
        self.register_buffer("std", std.reshape(1, -1).clone())
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (self.std + self.eps)


class ExportedPolicy(nn.Module):
    """normalizer (optional) -> actor MLP -> deterministic action (the mean)."""

    __jit_unused_properties__ = ["obs_dim", "action_dim"]

    def __init__(self, actor: nn.Sequential, normalizer: nn.Module | None = None):
        super().__init__()
        self.normalizer = normalizer if normalizer is not None else nn.Identity()
        self.actor = actor

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(self.normalizer(obs))

    @property
    def obs_dim(self) -> int:
        return self.actor[0].in_features

    @property
    def action_dim(self) -> int:
        return [m for m in self.actor if isinstance(m, nn.Linear)][-1].out_features


def actor_from_state_dict(state: dict[str, torch.Tensor], activation: str = "elu") -> ExportedPolicy:
    layer_ids = sorted({int(m.group(1)) for k in state if (m := re.fullmatch(r"actor\.(\d+)\.weight", k))})
    if not layer_ids:
        raise KeyError("no 'actor.<i>.weight' entries in the state dict (not an rsl_rl ActorCritic checkpoint?)")
    act_cls = ACTIVATIONS[activation.lower()]
    modules: list[nn.Module] = []
    for n, i in enumerate(layer_ids):
        w, b = state[f"actor.{i}.weight"], state[f"actor.{i}.bias"]
        lin = nn.Linear(w.shape[1], w.shape[0])
        lin.weight.data.copy_(w)
        lin.bias.data.copy_(b)
        modules.append(lin)
        if n < len(layer_ids) - 1:
            modules.append(act_cls())
    normalizer = None
    if "actor_obs_normalizer._mean" in state:
        normalizer = Normalizer(state["actor_obs_normalizer._mean"], state["actor_obs_normalizer._std"])
    policy = ExportedPolicy(nn.Sequential(*modules), normalizer)
    policy.eval()
    return policy


def load_policy(checkpoint: str | Path, activation: str = "elu") -> ExportedPolicy:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    return actor_from_state_dict(state, activation)


def export_torchscript(policy: ExportedPolicy, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.script(policy).save(str(path))
    return path


def export_onnx(policy: ExportedPolicy, path: str | Path, opset: int = 18) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, policy.obs_dim)
    torch.onnx.export(
        policy,
        dummy,
        str(path),
        export_params=True,
        opset_version=opset,
        input_names=["obs"],
        output_names=["actions"],
        dynamic_axes={},
        dynamo=False,
    )
    return path
