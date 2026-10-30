"""[No Isaac] rsl_rl checkpoint -> TorchScript + ONNX, checked against the run's deploy.yaml.

    python scripts/export_policy.py --checkpoint logs/rsl_rl/go2w_wheel_leg/<run>/best.pt
    python scripts/export_policy.py --run_dir logs/rsl_rl/go2w_wheel_leg/<run>      # latest model_*.pt

Writes ``<run>/exported/policy.pt`` and ``policy.onnx`` (input ``obs`` [1, obs_dim], output ``actions``), the
layout read by ``PolicyInterface`` and by ``unitree_rl_lab/deploy``. The activation is read from
``params/agent.yaml`` when present. Fails if the policy's input/output size disagrees with ``deploy.yaml``.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import _paths  # noqa: F401
import yaml

from wheel_legged_rl.deployment.export_deploy_cfg import load_deploy_cfg
from wheel_legged_rl.deployment.policy_export import export_onnx, export_torchscript, load_policy


def latest_checkpoint(run_dir: Path) -> Path:
    ckpts = sorted(run_dir.glob("model_*.pt"), key=lambda p: int(re.search(r"(\d+)", p.stem).group(1)))
    if not ckpts:
        raise FileNotFoundError(f"no model_*.pt in {run_dir}")
    return ckpts[-1]


class _AgentYamlLoader(yaml.SafeLoader):
    """SafeLoader + ``!!python/tuple`` (Isaac Lab's dump_yaml writes tuples with that tag)."""


_AgentYamlLoader.add_constructor(
    "tag:yaml.org,2002:python/tuple", lambda loader, node: tuple(loader.construct_sequence(node))
)
_AgentYamlLoader.add_multi_constructor("tag:yaml.org,2002:python/", lambda loader, suffix, node: None)  # ignore others


def activation_from_agent_yaml(run_dir: Path) -> str | None:
    path = run_dir / "params" / "agent.yaml"
    if not path.is_file():
        return None
    with open(path) as f:
        agent = yaml.load(f, Loader=_AgentYamlLoader)
    return (agent.get("policy") or {}).get("activation")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint", type=Path)
    group.add_argument("--run_dir", type=Path)
    parser.add_argument("--out", type=Path, default=None, help="default: <run>/exported")
    parser.add_argument("--activation", default=None, help="default: from params/agent.yaml, else elu")
    parser.add_argument("--opset", type=int, default=18)
    args = parser.parse_args()

    ckpt = args.checkpoint or latest_checkpoint(args.run_dir)
    run_dir = args.run_dir or ckpt.parent
    activation = args.activation or activation_from_agent_yaml(run_dir) or "elu"
    policy = load_policy(ckpt, activation)
    print(f"checkpoint {ckpt}: obs_dim={policy.obs_dim} action_dim={policy.action_dim} activation={activation}")

    deploy = run_dir / "params" / "deploy.yaml"
    if deploy.is_file():
        cfg = load_deploy_cfg(deploy)
        expected = (cfg["policy"]["obs_dim"], cfg["policy"]["action_dim"])
        if (policy.obs_dim, policy.action_dim) != expected:
            dims = (policy.obs_dim, policy.action_dim)
            print(f"[FAIL] policy dims {dims} != deploy.yaml {expected}", file=sys.stderr)
            return 1
        if cfg["policy"].get("cpp_unsupported"):
            print(f"[NOTE] terms without a unitree_rl_lab C++ implementation: {cfg['policy']['cpp_unsupported']}")
    else:
        print(f"[WARN] {deploy} not found; dimensions not checked")

    out = args.out or run_dir / "exported"
    print(f"wrote {export_torchscript(policy, out / 'policy.pt')}")
    print(f"wrote {export_onnx(policy, out / 'policy.onnx', opset=args.opset)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
