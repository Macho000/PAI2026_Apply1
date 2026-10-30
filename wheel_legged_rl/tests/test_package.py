"""Package hygiene: everything compiles, pure modules import without Isaac Sim, tasks are registered."""

import importlib
import py_compile
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "source" / "wheel_legged_rl" / "wheel_legged_rl"

PURE_MODULES = [
    "wheel_legged_rl",
    "wheel_legged_rl.actuators",
    "wheel_legged_rl.actuators.qdd_model",
    "wheel_legged_rl.actuators.selection",
    "wheel_legged_rl.agents.critic_warmup",
    "wheel_legged_rl.robots",
    "wheel_legged_rl.robots.go2w",
    "wheel_legged_rl.robots.robot_config",
    "wheel_legged_rl.utils.urdf_tools",
    "wheel_legged_rl.utils.config_loader",
    "wheel_legged_rl.rewards.rewards",
    "wheel_legged_rl.observations.observations",
    "wheel_legged_rl.curriculum.levels",
    "wheel_legged_rl.terrains.layout",
    "wheel_legged_rl.evaluation.metrics",
    "wheel_legged_rl.evaluation.diagnostics",
    "wheel_legged_rl.deployment.export_deploy_cfg",
    "wheel_legged_rl.deployment.policy_interface",
    "wheel_legged_rl.deployment.policy_export",
]


@pytest.mark.parametrize(
    "path", sorted(PKG.rglob("*.py")) + sorted((ROOT / "scripts").glob("*.py")), ids=lambda p: str(p.relative_to(ROOT))
)
def test_compiles(path, tmp_path):
    py_compile.compile(str(path), cfile=str(tmp_path / "x.pyc"), doraise=True)


@pytest.mark.parametrize("module", PURE_MODULES)
def test_pure_modules_do_not_need_isaac(module):
    importlib.import_module(module)
    assert not any(m == "isaaclab" or m.startswith("isaaclab.") for m in sys.modules), module


def test_tasks_registered():
    gym = pytest.importorskip("gymnasium")
    importlib.import_module("wheel_legged_rl.envs")
    for task in (
        "WL-Go2W-Phase1-Flat-v0",
        "WL-Go2W-Phase1-Flat-Play-v0",
        "WL-Go2W-Phase2-Flat3Axis-v0",
        "WL-Go2W-Phase2-Flat3Axis-Play-v0",
        "WL-Go2W-Phase3-FlatCoord-v0",
        "WL-Go2W-Phase3-FlatCoord-Play-v0",
        "WL-Go2W-Phase4-Steps-v0",
        "WL-Go2W-Phase4-Steps-Play-v0",
    ):
        spec = gym.spec(task)
        assert spec.entry_point == "wheel_legged_rl.envs.wheel_legged_env:WheelLeggedEnv"
        module, cls = spec.kwargs["env_cfg_entry_point"].split(":")
        assert (PKG / "envs" / "go2w" / "env_cfg.py").read_text().count(f"class {cls}(") == 1
        module, cls = spec.kwargs["rsl_rl_cfg_entry_point"].split(":")
        assert f"class {cls}(" in (PKG / "agents" / "rsl_rl_ppo_cfg.py").read_text()


def test_phase_configs_are_package_data():
    text = (ROOT / "source" / "wheel_legged_rl" / "pyproject.toml").read_text()
    assert "configs/*/*.yaml" in text
