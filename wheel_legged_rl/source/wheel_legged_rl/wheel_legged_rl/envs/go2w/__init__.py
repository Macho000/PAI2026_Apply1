"""Gym registrations for Unitree Go2-W.

Phases 1-4 are registered. Later phases are added here (one ``gym.register`` + one YAML each).
"""

import gymnasium as gym

_ENV = "wheel_legged_rl.envs.wheel_legged_env:WheelLeggedEnv"
_AGENTS = "wheel_legged_rl.agents.rsl_rl_ppo_cfg"

gym.register(
    id="WL-Go2W-Phase1-Flat-v0",
    entry_point=_ENV,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:Go2WPhase1FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{_AGENTS}:Go2WPhase1PPORunnerCfg",
    },
)

gym.register(
    id="WL-Go2W-Phase1-Flat-Play-v0",
    entry_point=_ENV,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:Go2WPhase1FlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{_AGENTS}:Go2WPhase1PPORunnerCfg",
    },
)

gym.register(
    id="WL-Go2W-Phase2-Flat3Axis-v0",
    entry_point=_ENV,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:Go2WPhase2Flat3AxisEnvCfg",
        "rsl_rl_cfg_entry_point": f"{_AGENTS}:Go2WPhase2PPORunnerCfg",
    },
)

gym.register(
    id="WL-Go2W-Phase2-Flat3Axis-Play-v0",
    entry_point=_ENV,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:Go2WPhase2Flat3AxisEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{_AGENTS}:Go2WPhase2PPORunnerCfg",
    },
)

gym.register(
    id="WL-Go2W-Phase3-FlatCoord-v0",
    entry_point=_ENV,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:Go2WPhase3FlatCoordEnvCfg",
        "rsl_rl_cfg_entry_point": f"{_AGENTS}:Go2WPhase3PPORunnerCfg",
    },
)

gym.register(
    id="WL-Go2W-Phase3-FlatCoord-Play-v0",
    entry_point=_ENV,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:Go2WPhase3FlatCoordEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{_AGENTS}:Go2WPhase3PPORunnerCfg",
    },
)

gym.register(
    id="WL-Go2W-Phase4-Steps-v0",
    entry_point=_ENV,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:Go2WPhase4StepsEnvCfg",
        "rsl_rl_cfg_entry_point": f"{_AGENTS}:Go2WPhase4PPORunnerCfg",
    },
)

gym.register(
    id="WL-Go2W-Phase4-Steps-Play-v0",
    entry_point=_ENV,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:Go2WPhase4StepsEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{_AGENTS}:Go2WPhase4PPORunnerCfg",
    },
)
