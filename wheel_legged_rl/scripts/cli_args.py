# Adapted from Isaac Lab v2.3.0 scripts/reinforcement_learning/rsl_rl/cli_args.py
# Copyright (c) 2022-2025, The Isaac Lab Project Developers. SPDX-License-Identifier: BSD-3-Clause
"""RSL-RL command line arguments shared by train / play / evaluate."""

from __future__ import annotations

import argparse
import random


def add_rsl_rl_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("rsl_rl", description="Arguments for the RSL-RL agent.")
    group.add_argument("--experiment_name", type=str, default=None, help="Experiment folder under logs/rsl_rl.")
    group.add_argument("--run_name", type=str, default=None, help="Run name suffix of the log directory.")
    group.add_argument("--resume", action="store_true", default=False, help="Resume (weights + optimizer + iter).")
    group.add_argument("--load_run", type=str, default=None, help="Run folder (regex) to load from.")
    group.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file (regex or path) to load.")
    group.add_argument(
        "--logger", type=str, default=None, choices={"wandb", "tensorboard", "neptune"}, help="Logger backend."
    )
    group.add_argument("--log_project_name", type=str, default=None, help="W&B / Neptune project name.")


def update_rsl_rl_cfg(agent_cfg, args_cli: argparse.Namespace):
    if getattr(args_cli, "seed", None) is not None:
        if args_cli.seed == -1:
            args_cli.seed = random.randint(0, 10000)
        agent_cfg.seed = args_cli.seed
    if args_cli.experiment_name is not None:
        agent_cfg.experiment_name = args_cli.experiment_name
    if args_cli.resume:
        agent_cfg.resume = True
    if args_cli.load_run is not None:
        # Isaac Lab matches run folders with re.match (anchored at the start), but run folders are
        # "<timestamp>_<run_name>", so let a bare run name like "phase2_flat3axis" match too.
        load_run = args_cli.load_run
        agent_cfg.load_run = load_run if load_run.startswith((".*", "^")) else ".*" + load_run
    if args_cli.checkpoint is not None:
        agent_cfg.load_checkpoint = args_cli.checkpoint
    if args_cli.run_name is not None:
        agent_cfg.run_name = args_cli.run_name
    if args_cli.logger is not None:
        agent_cfg.logger = args_cli.logger
    if agent_cfg.logger in {"wandb", "neptune"} and args_cli.log_project_name:
        agent_cfg.wandb_project = args_cli.log_project_name
        agent_cfg.neptune_project = args_cli.log_project_name
    return agent_cfg
