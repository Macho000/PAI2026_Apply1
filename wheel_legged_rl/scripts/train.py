"""[Isaac] Train a wheel-leg policy with RSL-RL PPO.

    ./isaaclab.sh -p <repo>/wheel_legged_rl/scripts/train.py --task WL-Go2W-Phase1-Flat-v0 --headless \
        --logger wandb --log_project_name wheel_legged_rl

Options beyond Isaac Lab's train.py:
  --env_config PATH   phase YAML to use instead of the task's default (applied before Hydra overrides)
  --init_from PATH    initialise the policy from a checkpoint of an earlier phase (weights only, iteration 0);
                      a file, or a run directory (-> its best.pt). Default: the runner config's ``init_from``
                      (Phase 3: best.pt of the latest phase2_flat3axis run); ``agent.init_from=null`` -> scratch
  --checkpoint PATH   without --resume and given as a file path: same as --init_from, e.g.
                      --task WL-Go2W-Phase2-Flat3Axis-v0 --checkpoint logs/rsl_rl/go2w_wheel_leg/<phase1>/model_3999.pt
Resume an interrupted run of the same phase with --resume --load_run <run> --checkpoint model_XXXX.pt.
Host paths of this project (/home/<user>/quadra/wheel_legged_rl/...) also work inside the Docker container.
The starting checkpoint is recorded as ``init_checkpoint`` / ``init_mode`` in the runner config (W&B config
``runner_cfg.init_checkpoint``, ``params/agent.yaml``); paths inside this project are stored relative to it.
Hydra overrides go last, e.g. ``env.rewards.action_rate.weight=-0.01 agent.max_iterations=500``.
Writes ``params/deploy.yaml`` (C++ / PolicyInterface) next to the checkpoints before training starts.
A run started from a checkpoint with ``critic_warmup`` in the runner config (Phase 3) first trains the critic only
(actor frozen); see ``wheel_legged_rl/agents/runner.py``. Critic diagnostics are logged as ``Critic/*``.
"""

import argparse
import os
import sys

import _paths  # noqa: F401  (source checkout on sys.path)
from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", type=str, default="WL-Go2W-Phase1-Flat-v0")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--env_config", type=str, default=None, help="phase YAML overriding the task default")
parser.add_argument("--init_from", type=str, default=None, help="checkpoint file or run dir (-> best.pt)")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=200)
parser.add_argument("--video_interval", type=int, default=2000)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True
if args_cli.env_config:
    os.environ["WL_ENV_CONFIG"] = os.path.abspath(args_cli.env_config)
if args_cli.resume and args_cli.init_from:
    parser.error("--resume and --init_from are mutually exclusive")
if not args_cli.resume and args_cli.checkpoint:
    # without --resume a checkpoint file initialises the weights of a new run (e.g. Phase 1 -> Phase 2)
    if args_cli.init_from:
        parser.error("--checkpoint (without --resume) and --init_from are mutually exclusive")
    args_cli.init_from, args_cli.checkpoint = args_cli.checkpoint, None
if args_cli.init_from:
    try:
        args_cli.init_from = _paths.resolve_checkpoint(args_cli.init_from)
    except FileNotFoundError as e:
        parser.error(str(e))
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Everything below needs the simulator."""

import shutil  # noqa: E402
from datetime import datetime  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab.utils.dict import print_dict  # noqa: E402
from isaaclab.utils.io import dump_yaml  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import wheel_legged_rl.envs  # noqa: E402, F401  (registers the tasks)
from wheel_legged_rl.agents.runner import WheelLeggedOnPolicyRunner  # noqa: E402
from wheel_legged_rl.deployment.export_deploy_cfg import export_deploy_cfg  # noqa: E402
from wheel_legged_rl.envs.base_env_cfg import WheelLeggedEnvCfg, finalize  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: WheelLeggedEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    agent_cfg.device = env_cfg.sim.device
    finalize(env_cfg)  # re-derive update periods etc. after Hydra overrides

    log_root = os.path.join(_paths.PROJECT_ROOT, "logs", "rsl_rl", agent_cfg.experiment_name)
    run = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + (f"_{agent_cfg.run_name}" if agent_cfg.run_name else "")
    log_dir = os.path.join(log_root, run)
    env_cfg.log_dir = log_dir
    print(f"[INFO] logging to {log_dir}")

    resume_path = None
    if agent_cfg.resume:
        resume_path = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)
    elif args_cli.init_from is None and agent_cfg.init_from:
        # task default (e.g. Phase 3 <- best.pt of the latest Phase 2 run)
        args_cli.init_from = _paths.resolve_checkpoint(agent_cfg.init_from, log_root)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    deploy_path = export_deploy_cfg(env.unwrapped, log_dir)
    shutil.copy(env_cfg.wl.phase_config, os.path.join(log_dir, "params", "phase.yaml"))
    print(f"[INFO] wrote {deploy_path}")

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    # the runner config goes to W&B (runner_cfg.*) and params/agent.yaml: record where the weights came from
    runner_cfg = agent_cfg.to_dict()
    init_checkpoint = resume_path or args_cli.init_from
    runner_cfg["init_mode"] = "resume" if resume_path else ("init_from" if init_checkpoint else "scratch")
    runner_cfg["init_checkpoint"] = _paths.project_relative(init_checkpoint) if init_checkpoint else None
    runner = WheelLeggedOnPolicyRunner(env, runner_cfg, log_dir=log_dir, device=agent_cfg.device)
    runner.add_git_repo_to_log(__file__)
    if resume_path:
        print(f"[INFO] resuming from {resume_path}")
        runner.load(resume_path)
    elif args_cli.init_from:
        print(f"[INFO] initialising weights from {args_cli.init_from}")
        runner.load_init(args_cli.init_from)  # a grown critic input is expanded with zero weights

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), runner_cfg)

    env.unwrapped.desynchronize_timers()  # with the random episode lengths: stationary data from iteration 0
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
