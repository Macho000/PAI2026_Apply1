"""[Isaac] Quantitative evaluation of checkpoints on the phase's scenarios; picks the best checkpoint.

    ./isaaclab.sh -p <repo>/wheel_legged_rl/scripts/evaluate.py --task WL-Go2W-Phase1-Flat-Play-v0 \
        --load_run <run> [--checkpoint model_3000.pt | --all_checkpoints --every 5] [--select_best] [--wandb]

Scenarios, pass criteria and the checkpoint score come from the ``evaluation`` section of the phase YAML. Every
robot of the batch runs the same scenario (commands are written directly into the command term; no resampling,
no pushes, no curriculum). A scenario is a list of segments, or a mapping (terrain phases)::

    curbs_up_0.15: {terrain: curbs_up_0.15, group: steps, goal_distance: 3.5, segments: [...]}

``terrain`` names a sub-terrain of ``evaluation.terrain`` (which then replaces the training terrain): every
robot starts at the centre of that tile. With ``goal_distance`` [m] an episode succeeds when the robot did not
fall and ended at least that far (planar) from its start (``success_rate``). Scenarios with a ``group`` are also
summarised per group; criteria and selection weights may name ``<group>/<metric>``.

Results go to ``<run>/eval/<timestamp>/metrics.{json,csv}`` and, with ``--wandb``, to the W&B project under the
``eval/`` prefix. ``--select_best`` copies the best checkpoint to ``<run>/best.pt``.
Exit code 0 if the evaluated (or best) checkpoint meets every criterion, 1 otherwise.
"""

import argparse
import os
import sys

import _paths  # noqa: F401
from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", type=str, default="WL-Go2W-Phase1-Flat-Play-v0")
parser.add_argument("--num_envs", type=int, default=None, help="robots per scenario (default: YAML)")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--env_config", type=str, default=None)
parser.add_argument("--all_checkpoints", action="store_true", help="evaluate every model_*.pt of the run")
parser.add_argument("--every", type=int, default=1, help="with --all_checkpoints: keep every N-th checkpoint")
parser.add_argument("--select_best", action="store_true", help="copy the best checkpoint to <run>/best.pt")
parser.add_argument("--wandb", action="store_true", help="log to W&B (project from --log_project_name)")
parser.add_argument("--out", type=str, default=None, help="output directory (default <run>/eval/<timestamp>)")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.env_config:
    os.environ["WL_ENV_CONFIG"] = os.path.abspath(args_cli.env_config)
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import csv  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
from datetime import datetime  # noqa: E402

import gymnasium as gym  # noqa: E402
import isaaclab.utils.math as math_utils  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import wheel_legged_rl.envs  # noqa: E402, F401
from wheel_legged_rl.envs.base_env_cfg import WheelLeggedEnvCfg, finalize  # noqa: E402
from wheel_legged_rl.evaluation import metrics as M  # noqa: E402
from wheel_legged_rl.rewards.rewards import contact_mask, wheel_slip_from_tensors  # noqa: E402
from wheel_legged_rl.terrains.layout import column_terrain_names  # noqa: E402
from wheel_legged_rl.terrains.terrain_config import apply_terrain_section  # noqa: E402
from wheel_legged_rl.utils.config_loader import load_yaml  # noqa: E402


def checkpoint_iteration(path: str) -> int:
    m = re.search(r"model_(\d+)\.pt$", path)
    return int(m.group(1)) if m else -1


def collect_checkpoints(agent_cfg) -> tuple[str, list[str]]:
    if args_cli.checkpoint and os.path.isfile(args_cli.checkpoint):
        path = os.path.abspath(args_cli.checkpoint)
        return os.path.dirname(path), [path]
    log_root = os.path.join(_paths.PROJECT_ROOT, "logs", "rsl_rl", agent_cfg.experiment_name)
    latest = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)
    run_dir = os.path.dirname(latest)
    if not args_cli.all_checkpoints:
        return run_dir, [latest]
    ckpts = sorted(glob.glob(os.path.join(run_dir, "model_*.pt")), key=checkpoint_iteration)
    keep = ckpts[:: max(1, args_cli.every)]
    if ckpts and ckpts[-1] not in keep:
        keep.append(ckpts[-1])
    return run_dir, keep


class Recorder:
    """Collects per-step state of every robot until it terminates."""

    def __init__(self, env, wheel_radius: float, wheel_axis):
        self.env = env
        self.robot = env.scene["robot"]
        wl = env.cfg.wl
        self.wheel_ids, _ = self.robot.find_joints(wl.wheel_joint_names)
        self.wheel_bodies, _ = self.robot.find_bodies(wl.wheel_body_names, preserve_order=True)
        self.contact = env.scene.sensors["contact_forces"]
        self.wheel_sensor_ids, _ = self.contact.find_bodies(wl.wheel_body_names, preserve_order=True)
        self.r = wheel_radius
        self.axis = torch.tensor(wheel_axis, device=env.device, dtype=torch.float)
        self.rows: list[dict[str, np.ndarray]] = []

    @torch.no_grad()
    def record(self, command: torch.Tensor, alive: torch.Tensor) -> None:
        d = self.robot.data
        roll, pitch, _ = math_utils.euler_xyz_from_quat(d.root_quat_w)
        ids = self.wheel_bodies
        slip = wheel_slip_from_tensors(
            d.body_link_lin_vel_w[:, ids], d.body_link_ang_vel_w[:, ids], d.body_link_quat_w[:, ids], self.axis, self.r
        )
        self.rows.append(
            {
                "alive": alive.cpu().numpy().copy(),
                "command": command.cpu().numpy().copy(),
                "lin_vel_b": d.root_lin_vel_b.cpu().numpy(),
                "ang_vel_b": d.root_ang_vel_b.cpu().numpy(),
                "roll_pitch": torch.stack([math_utils.wrap_to_pi(roll), math_utils.wrap_to_pi(pitch)], dim=-1)
                .cpu()
                .numpy(),
                "power": (d.applied_torque * d.joint_vel).abs().sum(dim=-1).cpu().numpy(),
                "pos": d.root_pos_w[:, :2].cpu().numpy(),
                "slip": slip.cpu().numpy(),
                "rolling": (d.joint_vel[:, self.wheel_ids] * self.r).cpu().numpy(),
                "contact": contact_mask(self.contact.data.net_forces_w_history[:, :, self.wheel_sensor_ids], 1.0)
                .cpu()
                .numpy(),
            }
        )

    def episode_logs(
        self, fell: np.ndarray, dt: float, settle_steps: int, goal_distance: float | None = None
    ) -> list[M.EpisodeLog]:
        stack = {k: np.stack([r[k] for r in self.rows], axis=1) for k in self.rows[0]}  # (N, T, ...)
        logs = []
        for i in range(stack["alive"].shape[0]):
            T = int(stack["alive"][i].sum())  # alive is a prefix (never revived); T == 0 still counts as a fall
            pos = stack["pos"][i, :T]
            cmd = stack["command"][i, :T]
            logs.append(
                M.EpisodeLog(
                    dt=dt,
                    command=cmd,
                    lin_vel_b=stack["lin_vel_b"][i, :T],
                    ang_vel_b=stack["ang_vel_b"][i, :T],
                    roll_pitch=stack["roll_pitch"][i, :T],
                    power=stack["power"][i, :T],
                    fell=bool(fell[i]),
                    distance=float(np.linalg.norm(np.diff(pos, axis=0), axis=-1).sum()) if T > 1 else 0.0,
                    wheel_slip=stack["slip"][i, :T],
                    wheel_rolling_speed=stack["rolling"][i, :T],
                    wheel_contact=stack["contact"][i, :T],
                    valid=M.transient_mask(cmd, settle_steps),
                    reached_goal=None if goal_distance is None else M.reached_goal(pos, bool(fell[i]), goal_distance),
                )
            )
        return logs


def normalize_scenario(spec) -> dict:
    """List of segments (phases 1-3) or ``{segments, terrain?, group?, goal_distance?}``."""
    spec = {"segments": spec} if isinstance(spec, list) else dict(spec)
    unknown = set(spec) - {"segments", "terrain", "group", "goal_distance"}
    if unknown:
        raise KeyError(f"unknown scenario keys {sorted(unknown)}")
    return spec


class TerrainPlacer:
    """Puts every robot on one tile of the evaluation terrain (``evaluation.terrain``, a single curriculum row)."""

    def __init__(self, env, section: dict | None):
        self.terrain = env.scene.terrain
        self.columns: dict[str, int] = {}
        if not section:
            return
        gen = section["generator"]
        names = column_terrain_names(
            {k: v.get("proportion", 1.0) for k, v in section["sub_terrains"].items()}, gen["num_cols"]
        )
        for name in dict.fromkeys(names):
            cols = [c for c, n in enumerate(names) if n == name]
            self.columns[name] = cols[len(cols) // 2]  # middle column: wide terrains keep neighbours of their type

    def place(self, name: str | None) -> None:
        if name is None:
            if self.columns:
                raise ValueError("evaluation.terrain is set: every scenario needs a 'terrain'")
            return
        if name not in self.columns:
            raise KeyError(f"scenario terrain '{name}' is not in evaluation.terrain ({sorted(self.columns)})")
        t = self.terrain
        t.terrain_levels[:] = 0
        t.terrain_types[:] = self.columns[name]
        t.env_origins[:] = t.terrain_origins[0, self.columns[name]]


def run_scenario(env, policy, segments: list[dict], recorder: Recorder) -> np.ndarray:
    """Run one scenario on every robot; returns which robots fell. Per-step data is left in ``recorder``.

    Everything (reset included) runs in inference mode: stepping there leaves inference tensors in Isaac Lab's
    buffers, and a later reset outside it would fail on the in-place writes.
    """
    base = env.unwrapped
    term = base.command_manager.get_term("base_velocity")
    n = base.num_envs
    recorder.rows = []
    with torch.inference_mode():
        env.reset()
        alive = torch.ones(n, dtype=torch.bool, device=base.device)
        fell = torch.zeros(n, dtype=torch.bool, device=base.device)
        for seg in segments:
            command = torch.tensor(seg["command"], device=base.device, dtype=torch.float).repeat(n, 1)
            for _ in range(int(round(seg["duration"] / base.step_dt))):
                term.vel_command_b[:] = command
                obs = env.get_observations()
                _, _, dones, extras = env.step(policy(obs))
                dones = dones.bool()
                time_outs = extras.get("time_outs", torch.zeros_like(dones)).bool()
                fell |= alive & dones & ~time_outs
                alive &= ~dones
                recorder.record(command, alive)
    return fell.cpu().numpy()


EXIT_CODE = 1  # set by main(); Hydra's wrapper drops return values


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: WheelLeggedEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    phase = load_yaml(env_cfg.wl.phase_config)
    ev = phase.get("evaluation") or {}
    scenarios = {name: normalize_scenario(spec) for name, spec in ev["scenarios"].items()}
    longest = max(sum(s["duration"] for s in spec["segments"]) for spec in scenarios.values())

    env_cfg.scene.num_envs = args_cli.num_envs or ev.get("num_envs", 64)
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    agent_cfg.device = env_cfg.sim.device
    env_cfg.episode_length_s = longest + 5.0  # no time-outs inside a scenario
    cmd = env_cfg.commands.base_velocity
    cmd.resampling_time_range = (1.0e9, 1.0e9)
    cmd.rel_standing_envs = 0.0
    env_cfg.events.push_robot = None
    for name in list(vars(env_cfg.curriculum)):
        setattr(env_cfg.curriculum, name, None)
    if ev.get("terrain"):
        apply_terrain_section(env_cfg.scene.terrain, ev["terrain"])
    finalize(env_cfg)
    if ev.get("terrain"):
        env_cfg.scene.terrain.terrain_generator.curriculum = True  # fixed column layout (finalize turned it off)

    run_dir, checkpoints = collect_checkpoints(agent_cfg)
    out_dir = args_cli.out or os.path.join(run_dir, "eval", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(out_dir, exist_ok=True)

    env = RslRlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
    base = env.unwrapped
    mass = float(base.scene["robot"].root_physx_view.get_masses().sum(dim=1).mean())  # mean over randomised robots
    recorder = Recorder(base, env_cfg.wl.wheel_radius, env_cfg.wl.wheel_axis)
    placer = TerrainPlacer(base, ev.get("terrain"))
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    settle = int(round(ev.get("skip_s", 1.0) / base.step_dt))

    wandb_run = None
    if args_cli.wandb:
        import wandb

        wandb_run = wandb.init(
            project=args_cli.log_project_name or agent_cfg.wandb_project,
            name=f"eval_{os.path.basename(run_dir)}",
            job_type="eval",
            config={"task": args_cli.task, "run_dir": run_dir, "num_envs": env_cfg.scene.num_envs},
        )

    results: dict[str, dict] = {}
    rows = []
    for ckpt in checkpoints:
        runner.load(ckpt, load_optimizer=False)
        policy = runner.get_inference_policy(device=base.device)
        per_scenario, all_logs, group_logs = {}, [], {}
        for name, spec in scenarios.items():
            placer.place(spec.get("terrain"))
            fell = run_scenario(env, policy, spec["segments"], recorder)
            logs = recorder.episode_logs(fell, base.step_dt, settle, spec.get("goal_distance"))
            per_scenario[name] = M.summarize(logs, mass, skip_s=0.0)  # transients handled by `valid`
            all_logs += logs
            if spec.get("group"):
                group_logs.setdefault(spec["group"], []).extend(logs)
        overall = M.summarize(all_logs, mass, skip_s=0.0)
        groups = {g: M.summarize(logs, mass, skip_s=0.0) for g, logs in group_logs.items()}
        judged = M.with_groups(overall, groups)
        criteria = M.check_criteria(judged, ev.get("criteria", {}))
        entry = {
            "iteration": checkpoint_iteration(ckpt),
            "scenarios": per_scenario,
            "groups": groups,
            "all": overall,
            "criteria": criteria,
            "passed": all(criteria.values()),
            "score": M.score(judged, ev.get("selection_weights", {"vel_rmse": 1.0, "fall_rate": 10.0})),
        }
        results[ckpt] = entry
        print(f"[EVAL] {os.path.basename(ckpt)}  score={entry['score']:.4f}  passed={entry['passed']}  {criteria}")
        for scen, summary in {**per_scenario, **{f"group/{g}": v for g, v in groups.items()}, "all": overall}.items():
            rows.append(
                {"checkpoint": os.path.basename(ckpt), "iteration": entry["iteration"], "scenario": scen, **summary}
            )
        if wandb_run is not None:
            summaries = {**per_scenario, **{f"group/{g}": v for g, v in groups.items()}, "all": overall}
            log = {f"eval/{s}/{k}": v for s, summ in summaries.items() for k, v in summ.items()}
            log["eval/score"] = entry["score"]
            log["eval/passed"] = float(entry["passed"])
            wandb_run.log(log, step=max(entry["iteration"], 0))

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)
    with open(os.path.join(out_dir, "metrics.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[EVAL] wrote {out_dir}/metrics.json and metrics.csv")

    best = min(results, key=lambda c: results[c]["score"])
    if args_cli.select_best:
        shutil.copy(best, os.path.join(run_dir, "best.pt"))
        with open(os.path.join(run_dir, "best.json"), "w") as f:
            json.dump({"checkpoint": best, **results[best]}, f, indent=2, default=float)
        print(f"[EVAL] best checkpoint: {best} -> {os.path.join(run_dir, 'best.pt')}")
    if wandb_run is not None:
        wandb_run.summary["best_checkpoint"] = os.path.basename(best)
        wandb_run.finish()
    env.close()
    global EXIT_CODE
    EXIT_CODE = 0 if results[best if args_cli.select_best else checkpoints[-1]]["passed"] else 1


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    # SimulationApp.close() terminates the process itself in Isaac Sim 5.1 (exit code 0), which would hide a failed
    # evaluation; results are written and W&B is finished at this point, so exit directly with the verdict.
    os._exit(EXIT_CODE)
