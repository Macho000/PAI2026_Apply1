"""[Isaac] Verify the imported robot before any RL: joint order, limits, inertia, joint directions, stance, rolling.

    ./isaaclab.sh -p <repo>/wheel_legged_rl/scripts/joint_test.py --robot go2w --headless [--mode inspect joints]

Modes (default: all)
  inspect  sim joint / body order vs the URDF; legs must be sim joints 0..11 and wheels 12..15 (deploy layout);
           joint limits, merged body masses / CoMs / inertia eigenvalues vs the URDF; PhysX drive gains = 0
  joints   fixed-base robot: step every leg joint by +delta and every wheel by +omega; checks the sign of the motion,
           the tracking and that the child body rotates about the URDF axis (right-hand rule)
  stand    free robot holding the default pose: base height vs the URDF stance, attitude, bodies in contact
  roll     free robot, all wheels at +omega: must drive forward; effective rolling radius vs the spec
Writes a JSON report (``--out``) and exits 1 if any check fails.
"""

import argparse
import sys

import _paths  # noqa: F401
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--robot", default="go2w", choices=sorted(_paths.ROBOTS))
parser.add_argument(
    "--mode", nargs="*", default=["inspect", "joints", "stand", "roll"], choices=["inspect", "joints", "stand", "roll"]
)
parser.add_argument("--leg_delta", type=float, default=0.2, help="position step for leg joints [rad]")
parser.add_argument("--wheel_speed", type=float, default=5.0, help="wheel speed for joints/roll tests [rad/s]")
parser.add_argument("--stand_kp", type=float, default=80.0, help="leg Kp during the stand check (Unitree FixStand)")
parser.add_argument("--stand_kd", type=float, default=4.0, help="leg Kd during the stand check")
parser.add_argument("--stand_wheel_kp", type=float, default=20.0, help="wheel position hold during the stand check")
parser.add_argument(
    "--mesh_wheels", action="store_true", help="use the URDF wheel meshes (convex hull) instead of fitted cylinders"
)
parser.add_argument("--out", type=str, default=None, help="JSON report path")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import dataclasses  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
from dataclasses import asdict, dataclass  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
import isaaclab.terrains as terrain_gen  # noqa: E402
import isaaclab.utils.math as math_utils  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.sensors import ContactSensor, ContactSensorCfg  # noqa: E402
from isaaclab.terrains import TerrainImporter, TerrainImporterCfg  # noqa: E402

from wheel_legged_rl.robots.builder import build_articulation_cfg  # noqa: E402
from wheel_legged_rl.utils import urdf_tools as ut  # noqa: E402

DT = 0.005


@dataclass
class Check:
    mode: str
    name: str
    ok: bool
    detail: str


class Bench:
    def __init__(self, spec):
        self.spec = spec
        self.sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=DT, device=args_cli.device or "cuda:0"))
        self.sim.set_camera_view([2.5, -2.5, 1.5], [0.0, 1.0, 0.3])
        TerrainImporter(
            TerrainImporterCfg(
                prim_path="/World/ground",
                terrain_type="generator",
                terrain_generator=terrain_gen.TerrainGeneratorCfg(
                    size=(20.0, 20.0),
                    num_rows=1,
                    num_cols=1,
                    border_width=0.0,
                    sub_terrains={"flat": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0)},
                ),
                num_envs=1,
                env_spacing=1.0,
            )
        )
        sim_utils.DomeLightCfg(intensity=2000.0).func("/World/light", sim_utils.DomeLightCfg(intensity=2000.0))
        free_cfg = build_articulation_cfg(spec, prim_path="/World/Robot")
        fixed_cfg = build_articulation_cfg(spec, prim_path="/World/RobotFixed", fix_base=True)
        fixed_cfg.init_state.pos = (0.0, 2.0, 1.0)
        self.free = Articulation(free_cfg)
        self.fixed = Articulation(fixed_cfg)
        self.contact = ContactSensor(ContactSensorCfg(prim_path="/World/Robot/.*", history_length=1))
        self.sim.reset()
        self.checks: list[Check] = []
        self.model = ut.load_urdf(spec.urdf_path)
        # mirror the importer: the bodies it kept are the reference; everything else is merged into them
        sim_bodies = set(self.free.data.body_names)
        self.unknown_bodies = sorted(sim_bodies - set(self.model.links))
        self.bodies = ut.merged_bodies(self.model, keep=sim_bodies & set(self.model.links))
        self.body_of_link = {m: b for b, mb in self.bodies.items() for m in mb.members}

    # ------------------------------------------------------------------ helpers

    def add(self, mode, name, ok, detail):
        self.checks.append(Check(mode, name, bool(ok), detail))
        print(f"[{'PASS' if ok else 'FAIL'}] {mode:8s} {name:40s} {detail}")

    def reset(self, robot: Articulation):
        root = robot.data.default_root_state.clone()
        robot.write_root_pose_to_sim(root[:, :7])
        robot.write_root_velocity_to_sim(root[:, 7:])
        robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone())
        self.hold(robot)
        robot.reset()

    def hold(self, robot: Articulation, pos=None, vel=None):
        robot.set_joint_position_target(robot.data.default_joint_pos.clone() if pos is None else pos)
        robot.set_joint_velocity_target(torch.zeros_like(robot.data.joint_vel) if vel is None else vel)

    def run(self, seconds: float, callback=None):
        for _ in range(int(round(seconds / DT))):
            for robot in (self.free, self.fixed):
                robot.write_data_to_sim()
            self.sim.step(render=not args_cli.headless)
            for robot in (self.free, self.fixed):
                robot.update(DT)
            self.contact.update(DT)
            if callback is not None:
                callback()

    # ------------------------------------------------------------------ inspect

    def inspect(self):
        r, spec = self.free, self.spec
        names = list(r.data.joint_names)
        print(f"sim joint order: {names}")
        print(f"sim body order : {list(r.data.body_names)}")
        legs = [names.index(n) for n in spec.leg_joint_names]
        wheels = [names.index(n) for n in spec.wheel_joint_names]
        self.add("inspect", "legs are sim joints 0..11", sorted(legs) == list(range(len(legs))), f"{sorted(legs)}")
        self.add(
            "inspect",
            "wheels are sim joints 12..15",
            sorted(wheels) == list(range(len(legs), len(names))),
            f"{sorted(wheels)}",
        )
        kept_fixed = sorted(set(r.data.body_names) - set(ut.merged_bodies(self.model)))
        self.add(
            "inspect",
            "sim bodies are URDF links",
            not self.unknown_bodies,
            f"unknown {self.unknown_bodies}; fixed links kept by the importer: {kept_fixed}",
        )

        limits = r.data.joint_pos_limits[0].cpu().numpy()
        worst = 0.0
        for i, n in enumerate(names):
            j = self.model.joint(n)
            if j.type == "revolute":
                worst = max(worst, abs(limits[i, 0] - j.lower), abs(limits[i, 1] - j.upper))
        self.add("inspect", "revolute limits = URDF", worst < 1e-3, f"max |diff| {worst:.2e} rad")

        masses = r.root_physx_view.get_masses()[0].cpu().numpy()
        coms = r.root_physx_view.get_coms()[0].cpu().numpy()[:, :3]
        inertias = r.root_physx_view.get_inertias()[0].cpu().numpy().reshape(-1, 3, 3)
        dm = dc = di = 0.0
        for k, b in enumerate(r.data.body_names):
            ref = self.bodies.get(b)
            if ref is None:
                continue
            dm = max(dm, abs(masses[k] - ref.mass))
            dc = max(dc, float(np.linalg.norm(coms[k] - ref.com)))
            ev_sim = np.sort(np.linalg.eigvalsh(inertias[k]))
            ev_ref = np.sort(np.linalg.eigvalsh(ref.inertia))
            di = max(di, float(np.max(np.abs(ev_sim - ev_ref) / np.maximum(ev_ref, 1e-6))))
        self.add(
            "inspect", "body masses = URDF (merged)", dm < 1e-3, f"max |diff| {dm:.2e} kg, total {masses.sum():.3f}"
        )
        self.add("inspect", "body CoMs = URDF (merged)", dc < 2e-3, f"max |diff| {dc * 1000:.2f} mm")
        self.add("inspect", "inertia eigenvalues = URDF", di < 0.05, f"max rel diff {di:.3f}")

        stiff = r.root_physx_view.get_dof_stiffnesses()[0].cpu().numpy()
        damp = r.root_physx_view.get_dof_dampings()[0].cpu().numpy()
        self.add(
            "inspect",
            "PhysX drive gains are zero (explicit)",
            np.all(stiff == 0) and np.all(damp == 0),
            f"max Kp {stiff.max():.3g}, max Kd {damp.max():.3g}",
        )
        kp = r.data.default_joint_stiffness[0].cpu().numpy()
        expected = np.array([spec.actuator_for(n).stiffness for n in names])
        self.add("inspect", "actuator Kp = spec", np.allclose(kp, expected), f"{kp.tolist()}")

    # ------------------------------------------------------------------ joints (fixed base)

    def _axis_alignment(self, joint_name: str, idx: int) -> float:
        """cos(angle) between the child's rotation relative to its parent and the URDF axis, signed by dq.

        NaN while the joint is (almost) still, so only samples with a well-defined direction are averaged.
        """
        r = self.fixed
        dq = float(r.data.joint_vel[0, idx])
        if abs(dq) < 0.05:
            return float("nan")
        sign = math.copysign(1.0, dq)
        j = self.model.joint(joint_name)
        child = r.find_bodies(self.body_of_link[j.child])[0][0]
        parent = r.find_bodies(self.body_of_link[j.parent])[0][0]
        w_rel = r.data.body_link_ang_vel_w[0, child] - r.data.body_link_ang_vel_w[0, parent]
        w_child = math_utils.quat_apply_inverse(r.data.body_link_quat_w[0, child][None], w_rel[None])[0]
        norm = float(w_child.norm())
        if norm < 1e-3:
            return float("nan")
        return float(torch.dot(w_child / norm, torch.tensor(j.axis, device=w_child.device, dtype=torch.float)) * sign)

    def joints(self):
        r, spec = self.fixed, self.spec
        names = list(r.data.joint_names)
        default = r.data.default_joint_pos.clone()
        for i, n in enumerate(names):
            self.reset(r)
            self.run(1.0)
            base_q = float(r.data.joint_pos[0, i])
            if n in spec.wheel_joint_names:
                vel = torch.zeros_like(r.data.joint_vel)
                vel[0, i] = args_cli.wheel_speed
                self.hold(r, vel=vel)
                align: list[float] = []
                self.run(1.0, lambda name=n, idx=i, out=align: out.append(self._axis_alignment(name, idx)))
                w = float(r.data.joint_vel[0, i])
                a = np.nanmean(align[-50:])
                ok = abs(w - args_cli.wheel_speed) < 0.2 * args_cli.wheel_speed and a > 0.95
                self.add("joints", n, ok, f"cmd {args_cli.wheel_speed:+.1f} rad/s -> {w:+.2f}, axis cos {a:+.3f}")
                continue
            j = self.model.joint(n)
            delta = args_cli.leg_delta
            if default[0, i] + delta > j.upper - 0.05:
                delta = -delta
            target = default.clone()
            target[0, i] += delta
            self.hold(r, pos=target)
            align = []
            self.run(1.0, lambda name=n, idx=i, out=align: out.append(self._axis_alignment(name, idx)))
            moved = float(r.data.joint_pos[0, i]) - base_q
            a = np.nanmean(align) if np.isfinite(align).any() else float("nan")
            # direction test, not a gain test: with Kp 25 the gravity-loaded joints (e.g. left hips lifting the leg)
            # keep a steady-state error of dtau_gravity / Kp ~ 0.08 rad, so only require half of the step
            ok = moved * delta > 0 and abs(moved) > 0.5 * abs(delta) and a > 0.95
            self.add(
                "joints", n, ok, f"step {delta:+.2f} rad -> moved {moved:+.3f} ({moved / delta:.0%}), axis cos {a:+.3f}"
            )

    # ------------------------------------------------------------------ stand / roll (free robot)

    def _set_gains(self, robot: Articulation, leg_kp, leg_kd, wheel_kp, wheel_kd):
        """Override explicit-actuator gains in place (returns the previous values for restoring)."""
        saved = {}
        for name, act in robot.actuators.items():
            saved[name] = (act.stiffness.clone(), act.damping.clone())
            wheel = any(n in self.spec.wheel_joint_names for n in act.joint_names)
            act.stiffness[:] = wheel_kp if wheel else leg_kp
            act.damping[:] = wheel_kd if wheel else leg_kd
        return saved

    def stand(self):
        """Geometry check, not a controller test: stiff FixStand-like leg gains and position-held wheels, so the
        pose stays at the default and the base height can be compared with the URDF stance."""
        r, spec = self.free, self.spec
        saved = self._set_gains(r, args_cli.stand_kp, args_cli.stand_kd, args_cli.stand_wheel_kp, 0.5)
        self.reset(r)
        heights, tilts = [], []

        def sample():
            heights.append(float(r.data.root_pos_w[0, 2]))
            roll, pitch, _ = math_utils.euler_xyz_from_quat(r.data.root_quat_w)
            tilts.append(max(abs(float(math_utils.wrap_to_pi(roll)[0])), abs(float(math_utils.wrap_to_pi(pitch)[0]))))

        self.run(3.0, sample)
        h = float(np.mean(heights[-200:]))
        tilt = float(np.max(tilts[-200:]))
        forces = self.contact.data.net_forces_w[0].norm(dim=-1).cpu().numpy()
        touching = sorted(self.contact.body_names[k] for k in np.nonzero(forces > 1.0)[0])
        # geometry check: measured height vs URDF forward kinematics at the *measured* joint angles (independent of
        # how much the PD controller sags); the sag itself is reported separately
        names = list(r.data.joint_names)
        q = {n: float(r.data.joint_pos[0, i]) for i, n in enumerate(names)}
        wheels = list(spec.wheel_joint_names)
        geos = {w: ut.wheel_geometry(self.model, w) for w in wheels}
        h_fk = ut.nominal_base_height(self.model, q, wheels, spec.wheel_radius, geos)
        dev, dev_joint = max((abs(q[n] - spec.default_position(n)), n) for n in spec.leg_joint_names)
        self.add(
            "stand",
            "base height = URDF FK at measured q",
            abs(h - h_fk) < 0.01,
            f"{h:.4f} m vs FK {h_fk:.4f} m (default-pose stance {spec.nominal_base_height:.4f}, "
            f"max leg deviation {dev:.3f} rad ({dev_joint}) at Kp {args_cli.stand_kp:g})",
        )
        self.add("stand", "level", tilt < 0.05, f"max |roll|,|pitch| {tilt:.4f} rad")
        self.add("stand", "only wheels touch the ground", touching == sorted(spec.wheel_body_names), f"{touching}")
        for name, (kp, kd) in saved.items():
            r.actuators[name].stiffness[:] = kp
            r.actuators[name].damping[:] = kd

    def roll(self):
        r, spec = self.free, self.spec
        self.reset(r)
        self.run(1.0)
        wheel_ids = [list(r.data.joint_names).index(n) for n in spec.wheel_joint_names]
        vel = torch.zeros_like(r.data.joint_vel)
        vel[0, wheel_ids] = args_cli.wheel_speed
        self.hold(r, vel=vel)
        self.run(1.0)  # accelerate
        p0 = r.data.root_pos_w[0, :2].clone()
        yaw0 = float(math_utils.euler_xyz_from_quat(r.data.root_quat_w)[2][0])
        speeds = []
        self.run(2.0, lambda: speeds.append(float(r.data.joint_vel[0, wheel_ids].mean())))
        d = (r.data.root_pos_w[0, :2] - p0).cpu().numpy()
        heading = np.array([math.cos(yaw0), math.sin(yaw0)])
        forward = float(d @ heading)
        lateral = float(abs(d @ np.array([-heading[1], heading[0]])))
        w = float(np.mean(speeds))
        r_eff = forward / 2.0 / w if abs(w) > 1e-3 else float("nan")
        yaw1 = float(math_utils.euler_xyz_from_quat(r.data.root_quat_w)[2][0])
        self.add("roll", "positive wheel speed drives forward", forward > 0, f"{forward:+.3f} m in 2 s")
        self.add(
            "roll",
            "effective radius ~ spec",
            abs(r_eff - spec.wheel_radius) < 0.1 * spec.wheel_radius,
            f"{r_eff:.4f} m (spec {spec.wheel_radius:.4f}, wheel {w:.2f} rad/s)",
        )
        self.add(
            "roll",
            "drives straight",
            lateral < 0.05 * max(abs(forward), 1e-3) + 0.02
            and abs(math_utils.wrap_to_pi(torch.tensor(yaw1 - yaw0))) < 0.1,
            f"lateral {lateral:.3f} m, yaw change {yaw1 - yaw0:+.3f} rad",
        )


def main():
    spec = _paths.load_spec(args_cli.robot)
    if args_cli.mesh_wheels:
        spec = dataclasses.replace(spec, cylinder_wheels=False)
    bench = Bench(spec)
    for mode in ("inspect", "joints", "stand", "roll"):
        if mode in args_cli.mode:
            getattr(bench, mode)()
    failed = [c for c in bench.checks if not c.ok]
    print(f"\n{len(bench.checks) - len(failed)}/{len(bench.checks)} checks passed")
    if args_cli.out:
        os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)), exist_ok=True)
        with open(args_cli.out, "w") as f:
            json.dump(
                {
                    "robot": spec.name,
                    "checks": [asdict(c) for c in bench.checks],
                    "sim_joint_names": list(bench.free.data.joint_names),
                },
                f,
                indent=2,
            )
    return 1 if failed else 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    # Isaac Sim 5.1 headless can hang in SimulationApp.close() after this script (seen in Docker); every result is
    # already printed / written, so leave without the shutdown sequence.
    os._exit(code)
