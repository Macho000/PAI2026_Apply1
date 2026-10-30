"""[Pure Python] Audit a URDF: joints, limits, masses, wheel geometry, nominal stance height.

    python scripts/urdf_audit.py --robot go2w --out docs/URDF_AUDIT_go2w_generated.md
    python scripts/urdf_audit.py --urdf my_robot.urdf --wheel-pattern ".*_wheel_joint" \
        --stance thigh=0.8 calf=-1.5

With ``--robot`` the URDF path, wheel joints and stance come from the RobotSpec, and the spec is checked against
the URDF (joint names, wheel radius, nominal height, actuator limits). Exit code 1 if a check fails.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import _paths  # noqa: F401

from wheel_legged_rl.utils import urdf_tools as ut

SIM_DT = 0.005  # physics step of the tasks (envs/base_env_cfg.py)


def spec_checks(spec, model: ut.UrdfModel, report: dict) -> list[str]:
    errors = []
    urdf_joints = {j.name: j for j in model.actuated_joints}
    for name in spec.all_joint_names:
        if name not in urdf_joints:
            errors.append(f"spec joint '{name}' not in URDF")
    if sorted(report["wheel_joints"]) != sorted(spec.wheel_joint_names):
        errors.append(f"URDF wheel joints {report['wheel_joints']} != spec {list(spec.wheel_joint_names)}")
    if not math.isclose(report["wheel_radius"], spec.wheel_radius, abs_tol=1e-3):
        errors.append(f"wheel radius URDF {report['wheel_radius']:.4f} != spec {spec.wheel_radius:.4f}")
    h = report["nominal_base_height"]
    if h is not None and not math.isclose(h, spec.nominal_base_height, abs_tol=2e-3):
        errors.append(f"nominal base height URDF {h:.4f} != spec {spec.nominal_base_height:.4f}")
    for name in spec.all_joint_names:
        j = urdf_joints.get(name)
        if j is None:
            continue
        act = spec.actuator_for(name)
        peak = act.resolved_curve().peak_torque
        if j.effort and not math.isclose(peak, j.effort, rel_tol=0.05):
            errors.append(f"{name}: actuator peak torque {peak:.2f} vs URDF effort {j.effort:.2f}")
        if j.velocity and not math.isclose(act.resolved_velocity_limit(), j.velocity, rel_tol=0.05):
            vlim = act.resolved_velocity_limit()
            errors.append(f"{name}: actuator velocity limit {vlim:.2f} vs URDF {j.velocity:.2f}")
        stance = {n: spec.default_position(n) for n in spec.all_joint_names}
        inertia = ut.joint_axis_inertia(model, name, stance)
        ratio = ut.explicit_damping_ratio(inertia, act.damping, act.resolved_armature(), SIM_DT)
        if ratio >= 1.0:
            errors.append(
                f"{name}: explicit damping unstable at dt={SIM_DT}: Kd*dt/(I+armature) = {ratio:.2f} >= 1 "
                f"(driven inertia about the axis {inertia:.2e}, armature {act.resolved_armature():.2e})"
            )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--robot", choices=sorted(_paths.ROBOTS), help="use a RobotSpec (path, wheels, stance)")
    parser.add_argument("--urdf", type=Path, help="URDF path (overrides the spec)")
    parser.add_argument("--wheel-pattern", default=None, help="regex for wheel joints (default: all continuous)")
    parser.add_argument("--stance", nargs="*", default=[], help="joint-regex=value pairs for the stance height")
    parser.add_argument("--out", type=Path, help="write the Markdown report here")
    parser.add_argument("--json", action="store_true", help="print a JSON summary")
    args = parser.parse_args()

    spec = _paths.load_spec(args.robot) if args.robot else None
    urdf = args.urdf or (Path(spec.urdf_path) if spec else None)
    if urdf is None:
        parser.error("give --urdf or --robot")
    model = ut.load_urdf(urdf)

    wheel_pattern = args.wheel_pattern or ("|".join(re.escape(n) for n in spec.wheel_joint_names) if spec else r".*")
    stance: dict[str, float] = {}
    if spec:
        stance = {j.name: spec.default_position(j.name) for j in model.actuated_joints if j.type == "revolute"}
    for item in args.stance:
        pattern, value = item.split("=")
        for j in model.actuated_joints:
            if re.search(pattern, j.name):
                stance[j.name] = float(value)

    report = ut.audit(model, stance or None, wheel_pattern)
    md = ut.audit_markdown(report)
    errors = spec_checks(spec, model, report) if spec else []
    if spec:
        md += "\n## RobotSpec consistency (`" + spec.name + "`)\n\n"
        md += "\n".join(f"- FAIL: {e}" for e in errors) if errors else "- all checks passed"
        md += "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(md)
        print(f"wrote {args.out}")
    if args.json:
        print(
            json.dumps(
                {
                    "total_mass": report["total_mass"],
                    "leg_joints": report["leg_joints"],
                    "wheel_joints": report["wheel_joints"],
                    "wheel_radius": report["wheel_radius"],
                    "nominal_base_height": report["nominal_base_height"],
                    "warnings": report["warnings"],
                    "spec_errors": errors,
                },
                indent=2,
            )
        )
    elif not args.out:
        print(md)
    for e in errors:
        print(f"[FAIL] {e}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
