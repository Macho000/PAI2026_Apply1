import math
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from wheel_legged_rl.robots.go2w import GO2W_SPEC
from wheel_legged_rl.utils import urdf_tools as ut


def test_parse_toy(toy_urdf):
    m = ut.load_urdf(toy_urdf)
    assert m.root_link == "base"
    assert [j.name for j in m.actuated_joints] == ["hip_joint", "thigh_joint", "calf_joint", "wheel_joint"]
    assert m.total_mass == pytest.approx(3.5)
    legs, wheels = ut.classify_joints(m)
    assert legs == ["hip_joint", "thigh_joint", "calf_joint"] and wheels == ["wheel_joint"]
    assert m.joint("calf_joint").effort == 15.0


def test_forward_kinematics_and_stance(toy_urdf):
    m = ut.load_urdf(toy_urdf)
    q = {"thigh_joint": 0.5, "calf_joint": -1.0}
    T = m.link_transform("wheel", q)
    # knee 0.2 m below the thigh joint rotated by +0.5 about y, shank rotated by -0.5 in total
    expected_x = 0.1 - 0.2 * math.sin(0.5) - 0.2 * math.sin(-0.5)
    expected_z = -0.2 * math.cos(0.5) - 0.2 * math.cos(-0.5)
    assert T[0, 3] == pytest.approx(expected_x)
    assert T[2, 3] == pytest.approx(expected_z)
    assert ut.nominal_base_height(m, q, ["wheel_joint"], 0.05) == pytest.approx(-expected_z + 0.05)


def test_rpy_matches_urdf_convention():
    R = ut.rpy_matrix((0.1, 0.2, 0.3))
    Rx = ut.rotation_about_axis((1, 0, 0), 0.1)[:3, :3]
    Ry = ut.rotation_about_axis((0, 1, 0), 0.2)[:3, :3]
    Rz = ut.rotation_about_axis((0, 0, 1), 0.3)[:3, :3]
    np.testing.assert_allclose(R, Rz @ Ry @ Rx, atol=1e-12)


def test_wheel_geometry_from_cylinder(toy_urdf):
    g = ut.wheel_geometry(ut.load_urdf(toy_urdf), "wheel_joint")
    assert g.radius == pytest.approx(0.05, abs=1e-6)
    assert g.width == pytest.approx(0.03, abs=1e-6)
    assert g.axial_offset == pytest.approx(0.02, abs=1e-6)


def test_prepare_urdf_cylinder_patch(toy_urdf, tmp_path):
    out = ut.prepare_urdf(toy_urdf, tmp_path, ["wheel_joint"], cylinder_wheels=True)
    assert out.exists() and "_cylwheels_" in out.name
    link = next(le for le in ET.parse(out).getroot().findall("link") if le.get("name") == "wheel")
    (col,) = link.findall("collision")
    cyl = col.find("geometry/cylinder")
    assert float(cyl.get("radius")) == pytest.approx(0.05, abs=1e-6)
    assert float(cyl.get("length")) == pytest.approx(0.03, abs=1e-6)
    # the refitted cylinder has the same geometry as the original
    g = ut.wheel_geometry(ut.load_urdf(out), "wheel_joint")
    assert (g.radius, g.width, g.axial_offset) == pytest.approx((0.05, 0.03, 0.02), abs=1e-6)
    # deterministic output name
    assert ut.prepare_urdf(toy_urdf, tmp_path, ["wheel_joint"], cylinder_wheels=True) == out


def test_audit_markdown(toy_urdf):
    report = ut.audit(ut.load_urdf(toy_urdf), {"thigh_joint": 0.5, "calf_joint": -1.0})
    md = ut.audit_markdown(report)
    assert "| 3 | wheel_joint | continuous |" in md
    assert "calf_guard" in md and "no inertial" in md


# ------------------------------------------------------------------ Go2-W (needs the fetched description)


def test_go2w_urdf_facts(go2w_urdf):
    """The numbers the plan and GO2W_SPEC rely on, re-derived from the URDF."""
    m = ut.load_urdf(go2w_urdf)
    assert m.total_mass == pytest.approx(19.523, abs=1e-3)
    legs, wheels = ut.classify_joints(m, r".*_foot_joint")
    assert sorted(legs) == sorted(GO2W_SPEC.leg_joint_names)
    assert sorted(wheels) == sorted(GO2W_SPEC.wheel_joint_names)
    for w, body in zip(GO2W_SPEC.wheel_joint_names, GO2W_SPEC.wheel_body_names, strict=True):
        j = m.joint(w)
        assert j.type == "continuous" and j.child == body
        assert j.axis == pytest.approx(GO2W_SPEC.wheel_axis)
        g = ut.wheel_geometry(m, w)
        assert g.radius == pytest.approx(GO2W_SPEC.wheel_radius, abs=5e-4)
        assert g.width == pytest.approx(0.0518, abs=5e-4)
        assert abs(g.axial_offset) == pytest.approx(0.048, abs=5e-4)
    stance = {n: GO2W_SPEC.default_position(n) for n in legs}
    h = ut.nominal_base_height(m, stance, wheels, GO2W_SPEC.wheel_radius)
    assert h == pytest.approx(GO2W_SPEC.nominal_base_height, abs=1e-3)
    assert GO2W_SPEC.init_base_height > h


def test_go2w_actuators_match_urdf_limits(go2w_urdf):
    """Peak torque / speed of every actuator group within 2 % of the URDF effort / velocity limits."""
    m = ut.load_urdf(go2w_urdf)
    for name in GO2W_SPEC.all_joint_names:
        j = m.joint(name)
        act = GO2W_SPEC.actuator_for(name)
        curve = act.resolved_curve()
        assert curve.y1 <= j.effort * 1.02, name
        assert curve.peak_torque == pytest.approx(j.effort, rel=0.02), name
        assert act.resolved_velocity_limit() == pytest.approx(j.velocity, rel=1e-3), name
        if j.type == "revolute":
            lo, hi = j.lower, j.upper
            assert lo <= GO2W_SPEC.default_position(name) <= hi, name


def test_go2w_prepare_cylinder_wheels(go2w_urdf, tmp_path):
    out = ut.prepare_urdf(go2w_urdf, tmp_path, list(GO2W_SPEC.wheel_joint_names), cylinder_wheels=True)
    m = ut.load_urdf(out)
    for w in GO2W_SPEC.wheel_joint_names:
        link = m.links[m.joint(w).child]
        assert [c.kind for c in link.collisions] == ["cylinder"]
    for link in m.links.values():  # every remaining mesh path is absolute and exists
        for g in link.collisions + link.visuals:
            if g.kind == "mesh":
                assert ut.resolve_mesh_path(g.params["filename"], out).is_file()


def test_merged_bodies_toy(toy_urdf):
    bodies = ut.merged_bodies(ut.load_urdf(toy_urdf))
    assert set(bodies) == {"base", "hip", "thigh", "calf", "wheel"}
    assert bodies["calf"].members == ["calf", "calf_guard"]  # massless fixed child merged
    assert bodies["calf"].mass == pytest.approx(0.3)
    np.testing.assert_allclose(bodies["calf"].com, [0, 0, -0.1])
    assert sum(b.mass for b in bodies.values()) == pytest.approx(3.5)


def test_merged_inertia_parallel_axis(tmp_path):
    """Two point-ish masses on a fixed joint: inertia about the combined CoM follows the parallel-axis theorem."""
    urdf = tmp_path / "two.urdf"
    point = '<inertial><mass value="1"/><inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/></inertial>'
    urdf.write_text(f"""<robot name="two">
      <link name="a">{point}</link>
      <link name="b">{point}</link>
      <joint name="f" type="fixed"><parent link="a"/><child link="b"/><origin xyz="0.2 0 0" rpy="0 0 1.0"/></joint>
    </robot>""")
    (body,) = ut.merged_bodies(ut.load_urdf(urdf)).values()
    np.testing.assert_allclose(body.com, [0.1, 0, 0], atol=1e-12)
    np.testing.assert_allclose(np.diag(body.inertia), [0.0, 0.02, 0.02], atol=1e-12)


def test_go2w_merged_bodies(go2w_urdf):
    bodies = ut.merged_bodies(ut.load_urdf(go2w_urdf))
    # 17 simulated bodies: base, 4 x (hip, thigh, calf, wheel)
    assert len(bodies) == 17
    assert "FL_foot_motor" in bodies["FL_calf"].members  # 0.646 kg motor merged into the calf
    assert bodies["FL_calf"].mass == pytest.approx(0.154 + 0.646)
    assert sum(b.mass for b in bodies.values()) == pytest.approx(19.523, abs=1e-3)


def test_go2w_merged_bodies_isaac_sim_51(go2w_urdf):
    """Isaac Sim 5.1 keeps fixed links with mass (foot motors, head) as separate bodies (seen in joint_test)."""
    model = ut.load_urdf(go2w_urdf)
    keep = set(ut.merged_bodies(model)) | {f"{leg}_foot_motor" for leg in ("FL", "FR", "RL", "RR")}
    keep |= {"Head_upper", "Head_lower"}
    bodies = ut.merged_bodies(model, keep=keep)
    assert set(bodies) == keep and len(bodies) == 23
    assert bodies["FL_calf"].mass == pytest.approx(0.154)
    assert bodies["FL_foot_motor"].mass == pytest.approx(0.646)
    assert "FL_calflower" in bodies["FL_calf"].members  # massless links are still merged
    assert sum(b.mass for b in bodies.values()) == pytest.approx(19.523, abs=1e-3)


def test_go2w_explicit_damping_stability(go2w_urdf):
    """Regression for the Isaac Sim finding: wheel Kd 0.5 at dt 0.005 is unstable without armature."""
    model = ut.load_urdf(go2w_urdf)
    stance = {n: GO2W_SPEC.default_position(n) for n in GO2W_SPEC.all_joint_names}
    inertia = ut.joint_axis_inertia(model, "FL_foot_joint", stance)
    assert inertia == pytest.approx(0.0012, rel=1e-6)  # wheel CoM lies on the axis -> just I_yy
    assert ut.joint_axis_inertia(model, "FL_hip_joint", stance) > 0.01  # the hip drives the whole leg
    assert ut.explicit_damping_ratio(inertia, 0.5, 0.0, 0.005) > 2.0
    for name in GO2W_SPEC.all_joint_names:
        act = GO2W_SPEC.actuator_for(name)
        ratio = ut.explicit_damping_ratio(
            ut.joint_axis_inertia(model, name, stance), act.damping, act.resolved_armature(), 0.005
        )
        assert ratio < 1.0, (name, ratio)


def test_stance_height_with_camber(go2w_urdf):
    """Offset tyre + camber: splaying the hips outward lowers the base (seen in Isaac Sim joint_test stand)."""
    model = ut.load_urdf(go2w_urdf)
    wheels = list(GO2W_SPEC.wheel_joint_names)
    geos = {w: ut.wheel_geometry(model, w) for w in wheels}
    stance = {n: GO2W_SPEC.default_position(n) for n in GO2W_SPEC.all_joint_names}
    level = ut.nominal_base_height(model, stance, wheels, GO2W_SPEC.wheel_radius, geos)
    assert level == pytest.approx(ut.nominal_base_height(model, stance, wheels, GO2W_SPEC.wheel_radius), abs=1e-4)
    splayed = dict(stance, FL_hip_joint=0.2, RL_hip_joint=0.2, FR_hip_joint=-0.2, RR_hip_joint=-0.2)
    plain = ut.nominal_base_height(model, splayed, wheels, GO2W_SPEC.wheel_radius)
    tyre = ut.nominal_base_height(model, splayed, wheels, GO2W_SPEC.wheel_radius, geos)
    assert plain - tyre == pytest.approx(
        0.048 * math.sin(0.2) - 0.026 * math.sin(0.2) + 0.086 * (1 - math.cos(0.2)), abs=2e-3
    )
