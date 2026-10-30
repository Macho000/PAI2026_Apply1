"""Isaac-independent URDF inspection utilities.

Used by ``scripts/urdf_audit.py`` (report generation), ``scripts/joint_test.py`` (comparison against the
imported articulation) and ``robots/builder.py`` (asset preparation). Only the standard library and numpy.
"""

from __future__ import annotations

import hashlib
import math
import re
import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------------------------------------------------


@dataclass
class UrdfGeometry:
    kind: str  # "mesh" | "box" | "cylinder" | "sphere"
    origin_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    origin_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)
    params: dict = field(default_factory=dict)  # mesh: filename, scale / box: size / cylinder: radius, length ...


@dataclass
class UrdfLink:
    name: str
    mass: float | None = None
    com_xyz: tuple[float, float, float] | None = None
    inertia: dict[str, float] | None = None
    collisions: list[UrdfGeometry] = field(default_factory=list)
    visuals: list[UrdfGeometry] = field(default_factory=list)


@dataclass
class UrdfJoint:
    name: str
    type: str
    parent: str
    child: str
    origin_xyz: tuple[float, float, float]
    origin_rpy: tuple[float, float, float]
    axis: tuple[float, float, float]
    lower: float | None = None
    upper: float | None = None
    effort: float | None = None
    velocity: float | None = None

    @property
    def actuated(self) -> bool:
        return self.type in ("revolute", "continuous", "prismatic")


@dataclass
class UrdfModel:
    name: str
    path: Path
    links: dict[str, UrdfLink]
    joints: list[UrdfJoint]  # in file order

    # ------------------------------------------------------------------ queries

    def joint(self, name: str) -> UrdfJoint:
        for j in self.joints:
            if j.name == name:
                return j
        raise KeyError(f"joint '{name}' not found in {self.path}")

    @property
    def actuated_joints(self) -> list[UrdfJoint]:
        return [j for j in self.joints if j.actuated]

    @property
    def root_link(self) -> str:
        children = {j.child for j in self.joints}
        roots = [name for name in self.links if name not in children]
        if len(roots) != 1:
            raise ValueError(f"expected exactly one root link, found {roots}")
        return roots[0]

    @property
    def total_mass(self) -> float:
        return sum(link.mass or 0.0 for link in self.links.values())

    def parent_joint(self, link: str) -> UrdfJoint | None:
        for j in self.joints:
            if j.child == link:
                return j
        return None

    def chain_to(self, link: str) -> list[UrdfJoint]:
        """Joints from the root link to ``link`` (root first)."""
        chain = []
        current = link
        while (j := self.parent_joint(current)) is not None:
            chain.append(j)
            current = j.parent
        return list(reversed(chain))

    def link_transform(self, link: str, joint_pos: dict[str, float] | None = None) -> np.ndarray:
        """Homogeneous transform root -> ``link`` at the given joint positions (missing joints = 0)."""
        joint_pos = joint_pos or {}
        T = np.eye(4)
        for j in self.chain_to(link):
            T = T @ origin_matrix(j.origin_xyz, j.origin_rpy)
            q = joint_pos.get(j.name, 0.0)
            if j.type in ("revolute", "continuous"):
                T = T @ rotation_about_axis(j.axis, q)
            elif j.type == "prismatic":
                M = np.eye(4)
                M[:3, 3] = np.asarray(j.axis) * q
                T = T @ M
        return T


# ---------------------------------------------------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------------------------------------------------


def _floats(text: str | None, default: tuple[float, ...]) -> tuple[float, ...]:
    if text is None:
        return default
    return tuple(float(v) for v in text.split())


def _parse_origin(elem: ET.Element | None) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    if elem is None:
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
    return _floats(elem.get("xyz"), (0.0, 0.0, 0.0)), _floats(elem.get("rpy"), (0.0, 0.0, 0.0))


def _parse_geometry(elem: ET.Element) -> UrdfGeometry | None:
    xyz, rpy = _parse_origin(elem.find("origin"))
    geom = elem.find("geometry")
    if geom is None or len(geom) == 0:
        return None
    shape = geom[0]
    if shape.tag == "mesh":
        params = {"filename": shape.get("filename"), "scale": _floats(shape.get("scale"), (1.0, 1.0, 1.0))}
    elif shape.tag == "box":
        params = {"size": _floats(shape.get("size"), (0.0, 0.0, 0.0))}
    elif shape.tag == "cylinder":
        params = {"radius": float(shape.get("radius")), "length": float(shape.get("length"))}
    elif shape.tag == "sphere":
        params = {"radius": float(shape.get("radius"))}
    else:
        params = {}
    return UrdfGeometry(kind=shape.tag, origin_xyz=xyz, origin_rpy=rpy, params=params)


def load_urdf(path: str | Path) -> UrdfModel:
    path = Path(path)
    root = ET.parse(path).getroot()
    links: dict[str, UrdfLink] = {}
    for le in root.findall("link"):
        link = UrdfLink(name=le.get("name"))
        inertial = le.find("inertial")
        if inertial is not None:
            mass = inertial.find("mass")
            link.mass = float(mass.get("value")) if mass is not None else None
            link.com_xyz, _ = _parse_origin(inertial.find("origin"))
            inertia = inertial.find("inertia")
            if inertia is not None:
                link.inertia = {k: float(v) for k, v in inertia.attrib.items()}
        for ce in le.findall("collision"):
            if (g := _parse_geometry(ce)) is not None:
                link.collisions.append(g)
        for ve in le.findall("visual"):
            if (g := _parse_geometry(ve)) is not None:
                link.visuals.append(g)
        links[link.name] = link
    joints: list[UrdfJoint] = []
    for je in root.findall("joint"):
        xyz, rpy = _parse_origin(je.find("origin"))
        axis_elem = je.find("axis")
        axis = _floats(axis_elem.get("xyz") if axis_elem is not None else None, (1.0, 0.0, 0.0))
        limit = je.find("limit")

        def _lim(key: str, _limit=limit) -> float | None:
            return float(_limit.get(key)) if _limit is not None and _limit.get(key) is not None else None

        joints.append(
            UrdfJoint(
                name=je.get("name"),
                type=je.get("type"),
                parent=je.find("parent").get("link"),
                child=je.find("child").get("link"),
                origin_xyz=xyz,
                origin_rpy=rpy,
                axis=axis,
                lower=_lim("lower"),
                upper=_lim("upper"),
                effort=_lim("effort"),
                velocity=_lim("velocity"),
            )
        )
    return UrdfModel(name=root.get("name", path.stem), path=path, links=links, joints=joints)


# ---------------------------------------------------------------------------------------------------------------------
# Kinematics helpers
# ---------------------------------------------------------------------------------------------------------------------


def rpy_matrix(rpy: tuple[float, float, float]) -> np.ndarray:
    """URDF fixed-axis roll-pitch-yaw: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def origin_matrix(xyz, rpy) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = rpy_matrix(rpy)
    T[:3, 3] = xyz
    return T


def rotation_about_axis(axis, angle: float) -> np.ndarray:
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    T = np.eye(4)
    T[:3, :3] = np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)
    return T


# ---------------------------------------------------------------------------------------------------------------------
# Mesh loading (vertices only; enough for bounding geometry)
# ---------------------------------------------------------------------------------------------------------------------


def resolve_mesh_path(filename: str, urdf_path: Path) -> Path:
    """Resolve ``package://pkg/rel`` or relative mesh paths.

    ``package://pkg`` is looked up as a directory named ``pkg`` among the URDF's parent directories, falling back to
    the parent of the URDF directory (the usual ``<pkg>/urdf/robot.urdf`` layout).
    """
    if filename.startswith("package://"):
        pkg, _, rel = filename[len("package://") :].partition("/")
        for parent in urdf_path.resolve().parents:
            if parent.name == pkg:
                return parent / rel
        return urdf_path.resolve().parent.parent / rel
    if filename.startswith("file://"):
        return Path(filename[len("file://") :])
    p = Path(filename)
    return p if p.is_absolute() else (urdf_path.parent / p).resolve()


def load_mesh_vertices(path: Path) -> np.ndarray:
    """Return (N, 3) vertices of a COLLADA (.dae) or STL mesh in meters (node transforms applied for .dae)."""
    suffix = path.suffix.lower()
    if suffix == ".stl":
        return _load_stl(path)
    if suffix == ".dae":
        return _load_dae(path)
    raise ValueError(f"unsupported mesh format: {path}")


def _load_stl(path: Path) -> np.ndarray:
    data = path.read_bytes()
    if len(data) >= 84:
        (n,) = struct.unpack_from("<I", data, 80)
        if 84 + n * 50 == len(data):
            tri = np.frombuffer(data, dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]), offset=84)
            return tri["v"].reshape(-1, 3).astype(float)
    verts = re.findall(rb"vertex\s+(\S+)\s+(\S+)\s+(\S+)", data)
    return np.array([[float(x) for x in v] for v in verts])


def _load_dae(path: Path) -> np.ndarray:
    root = ET.parse(path).getroot()
    ns = {"c": root.tag.split("}")[0].strip("{")} if root.tag.startswith("{") else {"c": ""}
    q = (lambda tag: f"c:{tag}") if ns["c"] else (lambda tag: tag)
    unit = root.find(f"{q('asset')}/{q('unit')}", ns)
    meter = float(unit.get("meter", "1")) if unit is not None else 1.0

    geometries: dict[str, np.ndarray] = {}
    for geom in root.iter(f"{{{ns['c']}}}geometry" if ns["c"] else "geometry"):
        mesh = geom.find(q("mesh"), ns)
        if mesh is None:
            continue
        vertices = mesh.find(q("vertices"), ns)
        src_id = None
        if vertices is not None:
            for inp in vertices.findall(q("input"), ns):
                if inp.get("semantic") == "POSITION":
                    src_id = inp.get("source", "").lstrip("#")
        for src in mesh.findall(q("source"), ns):
            if src.get("id") == src_id:
                arr = src.find(q("float_array"), ns)
                geometries[geom.get("id")] = np.array(arr.text.split(), dtype=float).reshape(-1, 3)

    # apply node transforms of the visual scene (matrix only; exporters we target write <matrix>)
    out = []
    scene_nodes = root.iter(f"{{{ns['c']}}}node" if ns["c"] else "node")
    used = False
    for node in scene_nodes:
        inst = node.find(q("instance_geometry"), ns)
        if inst is None:
            continue
        gid = inst.get("url", "").lstrip("#")
        if gid not in geometries:
            continue
        M = np.eye(4)
        mat = node.find(q("matrix"), ns)
        if mat is not None:
            M = np.array(mat.text.split(), dtype=float).reshape(4, 4)
        v = geometries[gid]
        out.append((np.c_[v, np.ones(len(v))] @ M.T)[:, :3])
        used = True
    if not used:
        out = list(geometries.values())
    return np.vstack(out) * meter


# ---------------------------------------------------------------------------------------------------------------------
# Wheel geometry and nominal stance
# ---------------------------------------------------------------------------------------------------------------------


@dataclass
class WheelGeometry:
    link: str
    joint: str
    radius: float
    width: float
    axial_offset: float  # centre of the tyre along the joint axis, in the wheel link frame [m]
    source: str  # "mesh:<file>" | "cylinder" | ...


def _geometry_points(geom: UrdfGeometry, urdf_path: Path) -> np.ndarray:
    if geom.kind == "mesh":
        pts = load_mesh_vertices(resolve_mesh_path(geom.params["filename"], urdf_path)) * np.asarray(
            geom.params["scale"]
        )
    elif geom.kind == "cylinder":
        r, h = geom.params["radius"], geom.params["length"]
        t = np.linspace(0, 2 * math.pi, 64, endpoint=False)
        ring = np.c_[r * np.cos(t), r * np.sin(t)]
        pts = np.vstack([np.c_[ring, np.full(64, -h / 2)], np.c_[ring, np.full(64, h / 2)]])
    elif geom.kind == "sphere":
        r = geom.params["radius"]
        pts = np.array([[r, 0, 0], [-r, 0, 0], [0, r, 0], [0, -r, 0], [0, 0, r], [0, 0, -r]])
    elif geom.kind == "box":
        sx, sy, sz = (s / 2 for s in geom.params["size"])
        pts = np.array([[x, y, z] for x in (-sx, sx) for y in (-sy, sy) for z in (-sz, sz)])
    else:
        raise ValueError(f"unsupported geometry {geom.kind}")
    T = origin_matrix(geom.origin_xyz, geom.origin_rpy)
    return (np.c_[pts, np.ones(len(pts))] @ T.T)[:, :3]


def wheel_geometry(model: UrdfModel, joint_name: str) -> WheelGeometry:
    """Radius/width of the wheel driven by ``joint_name``, from its collision geometry (link frame)."""
    joint = model.joint(joint_name)
    link = model.links[joint.child]
    if not link.collisions:
        raise ValueError(f"wheel link '{link.name}' has no collision geometry")
    pts = np.vstack([_geometry_points(g, model.path) for g in link.collisions])
    axis = np.asarray(joint.axis, dtype=float)
    axis /= np.linalg.norm(axis)
    along = pts @ axis
    radial = np.linalg.norm(pts - np.outer(along, axis), axis=1)
    kinds = ",".join(f"mesh:{Path(g.params['filename']).name}" if g.kind == "mesh" else g.kind for g in link.collisions)
    return WheelGeometry(
        link=link.name,
        joint=joint_name,
        radius=float(radial.max()),
        width=float(along.max() - along.min()),
        axial_offset=float((along.max() + along.min()) / 2),
        source=kinds,
    )


def nominal_base_height(
    model: UrdfModel,
    joint_pos: dict[str, float],
    wheel_joints: list[str],
    wheel_radius: float,
    wheel_geometries: dict[str, WheelGeometry] | None = None,
) -> float:
    """Base height above flat ground when the lowest wheel touches it (base kept level).

    Without ``wheel_geometries`` the tyre is a disc of ``wheel_radius`` centred on the joint origin. With them, the
    tyre is a cylinder (radius, width, axial offset along the axle), whose lowest point accounts for camber: a
    tilted axle lifts/lowers the offset tyre and lets the rim edge touch first.
    """
    heights = []
    for name in wheel_joints:
        joint = model.joint(name)
        T = model.link_transform(joint.child, joint_pos)
        geo = (wheel_geometries or {}).get(name)
        if geo is None:
            heights.append(-T[2, 3] + wheel_radius)
            continue
        axis = T[:3, :3] @ (np.asarray(joint.axis, dtype=float) / np.linalg.norm(joint.axis))
        centre_z = T[2, 3] + geo.axial_offset * axis[2]
        lowest = centre_z - geo.radius * math.sqrt(max(0.0, 1.0 - axis[2] ** 2)) - 0.5 * geo.width * abs(axis[2])
        heights.append(-lowest)
    return float(max(heights))


@dataclass
class MergedBody:
    """A rigid body after fixed joints are merged (what the simulator sees with ``merge_fixed_joints``)."""

    name: str
    members: list[str]
    mass: float
    com: np.ndarray  # (3,) in the body frame
    inertia: np.ndarray  # (3, 3) about the CoM, body-frame axes


def _inertia_matrix(values: dict[str, float] | None) -> np.ndarray:
    if not values:
        return np.zeros((3, 3))
    g = lambda k: values.get(k, 0.0)  # noqa: E731
    return np.array([[g("ixx"), g("ixy"), g("ixz")], [g("ixy"), g("iyy"), g("iyz")], [g("ixz"), g("iyz"), g("izz")]])


def merged_bodies(model: UrdfModel, keep: set[str] | None = None) -> dict[str, MergedBody]:
    """Combine links connected by fixed joints into their nearest kept ancestor (mass, CoM, inertia about the CoM).

    ``keep`` = links that stay separate rigid bodies. Default: the root and every child of a non-fixed joint (full
    merge). Pass the simulator's body names to mirror an importer that keeps some fixed links -- Isaac Sim 5.1 keeps
    fixed links that have an ``<inertial>`` and only merges massless ones.
    """
    fixed_parent = {j.child: j for j in model.joints if j.type == "fixed"}
    if keep is not None:
        fixed_parent = {c: j for c, j in fixed_parent.items() if c not in keep}
    inertial_rpy: dict[str, tuple] = {}
    root = ET.parse(model.path).getroot()
    for le in root.findall("link"):
        _, rpy = _parse_origin(le.find("inertial/origin"))
        inertial_rpy[le.get("name")] = rpy

    groups: dict[str, list[tuple[str, np.ndarray]]] = {}
    for name in model.links:
        T = np.eye(4)
        current = name
        while current in fixed_parent:
            j = fixed_parent[current]
            T = origin_matrix(j.origin_xyz, j.origin_rpy) @ T
            current = j.parent
        groups.setdefault(current, []).append((name, T))

    out = {}
    for body, members in groups.items():
        masses, coms, rots, inertias = [], [], [], []
        for name, T in members:
            link = model.links[name]
            if not link.mass:
                continue
            masses.append(link.mass)
            coms.append((T @ np.r_[np.asarray(link.com_xyz or (0, 0, 0)), 1.0])[:3])
            rots.append(T[:3, :3] @ rpy_matrix(inertial_rpy.get(name, (0.0, 0.0, 0.0))))
            inertias.append(_inertia_matrix(link.inertia))
        total = float(sum(masses))
        com = np.sum([m * c for m, c in zip(masses, coms, strict=True)], axis=0) / total if total > 0 else np.zeros(3)
        inertia = np.zeros((3, 3))
        for m, c, R, inertia_i in zip(masses, coms, rots, inertias, strict=True):
            d = c - com
            inertia += R @ inertia_i @ R.T + m * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
        out[body] = MergedBody(body, [n for n, _ in members], total, com, inertia)
    return out


def joint_axis_inertia(model: UrdfModel, joint_name: str, joint_pos: dict[str, float] | None = None) -> float:
    """Inertia of everything the joint moves (its child subtree, rigid at ``joint_pos``) about the joint axis."""
    joint = model.joint(joint_name)
    T_joint_inv = np.linalg.inv(model.link_transform(joint.child, joint_pos))  # joint frame = child frame
    axis = np.asarray(joint.axis, dtype=float)
    axis /= np.linalg.norm(axis)
    inertial_rpy = {
        le.get("name"): _parse_origin(le.find("inertial/origin"))[1]
        for le in ET.parse(model.path).getroot().findall("link")
    }
    subtree, frontier = [], [joint.child]
    while frontier:
        link = frontier.pop()
        subtree.append(link)
        frontier += [j.child for j in model.joints if j.parent == link]
    total = 0.0
    for name in subtree:
        link = model.links[name]
        if not link.mass:
            continue
        T = T_joint_inv @ model.link_transform(name, joint_pos)
        R = T[:3, :3] @ rpy_matrix(inertial_rpy.get(name, (0.0, 0.0, 0.0)))
        com = (T @ np.r_[np.asarray(link.com_xyz or (0, 0, 0)), 1.0])[:3]
        d = com - np.dot(com, axis) * axis  # CoM offset perpendicular to the axis
        total += float(axis @ (R @ _inertia_matrix(link.inertia) @ R.T) @ axis + link.mass * np.dot(d, d))
    return total


def explicit_damping_ratio(inertia: float, damping: float, armature: float, dt: float) -> float:
    """Kd * dt / (I + armature). An explicit (Python-side) damping term is unstable at >= 2 and rings above ~1."""
    return damping * dt / (inertia + armature)


# ---------------------------------------------------------------------------------------------------------------------
# Asset preparation (absolute mesh paths, optional analytic wheel colliders)
# ---------------------------------------------------------------------------------------------------------------------


def _axis_to_cylinder_rpy(axis) -> tuple[float, float, float]:
    """rpy rotating the URDF cylinder axis (+z) onto ``axis``."""
    a = np.asarray(axis, dtype=float)
    a /= np.linalg.norm(a)
    if np.allclose(a, [0, 0, 1]) or np.allclose(a, [0, 0, -1]):
        return (0.0, 0.0, 0.0)
    if np.allclose(np.abs(a), [0, 1, 0]):
        return (math.pi / 2, 0.0, 0.0)
    if np.allclose(np.abs(a), [1, 0, 0]):
        return (0.0, math.pi / 2, 0.0)
    raise ValueError(f"only principal wheel axes are supported, got {axis}")


def prepare_urdf(
    urdf_path: str | Path,
    out_dir: str | Path,
    wheel_joints: list[str] | None = None,
    cylinder_wheels: bool = False,
) -> Path:
    """Write an import-ready copy of ``urdf_path``.

    * mesh filenames are rewritten to absolute paths (``package://`` is not resolved by every importer);
    * with ``cylinder_wheels`` the wheel collision meshes are replaced by analytic cylinders fitted to the mesh
      (radius, width and axial offset from :func:`wheel_geometry`), which roll much more smoothly than the
      convex hull of a tessellated tyre.

    The output file name contains a content hash so repeated calls are cheap and never clobber each other.
    """
    urdf_path = Path(urdf_path).resolve()
    model = load_urdf(urdf_path)
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    for mesh in root.iter("mesh"):
        mesh.set("filename", str(resolve_mesh_path(mesh.get("filename"), urdf_path)))
    if cylinder_wheels:
        for jname in wheel_joints or []:
            geo = wheel_geometry(model, jname)
            joint = model.joint(jname)
            link_elem = next(le for le in root.findall("link") if le.get("name") == geo.link)
            for ce in link_elem.findall("collision"):
                link_elem.remove(ce)
            ce = ET.SubElement(link_elem, "collision")
            center = np.asarray(joint.axis, dtype=float) / np.linalg.norm(joint.axis) * geo.axial_offset
            rpy = _axis_to_cylinder_rpy(joint.axis)
            ET.SubElement(
                ce,
                "origin",
                xyz=" ".join(f"{v:.6f}" for v in center),
                rpy=" ".join(f"{v:.6f}" for v in rpy),
            )
            g = ET.SubElement(ce, "geometry")
            ET.SubElement(g, "cylinder", radius=f"{geo.radius:.6f}", length=f"{geo.width:.6f}")
    text = ET.tostring(root, encoding="unicode")
    digest = hashlib.sha1((text + str(cylinder_wheels)).encode()).hexdigest()[:10]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{urdf_path.stem}{'_cylwheels' if cylinder_wheels else ''}_{digest}.urdf"
    if not out.exists():
        out.write_text('<?xml version="1.0"?>\n' + text)
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Audit report
# ---------------------------------------------------------------------------------------------------------------------


def classify_joints(model: UrdfModel, wheel_pattern: str = r".*") -> tuple[list[str], list[str]]:
    """Split actuated joints into (legs, wheels): continuous joints matching ``wheel_pattern`` are wheels."""
    legs, wheels = [], []
    for j in model.actuated_joints:
        (wheels if j.type == "continuous" and re.fullmatch(wheel_pattern, j.name) else legs).append(j.name)
    return legs, wheels


def audit(model: UrdfModel, stance: dict[str, float] | None = None, wheel_pattern: str = r".*") -> dict:
    """Machine-readable audit: joints, links, wheels, nominal height and warnings."""
    legs, wheels = classify_joints(model, wheel_pattern)
    wheel_geo = []
    for w in wheels:
        try:
            wheel_geo.append(wheel_geometry(model, w))
        except (OSError, ValueError) as err:  # mesh missing etc.
            wheel_geo.append(WheelGeometry(model.joint(w).child, w, float("nan"), float("nan"), float("nan"), str(err)))
    warnings: list[str] = []
    for link in model.links.values():
        if link.collisions and link.mass is None:
            warnings.append(f"link '{link.name}' has collision geometry but no inertial (merged when fixed)")
    for g in wheel_geo:
        if g.source.startswith("mesh"):
            warnings.append(
                f"wheel '{g.link}' collides through a mesh ({g.source}); URDF import uses convex_hull -> "
                "polygonal tyre. Consider cylinder_wheels=True."
            )
    radius = float(np.nanmean([g.radius for g in wheel_geo])) if wheel_geo else float("nan")
    height = None
    if stance and wheels and not math.isnan(radius):
        height = nominal_base_height(model, stance, wheels, radius)
    return {
        "name": model.name,
        "path": str(model.path),
        "root_link": model.root_link,
        "total_mass": model.total_mass,
        "joints": model.joints,
        "links": model.links,
        "leg_joints": legs,
        "wheel_joints": wheels,
        "wheels": wheel_geo,
        "wheel_radius": radius,
        "stance": stance,
        "nominal_base_height": height,
        "warnings": warnings,
    }


def _fmt(v, digits=4) -> str:
    if v is None:
        return "–"
    if isinstance(v, float):
        return f"{v:.{digits}g}"
    return str(v)


def audit_markdown(report: dict) -> str:
    lines = [
        f"# URDF audit: `{report['name']}`",
        "",
        "Generated by `scripts/urdf_audit.py`. Do not edit by hand.",
        "",
        f"- Source: `{Path(report['path']).name}`",
        f"- Root link: `{report['root_link']}`",
        f"- Total mass: **{report['total_mass']:.3f} kg**",
        f"- Actuated joints: {len(report['leg_joints']) + len(report['wheel_joints'])} "
        f"(legs {len(report['leg_joints'])}, wheels {len(report['wheel_joints'])})",
        "",
        "## Actuated joints (URDF order)",
        "",
        "| # | joint | type | parent → child | axis | lower | upper | effort [Nm] | vel [rad/s] | group |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    wheels = set(report["wheel_joints"])
    for i, j in enumerate([j for j in report["joints"] if j.actuated]):
        lines.append(
            f"| {i} | {j.name} | {j.type} | {j.parent} → {j.child} | {' '.join(_fmt(a, 3) for a in j.axis)} | "
            f"{_fmt(j.lower)} | {_fmt(j.upper)} | {_fmt(j.effort)} | {_fmt(j.velocity)} | "
            f"{'wheel' if j.name in wheels else 'leg'} |"
        )
    lines += [
        "",
        "## Wheels",
        "",
        "| joint | link | radius [m] | width [m] | axial offset [m] | collision |",
        "|---|---|---|---|---|---|",
    ]
    for g in report["wheels"]:
        lines.append(
            f"| {g.joint} | {g.link} | {_fmt(g.radius)} | {_fmt(g.width)} | {_fmt(g.axial_offset, 3)} | {g.source} |"
        )
    if report["nominal_base_height"] is not None:
        stance = ", ".join(f"{k}={v}" for k, v in report["stance"].items())
        lines += ["", f"Nominal base height at stance ({stance}): **{report['nominal_base_height']:.4f} m**"]
    lines += ["", "## Links", "", "| link | mass [kg] | CoM xyz | #collision |", "|---|---|---|---|"]
    for link in report["links"].values():
        com = " ".join(_fmt(c, 3) for c in link.com_xyz) if link.com_xyz else "–"
        lines.append(f"| {link.name} | {_fmt(link.mass)} | {com} | {len(link.collisions)} |")
    lines += ["", "## Fixed joints", ""]
    for j in report["joints"]:
        if not j.actuated:
            lines.append(f"- `{j.name}`: {j.parent} → {j.child}")
    lines += ["", "## Warnings", ""]
    lines += [f"- {w}" for w in report["warnings"]] or ["- none"]
    return "\n".join(lines) + "\n"
