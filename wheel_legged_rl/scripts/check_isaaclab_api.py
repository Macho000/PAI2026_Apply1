"""[No Isaac] Static check of this package against an Isaac Lab / rsl_rl source checkout.

    git clone --depth 1 --branch v2.3.0 https://github.com/isaac-sim/IsaacLab.git /tmp/IsaacLab
    pip download --no-deps rsl-rl-lib==3.0.1 -d /tmp/rsl && unzip -o /tmp/rsl/*.whl -d /tmp/rsl
    python scripts/check_isaaclab_api.py --isaaclab /tmp/IsaacLab --rsl_rl /tmp/rsl

Checks, without running the simulator:
  1. every ``from isaaclab* / rsl_rl import X`` resolves to a definition (following package re-exports);
  2. every ``module_alias.X`` used on an imported Isaac module exists (``ilmdp.track_lin_vel_xy_exp`` ...);
  3. keyword arguments of config-class constructors are fields of that class or its bases;
  4. ``params={...}`` keys of manager terms (``RewTerm(func=..., params=...)`` ...) are parameters of ``func``;
  5. attributes read from Isaac objects (``robot.data.<x>``, ``env.<manager>.<x>`` ...) exist (curated list).
Exit code 1 on any finding.
"""

from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OWN_PKG = "wheel_legged_rl"
EXTERNAL = ("isaaclab", "isaaclab_rl", "isaaclab_tasks", "rsl_rl")
TERM_CFGS = {"RewardTermCfg", "ObservationTermCfg", "EventTermCfg", "TerminationTermCfg", "CurriculumTermCfg"}


@dataclass
class Symbol:
    module: str
    name: str
    node: ast.AST | None  # FunctionDef / ClassDef / Assign / None for modules
    kind: str  # "function" | "class" | "variable" | "module"


@dataclass
class Index:
    roots: dict[str, Path]  # top-level package -> directory containing it
    _cache: dict[str, ast.Module | None] = field(default_factory=dict)

    def module_file(self, module: str) -> Path | None:
        top = module.split(".")[0]
        base = self.roots.get(top)
        if base is None:
            return None
        p = base.joinpath(*module.split("."))
        if (p / "__init__.py").is_file():
            return p / "__init__.py"
        if p.with_suffix(".py").is_file():
            return p.with_suffix(".py")
        return None

    def tree(self, module: str) -> ast.Module | None:
        if module not in self._cache:
            f = self.module_file(module)
            self._cache[module] = ast.parse(f.read_text()) if f else None
        return self._cache[module]

    def is_package(self, module: str) -> bool:
        f = self.module_file(module)
        return f is not None and f.name == "__init__.py"

    def resolve(self, module: str, name: str, depth: int = 0) -> Symbol | None:
        if depth > 12:
            return None
        tree = self.tree(module)
        if tree is None:
            return None
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
                return Symbol(module, name, node, "function")
            if isinstance(node, ast.ClassDef) and node.name == name:
                return Symbol(module, name, node, "class")
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                if any(isinstance(t, ast.Name) and t.id == name for t in targets):
                    return Symbol(module, name, node, "variable")
        for node in ast.walk(tree):  # re-exports, including inside `if TYPE_CHECKING` / try blocks
            if isinstance(node, ast.ImportFrom):
                src = self._abs(module, node)
                for alias in node.names:
                    if alias.name == "*":
                        if (s := self.resolve(src, name, depth + 1)) is not None:
                            return s
                    elif (alias.asname or alias.name) == name:
                        if (s := self.resolve(src, alias.name, depth + 1)) is not None:
                            return s
                        if self.module_file(f"{src}.{alias.name}"):
                            return Symbol(f"{src}.{alias.name}", alias.name, None, "module")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if (alias.asname or alias.name.split(".")[0]) == name and self.module_file(alias.name):
                        return Symbol(alias.name, name, None, "module")
        if self.is_package(module) and self.module_file(f"{module}.{name}"):
            return Symbol(f"{module}.{name}", name, None, "module")
        return None

    def _abs(self, module: str, node: ast.ImportFrom) -> str:
        if not node.level:
            return node.module or ""
        parts = module.split(".")
        base = parts if self.is_package(module) else parts[:-1]
        base = base[: len(base) - (node.level - 1)] if node.level > 1 else base
        return ".".join(base + ([node.module] if node.module else []))

    # ---------------------------------------------------------------- class introspection

    def class_fields(self, sym: Symbol, depth: int = 0) -> set[str] | None:
        """Annotated / assigned class attributes + nested classes + methods, including bases. None = unknown base."""
        if depth > 10 or not isinstance(sym.node, ast.ClassDef):
            return None
        out: set[str] = set()
        for stmt in sym.node.body:
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                out.add(stmt.target.id)
            elif isinstance(stmt, ast.Assign):
                out |= {t.id for t in stmt.targets if isinstance(t, ast.Name)}
            elif isinstance(stmt, (ast.ClassDef, ast.FunctionDef)):
                out.add(stmt.name)
        for base in sym.node.bases:
            name = base.id if isinstance(base, ast.Name) else base.attr if isinstance(base, ast.Attribute) else None
            if name in (None, "object", "ABC", "Generic"):
                continue
            bsym = self.resolve(sym.module, name)
            if bsym is None:
                # base imported under an alias chain (e.g. converters.UrdfConverterCfg)
                if isinstance(base, ast.Attribute) and isinstance(base.value, ast.Name):
                    mod = self.resolve(sym.module, base.value.id)
                    bsym = self.resolve(mod.module, base.attr) if mod and mod.kind == "module" else None
            if bsym is None:
                return None
            bf = self.class_fields(bsym, depth + 1)
            if bf is None:
                return None
            out |= bf
        return out

    def nested(self, sym: Symbol, name: str) -> Symbol | None:
        if isinstance(sym.node, ast.ClassDef):
            for stmt in sym.node.body:
                if isinstance(stmt, ast.ClassDef) and stmt.name == name:
                    return Symbol(sym.module, name, stmt, "class")
            for base in sym.node.bases:
                bname = base.id if isinstance(base, ast.Name) else getattr(base, "attr", None)
                if bname and (b := self.resolve(sym.module, bname)) and (n := self.nested(b, name)):
                    return n
        return None

    def init_params(self, sym: Symbol, depth: int = 0) -> tuple[set[str], bool] | None:
        """``__init__`` parameters of a plain class (searching bases); None if it has none (-> configclass)."""
        if depth > 10 or not isinstance(sym.node, ast.ClassDef):
            return None
        is_cfg = any(
            (isinstance(d, ast.Name) and d.id in ("configclass", "dataclass"))
            or (isinstance(d, ast.Call) and getattr(d.func, "id", None) in ("configclass", "dataclass"))
            for d in sym.node.decorator_list
        )
        if is_cfg:
            return None
        init = next((s for s in sym.node.body if isinstance(s, ast.FunctionDef) and s.name == "__init__"), None)
        if init is not None:
            a = init.args
            return {x.arg for x in a.posonlyargs + a.args + a.kwonlyargs} - {"self"}, a.kwarg is not None
        for base in sym.node.bases:
            name = base.id if isinstance(base, ast.Name) else getattr(base, "attr", None)
            if name and (b := self.resolve(sym.module, name)) and (r := self.init_params(b, depth + 1)):
                return r
        return None

    def callable_params(self, sym: Symbol) -> tuple[set[str], bool] | None:
        """Parameter names of a function, or of a ManagerTermBase class' ``__call__`` (minus env / env_ids)."""
        fn = sym.node
        if isinstance(fn, ast.ClassDef):
            fn = next((s for s in fn.body if isinstance(s, ast.FunctionDef) and s.name == "__call__"), None)
        if not isinstance(fn, ast.FunctionDef):
            return None
        a = fn.args
        names = {x.arg for x in a.posonlyargs + a.args + a.kwonlyargs} - {"self", "env", "env_ids"}
        return names, a.kwarg is not None


def dotted(node: ast.AST) -> list[str] | None:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        return [node.id] + list(reversed(parts))
    return None


class FileChecker(ast.NodeVisitor):
    def __init__(self, index: Index, path: Path, module: str):
        self.index, self.path, self.module = index, path, module
        self.names: dict[str, Symbol] = {}
        self.findings: list[str] = []
        self.checked = 0

    def err(self, node: ast.AST, msg: str):
        self.findings.append(f"{self.path.relative_to(ROOT)}:{getattr(node, 'lineno', 0)}: {msg}")

    # ---------------------------------------------------------------- imports

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            top = alias.name.split(".")[0]
            if top in EXTERNAL or top == OWN_PKG:
                self.checked += 1
                if not self.index.module_file(alias.name):
                    self.err(node, f"module '{alias.name}' not found")
                elif alias.asname:
                    self.names[alias.asname] = Symbol(alias.name, alias.asname, None, "module")

    def visit_ImportFrom(self, node: ast.ImportFrom):
        src = self.index._abs(self.module, node) if node.level else (node.module or "")
        if src.split(".")[0] not in EXTERNAL + (OWN_PKG,):
            return
        for alias in node.names:
            self.checked += 1
            sym = self.index.resolve(src, alias.name)
            if sym is None:
                self.err(node, f"'{alias.name}' not found in '{src}'")
            else:
                self.names[alias.asname or alias.name] = sym

    # ---------------------------------------------------------------- attribute / call resolution

    def lookup(self, node: ast.AST) -> Symbol | None:
        parts = dotted(node)
        if not parts or parts[0] not in self.names:
            return None
        sym = self.names[parts[0]]
        for attr in parts[1:]:
            if sym.kind == "module":
                nxt = self.index.resolve(sym.module, attr)
                if nxt is None:
                    self.err(node, f"'{attr}' not found in module '{sym.module}'")
                    return None
                sym = nxt
            elif sym.kind == "class":
                nxt = self.index.nested(sym, attr)
                if nxt is None:
                    return None  # class attribute / method: not checked
                sym = nxt
            else:
                return None
        return sym

    def visit_Attribute(self, node: ast.Attribute):
        if isinstance(node.ctx, ast.Load):
            parts = dotted(node)
            if parts and parts[0] in self.names and self.names[parts[0]].kind == "module":
                self.checked += 1
                self.lookup(node)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        sym = self.lookup(node.func)
        if sym is not None and sym.kind == "class" and sym.module.split(".")[0] in EXTERNAL:
            init = self.index.init_params(sym)
            if init is not None:
                names, has_kwargs = init
                fields = None if has_kwargs else names
                what = "parameter"
            else:
                fields = self.index.class_fields(sym)
                what = "field"
            if fields is not None:
                for kw in node.keywords:
                    if kw.arg is not None:
                        self.checked += 1
                        if kw.arg not in fields:
                            self.err(node, f"{sym.name}(...) has no {what} '{kw.arg}'")
            if sym.name in TERM_CFGS:
                self.check_term_params(node)
        self.generic_visit(node)

    def check_term_params(self, node: ast.Call):
        kws = {kw.arg: kw.value for kw in node.keywords}
        func, params = kws.get("func"), kws.get("params")
        if func is None or not isinstance(params, ast.Dict):
            return
        fsym = self.lookup(func)
        if fsym is None:
            return
        sig = self.index.callable_params(fsym)
        if sig is None:
            return
        names, has_kwargs = sig
        for key in params.keys:
            if isinstance(key, ast.Constant) and not has_kwargs:
                self.checked += 1
                if key.value not in names:
                    self.err(node, f"'{fsym.name}' has no parameter '{key.value}' (has {sorted(names)})")


# Attributes read from live Isaac objects: (reference file, attribute). Kept explicit and short on purpose.
ATTRIBUTES = {
    "isaaclab/source/isaaclab/isaaclab/assets/articulation/articulation_data.py": [
        "joint_names",
        "body_names",
        "default_joint_pos",
        "default_joint_vel",
        "default_root_state",
        "default_joint_stiffness",
        "default_joint_damping",
        "applied_torque",
        "joint_pos",
        "joint_vel",
        "joint_pos_limits",
        "root_lin_vel_b",
        "root_ang_vel_b",
        "root_quat_w",
        "root_pos_w",
        "body_link_lin_vel_w",
        "body_link_ang_vel_w",
        "body_link_quat_w",
    ],
    "isaaclab/source/isaaclab/isaaclab/sensors/contact_sensor/contact_sensor_data.py": [
        "net_forces_w_history",
        "net_forces_w",
    ],
    "isaaclab/source/isaaclab/isaaclab/sensors/ray_caster/ray_caster_data.py": ["pos_w", "ray_hits_w"],
    "isaaclab/source/isaaclab/isaaclab/managers/action_manager.py": [
        "action",
        "prev_action",
        "active_terms",
        "_terms",
        "total_action_dim",
    ],
    "isaaclab/source/isaaclab/isaaclab/managers/observation_manager.py": [
        "active_terms",
        "_group_obs_term_cfgs",
        "group_obs_term_dim",
    ],
    "isaaclab/source/isaaclab/isaaclab/managers/reward_manager.py": ["_episode_sums", "get_term_cfg"],
    "isaaclab/source/isaaclab/isaaclab/managers/command_manager.py": ["get_term", "get_command"],
    "isaaclab/source/isaaclab/isaaclab/envs/manager_based_rl_env.py": [
        "common_step_counter",
        "max_episode_length",
        "max_episode_length_s",
        "step",
    ],
    "isaaclab/source/isaaclab/isaaclab/envs/manager_based_env.py": ["step_dt", "device", "num_envs", "scene"],
    "isaaclab/source/isaaclab/isaaclab/envs/mdp/commands/velocity_command.py": ["vel_command_b"],
    "isaaclab/source/isaaclab/isaaclab/envs/mdp/actions/joint_actions.py": [
        "_joint_ids",
        "_scale",
        "_offset",
        "_clip",
        "action_dim",
    ],
    "isaaclab/source/isaaclab/isaaclab/assets/articulation/articulation.py": [
        "find_joints",
        "find_bodies",
        "root_physx_view",
        "write_joint_state_to_sim",
        "write_root_pose_to_sim",
        "write_root_velocity_to_sim",
        "set_joint_position_target",
        "set_joint_velocity_target",
        "write_data_to_sim",
    ],
    # PhysX tensor-API view methods live in Isaac Sim; check that Isaac Lab itself calls them the same way
    "isaaclab/source/isaaclab/isaaclab/**/*.py": [
        "root_physx_view.get_masses",
        "root_physx_view.get_coms",
        "root_physx_view.get_inertias",
        "root_physx_view.get_dof_stiffnesses",
        "root_physx_view.get_dof_dampings",
    ],
    "isaaclab/source/isaaclab/isaaclab/terrains/terrain_importer.py": ["terrain_levels"],
    "isaaclab/source/isaaclab/isaaclab/actuators/actuator_base.py": [
        "_parse_joint_parameter",
        "computed_effort",
        "applied_effort",
        "stiffness",
        "_num_envs",
    ],
    "isaaclab/source/isaaclab_rl/isaaclab_rl/rsl_rl/vecenv_wrapper.py": ["get_observations", "unwrapped"],
    "rsl/rsl_rl/runners/on_policy_runner.py": [
        "load",
        "get_inference_policy",
        "current_learning_iteration",
        "add_git_repo_to_log",
        "learn",
    ],
    "rsl/rsl_rl/modules/actor_critic.py": ["actor_obs_normalizer", "actor"],
}


def check_attributes(isaaclab: Path, rsl: Path) -> tuple[list[str], int]:
    findings, n = [], 0
    for rel, attrs in ATTRIBUTES.items():
        base = rsl if rel.startswith("rsl/") else isaaclab
        pattern = rel.split("/", 1)[1]
        paths = sorted(base.glob(pattern)) if "*" in pattern else [base / pattern]
        if not paths or not all(p.is_file() for p in paths):
            findings.append(f"reference file missing: {base / pattern}")
            continue
        text = "\n".join(p.read_text() for p in paths)
        for a in attrs:
            n += 1
            if a not in text:
                findings.append(f"attribute '{a}' not found in {rel}")
    return findings, n


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--isaaclab", type=Path, required=True, help="Isaac Lab checkout (v2.3.0)")
    parser.add_argument("--rsl_rl", type=Path, required=True, help="directory containing the rsl_rl package")
    args = parser.parse_args()

    src = args.isaaclab / "source"
    index = Index(
        roots={
            "isaaclab": src / "isaaclab",
            "isaaclab_rl": src / "isaaclab_rl",
            "isaaclab_tasks": src / "isaaclab_tasks",
            "rsl_rl": args.rsl_rl,
            OWN_PKG: ROOT / "source" / "wheel_legged_rl",
        }
    )
    files = sorted((ROOT / "source" / "wheel_legged_rl" / OWN_PKG).rglob("*.py")) + sorted(
        (ROOT / "scripts").glob("*.py")
    )
    findings, checked = [], 0
    for f in files:
        if f.parent.name == "scripts":
            module = f"scripts.{f.stem}"
        else:
            rel = f.relative_to(ROOT / "source" / "wheel_legged_rl").with_suffix("")
            module = ".".join(p for p in rel.parts if p != "__init__")
        fc = FileChecker(index, f, module)
        fc.visit(ast.parse(f.read_text()))
        findings += fc.findings
        checked += fc.checked
    attr_findings, n_attr = check_attributes(args.isaaclab, args.rsl_rl)
    findings += attr_findings
    checked += n_attr
    for line in findings:
        print(f"[FAIL] {line}")
    print(f"{checked} references checked in {len(files)} files, {len(findings)} problem(s)")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
