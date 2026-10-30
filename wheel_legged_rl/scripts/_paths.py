"""Make ``wheel_legged_rl`` importable from a source checkout (no ``pip install -e`` needed for pure scripts)."""

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = PROJECT_ROOT / "source" / "wheel_legged_rl"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

ROBOTS = {"go2w": "wheel_legged_rl.robots.go2w:GO2W_SPEC"}


def _resolve_project_path(path: str, exists) -> Path | None:
    p = Path(path).expanduser()
    if exists(p):
        return p.resolve()
    parts = p.parts
    if PROJECT_ROOT.name in parts:
        i = len(parts) - 1 - parts[::-1].index(PROJECT_ROOT.name)
        q = PROJECT_ROOT.joinpath(*parts[i + 1 :])
        if exists(q):
            return q
    return None


def resolve_project_file(path: str) -> str | None:
    """Absolute path of an existing file, also accepting a path of another checkout of this project (e.g. a host
    path ``/home/<user>/quadra/wheel_legged_rl/logs/...`` used inside the Docker container, mounted at /quadra).
    Returns None if the file cannot be found."""
    p = _resolve_project_path(path, Path.is_file)
    return str(p) if p else None


def resolve_checkpoint(spec: str, log_root: str | Path | None = None) -> str:
    """Checkpoint to initialise a run from: a file, a run directory (-> its ``best.pt``) or, with ``log_root``, a
    run-name pattern (``re.search`` on the run folder names) -> ``best.pt`` of the latest matching run.
    Raises FileNotFoundError with a hint when there is no such checkpoint."""
    hint = "pick the best checkpoint with evaluate.py --select_best, or pass a model_*.pt file"
    f = resolve_project_file(spec)
    if f:
        return f
    run = _resolve_project_path(spec, Path.is_dir)
    if run is None and log_root is not None and Path(log_root).is_dir():
        runs = sorted(d for d in Path(log_root).iterdir() if d.is_dir() and re.search(spec, d.name))
        run = runs[-1] if runs else None  # run folders start with a timestamp
        if run is None:
            raise FileNotFoundError(f"no run matching '{spec}' in {log_root}")
    if run is None:
        raise FileNotFoundError(f"checkpoint not found: {spec}")
    best = run / "best.pt"
    if not best.is_file():
        raise FileNotFoundError(f"{run} has no best.pt: {hint}")
    return str(best)


def load_spec(name: str):
    import importlib

    module, attr = ROBOTS[name].split(":")
    return getattr(importlib.import_module(module), attr)


def project_relative(path: str) -> str:
    """``path`` relative to the project root if it lies inside it (same string on the host and in the container),
    else the absolute path."""
    p = Path(path).resolve()
    try:
        return str(p.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(p)
