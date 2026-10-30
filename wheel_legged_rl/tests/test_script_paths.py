import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import _paths  # noqa: E402


def test_resolve_project_file_maps_other_checkout(tmp_path):
    target = _paths.PROJECT_ROOT / "pyproject.toml"
    assert _paths.resolve_project_file(str(target)) == str(target.resolve())
    host = f"/home/someone/quadra/{_paths.PROJECT_ROOT.name}/pyproject.toml"
    assert _paths.resolve_project_file(host) == str(target)
    assert _paths.resolve_project_file(str(tmp_path / "missing.pt")) is None


def test_project_relative():
    target = _paths.PROJECT_ROOT / "logs" / "run" / "model_1.pt"
    assert _paths.project_relative(str(target)) == "logs/run/model_1.pt"
    assert _paths.project_relative("/elsewhere/model_1.pt") == "/elsewhere/model_1.pt"


def test_resolve_checkpoint(tmp_path):
    old = tmp_path / "2026-01-01_00-00-00_phase2_flat3axis"
    new = tmp_path / "2026-01-02_00-00-00_phase2_flat3axis"
    other = tmp_path / "2026-01-03_00-00-00_phase3_flatcoord"
    for run in (old, new, other):
        run.mkdir()
        (run / "best.pt").write_bytes(b"")
    (new / "model_10.pt").write_bytes(b"")
    # file, run directory (-> best.pt), run-name pattern (-> latest matching run)
    assert _paths.resolve_checkpoint(str(new / "model_10.pt")) == str(new / "model_10.pt")
    assert _paths.resolve_checkpoint(str(old)) == str(old / "best.pt")
    assert _paths.resolve_checkpoint("phase2_flat3axis", tmp_path) == str(new / "best.pt")
    with pytest.raises(FileNotFoundError, match="no run matching"):
        _paths.resolve_checkpoint("phase9", tmp_path)
    with pytest.raises(FileNotFoundError, match="not found"):
        _paths.resolve_checkpoint("phase2_flat3axis")  # a pattern needs log_root
    (new / "best.pt").unlink()
    with pytest.raises(FileNotFoundError, match="select_best"):
        _paths.resolve_checkpoint("phase2_flat3axis", tmp_path)  # latest run not evaluated: no silent fallback
