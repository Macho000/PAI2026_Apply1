"""YAML -> config object overrides. Isaac-independent (works on configclasses, dataclasses and plain objects).

Phase files (``configs/<robot>/phase*.yaml``) contain only the values that differ from the robot-agnostic defaults
in ``envs/base_env_cfg.py``. They are applied in ``__post_init__`` of the task config, before Hydra CLI overrides.

Semantics
---------
* ``key: value`` sets an attribute (recursively for mappings). Unknown keys raise ``KeyError`` so typos surface.
* ``key: null`` sets the attribute to ``None`` (for manager terms this disables the term).
* A mapping applied to a ``dict`` attribute updates the dict (keys may be new).
* Lists are converted to tuples when the current value is a tuple.
* The top-level ``rewards`` section is special (see :func:`apply_reward_section`): every reward term that is not
  listed gets weight 0, i.e. *unlisted = disabled*. Weights are never taken from source code defaults.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

CONFIG_ENV_VAR = "WL_ENV_CONFIG"
"""If set, the task config loads this YAML instead of its default phase file (used by ``train.py --env_config``)."""


def load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    base = data.pop("_base_", None)
    if base is None:
        return data
    base_path = (Path(path).parent / base).resolve()
    return deep_merge(load_yaml(base_path), data)


def deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _is_leaf(value: Any) -> bool:
    return value is None or isinstance(value, (bool, int, float, str, tuple, list))


def apply_overrides(target: Any, overrides: Mapping[str, Any], path: str = "") -> None:
    """Recursively set attributes of ``target`` from ``overrides``."""
    for key, value in overrides.items():
        where = f"{path}.{key}" if path else key
        if isinstance(target, dict):
            if isinstance(value, Mapping) and key in target and not _is_leaf(target[key]):
                apply_overrides(target[key], value, where)
            else:
                target[key] = _coerce(target.get(key), value)
            continue
        if not hasattr(target, key):
            raise KeyError(f"unknown config key '{where}' (no attribute on {type(target).__name__})")
        current = getattr(target, key)
        if isinstance(value, Mapping) and current is not None and not _is_leaf(current):
            apply_overrides(current, value, where)
        else:
            setattr(target, key, _coerce(current, value))


def _coerce(current: Any, value: Any) -> Any:
    if isinstance(value, list) and (isinstance(current, tuple) or current is None):
        return tuple(_coerce(None, v) for v in value)
    if isinstance(current, float) and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    return value


def reward_term_names(rewards_cfg: Any) -> list[str]:
    """Names of reward terms on a manager config object (attributes having a ``weight``)."""
    names = []
    for name in dir(rewards_cfg):
        if name.startswith("_"):
            continue
        term = getattr(rewards_cfg, name)
        if term is not None and hasattr(term, "weight") and hasattr(term, "func"):
            names.append(name)
    return names


def apply_reward_section(rewards_cfg: Any, section: Mapping[str, Any] | None) -> None:
    """Apply the ``rewards`` YAML section.

    Each entry is either a number (the weight), ``null`` (disabled), or a mapping with ``weight`` and optionally
    ``params`` (merged into the term params). Terms not listed get weight 0.0 and are pruned by the environment, so
    that Hydra can still re-enable them from the command line (``env.rewards.<term>.weight=-0.01``).
    """
    section = dict(section or {})
    known = set(reward_term_names(rewards_cfg))
    unknown = set(section) - known
    if unknown:
        raise KeyError(f"unknown reward terms {sorted(unknown)}; available: {sorted(known)}")
    for name in known:
        term = getattr(rewards_cfg, name)
        entry = section.get(name)
        if entry is None:
            term.weight = 0.0
        elif isinstance(entry, (int, float)) and not isinstance(entry, bool):
            term.weight = float(entry)
        elif isinstance(entry, Mapping):
            extra = set(entry) - {"weight", "params"}
            if extra:
                raise KeyError(f"reward '{name}': unexpected keys {sorted(extra)}")
            term.weight = float(entry.get("weight", 0.0))
            for pkey, pval in (entry.get("params") or {}).items():
                if pkey not in term.params:
                    raise KeyError(f"reward '{name}': unknown param '{pkey}' (has {sorted(term.params)})")
                cur = term.params[pkey]
                if isinstance(pval, Mapping) and not _is_leaf(cur):
                    apply_overrides(cur, pval, f"rewards.{name}.params.{pkey}")
                else:
                    term.params[pkey] = _coerce(cur, pval)
        else:
            raise TypeError(f"reward '{name}': expected number, null or mapping, got {entry!r}")


def resolve_config_path(default_path: str | Path) -> Path:
    """The YAML to load: ``$WL_ENV_CONFIG`` if set, else ``default_path``."""
    return Path(os.environ.get(CONFIG_ENV_VAR) or default_path)
