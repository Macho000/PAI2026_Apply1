"""YAML ``terrain`` section -> Isaac Lab ``TerrainGeneratorCfg``.

Rows of the generated grid are difficulty levels, columns are terrain types (``proportion`` sets how many columns
each type gets, see ``layout.column_terrain_names``). The ``terrain_levels`` curriculum moves robots between rows.
Example::

    terrain:
      generator:
        size: [8.0, 8.0]
        num_rows: 10
        num_cols: 20
        border_width: 20.0
        curriculum: true
      max_init_terrain_level: 3
      sub_terrains:
        flat:   {type: flat, proportion: 0.2}
        stairs: {type: pyramid_stairs, proportion: 0.4, step_height_range: [0.05, 0.15], step_width: 0.3}

Types map 1:1 to Isaac Lab sub-terrain configs (``TERRAIN_TYPES``); all remaining keys are passed as that config's
fields. Terrain types for later phases (composite course, per-terrain friction) are added here when those phases
are enabled.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg

TERRAIN_TYPES: dict[str, type] = {
    "flat": terrain_gen.MeshPlaneTerrainCfg,
    "random_grid": terrain_gen.MeshRandomGridTerrainCfg,
    "box": terrain_gen.MeshBoxTerrainCfg,
    "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg,
    "inv_pyramid_stairs": terrain_gen.MeshInvertedPyramidStairsTerrainCfg,
    "pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg,
    "inv_pyramid_slope": terrain_gen.HfInvertedPyramidSlopedTerrainCfg,
    "random_rough": terrain_gen.HfRandomUniformTerrainCfg,
    "wave": terrain_gen.HfWaveTerrainCfg,
    "discrete_obstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg,
    "repeated_boxes": terrain_gen.MeshRepeatedBoxesTerrainCfg,
}


def _tuplify(value: Any) -> Any:
    return tuple(_tuplify(v) for v in value) if isinstance(value, list) else value


def make_sub_terrain(spec: Mapping[str, Any]):
    spec = dict(spec)
    kind = spec.pop("type")
    if kind not in TERRAIN_TYPES:
        raise KeyError(f"unknown terrain type '{kind}'; available: {sorted(TERRAIN_TYPES)}")
    cls = TERRAIN_TYPES[kind]
    cfg = cls()
    for key, value in spec.items():
        if not hasattr(cfg, key):
            raise KeyError(f"terrain type '{kind}' has no field '{key}'")
        if key == "proportion":
            value = float(value)  # the generator normalises the proportions in place (fails on an int array)
        setattr(cfg, key, _tuplify(value))
    return cfg


def make_terrain_generator(section: Mapping[str, Any]) -> TerrainGeneratorCfg:
    gen = dict(section.get("generator", {}))
    subs = section.get("sub_terrains") or {}
    if not subs:
        raise ValueError("terrain.sub_terrains must list at least one terrain")
    cfg = TerrainGeneratorCfg(
        size=_tuplify(gen.pop("size", [8.0, 8.0])),
        border_width=gen.pop("border_width", 20.0),
        num_rows=gen.pop("num_rows", 10),
        num_cols=gen.pop("num_cols", 20),
        horizontal_scale=gen.pop("horizontal_scale", 0.1),
        vertical_scale=gen.pop("vertical_scale", 0.005),
        slope_threshold=gen.pop("slope_threshold", 0.75),
        curriculum=gen.pop("curriculum", False),
        use_cache=gen.pop("use_cache", False),
        sub_terrains={name: make_sub_terrain(s) for name, s in subs.items()},
    )
    for key, value in gen.items():
        if not hasattr(cfg, key):
            raise KeyError(f"TerrainGeneratorCfg has no field '{key}'")
        setattr(cfg, key, _tuplify(value))
    return cfg


def apply_terrain_section(terrain_importer_cfg, section: Mapping[str, Any] | None) -> None:
    """Configure ``scene.terrain`` (a ``TerrainImporterCfg``) from the YAML ``terrain`` section."""
    if not section:
        return
    section = dict(section)
    terrain_importer_cfg.terrain_type = "generator"
    terrain_importer_cfg.terrain_generator = make_terrain_generator(section)
    if "max_init_terrain_level" in section:
        terrain_importer_cfg.max_init_terrain_level = section["max_init_terrain_level"]
    friction = section.get("friction")
    if friction:
        terrain_importer_cfg.physics_material.static_friction = friction.get("static", 1.0)
        terrain_importer_cfg.physics_material.dynamic_friction = friction.get("dynamic", 1.0)
    unknown = set(section) - {"generator", "sub_terrains", "max_init_terrain_level", "friction"}
    if unknown:
        raise KeyError(f"unknown terrain keys {sorted(unknown)}")
