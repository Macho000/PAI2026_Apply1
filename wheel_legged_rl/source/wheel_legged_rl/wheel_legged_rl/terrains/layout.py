"""Layout of Isaac Lab's curriculum terrain grid (pure Python, unit-tested without Isaac Sim)."""

from __future__ import annotations

from collections.abc import Mapping


def column_terrain_names(proportions: Mapping[str, float], num_cols: int) -> list[str]:
    """Sub-terrain name of every column, as ``TerrainGenerator._generate_curriculum_terrains`` assigns them
    (Isaac Lab v2.3.0): column ``c`` gets the first type whose cumulative proportion exceeds ``c / num_cols``."""
    names = list(proportions)
    total = float(sum(proportions.values()))
    if not names or total <= 0:
        raise ValueError("need at least one sub-terrain with a positive proportion")
    cumulative, acc = [], 0.0
    for name in names:
        acc += proportions[name] / total
        cumulative.append(acc)
    return [names[next(i for i, c in enumerate(cumulative) if col / num_cols + 0.001 < c)] for col in range(num_cols)]
