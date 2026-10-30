"""Map an articulation's joint selection onto its actuator groups (duck-typed, no Isaac imports).

Per-joint actuator state (gains, strength, friction, ...) lives in each actuator as ``(num_envs, num_joints_of_group)``
tensors. Events and privileged observations address joints in articulation order (``SceneEntityCfg.joint_ids``);
:func:`actuator_joints` gives, per actuator, which of its columns are selected and where they go.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import torch


def actuator_joints(asset, joint_ids: Sequence[int] | slice, attr: str | None = None) -> Iterator[tuple]:
    """Yield ``(actuator, local, out)`` for every actuator (having ``attr``, if given) with selected joints.

    ``local`` indexes the actuator's columns, ``out`` the position of those joints in the selection (so that
    ``result[:, out] = actuator.<tensor>[:, local]`` builds a tensor in selection order).
    """
    num_joints = asset.num_joints
    selected = list(range(num_joints)) if isinstance(joint_ids, slice) else [int(j) for j in joint_ids]
    position = {j: i for i, j in enumerate(selected)}
    for actuator in asset.actuators.values():
        if attr is not None and not hasattr(actuator, attr):
            continue
        indices = actuator.joint_indices
        group = list(range(num_joints)) if isinstance(indices, slice) else [int(j) for j in indices]
        pairs = [(k, position[j]) for k, j in enumerate(group) if j in position]
        if not pairs:
            continue
        local = torch.tensor([k for k, _ in pairs], dtype=torch.long, device=asset.device)
        out = torch.tensor([o for _, o in pairs], dtype=torch.long, device=asset.device)
        yield actuator, local, out
