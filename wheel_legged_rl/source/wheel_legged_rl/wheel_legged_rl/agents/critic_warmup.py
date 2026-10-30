"""Phase transfer of an actor-critic with a larger critic input, and the critic-only warm-up (torch only, no Isaac).

* :func:`expand_state_dict` -- load a checkpoint whose critic saw fewer observations: the new input columns of the
  critic's first layer are zero, so ``V_new(obs, privileged) == V_old(obs)`` at the start.
* :func:`critic_metrics` -- value-function diagnostics on a rollout (explained variance, bias, ...).
* :class:`CriticWarmup` -- decides when to unfreeze the actor, from the metrics on *fresh* rollouts (before the
  update that trains on them), so that memorising a batch is not mistaken for convergence.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import torch


def expand_state_dict(
    source: Mapping[str, torch.Tensor], target: Mapping[str, torch.Tensor], expandable: Iterable[str]
) -> tuple[dict[str, torch.Tensor], list[str]]:
    """State dict for a model shaped like ``target`` from the checkpoint ``source``.

    Keys must match. Tensors of equal shape are copied. A key in ``expandable`` may be a 2-D weight with the same
    number of rows and fewer columns in ``source``: its columns are copied to the front and the rest set to 0.
    Anything else raises ``ValueError``. Returns the state dict and the expanded keys.
    """
    expandable = set(expandable)
    missing, unexpected = set(target) - set(source), set(source) - set(target)
    if missing or unexpected:
        raise ValueError(f"checkpoint keys differ: missing {sorted(missing)}, unexpected {sorted(unexpected)}")
    out, expanded = {}, []
    for key, dst in target.items():
        src = source[key]
        if src.shape == dst.shape:
            out[key] = src
            continue
        if key in expandable and src.dim() == dst.dim() == 2 and src.shape[0] == dst.shape[0]:
            if src.shape[1] < dst.shape[1]:
                weight = torch.zeros(dst.shape, dtype=src.dtype, device=src.device)
                weight[:, : src.shape[1]] = src
                out[key] = weight
                expanded.append(key)
                continue
        raise ValueError(f"cannot load '{key}': checkpoint {tuple(src.shape)}, model {tuple(dst.shape)}")
    return out, expanded


def explained_variance(values: torch.Tensor, returns: torch.Tensor) -> float:
    """``1 - Var(R - V) / Var(R)`` (NaN if the returns are constant). Blind to a constant offset of V."""
    var = returns.var()
    if var.item() < 1e-12:
        return float("nan")
    return float(1.0 - (returns - values).var() / var)


def critic_metrics(values: torch.Tensor, returns: torch.Tensor) -> dict[str, float]:
    """Diagnostics of value predictions ``values`` against their targets ``returns`` (any matching shape)."""
    v, r = values.flatten().float(), returns.flatten().float()
    err = r - v
    return_std = float(r.std())
    return {
        "ev": explained_variance(v, r),
        "value_mse": float(err.square().mean()),
        "value_bias": float(err.mean()),  # > 0: the critic underestimates
        "rel_bias": float(err.mean()) / max(return_std, 1e-8),
        "value_mean": float(v.mean()),
        "value_std": float(v.std()),
        "return_mean": float(r.mean()),
        "return_std": return_std,
    }


@dataclass
class CriticWarmup:
    """When to end the critic-only warm-up. ``updates`` counts the critic-only PPO updates done so far.

    :meth:`step` is called once per iteration with the metrics of the *fresh* rollout, before the update:

    * ``updates >= max_iterations``                            -> "max_iterations"
    * ``updates >= min_iterations`` and ``ev_gap > max_ev_gap`` -> "ev_gap" (the critic memorises its batches:
      post-update EV on the previous batch minus pre-update EV on this one)
    * ``updates >= min_iterations`` and the pre-update EV rose by less than ``ev_plateau_delta`` for ``patience``
      consecutive iterations and ``ev >= ev_min`` and ``|rel_bias| <= max_rel_bias`` -> "converged"
    """

    min_iterations: int = 2
    max_iterations: int = 20
    max_ev_gap: float = 0.05
    ev_plateau_delta: float = 0.01
    patience: int = 2
    ev_min: float = 0.5
    max_rel_bias: float = 0.1

    def __post_init__(self):
        if not 0 <= self.min_iterations <= self.max_iterations:
            raise ValueError(
                f"need 0 <= min_iterations <= max_iterations, got {self.min_iterations}, {self.max_iterations}"
            )
        self.active = True
        self.updates = 0
        self.plateau_streak = 0
        self.reason: str | None = None
        self._prev_ev: float | None = None

    def step(self, ev: float, ev_gap: float, rel_bias: float) -> str | None:
        """Metrics of the fresh rollout -> the reason to unfreeze now, or None to keep warming up."""
        if not self.active:
            return None
        if self._prev_ev is not None and math.isfinite(ev):
            self.plateau_streak = self.plateau_streak + 1 if ev - self._prev_ev < self.ev_plateau_delta else 0
        self._prev_ev = ev
        reason = None
        if self.updates >= self.max_iterations:
            reason = "max_iterations"
        elif self.updates >= self.min_iterations:
            if math.isfinite(ev_gap) and ev_gap > self.max_ev_gap:
                reason = "ev_gap"
            elif self.plateau_streak >= self.patience and ev >= self.ev_min and abs(rel_bias) <= self.max_rel_bias:
                reason = "converged"
        if reason is not None:
            self.active, self.reason = False, reason
        return reason

    def record_update(self) -> None:
        if self.active:
            self.updates += 1

    def undertrained(self, ev: float) -> bool:
        """Ended on the iteration cap below ``ev_min`` (worth a warning)."""
        return self.reason == "max_iterations" and not ev >= self.ev_min


REASON_CODES = {None: 0, "converged": 1, "ev_gap": 2, "max_iterations": 3}
"""``Warmup/unfreeze_reason`` in the logs."""
