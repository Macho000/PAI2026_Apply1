"""RSL-RL ``OnPolicyRunner`` with critic diagnostics, phase-transfer loading and a critic-only warm-up.

* Every iteration logs ``Critic/*``: metrics of the value function on the fresh rollout *before* the update
  (``*_pre``), the explained variance on the same batch *after* it (``ev_post``) and the generalisation gap
  ``ev_gap = ev_post(previous iteration) - ev_pre(this iteration)``.
* :meth:`load_init` loads the weights of an earlier phase (optimizer not loaded, iteration 0), expanding the
  critic's first layer with zero columns when the critic observation has grown (``critic_warmup.expand_state_dict``).
* With ``train_cfg["critic_warmup"]`` and ``train_cfg["init_mode"] == "init_from"``, the actor (MLP and action std)
  is frozen (``requires_grad=False``) until :class:`~wheel_legged_rl.agents.critic_warmup.CriticWarmup` decides to
  unfreeze it. Meanwhile the learning rate is fixed (a frozen actor has KL ~ 0, which would make the adaptive schedule
  raise the rate up to 1e-2) and ``num_learning_epochs`` is reduced; both are restored at the unfreeze.
  Resumed runs (``--resume``) and runs from scratch do not warm up.
"""

from __future__ import annotations

import math

import torch
from rsl_rl.runners import OnPolicyRunner
from torch import nn

from .critic_warmup import REASON_CODES, CriticWarmup, critic_metrics, expand_state_dict, explained_variance


class WheelLeggedOnPolicyRunner(OnPolicyRunner):
    def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device: str = "cpu"):
        super().__init__(env, train_cfg, log_dir=log_dir, device=device)
        warm = dict(train_cfg.get("critic_warmup") or {})
        self._warmup_epochs = warm.pop("num_learning_epochs", None)
        self._warmup_lr = warm.pop("learning_rate", None)
        self.warmup = CriticWarmup(**warm) if warm and train_cfg.get("init_mode") == "init_from" else None
        self._defaults = {
            "schedule": self.alg.schedule,
            "num_learning_epochs": self.alg.num_learning_epochs,
            "learning_rate": self.alg.learning_rate,
        }
        self._prev_ev_post = math.nan
        self._critic_log: dict[str, float] = {}
        self._ppo_update = self.alg.update
        self.alg.update = self._update_with_diagnostics

    # ------------------------------------------------------------------------------------------------ loading

    def load_init(self, path: str) -> list[str]:
        """Weights of ``path`` for a new run (iteration 0, fresh optimizer). Returns the expanded keys."""
        loaded = torch.load(path, weights_only=False, map_location=self.device)
        policy = self.alg.policy
        first = next(name for name, m in policy.critic.named_modules() if isinstance(m, nn.Linear))
        state, expanded = expand_state_dict(loaded["model_state_dict"], policy.state_dict(), [f"critic.{first}.weight"])
        policy.load_state_dict(state)
        self.current_learning_iteration = 0
        for key in expanded:
            print(f"[INFO] {key}: {tuple(loaded['model_state_dict'][key].shape)} -> {tuple(state[key].shape)}")
        return expanded

    # ------------------------------------------------------------------------------------------------ warm-up

    def _actor_parameters(self) -> list[nn.Parameter]:
        policy = self.alg.policy
        params = list(policy.actor.parameters()) + list(policy.actor_obs_normalizer.parameters())
        params += [getattr(policy, name) for name in ("std", "log_std") if hasattr(policy, name)]
        return params

    def _actor_vector(self) -> torch.Tensor:
        return torch.cat([p.detach().flatten() for p in self._actor_parameters()]).clone()

    def _set_learning_rate(self, lr: float) -> None:
        self.alg.learning_rate = lr
        for group in self.alg.optimizer.param_groups:
            group["lr"] = lr

    def _freeze_actor(self) -> None:
        for p in self._actor_parameters():
            p.requires_grad_(False)
        self.alg.schedule = "fixed"
        if self._warmup_epochs:
            self.alg.num_learning_epochs = self._warmup_epochs
        self._set_learning_rate(self._warmup_lr or self._defaults["learning_rate"])
        w = self.warmup
        print(
            f"[warmup] actor frozen: critic-only updates ({w.min_iterations}-{w.max_iterations} iterations, "
            f"lr {self.alg.learning_rate:g} fixed, {self.alg.num_learning_epochs} epochs)"
        )

    def _unfreeze_actor(self, reason: str, ev: float) -> None:
        for p in self._actor_parameters():
            p.requires_grad_(True)
        self.alg.schedule = self._defaults["schedule"]
        self.alg.num_learning_epochs = self._defaults["num_learning_epochs"]
        self._set_learning_rate(self._defaults["learning_rate"])
        print(f"[warmup] actor unfrozen after {self.warmup.updates} critic-only updates ({reason}, ev_pre {ev:.3f})")
        if self.warmup.undertrained(ev):
            print(f"[warmup] WARNING: ev_pre {ev:.3f} < ev_min {self.warmup.ev_min} at the iteration cap")

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        if self.warmup is not None and self.warmup.active:
            self._freeze_actor()
        return super().learn(num_learning_iterations, init_at_random_ep_len=init_at_random_ep_len)

    # ------------------------------------------------------------------------------------------------ update

    def _update_with_diagnostics(self):
        storage = self.alg.storage
        with torch.inference_mode():
            pre = critic_metrics(storage.values, storage.returns)
        ev_gap = self._prev_ev_post - pre["ev"]

        if self.warmup is not None and self.warmup.active:
            reason = self.warmup.step(pre["ev"], ev_gap, pre["rel_bias"])
            if reason is not None:
                self._unfreeze_actor(reason, pre["ev"])
        frozen = self.warmup is not None and self.warmup.active
        before = self._actor_vector() if frozen else None

        loss_dict = self._ppo_update()

        if frozen:
            if not torch.equal(before, self._actor_vector()):
                raise RuntimeError("the actor changed during the critic-only warm-up")
            self.warmup.record_update()
        with torch.inference_mode():
            values = torch.cat(
                [self.alg.policy.evaluate(storage.observations[t]) for t in range(storage.values.shape[0])]
            )
            ev_post = explained_variance(values.flatten(), storage.returns.flatten())
        self._prev_ev_post = ev_post

        log = {
            f"Critic/{k}_pre" if k in ("ev", "value_mse", "value_bias", "rel_bias") else f"Critic/{k}": v
            for k, v in pre.items()
        }
        log.update({"Critic/ev_post": ev_post, "Critic/ev_gap": ev_gap})
        if self.warmup is not None:
            log.update(
                {
                    "Warmup/active": float(frozen),
                    "Warmup/updates": float(self.warmup.updates),
                    "Warmup/unfreeze_reason": float(REASON_CODES[self.warmup.reason]),
                }
            )
            if frozen:
                print(
                    f"[warmup] update {self.warmup.updates}: ev_pre {pre['ev']:.3f} ev_post {ev_post:.3f} "
                    f"ev_gap {ev_gap:.3f} rel_bias {pre['rel_bias']:+.3f} value_mse {pre['value_mse']:.4f}"
                )
        self._critic_log = log
        return loss_dict

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        super().log(locs, width, pad)
        for key, value in self._critic_log.items():
            if math.isfinite(value):
                self.writer.add_scalar(key, value, locs["it"])
