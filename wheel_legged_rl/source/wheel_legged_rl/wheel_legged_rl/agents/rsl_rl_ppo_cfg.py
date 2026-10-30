"""PPO settings. Initial values = unitree_rl_lab ``BasePPORunnerCfg`` (identical for every phase so that
checkpoints can be fine-tuned from one phase to the next)."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class CriticWarmupCfg:
    """Critic-only warm-up after a phase transfer (see ``agents/critic_warmup.py`` and ``agents/runner.py``).

    Unfreeze the actor at the first check that passes, counted in critic-only PPO updates:
    ``updates >= max_iterations``, or ``updates >= min_iterations`` and either the generalisation gap
    ``ev_post(prev) - ev_pre > max_ev_gap`` (the critic starts memorising its batches) or the pre-update EV on fresh
    rollouts has plateaued (rise < ``ev_plateau_delta`` for ``patience`` iterations) with ``ev_pre >= ev_min`` and
    ``|mean(R - V)| / std(R) <= max_rel_bias``. ``min_iterations == max_iterations`` gives a fixed warm-up.
    """

    min_iterations: int = 2
    max_iterations: int = 20  # safety cap; over-fitting is handled by max_ev_gap and num_learning_epochs
    num_learning_epochs: int = 2  # during the warm-up only (PPO default 5): less memorising of each batch
    # fixed during the warm-up. 1e-3 (the PPO value) with the fresh Adam wrecked the loaded critic on the first update
    # (EV on its own batch 0.70 -> 0.26); 3e-4 did not (0.70 -> 0.77) and reached the same EV (~0.95)
    learning_rate: float = 3.0e-4
    max_ev_gap: float = 0.1  # the gap is 0-0.05 in a healthy warm-up (iteration noise): 0.05 fired falsely
    ev_plateau_delta: float = 0.01
    patience: int = 2
    ev_min: float = 0.5
    max_rel_bias: float = 0.1


@configclass
class WheelLeggedPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 100
    experiment_name = "go2w_wheel_leg"
    run_name = ""
    wandb_project = "wheel_legged_rl"
    init_from: str | None = None
    """Checkpoint a new run starts from when train.py gets no --init_from / --checkpoint / --resume: a file, a run
    directory, or a run-name pattern (-> <latest matching run>/best.pt). Hydra: ``agent.init_from=null``."""
    critic_warmup: CriticWarmupCfg | None = None
    """Critic-only warm-up of a run started from ``init_from`` (not on --resume / from scratch). None: off."""
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        # 0.01 (unitree_rl_lab) let the action std run away (0.15 -> 3.0 by iter 3000, wheels ~5.6) once the
        # task rewards saturated; 0.003 keeps exploration without the drift.
        entropy_coef=0.003,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class Go2WPhase1PPORunnerCfg(WheelLeggedPPORunnerCfg):
    run_name = "phase1_flat"
    # the vx curriculum (delta 0.05) reaches +-1 m/s around iter ~1200; leave time to consolidate at full speed
    max_iterations = 4000


@configclass
class Go2WPhase2PPORunnerCfg(WheelLeggedPPORunnerCfg):
    run_name = "phase2_flat3axis"
    # fine-tuned from Phase 1; the vy / yaw curricula reach their limits around iter ~500
    max_iterations = 3000


@configclass
class Go2WPhase3PPORunnerCfg(WheelLeggedPPORunnerCfg):
    run_name = "phase3_flatcoord"
    init_from = "phase2_flat3axis"  # best.pt of the latest Phase 2 run (evaluate.py --select_best)
    # the critic gains the privileged inputs (zero-initialised) and has to learn the new rewards / dynamics first
    critic_warmup = CriticWarmupCfg()
    # the vx curriculum reaches +-2 m/s after ~420 iterations; the rest adapts to the randomisation
    max_iterations = 3000


@configclass
class Go2WPhase4PPORunnerCfg(WheelLeggedPPORunnerCfg):
    run_name = "phase4_steps"
    init_from = "phase3_flatcoord"  # best.pt of the latest Phase 3 run (evaluate.py --select_best)
    # same critic inputs, but the height scan now varies and the terrain changes the returns: refit V first
    critic_warmup = CriticWarmupCfg()
    # the terrain curriculum climbs 10 rows; one move per episode (~42 iterations) at most
    max_iterations = 3000
