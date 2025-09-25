"""Train the Aliens reinforcement learning agent on CPU with Stable-Baselines3.

This script mirrors the GPU training workflow implemented in
:mod:`train_rl_agent`, but pins the computation device to the host CPU by
default.  It wraps :class:`env.AliensEnv` inside the Gymnasium API, encodes the
observations into feature vectors and then leverages the third-party
``stable-baselines3`` implementation of DQN to optimise a policy.

Usage
-----
1. Install the project dependencies (they include ``gymnasium`` and
   ``stable-baselines3``)::

       pip install -r requirements.txt

2. Launch CPU training for a given level.  The device is forced to ``cpu``
   unless you override it explicitly::

       python train_rl_agent_cpu.py --level 0 --total-timesteps 200000

   You may supply any of the regular hyper-parameters (learning rate,
   exploration schedule, replay buffer size, etc.).  Checkpoints and metrics are
   written to the same locations as in the GPU script.

3. Optional: enable periodic evaluation or checkpointing::

       python train_rl_agent_cpu.py --eval-frequency 10000 --eval-episodes 5 \
           --checkpoint-frequency 50000

   Evaluation environments always run without rendering so the expensive PNG
   output is avoided.

The resulting policy artefacts are fully compatible with
``stable_baselines3.DQN.load`` and can be consumed by ``test_rl.py`` or
``rl_play.py`` for automated evaluation and dataset generation.
"""

from __future__ import annotations

from typing import Optional, Sequence

from train_rl_agent import TrainingConfig, parse_args as gpu_parse_args, train


def parse_args(args: Optional[Sequence[str]] = None):
    """Parse CLI arguments while defaulting the computation device to CPU."""

    namespace = gpu_parse_args(args)
    if namespace.device == "auto":
        namespace.device = "cpu"
    return namespace


def main() -> None:
    args = parse_args()
    save_path = args.save_path.format(level=args.level)
    config = TrainingConfig(
        level=args.level,
        total_timesteps=args.total_timesteps,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        gamma=args.gamma,
        target_update_interval=args.target_update_interval,
        exploration_fraction=args.exploration_fraction,
        exploration_final_eps=args.exploration_final_eps,
        learning_starts=args.learning_starts,
        verbose=args.verbose,
        tensorboard_log=args.tensorboard_log,
        seed=args.seed,
        eval_frequency=args.eval_frequency,
        eval_episodes=args.eval_episodes,
        checkpoint_frequency=args.checkpoint_frequency,
        save_path=save_path,
        log_interval=args.log_interval,
        device=args.device,
    )
    train(config)


if __name__ == "__main__":
    main()
