"""Train a reinforcement learning agent for the Aliens environment.

This script relies on the `stable-baselines3`_ implementation of Deep
Q-Networks (DQN) instead of the hand-written NumPy agent that previously lived
in this repository.  The environment defined in :mod:`env` exposes a rich grid
world with deterministic dynamics and a discrete action space.  Here we wrap it
as a `Gymnasium <https://gymnasium.farama.org/>`_ environment so that it can be
used directly with third-party reinforcement learning packages.

Quick start
-----------
1. Install the dependencies::

       pip install -r requirements.txt

   ``requirements.txt`` now includes ``stable-baselines3`` and
   ``gymnasium``.  These packages pull in PyTorch and other prerequisites
   automatically.

2. Launch training (saving the model under ``models/``)::

       python train_rl_agent.py --level 0 --total-timesteps 200000

   During training Stable-Baselines3 logs progress to stdout.  The final model
   checkpoint is written to ``models/dqn_level0`` (or the path supplied via
   ``--save-path``).

3. (Optional) Run evaluation episodes and periodic checkpoints without
   generating PNG renders::

       python train_rl_agent.py --level 0 --total-timesteps 3000000 --eval-frequency 10000 --eval-episodes 10 --checkpoint-frequency 50000

   Evaluation always runs headlessly to avoid the expensive PNG rendering that
   :class:`env.AliensEnv` performs when ``render=True``.

4. Train on GPU by pointing Stable-Baselines3 at your CUDA device::

       python train_rl_agent.py --device cuda

   ``cuda`` automatically selects the default CUDA device.  You can also supply
   ``cuda:1`` (or similar) to target a specific GPU, or leave the default of
   ``auto`` to let Stable-Baselines3 pick the best available accelerator.

The resulting policy can be loaded with ``stable_baselines3.DQN.load`` and used
for inference inside ``play.py``/``test.py`` or custom evaluation scripts.

.. _stable-baselines3: https://stable-baselines3.readthedocs.io/
"""

from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np

from env import AliensEnv

try:  # Third-party RL ecosystem imports.
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover - helpful runtime message.
    raise ImportError(
        "Gymnasium is required. Install the optional dependencies with"
        " `pip install -r requirements.txt`."
    ) from exc

try:
    from stable_baselines3 import DQN
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
    from stable_baselines3.common.utils import set_random_seed
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
except ImportError as exc:  # pragma: no cover - helpful runtime message.
    raise ImportError(
        "stable-baselines3 is required. Install the optional dependencies with"
        " `pip install -r requirements.txt`."
    ) from exc


# ---------------------------------------------------------------------------
# Observation encoding helpers (ported from the original NumPy agent)
# ---------------------------------------------------------------------------

ENTITY_CHANNELS: tuple[str, ...] = (
    "wall",
    "base",
    "portalSlow",
    "portalFast",
    "alien",
    "sam",
    "bomb",
    "avatar",
)
ENTITY_INDEX = {name: idx for idx, name in enumerate(ENTITY_CHANNELS)}
EXTRA_FEATURE_COUNT = 7


def set_global_seeds(seed: int) -> None:
    """Seed every source of randomness used by the trainer."""

    random.seed(seed)
    np.random.seed(seed)
    set_random_seed(seed)


def encode_observation(obs: Sequence[Sequence[Sequence[str]]]) -> np.ndarray:
    """Convert the nested list observation into a feature vector."""

    height = len(obs)
    width = len(obs[0]) if height else 0
    grid = np.zeros((len(ENTITY_CHANNELS), height, width), dtype=np.float32)

    avatar_x, avatar_y = 0, 0
    portals = 0
    alien_positions: list[tuple[int, int]] = []
    bomb_positions: list[tuple[int, int]] = []

    for y in range(height):
        for x in range(width):
            cell = obs[y][x]
            for entity in cell:
                idx = ENTITY_INDEX.get(entity)
                if idx is None:
                    continue
                grid[idx, y, x] = 1.0
                if entity == "avatar":
                    avatar_x, avatar_y = x, y
                elif entity in ("portalSlow", "portalFast"):
                    portals += 1
                elif entity == "alien":
                    alien_positions.append((x, y))
                elif entity == "bomb":
                    bomb_positions.append((x, y))

    avatar_x_norm = (2.0 * avatar_x / max(width - 1, 1)) - 1.0
    avatar_y_norm = (2.0 * avatar_y / max(height - 1, 1)) - 1.0

    grid_size = float(width * height if width and height else 1.0)
    alien_count = len(alien_positions) / grid_size
    bomb_count = len(bomb_positions) / grid_size

    perim = float(width + height)
    if alien_positions:
        alien_dist = min(abs(ax - avatar_x) + abs(ay - avatar_y) for ax, ay in alien_positions)
        alien_dist_norm = alien_dist / max(perim, 1.0)
    else:
        alien_dist_norm = 1.0

    if bomb_positions:
        bomb_dist = min(abs(bx - avatar_x) + abs(by - avatar_y) for bx, by in bomb_positions)
        bomb_dist_norm = bomb_dist / max(perim, 1.0)
    else:
        bomb_dist_norm = 1.0

    portal_fraction = portals / grid_size

    extras = np.array(
        [
            avatar_x_norm,
            avatar_y_norm,
            alien_count,
            bomb_count,
            alien_dist_norm,
            bomb_dist_norm,
            portal_fraction,
        ],
        dtype=np.float32,
    )

    return np.concatenate([grid.reshape(-1), extras], axis=0)


# ---------------------------------------------------------------------------
# Gymnasium wrapper
# ---------------------------------------------------------------------------


class AliensGymWrapper(gym.Env):
    """Expose :class:`env.AliensEnv` through the Gymnasium API."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 15}

    def __init__(self, level: int, render: bool = False):
        super().__init__()
        self._level = level
        self._render = render
        self._env = AliensEnv(level=level, render=render)

        sample_obs = encode_observation(self._env.reset())
        feature_dim = sample_obs.shape[0]

        low = np.zeros(feature_dim, dtype=np.float32)
        high = np.ones(feature_dim, dtype=np.float32)
        # Avatar coordinates are normalised to [-1, 1].
        low[-EXTRA_FEATURE_COUNT] = -1.0
        low[-EXTRA_FEATURE_COUNT + 1] = -1.0
        high[-EXTRA_FEATURE_COUNT] = 1.0
        high[-EXTRA_FEATURE_COUNT + 1] = 1.0

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Discrete(len(self._env.action_space))

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        obs = encode_observation(self._env.reset())
        return obs.astype(np.float32), {}

    def step(self, action: int):
        obs, reward, done, info = self._env.step(int(action))
        encoded = encode_observation(obs).astype(np.float32)
        terminated = bool(done)
        truncated = False
        return encoded, float(reward), terminated, truncated, info

    def render(self):
        if not self._render:
            raise RuntimeError(
                "Rendering was disabled. Re-create the environment with render=True."
            )
        # The underlying environment writes PNG files when render=True.
        return None

    def close(self):
        # Nothing to clean up: AliensEnv writes images to disk directly.
        return None


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------


@dataclass
class TrainingConfig:
    level: int
    total_timesteps: int
    learning_rate: float
    buffer_size: int
    batch_size: int
    train_freq: int
    gradient_steps: int
    gamma: float
    target_update_interval: int
    exploration_fraction: float
    exploration_final_eps: float
    learning_starts: int
    verbose: int
    tensorboard_log: Optional[str]
    seed: int
    eval_frequency: int
    eval_episodes: int
    checkpoint_frequency: int
    save_path: str
    log_interval: int
    device: str


def make_env(level: int, render: bool, seed: int) -> Callable[[], gym.Env]:
    """Factory to create wrapped environments with deterministic seeding."""

    def _init() -> gym.Env:
        env = AliensGymWrapper(level=level, render=render)
        env.reset(seed=seed)
        return env

    return _init


def train(config: TrainingConfig) -> None:
    """Train a DQN agent using Stable-Baselines3."""

    os.makedirs(os.path.dirname(config.save_path), exist_ok=True)

    set_global_seeds(config.seed)

    eval_env: Optional[VecMonitor] = None
    train_env: Optional[VecMonitor] = None
    try:
        train_vec = DummyVecEnv([make_env(config.level, render=False, seed=config.seed)])
        train_vec.seed(config.seed)
        train_env = VecMonitor(train_vec)
        train_env.reset()

        callbacks = []

        if config.eval_frequency > 0:
            eval_vec = DummyVecEnv(
                [make_env(config.level, render=False, seed=config.seed + 1)]
            )
            eval_vec.seed(config.seed + 1)
            eval_env = VecMonitor(eval_vec)
            eval_env.reset()
            best_model_dir = os.path.join(os.path.dirname(config.save_path), "best")
            os.makedirs(best_model_dir, exist_ok=True)
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=best_model_dir,
                log_path=best_model_dir,
                eval_freq=config.eval_frequency,
                n_eval_episodes=config.eval_episodes,
                deterministic=True,
                render=False,
            )
            callbacks.append(eval_callback)

        if config.checkpoint_frequency > 0:
            checkpoint_dir = os.path.join(os.path.dirname(config.save_path), "checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_callback = CheckpointCallback(
                save_freq=config.checkpoint_frequency,
                save_path=checkpoint_dir,
                name_prefix=f"dqn_level{config.level}",
            )
            callbacks.append(checkpoint_callback)

        model = DQN(
            "MlpPolicy",
            train_env,
            learning_rate=config.learning_rate,
            buffer_size=config.buffer_size,
            learning_starts=config.learning_starts,
            batch_size=config.batch_size,
            train_freq=config.train_freq,
            gradient_steps=config.gradient_steps,
            gamma=config.gamma,
            target_update_interval=config.target_update_interval,
            exploration_fraction=config.exploration_fraction,
            exploration_final_eps=config.exploration_final_eps,
            verbose=config.verbose,
            tensorboard_log=config.tensorboard_log,
            device=config.device,
        )

        callback = CallbackList(callbacks) if callbacks else None
        model.learn(
            total_timesteps=config.total_timesteps,
            log_interval=config.log_interval,
            callback=callback,
        )

        model.save(config.save_path)
        print(f"Saved trained model to {config.save_path}")
    finally:
        if eval_env is not None:
            eval_env.close()
        if train_env is not None:
            train_env.close()


def parse_args(args: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DQN agent for Aliens using Stable-Baselines3.")
    parser.add_argument("--level", type=int, default=0, help="Environment level to train on.")
    parser.add_argument("--total-timesteps", type=int, default=200_000, help="Total environment steps to collect.")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Optimizer learning rate.")
    parser.add_argument("--buffer-size", type=int, default=100_000, help="Replay buffer size.")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size.")
    parser.add_argument("--train-freq", type=int, default=4, help="Frequency of gradient updates in steps.")
    parser.add_argument("--gradient-steps", type=int, default=1, help="Gradient steps to run after each rollout.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument("--target-update-interval", type=int, default=10_000, help="Steps between target network syncs.")
    parser.add_argument("--exploration-fraction", type=float, default=0.1, help="Fraction of training for epsilon decay.")
    parser.add_argument("--exploration-final-eps", type=float, default=0.02, help="Final epsilon after decay.")
    parser.add_argument("--learning-starts", type=int, default=1_000, help="How many steps to collect before updates.")
    parser.add_argument("--verbose", type=int, default=1, choices=[0, 1, 2], help="Stable-Baselines3 verbosity level.")
    parser.add_argument("--tensorboard-log", type=str, default=None, help="Optional TensorBoard log directory.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    parser.add_argument("--eval-frequency", type=int, default=0, help="How often (in steps) to run evaluation. 0 disables it.")
    parser.add_argument("--eval-episodes", type=int, default=5, help="Number of evaluation episodes per run.")
    parser.add_argument("--checkpoint-frequency", type=int, default=0, help="How often (in steps) to save checkpoints. 0 disables it.")
    parser.add_argument("--save-path", type=str, default=os.path.join("models", "dqn_level{level}"), help="Where to save the trained policy.")
    parser.add_argument("--log-interval", type=int, default=100, help="Logging interval passed to Stable-Baselines3.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help=(
            "Computation device passed to Stable-Baselines3 (e.g. 'auto', 'cpu', 'cuda', 'cuda:0')."
        ),
    )
    return parser.parse_args(args)


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

