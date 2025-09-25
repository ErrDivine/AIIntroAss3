"""Evaluate a trained Stable-Baselines3 agent in the Aliens environment.

This utility complements ``train_rl_agent.py`` by loading a saved DQN policy
and letting it play a configurable number of episodes.  Typical usage::

    python test_rl.py --level 0 --model-path models/dqn_level{level}.zip --episodes 10

The script prints the episodic reward, number of environment steps, and the
terminal ``info['message']`` for each run.  Use ``--render`` if you want the
environment to dump PNG frames to ``figs/`` during evaluation.
"""

from __future__ import annotations

import argparse
import os
from statistics import mean, pstdev
from typing import Optional

from train_rl_agent import AliensGymWrapper, set_global_seeds

try:  # Stable-Baselines3 is required to load and run the trained policy.
    from stable_baselines3 import DQN
except ImportError as exc:  # pragma: no cover - more informative error message.
    raise ImportError(
        "stable-baselines3 is required. Install the optional dependencies with"
        " `pip install -r requirements.txt`."
    ) from exc


def run_episode(
    model: DQN,
    env: AliensGymWrapper,
    *,
    deterministic: bool,
    seed: Optional[int] = None,
) -> tuple[float, int, dict]:
    """Roll out a single episode and return the total reward, steps, and info."""

    obs, _ = env.reset(seed=seed)
    done = False
    total_reward = 0.0
    steps = 0
    info: dict = {}

    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        total_reward += float(reward)
        steps += 1

    return total_reward, steps, info


def parse_args(args: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for RL agent evaluation."""

    parser = argparse.ArgumentParser(
        description="Evaluate a Stable-Baselines3 DQN agent on the Aliens environment.")
    parser.add_argument(
        "--model-path",
        type=str,
        default=os.path.join("models", "dqn_level{level}.zip"),
        help="Path to the saved DQN .zip file. Supports {level} formatting.",
    )
    parser.add_argument("--level", type=int, default=0, help="Environment level to evaluate.")
    parser.add_argument("--episodes", type=int, default=5, help="Number of evaluation episodes to run.")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use a deterministic policy instead of sampling from the Q-values.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Enable rendering during evaluation (writes PNGs to figs/).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Base random seed for reproducibility.")
    return parser.parse_args(args)


def main() -> None:
    args = parse_args()
    model_path = args.model_path.format(level=args.level)

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Could not find trained model at '{model_path}'. "
            "Run train_rl_agent.py to create it."
        )

    set_global_seeds(args.seed)
    env = AliensGymWrapper(level=args.level, render=args.render)

    try:
        # ``print_system_info`` is only available in SB3 >= 2.1.0.
        model = DQN.load(model_path, print_system_info=False)
    except TypeError:
        model = DQN.load(model_path)

    episode_rewards: list[float] = []
    step_counts: list[int] = []

    for episode in range(args.episodes):
        # Derive a deterministic seed for each episode for reproducibility while
        # still exploring different trajectories when ``--deterministic`` is off.
        total_reward, steps, info = run_episode(
            model,
            env,
            deterministic=args.deterministic,
            seed=args.seed + episode,
        )
        episode_rewards.append(total_reward)
        step_counts.append(steps)
        message = info.get("message", "")
        print(
            f"Episode {episode + 1}/{args.episodes}: reward={total_reward:.2f}, "
            f"steps={steps}, message='{message}'"
        )

    env.close()

    if episode_rewards:
        avg_reward = mean(episode_rewards)
        reward_std = pstdev(episode_rewards) if len(episode_rewards) > 1 else 0.0
        avg_steps = mean(step_counts)
        print(
            "Summary: "
            f"reward_mean={avg_reward:.2f}, reward_std={reward_std:.2f}, "
            f"avg_steps={avg_steps:.1f}"
        )


if __name__ == "__main__":
    main()

