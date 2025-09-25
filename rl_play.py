"""Collect winning trajectories from a trained RL agent.

This utility mirrors :mod:`play` but drives the environment with a
Stable-Baselines3 policy instead of human keyboard input.  Every time the agent
wins a game the observation-action history is dumped to
``logs/game_records_lvl{level}_{timestamp}/data.pkl`` so that the resulting data
matches the format produced by manual play sessions.
"""

from __future__ import annotations

import argparse
import os
import pickle
from typing import List, Tuple

from env import AliensEnv
from train_rl_agent import encode_observation, set_global_seeds

try:  # Stable-Baselines3 is required to load and run the trained policy.
    from stable_baselines3 import DQN
except ImportError as exc:  # pragma: no cover - more informative error message.
    raise ImportError(
        "stable-baselines3 is required. Install the optional dependencies with"
        " `pip install -r requirements.txt`."
    ) from exc


class AliensEnvRecorder(AliensEnv):
    """Aliens environment variant that creates log folders on reset."""

    def __init__(self, level: int = 0, render: bool = False):
        super().__init__(level=level, render=render)
        self._update_log_folder()

    def _update_log_folder(self) -> None:
        self.log_folder = f"logs/game_records_lvl{self.level}_{self.timing}"
        os.makedirs(self.log_folder, exist_ok=True)

    def reset(self):  # type: ignore[override]
        observation = super().reset()
        self._update_log_folder()
        return observation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Have a trained RL agent play Aliens and store winning runs.")
    parser.add_argument(
        "--model-path",
        type=str,
        default=os.path.join("models", "dqn_level{level}.zip"),
        help="Path to the trained DQN policy. Supports {level} formatting.",
    )
    parser.add_argument("--level", type=int, default=0, help="Environment level to load.")
    parser.add_argument(
        "--wins",
        type=int,
        default=10,
        help="Number of successful games to collect before stopping.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use a deterministic policy (no exploration during play).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Base random seed for reproducibility.")
    return parser.parse_args()


def run_episode(model: DQN, env: AliensEnvRecorder, *, deterministic: bool) -> Tuple[List[Tuple[list, int]], dict]:
    """Roll out a single episode and capture the observation-action history."""

    data: List[Tuple[list, int]] = []
    observation = env.reset()
    done = False
    info: dict = {}

    while not done:
        encoded_obs = encode_observation(observation)
        action, _ = model.predict(encoded_obs, deterministic=deterministic)
        int_action = int(action)
        data.append((observation, int_action))
        observation, _, done, info = env.step(int_action)

    return data, info


def main() -> None:
    args = parse_args()
    model_path = args.model_path.format(level=args.level)

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Could not find trained model at '{model_path}'. "
            "Run train_rl_agent.py to create it."
        )

    set_global_seeds(args.seed)
    env = AliensEnvRecorder(level=args.level, render=False)

    try:
        model = DQN.load(model_path, print_system_info=False)
    except TypeError:
        model = DQN.load(model_path)

    wins = 0
    attempts = 0

    while wins < args.wins:
        attempts += 1
        data, info = run_episode(model, env, deterministic=args.deterministic)
        message = info.get("message", "")

        if message.endswith("You win."):
            output_path = os.path.join(env.log_folder, "data.pkl")
            with open(output_path, "wb") as f:
                pickle.dump(data, f)
            wins += 1
            print(
                f"Win {wins}/{args.wins}: saved trajectory to '{output_path}'. "
                f"Attempts so far: {attempts}."
            )
        else:
            print(f"Attempt {attempts} ended without a win: '{message}'")
            try:
                os.rmdir(env.log_folder)
            except Exception as e:
                print(f"Could not remove '{e}'")

    print(f"Collected {wins} winning games in {attempts} attempts.")


if __name__ == "__main__":
    main()
