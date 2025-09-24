"""Train a deep Q-learning agent for the Aliens environment.

This module inspects :mod:`env` to extract the discrete game mechanics and
implements a light-weight Deep Q-Network (DQN) trainer using only NumPy and the
standard library.  The environment exposes a grid of categorical entities
(`wall`, `alien`, `bomb`, etc.), a discrete action space (no-op, move left,
move right, shoot), and deterministic physics aside from alien bomb drops.

The agent learns from self-play by interacting with :class:`env.AliensEnv` and
storing transitions in a replay buffer.  Each observation is converted into a
binary feature map describing the presence of every entity on the grid along
with a handful of high-level summary statistics (avatar position, nearest
threat distance, remaining portals).  A small fully-connected neural network is
implemented directly in NumPy to predict Q-values.  Training uses mini-batch
TD(0) updates with a slowly-updated target network for stability.

Example
-------
    $ python train_rl_agent.py --level 0 --episodes 2000 --render-eval

The script will periodically print progress, evaluate the greedy policy, and
save the learned parameters under ``models/dqn_level0.npz`` by default.

Note
----
The implementation avoids third-party deep learning frameworks so that it can
run in restricted environments (only NumPy and scikit-learn are dependencies of
this repository).  The NumPy-based neural network supports saving/loading so
that the trained agent can later be deployed for inference in ``test.py`` or a
custom evaluation script.
"""
from __future__ import annotations

import argparse
import math
import os
import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from env import AliensEnv


ENTITY_CHANNELS: Tuple[str, ...] = (
    "wall",
    "base",
    "portalSlow",
    "portalFast",
    "alien",
    "sam",
    "bomb",
    "avatar",
)
ENTITY_INDEX: Dict[str, int] = {name: idx for idx, name in enumerate(ENTITY_CHANNELS)}


def set_global_seeds(seed: int) -> None:
    """Seed every source of randomness used by the trainer."""
    random.seed(seed)
    np.random.seed(seed)


def encode_observation(obs: Sequence[Sequence[Sequence[str]]]) -> np.ndarray:
    """Convert the nested list observation into a feature vector.

    The base encoding is an 8-channel binary image where each channel indicates
    the presence of one entity type at every grid coordinate.  To give the
    learner a sense of global structure, several aggregate statistics are
    appended:

    * Avatar x/y position (normalized to [-1, 1]).
    * Count of aliens and bombs (scaled by grid size).
    * Manhattan distance from the avatar to the nearest alien and bomb
      (normalized by grid perimeter).
    * Fraction of portals still active (derived from the observation).
    """

    height = len(obs)
    width = len(obs[0]) if height else 0
    grid = np.zeros((len(ENTITY_CHANNELS), height, width), dtype=np.float32)

    avatar_x, avatar_y = 0, 0
    portals = 0
    alien_positions: List[Tuple[int, int]] = []
    bomb_positions: List[Tuple[int, int]] = []

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

    # Estimate active portals by counting portal tiles that still contain the portal.
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


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """Experience replay with a fixed maximum capacity."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.buffer: Deque[Transition] = deque(maxlen=capacity)

    def push(self, transition: Transition) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


class QNetwork:
    """Simple fully-connected neural network implemented with NumPy."""

    def __init__(self, input_dim: int, action_dim: int, hidden: Sequence[int], lr: float = 1e-3,
                 weight_scale: float = 0.02) -> None:
        layer_sizes = [input_dim, *hidden, action_dim]
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
            w = np.random.randn(in_dim, out_dim).astype(np.float32) * weight_scale
            b = np.zeros(out_dim, dtype=np.float32)
            self.weights.append(w)
            self.biases.append(b)
        self.lr = lr

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, List[Tuple[np.ndarray, np.ndarray]]]:
        activations: List[Tuple[np.ndarray, np.ndarray]] = []
        out = x
        for idx, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = out @ w + b
            activations.append((out, z))
            if idx < len(self.weights) - 1:
                out = np.maximum(z, 0.0)
            else:
                out = z
        return out, activations

    def predict(self, x: np.ndarray) -> np.ndarray:
        out, _ = self.forward(x)
        return out

    def update(self, states: np.ndarray, actions: np.ndarray, targets: np.ndarray) -> float:
        q_values, activations = self.forward(states)
        batch_size = states.shape[0]

        chosen_q = q_values[np.arange(batch_size), actions]
        td_error = chosen_q - targets
        loss = float(0.5 * np.mean(td_error ** 2))

        grad_output = np.zeros_like(q_values)
        grad_output[np.arange(batch_size), actions] = td_error / batch_size

        grad = grad_output
        for layer in reversed(range(len(self.weights))):
            inputs, pre_act = activations[layer]
            dW = inputs.T @ grad
            db = np.sum(grad, axis=0)

            # Gradient descent step
            self.weights[layer] -= self.lr * dW.astype(np.float32)
            self.biases[layer] -= self.lr * db.astype(np.float32)

            if layer > 0:
                prev_inputs, prev_pre_act = activations[layer - 1]
                grad = grad @ self.weights[layer].T
                grad = grad * (prev_pre_act > 0)
            else:
                # No further layers; break after computing gradient for input layer.
                break

        return loss

    def copy_from(self, other: "QNetwork") -> None:
        for idx in range(len(self.weights)):
            self.weights[idx][...] = other.weights[idx]
            self.biases[idx][...] = other.biases[idx]

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, *self.weights, *self.biases, allow_pickle=False)

    @classmethod
    def load(cls, path: str, input_dim: int, action_dim: int, hidden: Sequence[int], lr: float) -> "QNetwork":
        network = cls(input_dim, action_dim, hidden, lr)
        data = np.load(path)
        total_layers = len(network.weights)
        for idx in range(total_layers):
            network.weights[idx][...] = data[f'arr_{idx}']
            network.biases[idx][...] = data[f'arr_{idx + total_layers}']
        return network


class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int] = (256, 256),
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        target_sync_interval: int = 250,
    ) -> None:
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.target_sync_interval = target_sync_interval
        self.step_count = 0

        self.online_net = QNetwork(state_dim, action_dim, hidden_sizes, lr)
        self.target_net = QNetwork(state_dim, action_dim, hidden_sizes, lr)
        self.target_net.copy_from(self.online_net)

    def act(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        q_values = self.online_net.predict(state[None, :])[0]
        return int(np.argmax(q_values))

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def learn(self, buffer: ReplayBuffer, batch_size: int) -> Optional[float]:
        if len(buffer) < batch_size:
            return None
        transitions = buffer.sample(batch_size)
        states = np.stack([t.state for t in transitions])
        actions = np.array([t.action for t in transitions], dtype=np.int64)
        rewards = np.array([t.reward for t in transitions], dtype=np.float32)
        next_states = np.stack([t.next_state for t in transitions])
        dones = np.array([t.done for t in transitions], dtype=np.float32)

        next_q = self.target_net.predict(next_states)
        max_next_q = np.max(next_q, axis=1)
        targets = rewards + self.gamma * (1.0 - dones) * max_next_q

        loss = self.online_net.update(states, actions, targets)

        self.step_count += 1
        if self.step_count % self.target_sync_interval == 0:
            self.target_net.copy_from(self.online_net)

        return loss

    def save(self, path: str) -> None:
        self.online_net.save(path)


def run_episode(
    env: AliensEnv,
    agent: DQNAgent,
    buffer: ReplayBuffer,
    max_steps: int,
    gamma: float,
    batch_size: int,
) -> Tuple[float, int]:
    obs = env.reset()
    state = encode_observation(obs)
    total_reward = 0.0

    for step in range(max_steps):
        action = agent.act(state)
        next_obs, reward, done, _ = env.step(action)
        next_state = encode_observation(next_obs)

        buffer.push(Transition(state, action, reward, next_state, done))
        loss = agent.learn(buffer, batch_size)
        if loss is not None and math.isnan(loss):
            raise RuntimeError("Loss became NaN; try lowering the learning rate or adjusting features.")

        total_reward += reward
        state = next_state

        if done:
            break

    agent.decay_epsilon()
    return total_reward, step + 1


def evaluate_agent(env: AliensEnv, agent: DQNAgent, episodes: int, max_steps: int) -> Tuple[float, float]:
    """Run deterministic evaluations (epsilon=0) and report stats."""
    prev_epsilon = agent.epsilon
    agent.epsilon = 0.0
    rewards = []
    lengths = []
    for _ in range(episodes):
        obs = env.reset()
        state = encode_observation(obs)
        total_reward = 0.0
        for step in range(max_steps):
            action = agent.act(state)
            next_obs, reward, done, _ = env.step(action)
            state = encode_observation(next_obs)
            total_reward += reward
            if done:
                break
        rewards.append(total_reward)
        lengths.append(step + 1)
    agent.epsilon = prev_epsilon
    return float(np.mean(rewards)), float(np.mean(lengths))


def train(
    level: int,
    episodes: int,
    buffer_capacity: int,
    batch_size: int,
    max_steps: int,
    hidden_sizes: Sequence[int],
    lr: float,
    gamma: float,
    epsilon: float,
    epsilon_min: float,
    epsilon_decay: float,
    target_sync: int,
    seed: int,
    eval_interval: int,
    eval_episodes: int,
    render_eval: bool,
    save_path: str,
) -> None:
    set_global_seeds(seed)
    env = AliensEnv(level=level, render=False)
    eval_env = AliensEnv(level=level, render=render_eval)

    sample_state = encode_observation(env.reset())
    agent = DQNAgent(
        state_dim=sample_state.shape[0],
        action_dim=len(env.action_space),
        hidden_sizes=hidden_sizes,
        lr=lr,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        target_sync_interval=target_sync,
    )

    buffer = ReplayBuffer(buffer_capacity)

    for episode in range(1, episodes + 1):
        reward, steps = run_episode(env, agent, buffer, max_steps, gamma, batch_size)

        if episode % eval_interval == 0:
            avg_reward, avg_len = evaluate_agent(eval_env, agent, eval_episodes, max_steps)
            print(
                f"Episode {episode:05d} | epsilon={agent.epsilon:.3f} | "
                f"train_reward={reward:.2f} | steps={steps} | "
                f"eval_reward={avg_reward:.2f} | eval_len={avg_len:.1f}"
            )
        else:
            print(
                f"Episode {episode:05d} | epsilon={agent.epsilon:.3f} | "
                f"train_reward={reward:.2f} | steps={steps}"
            )

    agent.save(save_path)
    print(f"Saved trained agent to {save_path}")


def parse_args(args: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DQN agent for Aliens.")
    parser.add_argument("--level", type=int, default=0, help="Environment level to train on.")
    parser.add_argument("--episodes", type=int, default=1500, help="Number of training episodes.")
    parser.add_argument("--buffer-capacity", type=int, default=20000, help="Replay buffer size.")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size for updates.")
    parser.add_argument("--max-steps", type=int, default=2000, help="Maximum steps per episode.")
    parser.add_argument(
        "--hidden-sizes",
        type=lambda s: tuple(int(x) for x in s.split(",")),
        default="256,256",
        help="Comma-separated hidden layer sizes for the Q-network.",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for the Q-network.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument("--epsilon", type=float, default=1.0, help="Initial epsilon for exploration.")
    parser.add_argument("--epsilon-min", type=float, default=0.05, help="Minimum epsilon.")
    parser.add_argument(
        "--epsilon-decay",
        type=float,
        default=0.995,
        help="Multiplicative epsilon decay applied after each episode.",
    )
    parser.add_argument(
        "--target-sync",
        type=int,
        default=250,
        help="Number of gradient steps between target network synchronizations.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=50,
        help="How often to run policy evaluation episodes.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=5,
        help="Number of deterministic evaluation runs per interval.",
    )
    parser.add_argument(
        "--render-eval",
        action="store_true",
        help="Render frames during evaluation runs (useful for debugging).",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=os.path.join("models", "dqn_level{level}.npz"),
        help="Destination path for the trained weights.",
    )
    return parser.parse_args(args)


def main() -> None:
    args = parse_args()
    save_path = args.save_path.format(level=args.level)
    train(
        level=args.level,
        episodes=args.episodes,
        buffer_capacity=args.buffer_capacity,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        hidden_sizes=args.hidden_sizes,
        lr=args.lr,
        gamma=args.gamma,
        epsilon=args.epsilon,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        target_sync=args.target_sync,
        seed=args.seed,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        render_eval=args.render_eval,
        save_path=save_path,
    )


if __name__ == "__main__":
    main()
