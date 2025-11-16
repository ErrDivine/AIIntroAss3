"""Batch evaluation script for supervised Aliens agents."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from env import AliensEnv
from supervised import FEATURE_EXTRACTORS, load_model_bundle
from concurrent.futures import ProcessPoolExecutor, as_completed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-paths",
        nargs="*",
        default=None,
        help="Explicit list of model bundle paths to evaluate.",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models",
        help="Directory to search when --model-paths is omitted.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.pkl",
        help="Glob pattern used with --model-dir to discover bundles.",
    )
    parser.add_argument("--episodes", type=int, default=30, help="Episodes per model/level combination.")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=2000,
        help="Safety cap on the number of steps per episode.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2024,
        help="Base random seed to reproduce stochastic behaviour.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="analysis/scores.json",
        help="Where to store the raw evaluation scores (JSON).",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=0,
        help="Parallel workers for evaluating bundles (0 auto-detects cores).",
    )
    return parser.parse_args(argv)


def discover_model_paths(model_dir: str, pattern: str) -> List[Path]:
    base = Path(model_dir)
    if not base.exists():
        raise FileNotFoundError(f"Model directory '{model_dir}' does not exist.")
    return sorted(base.glob(pattern))


def extract_feature_fn(name: str):
    try:
        return FEATURE_EXTRACTORS[name]
    except KeyError as exc:
        raise ValueError(
            f"Model references unknown feature extractor '{name}'."
        ) from exc


def ensure_level(level: int | str) -> int | str:
    if isinstance(level, str) and level != "all":
        try:
            return int(level)
        except ValueError:
            return level
    return level


def evaluation_levels(level: int | str) -> List[int]:
    level = ensure_level(level)
    if level == "all":
        return [0, 1, 2, 3, 4]
    if isinstance(level, int):
        return [level]
    raise ValueError(f"Unsupported level specifier: {level!r}")


def play_episode(env: AliensEnv, model, feature_fn, max_steps: int) -> Dict:
    observation = env.reset()
    done = False
    total_reward = 0.0
    steps = 0
    info: Dict = {}

    while not done and steps < max_steps:
        features = feature_fn(observation).reshape(1, -1)
        action = int(model.predict(features)[0])
        observation, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1

    message = info.get("message", "")
    win = isinstance(message, str) and message.endswith("You win.")
    return {
        "score": float(total_reward),
        "steps": steps,
        "done": bool(done),
        "message": message,
        "win": bool(win),
    }


def evaluate_bundle(bundle_path: Path, *, episodes: int, max_steps: int, seed: int) -> List[Dict]:
    bundle = load_model_bundle(bundle_path)
    feature_fn = extract_feature_fn(bundle.metadata.feature_extractor)
    model = bundle.model

    results: List[Dict] = []
    for level in evaluation_levels(bundle.metadata.level):
        env = AliensEnv(level=level, render=False)
        for episode in range(episodes):
            random.seed(seed + episode)
            np.random.seed(seed + episode)
            outcome = play_episode(env, model, feature_fn, max_steps)
            outcome.update(
                {
                    "model_path": str(bundle_path),
                    "method": bundle.metadata.method,
                    "feature_extractor": bundle.metadata.feature_extractor,
                    "train_level": bundle.metadata.level,
                    "eval_level": level,
                    "episode": episode,
                }
            )
            results.append(outcome)
    return results


def _resolve_jobs(requested: int, tasks: int) -> int:
    if tasks <= 0:
        return 0
    if requested > 0:
        return max(1, min(requested, tasks))
    detected = os.cpu_count() or 1
    return max(1, min(detected, tasks))


def _evaluate_worker(path: str, episodes: int, max_steps: int, seed: int) -> List[Dict]:
    return evaluate_bundle(Path(path), episodes=episodes, max_steps=max_steps, seed=seed)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    if args.model_paths:
        model_paths = [Path(p) for p in args.model_paths]
    else:
        model_paths = discover_model_paths(args.model_dir, args.pattern)

    if not model_paths:
        raise RuntimeError("No model bundles found for evaluation.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_results: List[Dict] = []
    jobs = _resolve_jobs(args.jobs, len(model_paths))

    if jobs <= 1 or len(model_paths) <= 1:
        for model_path in model_paths:
            print(f"[eval] Evaluating {model_path}")
            all_results.extend(
                evaluate_bundle(
                    model_path,
                    episodes=args.episodes,
                    max_steps=args.max_steps,
                    seed=args.seed,
                )
            )
    else:
        print(f"[eval] Using {jobs} parallel workers")
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = {
                pool.submit(
                    _evaluate_worker,
                    str(model_path),
                    args.episodes,
                    args.max_steps,
                    args.seed,
                ): model_path
                for model_path in model_paths
            }
            for future in as_completed(futures):
                model_path = futures[future]
                print(f"[eval] Completed {model_path}")
                all_results.extend(future.result())

    metadata = {
        "episodes_per_combination": args.episodes,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "models_evaluated": [str(p) for p in model_paths],
        "total_records": len(all_results),
    }

    with output_path.open("w", encoding="utf-8") as fh:
        json.dump({"metadata": metadata, "results": all_results}, fh, indent=2)

    print(f"[done] Stored evaluation results in {output_path}")


if __name__ == "__main__":
    main()
