"""Batch-training harness for the supervised Aliens agents."""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import learn_gradient_boost
import learn_gradient_boost_enhanced
import learn_logistic


TrainerFn = Callable[[Sequence[str]], None]


TRAINERS: Dict[str, TrainerFn] = {
    "logistic": learn_logistic.main,
    "gradient_boost": learn_gradient_boost.main,
    "gradient_boost_enhanced": learn_gradient_boost_enhanced.main,
}

DEFAULT_LEVELS: List[str] = ["0", "1", "2", "3", "4", "all"]


def parse_levels(values: Iterable[str]) -> List[str]:
    levels: List[str] = []
    for value in values:
        token = value.strip().lower()
        if token == "all":
            levels.append("all")
            continue
        try:
            level_int = int(token)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid level token: {value!r}") from exc
        if not 0 <= level_int <= 4:
            raise argparse.ArgumentTypeError("Level must be between 0 and 4 or 'all'.")
        levels.append(str(level_int))
    return levels


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--levels",
        nargs="*",
        default=DEFAULT_LEVELS,
        help="Levels to train on (0-4 or 'all').",
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        choices=sorted(TRAINERS.keys()),
        default=sorted(TRAINERS.keys()),
        help="Subset of training pipelines to execute.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Directory where trained bundles will be stored.",
    )
    parser.add_argument(
        "--max-logs",
        type=int,
        default=150,
        help="Cap the number of log folders per level (mirrors learn.py default of 150).",
    )
    parser.add_argument(
        "--wins-only",
        action="store_true",
        help="When discovering logs automatically, only use winning trajectories.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=0,
        help="Parallel workers for training (0 auto-detects CPU cores).",
    )
    parser.add_argument("--logistic-c", type=float, default=1.0, dest="logistic_c")
    parser.add_argument("--logistic-max-iter", type=int, default=1000, dest="logistic_max_iter")
    parser.add_argument("--logistic-random-state", type=int, default=42, dest="logistic_random_state")
    parser.add_argument("--gb-learning-rate", type=float, default=0.05, dest="gb_learning_rate")
    parser.add_argument("--gb-n-estimators", type=int, default=400, dest="gb_n_estimators")
    parser.add_argument("--gb-max-depth", type=int, default=3, dest="gb_max_depth")
    parser.add_argument("--gb-random-state", type=int, default=13, dest="gb_random_state")
    parser.add_argument("--enh-learning-rate", type=float, default=0.05, dest="enh_learning_rate")
    parser.add_argument("--enh-n-estimators", type=int, default=400, dest="enh_n_estimators")
    parser.add_argument("--enh-max-depth", type=int, default=3, dest="enh_max_depth")
    parser.add_argument("--enh-random-state", type=int, default=99, dest="enh_random_state")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands that would be executed without training.",
    )
    return parser.parse_args(argv)


def build_arguments(method: str, level: str, args: argparse.Namespace) -> List[str]:
    base = ["--level", level, "--output-dir", args.output_dir]
    if args.max_logs is not None:
        base.extend(["--max-logs", str(args.max_logs)])
    if args.wins_only:
        base.append("--wins-only")
    if method == "logistic":
        base.extend(
            [
                "--c",
                str(args.logistic_c),
                "--max-iter",
                str(args.logistic_max_iter),
                "--random-state",
                str(args.logistic_random_state),
            ]
        )
    elif method == "gradient_boost":
        base.extend(
            [
                "--learning-rate",
                str(args.gb_learning_rate),
                "--n-estimators",
                str(args.gb_n_estimators),
                "--max-depth",
                str(args.gb_max_depth),
                "--random-state",
                str(args.gb_random_state),
            ]
        )
    elif method == "gradient_boost_enhanced":
        base.extend(
            [
                "--learning-rate",
                str(args.enh_learning_rate),
                "--n-estimators",
                str(args.enh_n_estimators),
                "--max-depth",
                str(args.enh_max_depth),
                "--random-state",
                str(args.enh_random_state),
            ]
        )
    else:  # pragma: no cover - defensive guard
        raise ValueError(f"Unsupported method: {method}")
    return base


def _invoke_trainer(method: str, cmd_args: Sequence[str]) -> None:
    trainer = TRAINERS[method]
    trainer(cmd_args)


def _resolve_jobs(requested: int, tasks: int) -> int:
    if tasks <= 0:
        return 0
    if requested > 0:
        return max(1, min(requested, tasks))
    detected = os.cpu_count() or 1
    return max(1, min(detected, tasks))


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    levels = parse_levels(args.levels)

    tasks: List[Tuple[str, str, List[str]]] = []
    for method in args.methods:
        for level in levels:
            cmd_args = build_arguments(method, level, args)
            tasks.append((method, level, cmd_args))

    for method, level, cmd_args in tasks:
        printable = " ".join(cmd_args)
        print(f"[train] {method} level={level} :: {printable}")

    if args.dry_run:
        return

    jobs = _resolve_jobs(args.jobs, len(tasks))

    if jobs <= 1 or len(tasks) <= 1:
        for method, _, cmd_args in tasks:
            _invoke_trainer(method, cmd_args)
        return

    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futures = {
            pool.submit(_invoke_trainer, method, cmd_args): (method, level)
            for method, level, cmd_args in tasks
        }
        for future in as_completed(futures):
            method, level = futures[future]
            future.result()
            print(f"[done] {method} level={level}")


if __name__ == "__main__":
    main()

