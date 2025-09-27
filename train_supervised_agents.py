"""Batch-training harness for the supervised Aliens agents."""

from __future__ import annotations

import argparse
from typing import Callable, Dict, Iterable, List, Sequence

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


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    levels = parse_levels(args.levels)

    for method in args.methods:
        trainer = TRAINERS[method]
        for level in levels:
            cmd_args = build_arguments(method, level, args)
            printable = " ".join(cmd_args)
            print(f"[train] {method} level={level} :: {printable}")
            if args.dry_run:
                continue
            trainer(cmd_args)


if __name__ == "__main__":
    main()

