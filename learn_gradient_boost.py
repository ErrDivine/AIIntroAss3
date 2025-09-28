"""Train a supervised Aliens agent using Gradient Boosting."""

from __future__ import annotations

import argparse
from typing import Sequence

from sklearn.ensemble import GradientBoostingClassifier

from supervised import train_and_save_model


def parse_level(value: str) -> int | str:
    return value if value == "all" else int(value)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", type=parse_level, default=0, help="Level to train on (0-4 or 'all').")
    parser.add_argument(
        "--logs",
        nargs="*",
        default=None,
        help="Optional list of log folder names to train on.",
    )
    parser.add_argument(
        "--max-logs",
        type=int,
        default=150,
        help="Cap the number of log folders used (mirrors learn.py default of 150).",
    )
    parser.add_argument(
        "--wins-only",
        action="store_true",
        help="When discovering logs automatically, only use winning trajectories.",
    )
    parser.add_argument("--learning-rate", type=float, default=0.05, help="Gradient boosting learning rate.")
    parser.add_argument("--n-estimators", type=int, default=400, help="Number of boosting stages.")
    parser.add_argument("--max-depth", type=int, default=3, help="Maximum depth of individual estimators.")
    parser.add_argument("--random-state", type=int, default=13, help="Random seed for reproducibility.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Directory to store the trained model bundle.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    estimator = GradientBoostingClassifier(
        learning_rate=args.learning_rate,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state,
    )

    train_and_save_model(
        level=args.level,
        estimator=estimator,
        method="gradient_boosting",
        feature_extractor="extract_binary_features",
        logs=args.logs,
        max_logs=args.max_logs,
        wins_only=args.wins_only,
        output_dir=args.output_dir,
        extra_metadata={
            "learning_rate": args.learning_rate,
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
        },
    )


if __name__ == "__main__":
    main()
