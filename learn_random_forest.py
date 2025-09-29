"""Train a supervised Aliens agent using Random Forests."""

from __future__ import annotations

import argparse
from typing import Sequence

from sklearn.ensemble import RandomForestClassifier

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
        help="Optional list of log folder names to train on. Overrides automatic discovery.",
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
    parser.add_argument("--n-estimators", type=int, default=300, help="Number of trees in the forest.")
    parser.add_argument("--max-depth", type=int, default=None, help="Maximum depth of each tree.")
    parser.add_argument(
        "--min-samples-leaf",
        type=int,
        default=1,
        help="Minimum samples required at a leaf node.",
    )
    parser.add_argument("--random-state", type=int, default=7, help="Random seed for reproducibility.")
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallel workers for tree construction (mirrors scikit-learn's n_jobs).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Directory to store the trained model bundle.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    estimator = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        verbose=0,
    )

    train_and_save_model(
        level=args.level,
        estimator=estimator,
        method="random_forest",
        feature_extractor="extract_binary_features",
        logs=args.logs,
        max_logs=args.max_logs,
        wins_only=args.wins_only,
        output_dir=args.output_dir,
        extra_metadata={
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "min_samples_leaf": args.min_samples_leaf,
            "n_jobs": args.n_jobs,
        },
    )


if __name__ == "__main__":
    main()
