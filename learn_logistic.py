"""Train a supervised Aliens agent using Logistic Regression."""

from __future__ import annotations

import argparse
from typing import Sequence

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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
    parser.add_argument("--c", type=float, default=1.0, help="Inverse regularisation strength for LogisticRegression.")
    parser.add_argument("--max-iter", type=int, default=1000, help="Maximum iterations for LogisticRegression.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Directory to store the trained model bundle.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    estimator = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=False)),
            (
                "clf",
                LogisticRegression(
                    C=args.c,
                    max_iter=args.max_iter,
                    random_state=args.random_state,
                    multi_class="auto",
                    solver="saga",
                    n_jobs=-1,
                ),
            ),
        ]
    )

    train_and_save_model(
        level=args.level,
        estimator=estimator,
        method="logistic_regression",
        feature_extractor="extract_binary_features",
        logs=args.logs,
        max_logs=args.max_logs,
        wins_only=args.wins_only,
        output_dir=args.output_dir,
        extra_metadata={"C": args.c, "max_iter": args.max_iter},
    )


if __name__ == "__main__":
    main()
