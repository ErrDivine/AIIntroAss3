"""Generate tables and figures from supervised agent evaluation scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


plt.switch_backend("Agg")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=str, default="analysis/scores.json", help="Evaluation JSON file.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="analysis/figures",
        help="Directory where tables and figures will be stored.",
    )
    return parser.parse_args(argv)


def load_scores(path: str | Path) -> pd.DataFrame:
    with Path(path).open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return pd.DataFrame(payload["results"])


def save_table(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)
    with path.with_suffix(".md").open("w", encoding="utf-8") as fh:
        fh.write(df.to_markdown(index=False))


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_scores(args.scores)
    if df.empty:
        raise RuntimeError("Score file contains no results.")

    df["win"] = df["win"].astype(bool)

    summary_cols = ["method", "feature_extractor", "train_level", "eval_level"]
    grouped = (
        df.groupby(summary_cols)
        .agg(
            score_mean=("score", "mean"),
            score_median=("score", "median"),
            score_std=("score", "std"),
            games_played=("score", "count"),
            win_rate=("win", "mean"),
            avg_steps=("steps", "mean"),
            steps_std=("steps", "std"),
        )
        .reset_index()
    )
    grouped["win_rate"] = grouped["win_rate"].fillna(0.0)
    save_table(grouped, output_dir / "score_summary.csv")

    pivot = (
        df.pivot_table(
            index="eval_level",
            columns="method",
            values="score",
            aggfunc="mean",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    save_table(pivot, output_dir / "mean_scores_by_level.csv")

    win_pivot = (
        df.pivot_table(
            index="eval_level",
            columns="method",
            values="win",
            aggfunc="mean",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    win_pivot.rename(columns=lambda c: str(c) if c == "eval_level" else f"{c}_win_rate", inplace=True)
    save_table(win_pivot, output_dir / "win_rates_by_level.csv")

    feature_pivot = (
        df.pivot_table(
            index="feature_extractor",
            columns="method",
            values="score",
            aggfunc="mean",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    save_table(feature_pivot, output_dir / "mean_scores_by_feature.csv")

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="eval_level", y="score", hue="method")
    plt.title("Score Distribution by Method and Level")
    plt.xlabel("Evaluation Level")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig(output_dir / "score_distribution_boxplot.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=df,
        x="method",
        y="score",
        hue="feature_extractor",
        estimator="mean",
        errorbar="sd",
    )
    plt.title("Average Score by Method and Feature Set")
    plt.xlabel("Method")
    plt.ylabel("Mean Score")
    plt.tight_layout()
    plt.savefig(output_dir / "mean_scores_by_method.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.pointplot(
        data=df,
        x="eval_level",
        y="win",
        hue="method",
        errorbar="sd",
        dodge=True,
    )
    plt.title("Win Rate by Method and Level")
    plt.xlabel("Evaluation Level")
    plt.ylabel("Win Rate")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(output_dir / "win_rate_by_level.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=df,
        x="method",
        y="steps",
        hue="feature_extractor",
        estimator="mean",
        errorbar="sd",
    )
    plt.title("Average Steps by Method and Feature Set")
    plt.xlabel("Method")
    plt.ylabel("Steps")
    plt.tight_layout()
    plt.savefig(output_dir / "steps_by_method.png", dpi=200)
    plt.close()

    print(f"[done] Tables and figures saved to {output_dir}")


if __name__ == "__main__":
    main()
