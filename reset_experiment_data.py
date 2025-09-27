"""Utility to remove stale logs, models, and figures before new experiments."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Sequence

DEFAULT_TARGETS = ("logs", "models", "figs", "analysis")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--targets",
        nargs="*",
        default=DEFAULT_TARGETS,
        help="Directories whose contents should be deleted.",
    )
    parser.add_argument(
        "--include-hidden",
        action="store_true",
        help="Also delete hidden files (starting with a dot).",
    )
    parser.add_argument(
        "--include-agent-logs",
        action="store_true",
        help="Delete the persistent logs/agent folder as well.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation prompt.",
    )
    return parser.parse_args(argv)


def iter_children(path: Path, include_hidden: bool):
    for child in path.iterdir():
        if not include_hidden and child.name.startswith('.'):
            continue
        yield child


def remove_child(child: Path) -> None:
    if child.is_dir():
        shutil.rmtree(child)
    else:
        child.unlink()


def purge_directory(path: Path, *, include_hidden: bool, include_agent_logs: bool) -> None:
    if not path.exists():
        return

    for child in iter_children(path, include_hidden):
        if path.name == "logs" and child.name == "agent" and not include_agent_logs:
            continue
        remove_child(child)


def confirm(prompt: str) -> bool:
    reply = input(f"{prompt} [y/N]: ").strip().lower()
    return reply in {"y", "yes"}


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    targets = [Path(t) for t in args.targets]

    existing = [t for t in targets if t.exists()]
    if not existing:
        print("No targets exist. Nothing to clean.")
        return

    if not args.yes and not confirm(
        "This will permanently delete experiment artefacts. Continue?"
    ):
        print("Aborted.")
        return

    for target in existing:
        purge_directory(
            target,
            include_hidden=args.include_hidden,
            include_agent_logs=args.include_agent_logs,
        )
        print(f"[clean] Cleared contents of {target}")


if __name__ == "__main__":
    main()
