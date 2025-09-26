"""Data loading helpers for supervised Aliens agents."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Callable, Iterable, List, Sequence, Tuple

import numpy as np

Observation = List[List[List[str]]]
Trajectory = Tuple[Observation, int]


def load_game_records(log_names: Sequence[str], root: str | Path = "logs") -> List[Trajectory]:
    """Load observation-action pairs from the given log folders."""

    records: List[Trajectory] = []
    base = Path(root)
    for name in log_names:
        data_path = base / name / "data.pkl"
        if not data_path.exists():
            print(f"[skip] missing log file: {data_path}")
            continue
        if data_path.stat().st_size == 0:
            print(f"[skip] empty log file: {data_path}")
            continue

        loaded_any = False
        with data_path.open("rb") as fh:
            while True:
                try:
                    obj = pickle.load(fh)
                except EOFError:
                    break
                except pickle.UnpicklingError as exc:
                    print(f"[skip] corrupt pickle '{data_path}': {exc}")
                    break
                else:
                    loaded_any = True
                    if isinstance(obj, list):
                        records.extend(tuple(sample) for sample in obj)  # type: ignore[arg-type]
                    elif isinstance(obj, dict) and "trajectory" in obj:
                        seq = obj["trajectory"]
                        records.extend(tuple(sample) for sample in seq)  # type: ignore[arg-type]
                    else:
                        records.append(tuple(obj))  # type: ignore[arg-type]
        if not loaded_any:
            print(f"[warn] no trajectories found in {data_path}")
    return records


def build_dataset(
    records: Sequence[Trajectory],
    feature_fn: Callable[[Observation], np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert observation-action pairs into arrays suitable for scikit-learn."""

    if not records:
        raise ValueError("No records provided. Cannot build dataset.")

    features: List[np.ndarray] = []
    labels: List[int] = []
    for observation, action in records:
        feat = feature_fn(observation)
        if feat.ndim != 1:
            raise ValueError(
                "Feature extractor must return a 1-D vector. "
                f"Got shape {feat.shape} from {feature_fn.__name__}."
            )
        features.append(feat.astype(np.float32, copy=False))
        labels.append(int(action))

    X = np.vstack(features)
    y = np.array(labels, dtype=np.int64)
    return X, y


__all__ = ["load_game_records", "build_dataset", "Observation", "Trajectory"]
