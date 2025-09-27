"""Training helpers for supervised Aliens agents."""

from __future__ import annotations

import datetime as _dt
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from sklearn.base import BaseEstimator

import plugins
from .data import Observation, build_dataset, load_game_records
from .features import FEATURE_EXTRACTORS


@dataclass
class ModelMetadata:
    method: str
    level: int | str
    feature_extractor: str
    logs: list[str]
    created_at: str
    extra: dict | None = None


@dataclass
class ModelBundle:
    model: BaseEstimator
    metadata: ModelMetadata


FeatureFn = Callable[[Observation], "np.ndarray"]


def _resolve_feature_fn(name: str) -> FeatureFn:
    try:
        return FEATURE_EXTRACTORS[name]
    except KeyError as exc:  # pragma: no cover - defensive programming
        raise ValueError(f"Unknown feature extractor '{name}'.") from exc


def train_and_save_model(
    *,
    level: int | str,
    estimator: BaseEstimator,
    method: str,
    feature_extractor: str,
    logs: Sequence[str] | None = None,
    output_dir: str | Path = "models",
    filename_template: str | None = None,
    extra_metadata: dict | None = None,
) -> Path:
    """Train `estimator` on data from `level` and persist the fitted bundle."""

    if logs is None:
        logs = plugins.get_level_logs(level)

    records = load_game_records(logs)
    feature_fn = _resolve_feature_fn(feature_extractor)
    X, y = build_dataset(records, feature_fn)

    estimator.fit(X, y)

    timestamp = _dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    metadata = ModelMetadata(
        method=method,
        level=level,
        feature_extractor=feature_extractor,
        logs=list(logs),
        created_at=timestamp,
        extra=extra_metadata,
    )
    bundle = ModelBundle(model=estimator, metadata=metadata)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if filename_template is None:
        filename_template = "{method}_lvl{level}_{timestamp}.pkl"

    safe_level = str(level)
    file_name = filename_template.format(
        method=method,
        level=safe_level,
        timestamp=timestamp,
        feature=feature_extractor,
    )
    path = output_path / file_name

    with path.open("wb") as fh:
        pickle.dump(bundle, fh)

    print(f"[save] model saved to {path}")
    return path


def load_model_bundle(path: str | Path) -> ModelBundle:
    """Load a model bundle from disk, upgrading legacy payloads when necessary."""

    with Path(path).open("rb") as fh:
        payload = pickle.load(fh)

    if isinstance(payload, ModelBundle):
        return payload

    # Legacy payload: bare estimator without metadata (from learn.py)
    if isinstance(payload, BaseEstimator):
        metadata = ModelMetadata(
            method="random_forest",
            level="unknown",
            feature_extractor="extract_binary_features",
            logs=[],
            created_at="unknown",
            extra={"legacy": True},
        )
        return ModelBundle(model=payload, metadata=metadata)

    raise TypeError(
        "Unsupported model payload encountered. Expected ModelBundle or BaseEstimator."
    )


__all__ = [
    "ModelMetadata",
    "ModelBundle",
    "train_and_save_model",
    "load_model_bundle",
]
