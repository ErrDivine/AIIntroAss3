"""Feature extraction utilities for supervised Aliens agents."""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np

from .data import Observation

OBJECT_MAPPING: Dict[str, int] = {
    "floor": 0,
    "wall": 1,
    "avatar": 2,
    "alien": 3,
    "bomb": 4,
    "portalSlow": 5,
    "portalFast": 6,
    "sam": 7,
    "base": 8,
}


def _cell_to_feature(cell: Iterable[str]) -> List[int]:
    vector = [0] * len(OBJECT_MAPPING)
    for obj in cell:
        idx = OBJECT_MAPPING.get(obj)
        if idx is not None:
            vector[idx] = 1
    return vector


def extract_binary_features(observation: Observation) -> np.ndarray:
    """Baseline feature extractor matching the original `learn.py` implementation."""

    features: List[int] = []
    for row in observation:
        for cell in row:
            features.extend(_cell_to_feature(cell))
    return np.array(features, dtype=np.float32)


def _count_objects(observation: Observation) -> np.ndarray:
    counts = np.zeros(len(OBJECT_MAPPING), dtype=np.float32)
    for row in observation:
        for cell in row:
            for obj in cell:
                idx = OBJECT_MAPPING.get(obj)
                if idx is not None:
                    counts[idx] += 1
    return counts


def _avatar_position(observation: Observation) -> Tuple[int, int] | None:
    for y, row in enumerate(observation):
        for x, cell in enumerate(row):
            if "avatar" in cell:
                return x, y
    return None


def _object_positions(observation: Observation, token: str) -> List[Tuple[int, int]]:
    coords: List[Tuple[int, int]] = []
    for y, row in enumerate(observation):
        for x, cell in enumerate(row):
            if token in cell:
                coords.append((x, y))
    return coords


def extract_enhanced_features(observation: Observation) -> np.ndarray:
    """Improved feature extractor with spatial statistics.

    The enhanced representation augments the binary encoding with counts of
    each token, the avatar's normalised position, and the minimum Manhattan
    distance to key hazards and objectives.
    """

    base = extract_binary_features(observation)
    counts = _count_objects(observation)

    height = len(observation)
    width = len(observation[0]) if height else 0
    max_dist = width + height if width and height else 1

    avatar_pos = _avatar_position(observation)
    if avatar_pos is None:
        avatar_norm = np.array([0.0, 0.0], dtype=np.float32)
    else:
        avatar_norm = np.array([
            avatar_pos[0] / max(1, width - 1),
            avatar_pos[1] / max(1, height - 1),
        ], dtype=np.float32)

    def min_distance(token: str) -> float:
        positions = _object_positions(observation, token)
        if not positions or avatar_pos is None:
            return float(max_dist)
        ax, ay = avatar_pos
        return min(abs(ax - x) + abs(ay - y) for x, y in positions)

    distances = np.array(
        [
            min_distance("alien") / max_dist,
            min_distance("bomb") / max_dist,
            min_distance("portalSlow") / max_dist,
            min_distance("portalFast") / max_dist,
            min_distance("base") / max_dist,
        ],
        dtype=np.float32,
    )

    enhanced = np.concatenate([base, counts, avatar_norm, distances])
    return enhanced.astype(np.float32, copy=False)


FEATURE_EXTRACTORS = {
    "extract_binary_features": extract_binary_features,
    "extract_enhanced_features": extract_enhanced_features,
}

__all__ = [
    "OBJECT_MAPPING",
    "extract_binary_features",
    "extract_enhanced_features",
    "FEATURE_EXTRACTORS",
]
