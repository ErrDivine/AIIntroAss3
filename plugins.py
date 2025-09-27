from pathlib import Path
import shutil
import pickle
from typing import Any, Iterable, Tuple

from tqdm import tqdm
from collections import defaultdict

import numpy as np

def get_level_logs(level=0):
    #create folder path
    logs = Path('logs')

    #iterate the file names
    if level == 'all':
        paths = [p.name for p in logs.iterdir() if "lvl" in p.name]
    else:
        paths = [p.name for p in logs.iterdir() if f"lvl{level}" in p.name]

    #print
    print("successfully found:")
    for p in paths:
        print(p)

    return paths

def get_level_logs_win(level=0):
    #create folder path
    logs = Path('logs')

    #iterate the file names
    if level == 'all':
        paths = [p.name for p in logs.iterdir() if "lvl" in p.name and check_outcome(f"logs/{p.name}/data.pkl")[0]=='WIN']
    else:
        paths = [p.name for p in logs.iterdir() if f"lvl{level}" in p.name and check_outcome(f"logs/{p.name}/data.pkl")[0]=='WIN']

    #print
    print("successfully found:")
    for p in paths:
        print(p)

    return paths

def get_level_model(level=0):
    """
    The function which you can use to get the latest model you saved.
    :param level: the game level
    :return:
    """
    models = Path('models')

    iterated = sorted(models.iterdir(), key=lambda p: p.name.lower(), reverse=True)
    if level == 'all':
        paths = [p.name for p in iterated if "lvlall" in p.name]
    else:
        paths = [p.name for p in iterated if f"lvl{level}" in p.name]
    model_path = paths[0]
    print(paths)
    print(f"Using model:{model_path}")
    return "models/"+model_path



def save_agent_differently(src_path):
    src = Path(src_path)
    dst = Path("logs/agent")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(src, dst)

class pkl_outcome():
    # pkl_outcome.py

    def __init__(self):
        # Adjust these if your token names differ
        self.FOE_TOKENS = ("alien",)
        self.PORTAL_TOKENS = ("portalSlow", "portalFast")
        self.AVATAR_TOKEN = "avatar"

    def _flatten_tokens(self,grid: Iterable) -> Iterable[str]:
        """
        Yields all cell tokens from an observation grid shaped like:
          grid[row][col] -> List[str]  (tokens in that cell)
        """
        for row in grid:
            for cell in row:
                for tok in cell:
                    yield tok

    def _infer_outcome_from_last_obs(self,last_obs) -> Tuple[str, str]:
        """
        Apply the same terminal rule used by the env:
          - LOSE: avatar absent
          - WIN: no foes and no portals remain
          - otherwise: UNKNOWN (last frame isn't terminal)
        """
        toks = list(self._flatten_tokens(last_obs))
        print(toks)
        avatar_exists = self.AVATAR_TOKEN in toks
        foe_exists = toks.count('alien') - 1
        print(foe_exists)
        portal_exists = any(p in toks for p in self.PORTAL_TOKENS)

        if not avatar_exists:
            return "LOSE", "Avatar destroyed / missing in final frame"
        if not foe_exists and not portal_exists:
            return "WIN", "All foes and portals cleared in final frame"
        return "UNKNOWN", "Final frame is non-terminal by rule"

    def outcome_from_pkl(self,path: str) -> Tuple[str, str]:
        """
        Load a saved game .pkl and return (RESULT, REASON)
          RESULT ∈ {"WIN","LOSE","UNKNOWN"}
        No framework edits required.

        Supports:
          - legacy format: list/tuple of (observation, action) pairs
          - dict format: {"trajectory": ..., "meta": ...} (if meta has 'result', it will be used)
        """
        with open(path, "rb") as f:
            blob: Any = pickle.load(f)

        # If meta already stores a result, use it directly

        seq = blob

        if not seq:
            return "UNKNOWN", "Empty trajectory"

        last = seq[-1]
        #print(last)
        # Expect (obs, action, ...). First item is the observation grid.
        last_obs = last[0] if isinstance(last, (list, tuple)) and last else last
        print(last_obs)
        return self._infer_outcome_from_last_obs(last_obs)

def check_outcome(path):
    checker = pkl_outcome()
    return checker.outcome_from_pkl(path)



def remove_duplicate_states(X, y, max_diff_bits=1):
    """Remove duplicate or near-duplicate feature vectors.

    Parameters
    ----------
    X : list[np.ndarray]
        Feature vectors extracted from observations.
    y : list
        Corresponding action labels for each feature vector.
    max_diff_bits : int, optional
        Maximum number of differing feature positions for two states to be
        considered similar.  The default of 1 keeps the similarity check
        strict so that only almost-identical states are filtered.

    Returns
    -------
    tuple[list[np.ndarray], list]
        Filtered lists without duplicate states.
    """

    filtered_X = []
    filtered_y = []
    # Group candidate states by the number of active features; states that are
    # within ``max_diff_bits`` distance must have similar activation counts, so
    # this dramatically cuts down the comparisons that need to be made.
    buckets = defaultdict(list)
    removed_duplicates = 0
    removed_similar = 0
    conflicting_labels = 0

    for features, action in tqdm(zip(X, y),desc="Removing duplicate states",total=len(X)):
        features = np.asarray(features)
        ones_count = int(np.count_nonzero(features))

        similar_index = None
        is_exact_match = False
        lower = max(0, ones_count - max_diff_bits)
        upper = ones_count + max_diff_bits

        for count_key in range(lower, upper + 1):
            for candidate_idx in buckets.get(count_key, ()):
                candidate = filtered_X[candidate_idx]
                if candidate.shape != features.shape:
                    continue
                diff = int(np.count_nonzero(candidate != features))
                if diff <= max_diff_bits:
                    similar_index = candidate_idx
                    is_exact_match = diff == 0
                    break
            if similar_index is not None:
                break

        if similar_index is not None:
            removed_duplicates += 1
            if not is_exact_match:
                removed_similar += 1
            if filtered_y[similar_index] != action:
                conflicting_labels += 1
            continue

        buckets[ones_count].append(len(filtered_X))
        filtered_X.append(features)
        filtered_y.append(action)

    if removed_duplicates:
        message = (
            f"[info] removed {removed_duplicates} near-duplicate states"
        )
        if removed_similar:
            message += f" ({removed_similar} with small feature differences)"
        if conflicting_labels:
            message += f"; {conflicting_labels} conflicting labels skipped"
        print(message)

    return filtered_X, filtered_y



#testification area
if __name__ == "__main__":
    check_outcome('logs/agent/game_records_lvl0_2025-09-24_22-21-20')