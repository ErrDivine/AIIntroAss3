from pathlib import Path
import shutil
import pickle
from typing import Any, Iterable, Tuple
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


#testification area
if __name__ == "__main__":
    check_outcome('logs/agent/game_records_lvl0_2025-09-24_22-21-20')