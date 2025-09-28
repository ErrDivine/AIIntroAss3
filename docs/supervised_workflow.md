# Supervised Workflow Usage Guide

This document explains how to prepare data, train supervised learning agents, evaluate them, and maintain experiment artefacts using the utilities provided in this repository. It covers the scripts introduced in the latest update and highlights how their functionality works together to streamline experimentation.

## 1. Environment Setup and Cleanup

1. Install Python 3.9 or newer.
2. From the repository root, install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Reset stale artefacts before starting a fresh round of experiments. This keeps the `logs/`, `models/`, `figs/`, and `analysis/` directories free from previous work as required in the assignment brief:
   ```bash
   python reset_experiment_data.py --yes
   ```
   Use `--targets` if you need to tailor which directories are purged.
4. (Optional) Recreate helper directories if the cleanup removed them:
   ```bash
   mkdir -p logs models analysis/figures figs
   ```

## 2. Collecting Training Trajectories

Supervised agents learn from demonstrations produced by reinforcement learning (RL) agents. Repeat the following for each game level `0`–`4` (and the aggregated `all` dataset when needed):

1. **Train RL specialists** for the level of interest:
   ```bash
   python train_rl_agent.py --level <LEVEL_ID> --total-timesteps 2500000 --num-envs auto
   ```
   Adjust `--total-timesteps` if you need longer training. The new `--num-envs`
   flag defaults to `auto`, spawning a vectorised `SubprocVecEnv` across all
   available CPU cores so data collection scales with your hardware. Use
   consistent seeds to keep runs comparable, and drop the value back to `1` if
   you need strictly serial environments for debugging.
2. **Collect 80 or more winning trajectories** with the trained RL agent:
   ```bash
   python rl_play.py --level <LEVEL_ID> --wins 80 --deterministic
   ```
   The command records successful games under `logs/`. Increase `--wins` for larger datasets. Remove any empty folders that appear due to RL logging quirks.
3. **Verify the dataset** using `plugins.get_level_logs_win` or by listing `logs/` to ensure each level has usable trajectories before moving on.

## 3. Batch Training the Supervised Agents

The repository provides three supervised pipelines driven by scikit-learn. Use the following cheat-sheet to pick the right scri
pt for your run:

| Script | Learning Method | Feature Type |
| ------ | ---------------- | ------------ |
| `learn.py` | Random forest (legacy baseline) | Binary grid features |
| `learn_logistic.py` | Logistic regression | Binary grid features |
| `learn_gradient_boost.py` | Gradient boosting | Binary grid features |
| `learn_gradient_boost_enhanced.py` | Gradient boosting | Enhanced spatial statistics |

Each script relies on the shared `supervised/` package to locate logs, build
features, train models, and persist rich metadata. Feature extraction now uses
Joblib to parallelise across CPU cores, so even the aggregated `all` dataset
loads quickly. To satisfy the assignment requirement of training **three
methods across six level specifications (0–4 plus `all`)**, use the automation
helper:

```bash
python train_supervised_agents.py
```

If you prefer a single command that reproduces the entire workflow—from
resetting artefacts to exporting analysis plots—the repository ships with two
automation harnesses located beside this document:

```bash
# macOS (or other systems with zsh available)
./docs/run_supervised_workflow.zsh

# Bash-friendly variant for Linux environments
./docs/run_supervised_workflow.sh
```

Running either script performs the following sequence with the defaults
described in this guide while automatically parallelising CPU-heavy stages:

1. **Clean start** – calls `reset_experiment_data.py --yes --targets logs models figs analysis` unless `--skip-reset` is supplied.
2. **RL training** – trains DQN specialists for levels `0`–`4` with
   `train_rl_agent.py --total-timesteps 2500000 --num-envs auto` and a configurable seed.
   The harness fans these jobs out across available CPU cores (override with
   `--rl-train-jobs`).
3. **Trajectory capture** – records 80 deterministic wins per level via
   `rl_play.py`, producing the datasets consumed by the supervised trainers.
   Collection processes can run concurrently through `--rl-play-jobs`.
4. **Supervised training** – launches `train_supervised_agents.py --jobs auto` for the
   logistic regression and both gradient boosting variants across levels
   `0`–`4` plus the aggregated `all` split (18 jobs by default). The script
   forwards flags such as `--max-logs` and `--wins-only` so you can mirror the
   data-selection rules introduced in the updated `learn.py`/`plugins.py`.
5. **Evaluation** – executes `evaluate_supervised_agents.py --jobs auto` to generate the
   900 scored episodes stored in `analysis/scores.json` (configurable through
   `--episodes`, `--max-steps`, `--scores-out`, and `--eval-seed`).
6. **Analysis** – runs `analyze_scores.py` to populate `analysis/figures/` with
   tables and PNG visualisations.

Useful toggles include `--dry-run` (preview commands without executing them)
plus `--skip-rl-train`, `--skip-rl-play`, `--skip-supervised`, `--skip-eval`,
and `--skip-analysis` for partial reruns. Repeatable options such as
`--rl-train-extra`, `--supervised-extra`, or `--eval-extra` let you append
additional tokens to the underlying Python invocations, enabling advanced
hyper-parameter tweaks without editing the harness. Concurrency knobs like
`--rl-train-jobs`, `--rl-play-jobs`, `--supervised-jobs`, and `--eval-jobs`
override the auto-detected worker counts when you need to bound resource usage.

Both helpers respect the refreshed log-handling defaults by capping automatic
discovery at 150 folders per level (override with `--max-logs` or `--max-logs
none`) and by supporting `--wins-only` filtering. The dataset loader
automatically removes duplicate or near-identical states when the updated
`plugins.remove_duplicate_states` helper is available on the refreshed `main`
branch, keeping the feature matrices compact and consistent with the latest
tooling.

By default this issues 18 training jobs (3 methods × 6 level specs) and stores timestamped `.pkl` bundles inside `models/`. Customise the run if needed:

```bash
python train_supervised_agents.py \
    --levels 0 1 2 3 4 all \
    --methods logistic gradient_boost_enhanced \
    --logistic-c 0.5 --logistic-max-iter 300 \
    --enh-learning-rate 0.03 --enh-n-estimators 500 \
    --wins-only
```

Use `--dry-run` to preview the generated commands without executing them. You can still launch individual scripts manually when experimenting with alternative hyperparameters.

Common one-off training commands if you prefer direct control:

```bash
python learn_logistic.py --level 0
python learn_gradient_boost.py --level all
python learn_gradient_boost_enhanced.py --level 2
```

`learn_logistic.py` now drives the multi-thread capable `saga` solver with
`n_jobs=-1`, so it automatically utilises every available core. The legacy
`learn.py` random forest baseline already benefits from `n_estimators=300` and
`n_jobs=-1` for parallel tree building.


## 4. Evaluating Trained Agents

After producing the 18 supervised bundles, run the evaluation harness once to execute the required 900 tests (30 episodes for each level/method combination):

```bash
python evaluate_supervised_agents.py \
    --episodes 30 \
    --max-steps 2000 \
    --jobs auto \
    --output analysis/scores.json
```

- Models trained on a specific level are evaluated on that level only.
- Models trained with `--level all` are evaluated sequentially on levels `0`–`4`.
- The script seeds NumPy and Python RNGs per episode to keep trials reproducible.
- Use `--jobs` to bound the number of parallel evaluation workers if the
  default `auto` setting is too aggressive for your machine.

The output file contains every episode result along with metadata that captures the training method, feature extractor, train level, evaluation level, and whether the agent won.

## 5. Analysing Evaluation Results

Use `analyze_scores.py` to build publication-ready tables and figures:

```bash
python analyze_scores.py --scores analysis/scores.json --output-dir analysis/figures
```

The analyser creates:

- `score_summary.csv` / `.md`: per-method-and-level aggregates (mean/median reward, win rate, average steps, counts).
- `mean_scores_by_level.csv`: comparison of mean scores across methods for each evaluation level.
- `win_rates_by_level.csv`: win-rate pivot table mirroring the assignment's performance comparison requirement.
- `mean_scores_by_feature.csv`: contrasts between baseline and enhanced feature sets.
- Visualisations saved as PNG files showing score distributions, average scores, win rates, and average steps.

Inspect both the CSV and Markdown exports when preparing your report—they provide the quantitative evidence required for the performance comparison section.

## 6. Enhanced Feature Experiments

To evaluate the improved feature extractor:

1. Use `train_supervised_agents.py --methods gradient_boost_enhanced --levels 0 1 2 3 4 all` (or call `learn_gradient_boost_enhanced.py` manually) to retrain models with the enhanced features.
2. Re-run `evaluate_supervised_agents.py` to append fresh results or write them to a separate JSON file (e.g., `analysis/enhanced_scores.json`).
3. Invoke `analyze_scores.py` on each JSON file to generate dedicated tables/figures. You can also merge evaluation files beforehand to compare baseline and enhanced features within a single analysis run.
4. Use the win-rate and score tables to argue which feature extractor performs better at each level.

## 7. Maintaining Experiments

- **Resetting quickly**: run `python reset_experiment_data.py --targets logs models figs analysis --yes` whenever you need a clean slate.
- **Reproducibility**: every training script exposes a `--random-state` argument. Set the seed explicitly in batch runs to make comparisons fair.
- **Automation**: the new `train_supervised_agents.py` script can be combined with shell loops or schedulers for repeated sweeps. Pair it with `evaluate_supervised_agents.py` in a single shell script to reproduce the 18-model experiment end to end.

### Handy Utility Scripts

- `reset_experiment_data.py`: Remove stale `logs/`, `models/`, `figs/`, or `analysis/` artefacts before a new run.
- `plugins.py`: Helper functions for locating logs, loading models, and working with metadata.
- `play.py`: Keyboard-controlled gameplay for manual data collection or quick smoke tests.


## 8. Troubleshooting Tips

| Issue | Possible Fix |
| --- | --- |
| `FileNotFoundError` during training | Confirm that `logs/<LEVEL>` folders contain `.pkl` trajectories generated by `rl_play.py`. |
| Poor evaluation performance | Collect more wins, rebalance the dataset, or tweak hyperparameters (`--logistic-c`, `--gb-learning-rate`, etc.). |
| Evaluation appears to hang | Lower `--max-steps` or inspect problematic trajectories within `logs/`. |
| Memory pressure in enhanced models | Reduce `--enh-n-estimators` or work on fewer levels per run. |

## 9. Quick Start Checklist

1. `python reset_experiment_data.py --targets logs models figs analysis --yes`
2. Train RL agents and gather wins for levels `0`–`4` using `train_rl_agent.py --num-envs auto` and `rl_play.py` (optionally with
   `--rl-play-jobs` for the harness).
3. `python train_supervised_agents.py --levels 0 1 2 3 4 all --jobs auto`
4. `python evaluate_supervised_agents.py --episodes 30 --jobs auto --output analysis/scores.json`
5. `python analyze_scores.py --scores analysis/scores.json --output-dir analysis/figures`

Following this pipeline produces reproducible supervised learning agents, comprehensive evaluation logs (900 test games), and ready-to-use tables and figures for the final report.

> Need everything in one command? Run `./docs/run_supervised_workflow.zsh`
> (or the `.sh` variant) to execute steps 1–5 automatically with the defaults
> described above.
