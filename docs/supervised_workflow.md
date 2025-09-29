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
   mkdir -p logs models analysis/baseline_figures analysis/enhanced_figures figs
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
| `learn_random_forest.py` | Random forest | Binary grid features |
| `learn_logistic.py` | Logistic regression | Binary grid features |
| `learn_gradient_boost.py` | Gradient boosting | Binary grid features |
| `learn_gradient_boost_enhanced.py` | Gradient boosting | Enhanced spatial statistics |

Each script relies on the shared `supervised/` package to locate logs, build
features, train models, and persist rich metadata. Feature extraction now uses
Joblib to parallelise across CPU cores, so even the aggregated `all` dataset
loads quickly. The training workload is intentionally split into **two
experiments** so you can compare the baseline binary representation with the new
engineered features without cross-contamination:

- **Phase 1 – Baseline (binary features):** trains `random_forest`, `logistic`,
  and `gradient_boost` models for levels `0`–`4` plus `all`.
- **Phase 2 – Enhanced features:** reruns gradient boosting with the
  engineered feature extractor across the same level specification.

Models are stored under `models/baseline/` or `models/enhanced/` depending on
the phase. Run the automation harness twice—once per experiment—to satisfy the
assignment's requirement of three supervised methods and to keep the feature
comparison isolated:

```bash
# Phase 1: binary-feature baselines (random forest, logistic, gradient boost)
python train_supervised_agents.py --experiments baseline --levels 0 1 2 3 4 all --jobs auto

# Phase 2: enhanced feature sweep (gradient boosting + engineered features)
python train_supervised_agents.py --experiments enhanced --levels 0 1 2 3 4 all --jobs auto
```

`--methods` still accepts fine-grained selections—for example,
`--methods random_forest logistic` with `--experiments baseline` retrains only
the requested pipelines. Use `--dry-run` to preview the generated commands
without executing them. Individual scripts remain callable for bespoke runs:

```bash
python learn_random_forest.py --level all --n-estimators 400 --max-depth 18
python learn_logistic.py --level 0 --c 0.25 --wins-only
python learn_gradient_boost.py --level 2 --n-estimators 600 --learning-rate 0.03
python learn_gradient_boost_enhanced.py --level 3 --n-estimators 500 --learning-rate 0.04
```

`learn_logistic.py` still drives the multi-thread capable `saga` solver with
`n_jobs=-1`, while the tree-based pipelines expose explicit seeds and
parallelism controls to keep comparisons reproducible.

## 4. Evaluating Trained Agents

Because the two experiments should be analysed separately, evaluate the saved models per phase to generate distinct score logs (still 900 games per phase by default):

```bash
# Phase 1 baseline evaluation
python evaluate_supervised_agents.py \
    --model-dir models/baseline \
    --episodes 30 \
    --max-steps 2000 \
    --jobs auto \
    --output analysis/baseline_scores.json

# Phase 2 enhanced-feature evaluation
python evaluate_supervised_agents.py \
    --model-dir models/enhanced \
    --episodes 30 \
    --max-steps 2000 \
    --jobs auto \
    --output analysis/enhanced_scores.json
```

- Models trained on a specific level are evaluated on that level only.
- Models trained with `--level all` are evaluated sequentially on levels `0`–`4`.
- The script seeds NumPy and Python RNGs per episode to keep trials reproducible.
- Use `--jobs` to bound the number of parallel evaluation workers if the
  default `auto` setting is too aggressive for your machine.

Each JSON file contains the per-episode scores and metadata for the corresponding phase, making downstream comparisons straightforward.

## 5. Analysing Evaluation Results

Run the analyser once per JSON file to keep the reporting artefacts grouped by experiment:

```bash
python analyze_scores.py --scores analysis/baseline_scores.json --output-dir analysis/baseline_figures
python analyze_scores.py --scores analysis/enhanced_scores.json --output-dir analysis/enhanced_figures
```

The analyser creates:

- `score_summary.csv` / `.md`: per-method-and-level aggregates (mean/median reward, win rate, average steps, counts).
- `mean_scores_by_level.csv`: comparison of mean scores across methods for each evaluation level.
- `win_rates_by_level.csv`: win-rate pivot table mirroring the assignment's performance comparison requirement.
- `mean_scores_by_feature.csv`: contrasts between baseline and enhanced feature sets.
- Visualisations saved as PNG files showing score distributions, average scores, win rates, and average steps.

Inspect both the CSV and Markdown exports for each phase—they provide the quantitative evidence required for the performance comparison section. You can merge the JSON files later if a combined analysis becomes necessary, but keep the primary artefacts separate to follow the assignment brief.

## 6. Enhanced Feature Experiments

The enhanced-feature phase described above already encapsulates the improved feature extractor workflow. Re-run it whenever you tweak the engineered features or adjust gradient boosting hyperparameters. Keep the outputs under `analysis/enhanced_*` to maintain a clear comparison with the baseline figures generated in `analysis/baseline_*`.

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
3. `python train_supervised_agents.py --experiments baseline --levels 0 1 2 3 4 all --jobs auto`
4. `python train_supervised_agents.py --experiments enhanced --levels 0 1 2 3 4 all --jobs auto`
5. `python evaluate_supervised_agents.py --model-dir models/baseline --episodes 30 --jobs auto --output analysis/baseline_scores.json`
6. `python evaluate_supervised_agents.py --model-dir models/enhanced --episodes 30 --jobs auto --output analysis/enhanced_scores.json`
7. `python analyze_scores.py --scores analysis/baseline_scores.json --output-dir analysis/baseline_figures`
8. `python analyze_scores.py --scores analysis/enhanced_scores.json --output-dir analysis/enhanced_figures`

Following this pipeline produces reproducible supervised learning agents for both experiments, comprehensive evaluation logs (900 test games per phase), and ready-to-use tables and figures for the final report.

> Need everything in one command? Run `./docs/run_supervised_workflow.zsh`
> (or the `.sh` variant) to execute steps 1–5 automatically with the defaults
> described above.
 

## A different script(beta)
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
4. **Supervised training** – launches `train_supervised_agents.py --jobs auto`
   twice: once for the baseline phase (`random_forest`, `logistic`,
   `gradient_boost`) and once for the enhanced-feature sweep
   (`gradient_boost_enhanced`). Bundles are stored under
   `models/baseline/` and `models/enhanced/` respectively. The script forwards
   flags such as `--max-logs` and `--wins-only` so you can mirror the
   data-selection rules introduced in the updated `learn.py`/`plugins.py`.
5. **Evaluation** – executes `evaluate_supervised_agents.py --jobs auto` twice
   (once per model directory) to generate two sets of 900 scored episodes:
   `analysis/baseline_scores.json` and `analysis/enhanced_scores.json`. Adjust
   `--episodes`, `--max-steps`, or `--eval-seed` as needed.
6. **Analysis** – runs `analyze_scores.py` for each JSON file to populate
   `analysis/baseline_figures/` and `analysis/enhanced_figures/` with tables and
   PNG visualisations.

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

By default the harness now launches 18 baseline jobs (3 methods × 6 levels) plus 6 enhanced-feature jobs, storing bundles under `models/baseline/` and `models/enhanced/`. Customise the run if needed:
