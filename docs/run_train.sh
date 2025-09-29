#!/usr/bin/env bash
# file: run_supervised_workflow.sh

# Strict mode: stop on first error, unset vars are errors, pipeline fails if any part fails
set -euo pipefail
# set -x   # uncomment to trace commands as they execute

echo "== Step 0: (once) install dependencies =="
pip install -r requirements.txt

echo "== Step 1: reset stale artefacts =="
python reset_experiment_data.py --targets logs models figs analysis --yes

echo "== Step 2: ensure helper directories exist =="
mkdir -p logs models analysis/figures figs

echo "== Step 3: train RL specialists for levels 0-4 =="
python train_rl_agent.py --level 0 --total-timesteps 3500000 --num-envs auto
python train_rl_agent.py --level 1 --total-timesteps 3500000 --num-envs auto
python train_rl_agent.py --level 2 --total-timesteps 3500000 --num-envs auto
python train_rl_agent.py --level 3 --total-timesteps 3500000 --num-envs auto
python train_rl_agent.py --level 4 --total-timesteps 3500000 --num-envs auto

echo "== Step 4: collect ≥80 deterministic wins per level =="
python rl_play.py --level 0 --wins 100 --deterministic
python rl_play.py --level 1 --wins 100 --deterministic
python rl_play.py --level 2 --wins 100 --deterministic
python rl_play.py --level 3 --wins 100 --deterministic
python rl_play.py --level 4 --wins 100 --deterministic

echo "== Step 5: train supervised agents across methods and levels (incl. 'all') =="
# Default helper trains the required methods over levels 0–4 plus 'all'
python train_supervised_agents.py --levels 0 1 2 3 4 all --jobs auto

echo "== Step 6: evaluate trained agents (900 episodes total by default settings) =="
python evaluate_supervised_agents.py \
  --episodes 30 \
  --max-steps 2500 \
  --jobs auto \
  --output analysis/scores.json

echo "== Step 7: analyze and export tables/figures =="
python analyze_scores.py --scores analysis/scores.json --output-dir analysis/figures

echo "✅ Done. Artifacts are under models/, analysis/, and analysis/figures/"
