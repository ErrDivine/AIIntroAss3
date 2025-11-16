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
mkdir -p logs models analysis/baseline_figures analysis/enhanced_figures figs

echo "== Step 3: train RL specialists for levels 0-4 =="
python train_rl_agent.py --level 0 --total-timesteps 2600000 
python train_rl_agent.py --level 1 --total-timesteps 2600000 
python train_rl_agent.py --level 2 --total-timesteps 3000000 
python train_rl_agent.py --level 3 --total-timesteps 3000000 
python train_rl_agent.py --level 4 --total-timesteps 3000000 

echo "== Step 4: collect  deterministic wins per level =="
python rl_play.py --level 0 --wins 50 
python rl_play.py --level 1 --wins 50 
python rl_play.py --level 2 --wins 50 
python rl_play.py --level 3 --wins 50 
python rl_play.py --level 4 --wins 50 

echo "== Step 5a: train baseline supervised agents (random forest, logistic, gradient boost) =="
python train_supervised_agents.py --experiments baseline --levels 0 1 2 3 4 all 

echo "== Step 5b: train enhanced-feature supervised agents (gradient boost + engineered features) =="
python train_supervised_agents.py --experiments enhanced --levels 0 1 2 3 4 all 

echo "== Step 6a: evaluate baseline agents (900 episodes) =="
python evaluate_supervised_agents.py \
  --model-dir models/baseline \
  --episodes 30 \
  --max-steps 2500 \
  --output analysis/baseline_scores.json

echo "== Step 6b: evaluate enhanced-feature agents (900 episodes) =="
python evaluate_supervised_agents.py \
  --model-dir models/enhanced \
  --episodes 30 \
  --max-steps 2500 \
  --output analysis/enhanced_scores.json

echo "== Step 7a: analyze baseline evaluation results =="
python analyze_scores.py --scores analysis/baseline_scores.json --output-dir analysis/baseline_figures

echo "== Step 7b: analyze enhanced-feature evaluation results =="
python analyze_scores.py --scores analysis/enhanced_scores.json --output-dir analysis/enhanced_figures

echo "✅ Done. Artifacts are under models/, analysis/baseline_figures/, and analysis/enhanced_figures/"
