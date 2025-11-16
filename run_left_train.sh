#!/usr/bin/env bash
set -euo pipefail

echo "== Step 5a: train baseline supervised agents (random forest, logistic, gradient boost) =="
python train_supervised_agents.py --experiments baseline --levels 0 1 2 3 4 all --jobs 1 

echo "== Step 5b: train enhanced-feature supervised agents (gradient boost + engineered features) =="
python train_supervised_agents.py --experiments enhanced --levels 0 1 2 3 4 all --jobs 1 

echo "== Step 6a: evaluate baseline agents (900 episodes) =="
python evaluate_supervised_agents.py \
  --model-dir models/baseline \
  --episodes 30 \
  --max-steps 2500 \
  --jobs 1 \
  --output analysis/baseline_scores.json

echo "== Step 6b: evaluate enhanced-feature agents (900 episodes) =="
python evaluate_supervised_agents.py \
  --model-dir models/enhanced \
  --episodes 30 \
  --max-steps 2500 \
  --jobs 1 \
  --output analysis/enhanced_scores.json

echo "== Step 7a: analyze baseline evaluation results =="
python analyze_scores.py --scores analysis/baseline_scores.json --output-dir analysis/baseline_figures

echo "== Step 7b: analyze enhanced-feature evaluation results =="
python analyze_scores.py --scores analysis/enhanced_scores.json --output-dir analysis/enhanced_figures

echo "✅ Done. Artifacts are under models/, analysis/baseline_figures/, and analysis/enhanced_figures/"

