#!/usr/bin/env bash
# Comprehensive automation harness for the supervised experiment pipeline.
# Mirrors docs/run_supervised_workflow.zsh but uses Bash for portability.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"

# ---------------------------------------------------------------------------
# Defaults that mirror the documented workflow
# ---------------------------------------------------------------------------
declare -a RL_LEVELS=(0 1 2 3 4)
declare -a SUPER_LEVELS=(0 1 2 3 4 all)
declare -a METHODS=(logistic gradient_boost gradient_boost_enhanced)
declare -a RESET_TARGETS=(logs models figs analysis)
MODELS_DIR="models"
LOGS_DIR="logs"
SCORES_OUT="analysis/scores.json"
ANALYSIS_DIR="analysis/figures"
RL_TOTAL_TIMESTEPS="2500000"
RL_SEED="0"
RL_WINS="80"
RL_DETERMINISTIC=1
MAX_LOGS="150"
WINS_ONLY=0
EVAL_EPISODES="30"
EVAL_MAX_STEPS="2000"
EVAL_SEED="2024"
DRY_RUN=0
SKIP_RESET=0
SKIP_RL_TRAIN=0
SKIP_RL_PLAY=0
SKIP_SUPERVISED=0
SKIP_EVAL=0
SKIP_ANALYSIS=0
RL_TRAIN_JOBS="auto"
RL_PLAY_JOBS="auto"
SUPERVISED_JOBS="auto"
EVAL_JOBS="auto"

# Extra per-stage arguments. Repeat the associated flag to append multiple tokens.
declare -a RL_TRAIN_EXTRA=()
declare -a RL_PLAY_EXTRA=()
declare -a SUPERVISED_EXTRA=()
declare -a EVAL_EXTRA=()
declare -a ANALYZE_EXTRA=()

usage() {
  cat <<'USAGE'
Usage: run_supervised_workflow.sh [options]

General options:
  --python <bin>              Python interpreter to use (default: $PYTHON_BIN or 'python').
  --dry-run                   Print the planned commands without executing them.

Workflow toggles:
  --skip-reset                Do not clear previous artefacts with reset_experiment_data.py.
  --skip-rl-train             Skip RL training (expects existing DQN checkpoints).
  --skip-rl-play              Skip RL trajectory collection (expects logs/ to exist).
  --skip-supervised           Skip train_supervised_agents.py.
  --skip-eval                 Skip evaluate_supervised_agents.py.
  --skip-analysis             Skip analyze_scores.py.

Environment reset:
  --reset-targets <vals...>   Override directories cleared before the run.

Reinforcement learning:
  --rl-levels <vals...>       Levels to train/collect RL data for (default: 0 1 2 3 4).
  --rl-total-timesteps <int>  Timesteps per RL training job (default: 2500000).
  --rl-seed <int>             Base RNG seed for RL training (default: 0).
  --rl-wins <int>             Winning trajectories to record per level (default: 80).
  --rl-deterministic          Use deterministic policies when collecting wins (default).
  --no-rl-deterministic       Allow exploration during RL play.
  --rl-train-extra <token>    Additional token forwarded to train_rl_agent.py (repeatable).
  --rl-play-extra <token>     Additional token forwarded to rl_play.py (repeatable).
  --rl-train-jobs <n|auto>    Parallel RL training jobs (default: auto, matches CPU cores).
  --rl-play-jobs <n|auto>     Parallel RL play jobs (default: auto).

Supervised training:
  --super-levels <vals...>    Levels to train supervised agents for (default: 0 1 2 3 4 all).
  --methods <names...>        Supervised methods to run (default: logistic gradient_boost gradient_boost_enhanced).
  --models-dir <path>         Output directory for supervised bundles (default: models).
  --max-logs <count|none>     Cap on log folders per level (default: 150, use 'none' for no cap).
  --wins-only                 Restrict datasets to winning logs only.
  --no-wins-only              Use every discovered log (default).
  --supervised-extra <token>  Additional token forwarded to train_supervised_agents.py (repeatable).
  --supervised-jobs <n|auto>  Parallel workers for supervised training (default: auto).

Evaluation:
  --episodes <int>            Episodes per model/level pair (default: 30).
  --max-steps <int>           Max steps per evaluation episode (default: 2000).
  --eval-seed <int>           Base RNG seed for evaluation (default: 2024).
  --scores-out <path>         Destination JSON for evaluation results (default: analysis/scores.json).
  --eval-extra <token>        Additional token forwarded to evaluate_supervised_agents.py (repeatable).
  --eval-jobs <n|auto>        Parallel workers passed to evaluate_supervised_agents.py (default: auto).

Analysis:
  --analysis-dir <path>       Directory to store tables and figures (default: analysis/figures).
  --analyze-extra <token>     Additional token forwarded to analyze_scores.py (repeatable).

Repeat flags such as --rl-train-extra to construct complex option sequences for
individual stages.
USAGE
}

parse_list_flag() {
  declare -n target=$1
  shift
  target=()
  if [[ $# -eq 0 ]]; then
    echo "run_supervised_workflow.sh: expected at least one value for list flag" >&2
    exit 64
  fi
  local token
  for token in "$@"; do
    if [[ $token == --* ]]; then
      echo "run_supervised_workflow.sh: list flag missing value before '$token'" >&2
      exit 64
    fi
    target+=("$token")
  done
}

append_token() {
  declare -n target=$1
  local value=$2
  target+=("$value")
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)
      [[ $# -ge 2 ]] || { echo "run_supervised_workflow.sh: --python expects a value" >&2; exit 64; }
      PYTHON_BIN=$2
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --skip-reset)
      SKIP_RESET=1
      shift
      ;;
    --skip-rl-train)
      SKIP_RL_TRAIN=1
      shift
      ;;
    --skip-rl-play)
      SKIP_RL_PLAY=1
      shift
      ;;
    --skip-supervised)
      SKIP_SUPERVISED=1
      shift
      ;;
    --skip-eval)
      SKIP_EVAL=1
      shift
      ;;
    --skip-analysis)
      SKIP_ANALYSIS=1
      shift
      ;;
    --reset-targets)
      shift
      declare -a values=()
      while [[ $# -gt 0 && $1 != --* ]]; do
        values+=("$1")
        shift
      done
      parse_list_flag RESET_TARGETS "${values[@]}"
      ;;
    --rl-levels)
      shift
      declare -a rl_values=()
      while [[ $# -gt 0 && $1 != --* ]]; do
        rl_values+=("$1")
        shift
      done
      parse_list_flag RL_LEVELS "${rl_values[@]}"
      ;;
    --rl-total-timesteps)
      [[ $# -ge 2 ]] || { echo "run_supervised_workflow.sh: --rl-total-timesteps expects a value" >&2; exit 64; }
      RL_TOTAL_TIMESTEPS=$2
      shift 2
      ;;
    --rl-seed)
      [[ $# -ge 2 ]] || { echo "run_supervised_workflow.sh: --rl-seed expects a value" >&2; exit 64; }
      RL_SEED=$2
      shift 2
      ;;
    --rl-wins)
      [[ $# -ge 2 ]] || { echo "run_supervised_workflow.sh: --rl-wins expects a value" >&2; exit 64; }
      RL_WINS=$2
      shift 2
      ;;
    --rl-deterministic)
      RL_DETERMINISTIC=1
      shift
      ;;
    --no-rl-deterministic)
      RL_DETERMINISTIC=0
      shift
      ;;
    --rl-train-extra)
      [[ $# -ge 2 ]] || { echo "run_supervised_workflow.sh: --rl-train-extra expects a token" >&2; exit 64; }
      append_token RL_TRAIN_EXTRA "$2"
      shift 2
      ;;
    --rl-play-extra)
      [[ $# -ge 2 ]] || { echo "run_supervised_workflow.sh: --rl-play-extra expects a token" >&2; exit 64; }
      append_token RL_PLAY_EXTRA "$2"
      shift 2
      ;;
    --rl-train-jobs)
      [[ $# -ge 2 ]] || { echo "run_supervised_workflow.sh: --rl-train-jobs expects a value" >&2; exit 64; }
      RL_TRAIN_JOBS=$2
      shift 2
      ;;
    --rl-play-jobs)
      [[ $# -ge 2 ]] || { echo "run_supervised_workflow.sh: --rl-play-jobs expects a value" >&2; exit 64; }
      RL_PLAY_JOBS=$2
      shift 2
      ;;
    --super-levels)
      shift
      declare -a super_values=()
      while [[ $# -gt 0 && $1 != --* ]]; do
        super_values+=("$1")
        shift
      done
      parse_list_flag SUPER_LEVELS "${super_values[@]}"
      ;;
    --methods)
      shift
      declare -a method_values=()
      while [[ $# -gt 0 && $1 != --* ]]; do
        method_values+=("$1")
        shift
      done
      parse_list_flag METHODS "${method_values[@]}"
      ;;
    --models-dir)
      [[ $# -ge 2 ]] || { echo "run_supervised_workflow.sh: --models-dir expects a value" >&2; exit 64; }
      MODELS_DIR=$2
      shift 2
      ;;
    --max-logs)
      [[ $# -ge 2 ]] || { echo "run_supervised_workflow.sh: --max-logs expects a value" >&2; exit 64; }
      MAX_LOGS=$2
      shift 2
      ;;
    --wins-only)
      WINS_ONLY=1
      shift
      ;;
    --no-wins-only)
      WINS_ONLY=0
      shift
      ;;
    --supervised-extra)
      [[ $# -ge 2 ]] || { echo "run_supervised_workflow.sh: --supervised-extra expects a token" >&2; exit 64; }
      append_token SUPERVISED_EXTRA "$2"
      shift 2
      ;;
    --supervised-jobs)
      [[ $# -ge 2 ]] || { echo "run_supervised_workflow.sh: --supervised-jobs expects a value" >&2; exit 64; }
      SUPERVISED_JOBS=$2
      shift 2
      ;;
    --episodes)
      [[ $# -ge 2 ]] || { echo "run_supervised_workflow.sh: --episodes expects a value" >&2; exit 64; }
      EVAL_EPISODES=$2
      shift 2
      ;;
    --max-steps)
      [[ $# -ge 2 ]] || { echo "run_supervised_workflow.sh: --max-steps expects a value" >&2; exit 64; }
      EVAL_MAX_STEPS=$2
      shift 2
      ;;
    --eval-seed)
      [[ $# -ge 2 ]] || { echo "run_supervised_workflow.sh: --eval-seed expects a value" >&2; exit 64; }
      EVAL_SEED=$2
      shift 2
      ;;
    --scores-out)
      [[ $# -ge 2 ]] || { echo "run_supervised_workflow.sh: --scores-out expects a value" >&2; exit 64; }
      SCORES_OUT=$2
      shift 2
      ;;
    --eval-extra)
      [[ $# -ge 2 ]] || { echo "run_supervised_workflow.sh: --eval-extra expects a token" >&2; exit 64; }
      append_token EVAL_EXTRA "$2"
      shift 2
      ;;
    --eval-jobs)
      [[ $# -ge 2 ]] || { echo "run_supervised_workflow.sh: --eval-jobs expects a value" >&2; exit 64; }
      EVAL_JOBS=$2
      shift 2
      ;;
    --analysis-dir)
      [[ $# -ge 2 ]] || { echo "run_supervised_workflow.sh: --analysis-dir expects a value" >&2; exit 64; }
      ANALYSIS_DIR=$2
      shift 2
      ;;
    --analyze-extra)
      [[ $# -ge 2 ]] || { echo "run_supervised_workflow.sh: --analyze-extra expects a token" >&2; exit 64; }
      append_token ANALYZE_EXTRA "$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "run_supervised_workflow.sh: unrecognised argument '$1'" >&2
      exit 64
      ;;
  esac
done

if [[ $# -gt 0 ]]; then
  echo "run_supervised_workflow.sh: unexpected positional arguments: $*" >&2
  exit 64
fi

cd -- "$REPO_ROOT"

run_cmd() {
  local -a cmd=("$@")
  printf '  '
  printf '%q ' "${cmd[@]}"
  printf '\n'
  if [[ $DRY_RUN -eq 0 ]]; then
    "${cmd[@]}"
  fi
}

run_section() {
  printf '[workflow] %s\n' "$1"
}

CPU_COUNT_CACHE=""

get_cpu_count() {
  if [[ -n $CPU_COUNT_CACHE ]]; then
    printf '%s' "$CPU_COUNT_CACHE"
    return
  fi
  local result=""
  if command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    result=$("$PYTHON_BIN" - <<'PY'
import os
print(os.cpu_count() or 1)
PY
  ) || result=""
  fi
  if [[ -z $result && "$PYTHON_BIN" != "python3" ]]; then
    if command -v python3 >/dev/null 2>&1; then
      result=$(python3 - <<'PY'
import os
print(os.cpu_count() or 1)
PY
      ) || result=""
    fi
  fi
  if [[ -z $result ]] && command -v getconf >/dev/null 2>&1; then
    result=$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)
  fi
  if [[ -z $result ]]; then
    result=1
  fi
  CPU_COUNT_CACHE=$result
  printf '%s' "$result"
}

resolve_jobs() {
  local spec=$1
  local tasks=$2
  if [[ -z $spec || $spec == auto ]]; then
    local cpus
    cpus=$(get_cpu_count)
    if [[ -z $tasks || $tasks -le 0 ]]; then
      printf '%s' "$cpus"
      return
    fi
    if (( cpus > tasks )); then
      cpus=$tasks
    fi
    (( cpus < 1 )) && cpus=1
    printf '%s' "$cpus"
  else
    printf '%s' "$spec"
  fi
}

PIDS=()
CMDS=()

wait_for_slot() {
  local concurrency=$1
  while (( ${#PIDS[@]} >= concurrency && concurrency > 0 )); do
    local pid=${PIDS[0]}
    local cmd=${CMDS[0]}
    PIDS=(${PIDS[@]:1})
    CMDS=(${CMDS[@]:1})
    if [[ $DRY_RUN -eq 0 ]]; then
      wait "$pid"
      local status=$?
      if (( status != 0 )); then
        echo "[error] command failed (exit $status): $cmd" >&2
        exit $status
      fi
    fi
  done
}

launch_background() {
  local concurrency=$1
  shift
  local -a cmd=("$@")
  printf '  '
  printf '%q ' "${cmd[@]}"
  printf '\n'
  if [[ $DRY_RUN -eq 1 ]]; then
    return
  fi
  wait_for_slot "$concurrency"
  ("${cmd[@]}") &
  local pid=$!
  PIDS+=($pid)
  CMDS+=("$(printf '%q ' "${cmd[@]}")")
}

wait_for_all() {
  if [[ $DRY_RUN -eq 1 ]]; then
    PIDS=()
    CMDS=()
    return
  fi
  local idx=0
  for pid in "${PIDS[@]}"; do
    wait "$pid"
    local status=$?
    local cmd=${CMDS[$idx]}
    if (( status != 0 )); then
      echo "[error] command failed (exit $status): $cmd" >&2
      exit $status
    fi
    idx=$((idx + 1))
  done
  PIDS=()
  CMDS=()
}

rl_task_count=${#RL_LEVELS[@]}
rl_play_task_count=${#RL_LEVELS[@]}
super_task_count=$(( ${#METHODS[@]} * ${#SUPER_LEVELS[@]} ))

RL_TRAIN_PARALLEL=$(resolve_jobs "$RL_TRAIN_JOBS" "$rl_task_count")
RL_PLAY_PARALLEL=$(resolve_jobs "$RL_PLAY_JOBS" "$rl_play_task_count")
SUPERVISED_PARALLEL=$(resolve_jobs "$SUPERVISED_JOBS" "$super_task_count")
EVAL_PARALLEL=$(resolve_jobs "$EVAL_JOBS" "$(get_cpu_count)")

if [[ $SKIP_RESET -eq 0 ]]; then
  run_section "Resetting previous artefacts"
  reset_cmd=("$PYTHON_BIN" reset_experiment_data.py --yes)
  if [[ ${#RESET_TARGETS[@]} -gt 0 ]]; then
    reset_cmd+=(--targets "${RESET_TARGETS[@]}")
  fi
  run_cmd "${reset_cmd[@]}"
fi

if [[ $SKIP_RL_TRAIN -eq 0 ]]; then
  run_section "Training RL agents"
  if (( RL_TRAIN_PARALLEL <= 1 )); then
    for level in "${RL_LEVELS[@]}"; do
      cmd=("$PYTHON_BIN" train_rl_agent.py --level "$level" --total-timesteps "$RL_TOTAL_TIMESTEPS" --seed "$RL_SEED" --save-path "${MODELS_DIR}/dqn_level${level}" --num-envs auto)
      if [[ ${#RL_TRAIN_EXTRA[@]} -gt 0 ]]; then
        cmd+=("${RL_TRAIN_EXTRA[@]}")
      fi
      run_cmd "${cmd[@]}"
    done
  else
    PIDS=()
    CMDS=()
    for level in "${RL_LEVELS[@]}"; do
      cmd=("$PYTHON_BIN" train_rl_agent.py --level "$level" --total-timesteps "$RL_TOTAL_TIMESTEPS" --seed "$RL_SEED" --save-path "${MODELS_DIR}/dqn_level${level}" --num-envs auto)
      if [[ ${#RL_TRAIN_EXTRA[@]} -gt 0 ]]; then
        cmd+=("${RL_TRAIN_EXTRA[@]}")
      fi
      launch_background "$RL_TRAIN_PARALLEL" "${cmd[@]}"
    done
    wait_for_all
  fi
fi

if [[ $SKIP_RL_PLAY -eq 0 ]]; then
  run_section "Collecting winning trajectories"
  mkdir -p -- "$LOGS_DIR"
  if (( RL_PLAY_PARALLEL <= 1 )); then
    for level in "${RL_LEVELS[@]}"; do
      cmd=("$PYTHON_BIN" rl_play.py --level "$level" --wins "$RL_WINS" --seed "$RL_SEED" --model-path "${MODELS_DIR}/dqn_level${level}.zip")
      if [[ $RL_DETERMINISTIC -eq 1 ]]; then
        cmd+=(--deterministic)
      fi
      if [[ ${#RL_PLAY_EXTRA[@]} -gt 0 ]]; then
        cmd+=("${RL_PLAY_EXTRA[@]}")
      fi
      run_cmd "${cmd[@]}"
    done
  else
    PIDS=()
    CMDS=()
    for level in "${RL_LEVELS[@]}"; do
      cmd=("$PYTHON_BIN" rl_play.py --level "$level" --wins "$RL_WINS" --seed "$RL_SEED" --model-path "${MODELS_DIR}/dqn_level${level}.zip")
      if [[ $RL_DETERMINISTIC -eq 1 ]]; then
        cmd+=(--deterministic)
      fi
      if [[ ${#RL_PLAY_EXTRA[@]} -gt 0 ]]; then
        cmd+=("${RL_PLAY_EXTRA[@]}")
      fi
      launch_background "$RL_PLAY_PARALLEL" "${cmd[@]}"
    done
    wait_for_all
  fi
fi

if [[ $SKIP_SUPERVISED -eq 0 ]]; then
  run_section "Training supervised agents"
  cmd=("$PYTHON_BIN" train_supervised_agents.py --output-dir "$MODELS_DIR" --levels "${SUPER_LEVELS[@]}" --methods "${METHODS[@]}")
  if [[ $MAX_LOGS != "none" ]]; then
    cmd+=(--max-logs "$MAX_LOGS")
  fi
  if [[ $WINS_ONLY -eq 1 ]]; then
    cmd+=(--wins-only)
  fi
  if [[ -n $SUPERVISED_PARALLEL ]]; then
    cmd+=(--jobs "$SUPERVISED_PARALLEL")
  fi
  if [[ ${#SUPERVISED_EXTRA[@]} -gt 0 ]]; then
    cmd+=("${SUPERVISED_EXTRA[@]}")
  fi
  run_cmd "${cmd[@]}"
fi

if [[ $SKIP_EVAL -eq 0 ]]; then
  run_section "Evaluating supervised agents"
  mkdir -p -- "$(dirname -- "$SCORES_OUT")"
  cmd=("$PYTHON_BIN" evaluate_supervised_agents.py --model-dir "$MODELS_DIR" --episodes "$EVAL_EPISODES" --max-steps "$EVAL_MAX_STEPS" --seed "$EVAL_SEED" --output "$SCORES_OUT" --jobs "$EVAL_PARALLEL")
  if [[ ${#EVAL_EXTRA[@]} -gt 0 ]]; then
    cmd+=("${EVAL_EXTRA[@]}")
  fi
  run_cmd "${cmd[@]}"
fi

if [[ $SKIP_ANALYSIS -eq 0 ]]; then
  run_section "Analysing evaluation results"
  mkdir -p -- "$ANALYSIS_DIR"
  cmd=("$PYTHON_BIN" analyze_scores.py --scores "$SCORES_OUT" --output-dir "$ANALYSIS_DIR")
  if [[ ${#ANALYZE_EXTRA[@]} -gt 0 ]]; then
    cmd+=("${ANALYZE_EXTRA[@]}")
  fi
  run_cmd "${cmd[@]}"
fi

if [[ $DRY_RUN -eq 1 ]]; then
  printf '[workflow] Dry-run complete. No commands were executed.\n'
else
  printf '[workflow] Experiment pipeline finished. Artefacts live in:\n'
  printf '  Models:    %s\n' "$MODELS_DIR"
  printf '  Logs:      %s\n' "$LOGS_DIR"
  printf '  Scores:    %s\n' "$SCORES_OUT"
  printf '  Analysis:  %s\n' "$ANALYSIS_DIR"
fi
