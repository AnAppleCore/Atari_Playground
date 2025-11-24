#!/usr/bin/env bash

# Evaluate the continual-learning experiments run by run_continual_experiments.sh
# This script assumes you have already activated the `atari_rl` conda
# environment and are running from the Atari_Playground directory.
#
# The evaluation results (JSON) will be saved under
#   outputs/evaluate/

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

EPISODES=10
MAX_STEPS=10000

# List of games used in train_continual.py (adjust if you changed it)
GAMES=(Pong-v5 Breakout-v5 SpaceInvaders-v5)

# Experiment-specific eval directories for continual runs
DQN_NO_EWC_DIR="outputs/experiments/continual/dqn_ewcFalse/eval"
DQN_EWC_DIR="outputs/experiments/continual/dqn_ewcTrue/eval"
PPO_NO_EWC_DIR="outputs/experiments/continual/ppo_ewcFalse/eval"
PPO_EWC_DIR="outputs/experiments/continual/ppo_ewcTrue/eval"

mkdir -p "${DQN_NO_EWC_DIR}" "${DQN_EWC_DIR}" "${PPO_NO_EWC_DIR}" "${PPO_EWC_DIR}"

# DQN without EWC
python scripts/evaluate.py \
  --mode continual \
  --model checkpoints/continual/dqn_ewcFalse.pt \
  --algorithm dqn \
  --games "${GAMES[@]}" \
  --episodes "${EPISODES}" \
  --max-steps "${MAX_STEPS}" \
  --json-out "${DQN_NO_EWC_DIR}/metrics.json"

# DQN with EWC
python scripts/evaluate.py \
  --mode continual \
  --model checkpoints/continual/dqn_ewcTrue.pt \
  --algorithm dqn \
  --games "${GAMES[@]}" \
  --episodes "${EPISODES}" \
  --max-steps "${MAX_STEPS}" \
  --ewc \
  --ewc-lambda 50.0 \
  --json-out "${DQN_EWC_DIR}/metrics.json"

# PPO without EWC
python scripts/evaluate.py \
  --mode continual \
  --model checkpoints/continual/ppo_ewcFalse.pt \
  --algorithm ppo \
  --games "${GAMES[@]}" \
  --episodes "${EPISODES}" \
  --max-steps "${MAX_STEPS}" \
  --json-out "${PPO_NO_EWC_DIR}/metrics.json"

# PPO with EWC
python scripts/evaluate.py \
  --mode continual \
  --model checkpoints/continual/ppo_ewcTrue.pt \
  --algorithm ppo \
  --games "${GAMES[@]}" \
  --episodes "${EPISODES}" \
  --max-steps "${MAX_STEPS}" \
  --ewc \
  --ewc-lambda 50.0 \
  --json-out "${PPO_EWC_DIR}/metrics.json"

