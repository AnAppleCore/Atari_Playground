#!/usr/bin/env bash

# Evaluate the continual-learning experiments run by run_continual_experiments.sh
# This script assumes you have already activated the `atari_rl` conda
# environment and are running from the Atari_Playground directory.
#
# The evaluation results (JSON) will be saved under
#   outputs/continual/{algorithm}_ewc{True/False}/eval/

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

EPISODES=10
MAX_STEPS=10000

# List of games used in train_continual.py (adjust if you changed it)
GAMES=(Pong-v5 Breakout-v5 SpaceInvaders-v5)

# Evaluation results will be saved in the same directories as training outputs
# outputs/continual/{algorithm}_ewc{True/False}/eval/
# The evaluate.py script will automatically infer the directory from model path

# DQN without EWC
python scripts/evaluate.py \
  --mode continual \
  --model checkpoints/continual/dqn_ewcFalse.pt \
  --algorithm dqn \
  --games "${GAMES[@]}" \
  --episodes "${EPISODES}" \
  --max-steps "${MAX_STEPS}" \
  --json-out outputs/continual/dqn_ewcFalse/eval/metrics.json

# DQN with EWC
python scripts/evaluate.py \
  --mode continual \
  --model checkpoints/continual/dqn_ewcTrue.pt \
  --algorithm dqn \
  --games "${GAMES[@]}" \
  --episodes "${EPISODES}" \
  --max-steps "${MAX_STEPS}" \
  --ewc \
  --ewc-lambda 0.4 \
  --json-out outputs/continual/dqn_ewcTrue/eval/metrics.json

# PPO without EWC
python scripts/evaluate.py \
  --mode continual \
  --model checkpoints/continual/ppo_ewcFalse.pt \
  --algorithm ppo \
  --games "${GAMES[@]}" \
  --episodes "${EPISODES}" \
  --max-steps "${MAX_STEPS}" \
  --json-out outputs/continual/ppo_ewcFalse/eval/metrics.json

# PPO with EWC
python scripts/evaluate.py \
  --mode continual \
  --model checkpoints/continual/ppo_ewcTrue.pt \
  --algorithm ppo \
  --games "${GAMES[@]}" \
  --episodes "${EPISODES}" \
  --max-steps "${MAX_STEPS}" \
  --ewc \
  --ewc-lambda 0.4 \
  --json-out outputs/continual/ppo_ewcTrue/eval/metrics.json

