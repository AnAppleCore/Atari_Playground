#!/usr/bin/env bash

# Evaluate the multi-task joint training experiments run by run_multitask.sh
# This script assumes you have already activated the `atari_rl` conda
# environment and are running from the Atari_Playground directory.
#
# The evaluation results (JSON) will be saved under
#   outputs/multitask/{algorithm}/eval/

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

EPISODES=10
MAX_STEPS=10000

# List of games used in train_multitask.py (adjust if you changed it)
GAMES=(Pong-v5 Breakout-v5 SpaceInvaders-v5)

# Evaluation results will be saved in the same directories as training outputs
# outputs/multitask/{algorithm}/eval/
# The evaluate.py script will automatically infer the directory from model path

# DQN multi-task
python scripts/evaluate.py \
  --mode multitask \
  --model checkpoints/multitask/dqn.pt \
  --algorithm dqn \
  --games "${GAMES[@]}" \
  --episodes "${EPISODES}" \
  --max-steps "${MAX_STEPS}" \
  --json-out outputs/multitask/dqn/eval/metrics.json

# PPO multi-task
python scripts/evaluate.py \
  --mode multitask \
  --model checkpoints/multitask/ppo.pt \
  --algorithm ppo \
  --games "${GAMES[@]}" \
  --episodes "${EPISODES}" \
  --max-steps "${MAX_STEPS}" \
  --json-out outputs/multitask/ppo/eval/metrics.json

