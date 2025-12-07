#!/usr/bin/env bash

# Evaluate the multi-task joint training experiments SEQUENTIALLY on CPU ONLY
# This script assumes you have already activated the `atari_rl` conda
# environment and are running from the Atari_Playground directory.
#
# NOTE: This script forces CPU execution by setting CUDA_VISIBLE_DEVICES=""
# This script runs evaluations sequentially (one after another).
# The evaluation results (JSON) will be saved under
#   outputs/multitask/{algorithm}/eval/

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

# Force CPU execution by hiding all GPUs
export CUDA_VISIBLE_DEVICES=""

EPISODES=10
MAX_STEPS=10000

# List of games used in train_multitask.py (adjust if you changed it)
GAMES=(Pong-v5 Breakout-v5 SpaceInvaders-v5)

echo "=========================================="
echo "CPU-ONLY MODE + SEQUENTIAL: All evaluations will run on CPU, one after another"
echo "=========================================="
echo "Evaluating multi-task joint training experiments..."
echo "Episodes per game: ${EPISODES}, Max steps: ${MAX_STEPS}"
echo ""

# Evaluation results will be saved in the same directories as training outputs
# outputs/multitask/{algorithm}/eval/
# The evaluate.py script will automatically infer the directory from model path

# DQN multi-task
echo "[1/2] Evaluating DQN Multi-task..."
python scripts/evaluate.py \
  --mode multitask \
  --model checkpoints/multitask/dqn.pt \
  --algorithm dqn \
  --games "${GAMES[@]}" \
  --episodes "${EPISODES}" \
  --max-steps "${MAX_STEPS}" \
  --json-out outputs/multitask/dqn/eval/metrics.json
echo "✓ [1/2] DQN Multi-task evaluation completed"

# PPO multi-task
echo "[2/2] Evaluating PPO Multi-task..."
python scripts/evaluate.py \
  --mode multitask \
  --model checkpoints/multitask/ppo.pt \
  --algorithm ppo \
  --games "${GAMES[@]}" \
  --episodes "${EPISODES}" \
  --max-steps "${MAX_STEPS}" \
  --json-out outputs/multitask/ppo/eval/metrics.json
echo "✓ [2/2] PPO Multi-task evaluation completed"

echo ""
echo "=========================================="
echo "All multi-task evaluations finished."
echo "=========================================="

