#!/usr/bin/env bash

# Evaluate the single-task experiments SEQUENTIALLY on CPU ONLY
# This script assumes you have already activated the `atari_rl` conda
# environment and are running from the Atari_Playground directory.
#
# NOTE: This script forces CPU execution by setting CUDA_VISIBLE_DEVICES=""
# This script runs evaluations sequentially (one after another).
# The evaluation results (JSON, plots, and videos) will be saved under
#   outputs/single/{game}_{algorithm}/eval/

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

# Force CPU execution by hiding all GPUs
export CUDA_VISIBLE_DEVICES=""

EPISODES=10
MAX_STEPS=10000

echo "=========================================="
echo "CPU-ONLY MODE + SEQUENTIAL: All evaluations will run on CPU, one after another"
echo "=========================================="
echo "Evaluating single-task experiments..."
echo "Episodes per game: ${EPISODES}, Max steps: ${MAX_STEPS}"
echo ""

# Evaluation results will be saved in the same directories as training outputs
# outputs/single/{game}_{algorithm}/eval/
# The evaluate.py script will automatically infer the directory from model path

# Pong-v5 DQN
echo "[1/4] Evaluating Pong-v5 DQN..."
python scripts/evaluate.py \
  --mode single \
  --model checkpoints/single/Pong-v5_dqn.pt \
  --algorithm dqn \
  --game Pong-v5 \
  --episodes "${EPISODES}" \
  --max-steps "${MAX_STEPS}" \
  --json-out outputs/single/Pong-v5_dqn/eval/metrics.json
echo "✓ [1/4] Pong-v5 DQN evaluation completed"

# Pong-v5 PPO
echo "[2/4] Evaluating Pong-v5 PPO..."
python scripts/evaluate.py \
  --mode single \
  --model checkpoints/single/Pong-v5_ppo.pt \
  --algorithm ppo \
  --game Pong-v5 \
  --episodes "${EPISODES}" \
  --max-steps "${MAX_STEPS}" \
  --json-out outputs/single/Pong-v5_ppo/eval/metrics.json
echo "✓ [2/4] Pong-v5 PPO evaluation completed"

# Breakout-v5 DQN
echo "[3/4] Evaluating Breakout-v5 DQN..."
python scripts/evaluate.py \
  --mode single \
  --model checkpoints/single/Breakout-v5_dqn.pt \
  --algorithm dqn \
  --game Breakout-v5 \
  --episodes "${EPISODES}" \
  --max-steps "${MAX_STEPS}" \
  --json-out outputs/single/Breakout-v5_dqn/eval/metrics.json
echo "✓ [3/4] Breakout-v5 DQN evaluation completed"

# Breakout-v5 PPO
echo "[4/4] Evaluating Breakout-v5 PPO..."
python scripts/evaluate.py \
  --mode single \
  --model checkpoints/single/Breakout-v5_ppo.pt \
  --algorithm ppo \
  --game Breakout-v5 \
  --episodes "${EPISODES}" \
  --max-steps "${MAX_STEPS}" \
  --json-out outputs/single/Breakout-v5_ppo/eval/metrics.json
echo "✓ [4/4] Breakout-v5 PPO evaluation completed"

echo ""
echo "=========================================="
echo "All single-task evaluations finished."
echo "=========================================="

