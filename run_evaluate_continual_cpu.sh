#!/usr/bin/env bash

# Evaluate the continual-learning experiments on CPU ONLY
# This script assumes you have already activated the `atari_rl` conda
# environment and are running from the Atari_Playground directory.
#
# NOTE: This script forces CPU execution by setting CUDA_VISIBLE_DEVICES=""
# The evaluation results (JSON) will be saved under
#   outputs/continual/{algorithm}_ewc{True/False}/eval/

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Force CPU execution by hiding all GPUs
export CUDA_VISIBLE_DEVICES=""

EPISODES=10
MAX_STEPS=10000

# List of games used in train_continual.py (adjust if you changed it)
GAMES=(Pong-v5 Breakout-v5 SpaceInvaders-v5)

echo "=========================================="
echo "CPU-ONLY MODE: All evaluations will run on CPU"
echo "=========================================="
echo "Evaluating continual learning experiments..."
echo "Episodes per game: ${EPISODES}, Max steps: ${MAX_STEPS}"
echo ""

# Evaluation results will be saved in the same directories as training outputs
# outputs/continual/{algorithm}_ewc{True/False}/eval/
# The evaluate.py script will automatically infer the directory from model path

# DQN without EWC
echo "[1/4] Evaluating DQN WITHOUT EWC..."
python scripts/evaluate.py \
  --mode continual \
  --model checkpoints/continual/dqn_ewcFalse.pt \
  --algorithm dqn \
  --games "${GAMES[@]}" \
  --episodes "${EPISODES}" \
  --max-steps "${MAX_STEPS}" \
  --json-out outputs/continual/dqn_ewcFalse/eval/metrics.json

# DQN with EWC
echo "[2/4] Evaluating DQN WITH EWC..."
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
echo "[3/4] Evaluating PPO WITHOUT EWC..."
python scripts/evaluate.py \
  --mode continual \
  --model checkpoints/continual/ppo_ewcFalse.pt \
  --algorithm ppo \
  --games "${GAMES[@]}" \
  --episodes "${EPISODES}" \
  --max-steps "${MAX_STEPS}" \
  --json-out outputs/continual/ppo_ewcFalse/eval/metrics.json

# PPO with EWC
echo "[4/4] Evaluating PPO WITH EWC..."
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

echo ""
echo "=========================================="
echo "All continual learning evaluations finished."
echo "=========================================="

