#!/usr/bin/env bash

# Run 2 multi-task joint training experiments SEQUENTIALLY (one after another):
#   - DQN multi-task
#   - PPO multi-task
# Each experiment trains on 3 games (Pong-v5, Breakout-v5, SpaceInvaders-v5)
# with random task sampling per iteration for TOTAL_STEPS steps.
#
# NOTE: This script runs experiments sequentially (not in parallel).
# This script assumes you have already activated the `atari_rl` conda environment before running, e.g.:
#   conda activate atari_rl
#   cd Atari_Playground
#   bash non_parallel_bash/run_multitask.sh
#
# Check logs/ for per-experiment logs and outputs/ for videos & plots.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

LOG_DIR="logs"
mkdir -p "${LOG_DIR}"

# You can adjust this hyperparameter manually if needed
TOTAL_STEPS=50000

# Detect number of available GPUs (if any)
NUM_GPUS=0
if command -v nvidia-smi >/dev/null 2>&1; then
  NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
fi

if [ "${NUM_GPUS}" -ge 1 ]; then
  echo "Detected ${NUM_GPUS} GPU(s). All jobs will run sequentially on GPU 0."
else
  echo "No GPU detected (or nvidia-smi not available). All jobs will run on CPU."
fi

echo "=========================================="
echo "SEQUENTIAL MODE: Experiments will run one after another"
echo "=========================================="
echo "Starting 2 multi-task joint training experiments (DQN/PPO)..."
echo "Total steps: ${TOTAL_STEPS}"
echo ""

# Job 1: DQN multi-task
DQN_MULTITASK_TRAIN_DIR="outputs/multitask/dqn"
mkdir -p "${DQN_MULTITASK_TRAIN_DIR}"
if [ "${NUM_GPUS}" -ge 1 ]; then
  GPU_ID=0
  echo "[1/2] DQN Multi-task on GPU ${GPU_ID}"
  CUDA_VISIBLE_DEVICES=${GPU_ID} python scripts/train_multitask.py \
    --algorithm dqn \
    --steps "${TOTAL_STEPS}" \
    > "${LOG_DIR}/multitask_dqn.log" 2>&1
else
  echo "[1/2] DQN Multi-task on CPU"
  python scripts/train_multitask.py \
    --algorithm dqn \
    --steps "${TOTAL_STEPS}" \
    > "${LOG_DIR}/multitask_dqn.log" 2>&1
fi
echo "✓ [1/2] DQN Multi-task completed"

# Job 2: PPO multi-task
PPO_MULTITASK_TRAIN_DIR="outputs/multitask/ppo"
mkdir -p "${PPO_MULTITASK_TRAIN_DIR}"
if [ "${NUM_GPUS}" -ge 1 ]; then
  GPU_ID=0
  echo "[2/2] PPO Multi-task on GPU ${GPU_ID}"
  CUDA_VISIBLE_DEVICES=${GPU_ID} python scripts/train_multitask.py \
    --algorithm ppo \
    --steps "${TOTAL_STEPS}" \
    > "${LOG_DIR}/multitask_ppo.log" 2>&1
else
  echo "[2/2] PPO Multi-task on CPU"
  python scripts/train_multitask.py \
    --algorithm ppo \
    --steps "${TOTAL_STEPS}" \
    > "${LOG_DIR}/multitask_ppo.log" 2>&1
fi
echo "✓ [2/2] PPO Multi-task completed"

echo ""
echo "=========================================="
echo "All two multi-task joint training experiments finished."
echo "=========================================="

