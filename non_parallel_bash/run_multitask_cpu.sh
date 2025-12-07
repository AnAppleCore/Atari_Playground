#!/usr/bin/env bash

# Run 2 multi-task joint training experiments SEQUENTIALLY on CPU ONLY:
#   - DQN multi-task
#   - PPO multi-task
# Each experiment trains on 3 games (Pong-v5, Breakout-v5, SpaceInvaders-v5)
# with random task sampling per iteration for TOTAL_STEPS steps.
#
# NOTE: This script forces CPU execution by setting CUDA_VISIBLE_DEVICES=""
# This script runs experiments sequentially (not in parallel).
# This script assumes you have already activated the `atari_rl` conda environment before running, e.g.:
#   conda activate atari_rl
#   cd Atari_Playground
#   bash non_parallel_bash/run_multitask_cpu.sh
#
# Check logs/ for per-experiment logs and outputs/ for videos & plots.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

LOG_DIR="logs"
mkdir -p "${LOG_DIR}"

# You can adjust this hyperparameter manually if needed
TOTAL_STEPS=50000

# Force CPU execution by hiding all GPUs
export CUDA_VISIBLE_DEVICES=""

echo "=========================================="
echo "CPU-ONLY MODE + SEQUENTIAL: All experiments will run on CPU, one after another"
echo "=========================================="
echo "Starting 2 multi-task joint training experiments (DQN/PPO)..."
echo "Total steps: ${TOTAL_STEPS}"
echo ""

# Job 1: DQN multi-task
DQN_MULTITASK_TRAIN_DIR="outputs/multitask/dqn"
mkdir -p "${DQN_MULTITASK_TRAIN_DIR}"
echo "[1/2] DQN Multi-task on CPU"
python scripts/train_multitask.py \
  --algorithm dqn \
  --steps "${TOTAL_STEPS}" \
  > "${LOG_DIR}/multitask_dqn.log" 2>&1
echo "✓ [1/2] DQN Multi-task completed"

# Job 2: PPO multi-task
PPO_MULTITASK_TRAIN_DIR="outputs/multitask/ppo"
mkdir -p "${PPO_MULTITASK_TRAIN_DIR}"
echo "[2/2] PPO Multi-task on CPU"
python scripts/train_multitask.py \
  --algorithm ppo \
  --steps "${TOTAL_STEPS}" \
  > "${LOG_DIR}/multitask_ppo.log" 2>&1
echo "✓ [2/2] PPO Multi-task completed"

echo ""
echo "=========================================="
echo "All two multi-task joint training experiments finished."
echo "=========================================="

