#!/usr/bin/env bash

# Run 2 multi-task joint training experiments in parallel on CPU ONLY:
#   - DQN multi-task
#   - PPO multi-task
# Each experiment trains on 3 games (Pong-v5, Breakout-v5, SpaceInvaders-v5)
# with random task sampling per iteration for TOTAL_STEPS steps.
#
# NOTE: This script forces CPU execution by setting CUDA_VISIBLE_DEVICES=""
# This script assumes you have already activated the `atari_rl` conda environment before running, e.g.:
#   conda activate atari_rl
#   cd Atari_Playground
#   nohup ./run_multitask_cpu.sh > multitask_master_cpu.log 2>&1 &
#
# Check logs/ for per-experiment logs and outputs/ for videos & plots.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

LOG_DIR="logs"
mkdir -p "${LOG_DIR}"

# You can adjust this hyperparameter manually if needed
TOTAL_STEPS=50000

# Force CPU execution by hiding all GPUs
export CUDA_VISIBLE_DEVICES=""

echo "=========================================="
echo "CPU-ONLY MODE: All experiments will run on CPU"
echo "=========================================="
echo "Starting 2 multi-task joint training experiments (DQN/PPO)..."
echo "Total steps: ${TOTAL_STEPS}"
echo ""

# Job 1: DQN multi-task
JOB_IDX=0
DQN_MULTITASK_TRAIN_DIR="outputs/multitask/dqn"
mkdir -p "${DQN_MULTITASK_TRAIN_DIR}"
echo "[1/2] DQN Multi-task on CPU"
python scripts/train_multitask.py \
  --algorithm dqn \
  --steps "${TOTAL_STEPS}" \
  > "${LOG_DIR}/multitask_dqn.log" 2>&1 &
PID1=$!

# Job 2: PPO multi-task
JOB_IDX=1
PPO_MULTITASK_TRAIN_DIR="outputs/multitask/ppo"
mkdir -p "${PPO_MULTITASK_TRAIN_DIR}"
echo "[2/2] PPO Multi-task on CPU"
python scripts/train_multitask.py \
  --algorithm ppo \
  --steps "${TOTAL_STEPS}" \
  > "${LOG_DIR}/multitask_ppo.log" 2>&1 &
PID2=$!

echo ""
echo "All 2 jobs started. PIDs: ${PID1}, ${PID2}"
echo "Monitor progress in logs/ directory:"
echo "  - ${LOG_DIR}/multitask_dqn.log"
echo "  - ${LOG_DIR}/multitask_ppo.log"
echo ""

# Wait for all 2 experiments to finish
wait "${PID1}" "${PID2}"

echo "=========================================="
echo "All two multi-task joint training experiments finished."
echo "=========================================="

