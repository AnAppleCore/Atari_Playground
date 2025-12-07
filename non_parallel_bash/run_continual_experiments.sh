#!/usr/bin/env bash

# Run 4 continual learning experiments SEQUENTIALLY (one after another):
#   - DQN without EWC
#   - DQN with EWC
#   - PPO without EWC
#   - PPO with EWC
# Each experiment trains on 3 games (Pong-v5, Breakout-v5, SpaceInvaders-v5)
# for STEPS_PER_GAME steps per game (3 * STEPS_PER_GAME total steps).
#
# NOTE: This script runs experiments sequentially (not in parallel).
# This script assumes you have already activated the `atari_rl` conda environment before running, e.g.:
#   conda activate atari_rl
#   cd Atari_Playground
#   bash non_parallel_bash/run_continual_experiments.sh
#
# Check logs/ for per-experiment logs and outputs/ for videos & plots.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

LOG_DIR="logs"
mkdir -p "${LOG_DIR}"

# You can adjust these two hyperparameters manually if needed
STEPS_PER_GAME=500000
EWC_LAMBDA=0.4

# Detect number of available GPUs (if any)
NUM_GPUS=0
if command -v nvidia-smi >/dev/null 2>&1; then
  NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
fi

if [ "${NUM_GPUS}" -ge 2 ]; then
  echo "Detected ${NUM_GPUS} GPU(s). All jobs will run sequentially on GPU 0."
elif [ "${NUM_GPUS}" -eq 1 ]; then
  echo "Detected 1 GPU. All jobs will run sequentially on GPU 0."
else
  echo "No GPU detected (or nvidia-smi not available). All jobs will run on CPU."
fi

echo "=========================================="
echo "SEQUENTIAL MODE: Experiments will run one after another"
echo "=========================================="
echo "Starting 4 continual learning experiments (DQN/PPO x no-EWC/EWC)..."
echo "Steps per game: ${STEPS_PER_GAME}, EWC lambda: ${EWC_LAMBDA}"
echo ""

# Job 1: DQN without EWC
JOB_IDX=0
DQN_NO_EWC_TRAIN_DIR="outputs/continual/dqn_ewcFalse"
mkdir -p "${DQN_NO_EWC_TRAIN_DIR}"
if [ "${NUM_GPUS}" -ge 1 ]; then
  GPU_ID=0
  echo "[1/4] DQN WITHOUT EWC on GPU ${GPU_ID}"
  CUDA_VISIBLE_DEVICES=${GPU_ID} python scripts/train_continual.py \
    --algorithm dqn \
    --steps-per-game "${STEPS_PER_GAME}" \
    > "${LOG_DIR}/continual_dqn_no_ewc.log" 2>&1
else
  echo "[1/4] DQN WITHOUT EWC on CPU"
  python scripts/train_continual.py \
    --algorithm dqn \
    --steps-per-game "${STEPS_PER_GAME}" \
    > "${LOG_DIR}/continual_dqn_no_ewc.log" 2>&1
fi
echo "✓ [1/4] DQN WITHOUT EWC completed"

# Job 2: DQN with EWC
JOB_IDX=1
DQN_EWC_TRAIN_DIR="outputs/continual/dqn_ewcTrue"
mkdir -p "${DQN_EWC_TRAIN_DIR}"
if [ "${NUM_GPUS}" -ge 1 ]; then
  GPU_ID=0
  echo "[2/4] DQN WITH EWC on GPU ${GPU_ID}"
  CUDA_VISIBLE_DEVICES=${GPU_ID} python scripts/train_continual.py \
    --algorithm dqn \
    --use-ewc \
    --ewc-lambda "${EWC_LAMBDA}" \
    --steps-per-game "${STEPS_PER_GAME}" \
    > "${LOG_DIR}/continual_dqn_ewc.log" 2>&1
else
  echo "[2/4] DQN WITH EWC on CPU"
  python scripts/train_continual.py \
    --algorithm dqn \
    --use-ewc \
    --ewc-lambda "${EWC_LAMBDA}" \
    --steps-per-game "${STEPS_PER_GAME}" \
    > "${LOG_DIR}/continual_dqn_ewc.log" 2>&1
fi
echo "✓ [2/4] DQN WITH EWC completed"

# Job 3: PPO without EWC
JOB_IDX=2
PPO_NO_EWC_TRAIN_DIR="outputs/continual/ppo_ewcFalse"
mkdir -p "${PPO_NO_EWC_TRAIN_DIR}"
if [ "${NUM_GPUS}" -ge 1 ]; then
  GPU_ID=0
  echo "[3/4] PPO WITHOUT EWC on GPU ${GPU_ID}"
  CUDA_VISIBLE_DEVICES=${GPU_ID} python scripts/train_continual.py \
    --algorithm ppo \
    --steps-per-game "${STEPS_PER_GAME}" \
    > "${LOG_DIR}/continual_ppo_no_ewc.log" 2>&1
else
  echo "[3/4] PPO WITHOUT EWC on CPU"
  python scripts/train_continual.py \
    --algorithm ppo \
    --steps-per-game "${STEPS_PER_GAME}" \
    > "${LOG_DIR}/continual_ppo_no_ewc.log" 2>&1
fi
echo "✓ [3/4] PPO WITHOUT EWC completed"

# Job 4: PPO with EWC
JOB_IDX=3
PPO_EWC_TRAIN_DIR="outputs/continual/ppo_ewcTrue"
mkdir -p "${PPO_EWC_TRAIN_DIR}"
if [ "${NUM_GPUS}" -ge 1 ]; then
  GPU_ID=0
  echo "[4/4] PPO WITH EWC on GPU ${GPU_ID}"
  CUDA_VISIBLE_DEVICES=${GPU_ID} python scripts/train_continual.py \
    --algorithm ppo \
    --use-ewc \
    --ewc-lambda "${EWC_LAMBDA}" \
    --steps-per-game "${STEPS_PER_GAME}" \
    > "${LOG_DIR}/continual_ppo_ewc.log" 2>&1
else
  echo "[4/4] PPO WITH EWC on CPU"
  python scripts/train_continual.py \
    --algorithm ppo \
    --use-ewc \
    --ewc-lambda "${EWC_LAMBDA}" \
    --steps-per-game "${STEPS_PER_GAME}" \
    > "${LOG_DIR}/continual_ppo_ewc.log" 2>&1
fi
echo "✓ [4/4] PPO WITH EWC completed"

echo ""
echo "=========================================="
echo "All four continual learning experiments finished."
echo "=========================================="

