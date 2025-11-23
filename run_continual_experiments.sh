#!/usr/bin/env bash

# Run 4 continual learning experiments in parallel:
#   - DQN without EWC
#   - DQN with EWC
#   - PPO without EWC
#   - PPO with EWC
# Each experiment trains on 3 games (Pong-v5, Breakout-v5, SpaceInvaders-v5)
# for STEPS_PER_GAME steps per game (3 * STEPS_PER_GAME total steps).
#
# NOTE: This script assumes you have already activated the
# `atari_rl` conda environment before running, e.g.:
#   conda activate atari_rl
#   cd Atari_Playground
#   nohup ./run_continual_experiments.sh > continual_master.log 2>&1 &
#
# Check logs/ for per-experiment logs and outputs/ for videos & plots.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

LOG_DIR="logs"
mkdir -p "${LOG_DIR}"

# You can adjust these two hyperparameters manually if needed
STEPS_PER_GAME=50000
EWC_LAMBDA=50.0

# Detect number of available GPUs (if any)
NUM_GPUS=0
if command -v nvidia-smi >/dev/null 2>&1; then
  NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
fi

if [ "${NUM_GPUS}" -ge 2 ]; then
  echo "Detected ${NUM_GPUS} GPU(s). Jobs will be assigned round-robin across GPUs."
elif [ "${NUM_GPUS}" -eq 1 ]; then
  echo "Detected 1 GPU. All 4 jobs will share GPU 0."
else
  echo "No GPU detected (or nvidia-smi not available). All jobs will run on CPU."
fi

echo "Starting 4 continual learning experiments (DQN/PPO x no-EWC/EWC)..."
echo "Steps per game: ${STEPS_PER_GAME}, EWC lambda: ${EWC_LAMBDA}"

# Job 1: DQN without EWC
JOB_IDX=0
if [ "${NUM_GPUS}" -ge 1 ]; then
  GPU_ID=$((JOB_IDX % NUM_GPUS))
  echo "[1/4] DQN WITHOUT EWC on GPU ${GPU_ID}"
  CUDA_VISIBLE_DEVICES=${GPU_ID} python scripts/train_continual.py \
    --algorithm dqn \
    --steps-per-game "${STEPS_PER_GAME}" \
    > "${LOG_DIR}/continual_dqn_no_ewc.log" 2>&1 &
else
  echo "[1/4] DQN WITHOUT EWC on CPU (no GPU detected)"
  python scripts/train_continual.py \
    --algorithm dqn \
    --steps-per-game "${STEPS_PER_GAME}" \
    > "${LOG_DIR}/continual_dqn_no_ewc.log" 2>&1 &
fi
PID1=$!

# Job 2: DQN with EWC
JOB_IDX=1
if [ "${NUM_GPUS}" -ge 1 ]; then
  GPU_ID=$((JOB_IDX % NUM_GPUS))
  echo "[2/4] DQN WITH EWC on GPU ${GPU_ID}"
  CUDA_VISIBLE_DEVICES=${GPU_ID} python scripts/train_continual.py \
    --algorithm dqn \
    --use-ewc \
    --ewc-lambda "${EWC_LAMBDA}" \
    --steps-per-game "${STEPS_PER_GAME}" \
    > "${LOG_DIR}/continual_dqn_ewc.log" 2>&1 &
else
  echo "[2/4] DQN WITH EWC on CPU (no GPU detected)"
  python scripts/train_continual.py \
    --algorithm dqn \
    --use-ewc \
    --ewc-lambda "${EWC_LAMBDA}" \
    --steps-per-game "${STEPS_PER_GAME}" \
    > "${LOG_DIR}/continual_dqn_ewc.log" 2>&1 &
fi
PID2=$!

# Job 3: PPO without EWC
JOB_IDX=2
if [ "${NUM_GPUS}" -ge 1 ]; then
  GPU_ID=$((JOB_IDX % NUM_GPUS))
  echo "[3/4] PPO WITHOUT EWC on GPU ${GPU_ID}"
  CUDA_VISIBLE_DEVICES=${GPU_ID} python scripts/train_continual.py \
    --algorithm ppo \
    --steps-per-game "${STEPS_PER_GAME}" \
    > "${LOG_DIR}/continual_ppo_no_ewc.log" 2>&1 &
else
  echo "[3/4] PPO WITHOUT EWC on CPU (no GPU detected)"
  python scripts/train_continual.py \
    --algorithm ppo \
    --steps-per-game "${STEPS_PER_GAME}" \
    > "${LOG_DIR}/continual_ppo_no_ewc.log" 2>&1 &
fi
PID3=$!

# Job 4: PPO with EWC
JOB_IDX=3
if [ "${NUM_GPUS}" -ge 1 ]; then
  GPU_ID=$((JOB_IDX % NUM_GPUS))
  echo "[4/4] PPO WITH EWC on GPU ${GPU_ID}"
  CUDA_VISIBLE_DEVICES=${GPU_ID} python scripts/train_continual.py \
    --algorithm ppo \
    --use-ewc \
    --ewc-lambda "${EWC_LAMBDA}" \
    --steps-per-game "${STEPS_PER_GAME}" \
    > "${LOG_DIR}/continual_ppo_ewc.log" 2>&1 &
else
  echo "[4/4] PPO WITH EWC on CPU (no GPU detected)"
  python scripts/train_continual.py \
    --algorithm ppo \
    --use-ewc \
    --ewc-lambda "${EWC_LAMBDA}" \
    --steps-per-game "${STEPS_PER_GAME}" \
    > "${LOG_DIR}/continual_ppo_ewc.log" 2>&1 &
fi
PID4=$!

# Wait for all 4 experiments to finish
wait "${PID1}" "${PID2}" "${PID3}" "${PID4}"

echo "All four continual learning experiments finished."

