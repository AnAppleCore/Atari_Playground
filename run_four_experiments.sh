#!/usr/bin/env bash

# Run 4 experiments in parallel:
#   - Pong-v5 with DQN
#   - Pong-v5 with PPO
#   - Breakout-v5 with DQN
#   - Breakout-v5 with PPO
# Each experiment runs for 500,000 steps.
#
# NOTE: This script assumes you have already activated the
# `atari_rl` conda environment before running, e.g.:
#   conda activate atari_rl
#   cd Atari_Playground
#   bash run_four_experiments.sh

set -e

TOTAL_STEPS=500000

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

LOG_DIR="logs"
mkdir -p "${LOG_DIR}"

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

echo "Starting 4 experiments (2 algorithms x 2 games, 500k steps each)..."

# Job 1: Pong DQN
JOB_IDX=0
PONG_DQN_EXP_DIR="outputs/single/Pong-v5_dqn"
mkdir -p "${PONG_DQN_EXP_DIR}"
if [ "${NUM_GPUS}" -ge 1 ]; then
  GPU_ID=$((JOB_IDX % NUM_GPUS))
  echo "[1/4] Pong-v5 with DQN on GPU ${GPU_ID}"
  CUDA_VISIBLE_DEVICES=${GPU_ID} python scripts/train_single.py \
    --game Pong-v5 --algorithm dqn --steps ${TOTAL_STEPS} \
    > "${LOG_DIR}/pong_dqn.log" 2>&1 &
else
  echo "[1/4] Pong-v5 with DQN on CPU (no GPU detected)"
  python scripts/train_single.py \
    --game Pong-v5 --algorithm dqn --steps ${TOTAL_STEPS} \
    > "${LOG_DIR}/pong_dqn.log" 2>&1 &
fi
PID1=$!

# Job 2: Pong PPO
JOB_IDX=1
PONG_PPO_EXP_DIR="outputs/single/Pong-v5_ppo"
mkdir -p "${PONG_PPO_EXP_DIR}"
if [ "${NUM_GPUS}" -ge 1 ]; then
  GPU_ID=$((JOB_IDX % NUM_GPUS))
  echo "[2/4] Pong-v5 with PPO on GPU ${GPU_ID}"
  CUDA_VISIBLE_DEVICES=${GPU_ID} python scripts/train_single.py \
    --game Pong-v5 --algorithm ppo --steps ${TOTAL_STEPS} \
    > "${LOG_DIR}/pong_ppo.log" 2>&1 &
else
  echo "[2/4] Pong-v5 with PPO on CPU (no GPU detected)"
  python scripts/train_single.py \
    --game Pong-v5 --algorithm ppo --steps ${TOTAL_STEPS} \
    > "${LOG_DIR}/pong_ppo.log" 2>&1 &
fi
PID2=$!

# Job 3: Breakout DQN
JOB_IDX=2
BRK_DQN_EXP_DIR="outputs/single/Breakout-v5_dqn"
mkdir -p "${BRK_DQN_EXP_DIR}"
if [ "${NUM_GPUS}" -ge 1 ]; then
  GPU_ID=$((JOB_IDX % NUM_GPUS))
  echo "[3/4] Breakout-v5 with DQN on GPU ${GPU_ID}"
  CUDA_VISIBLE_DEVICES=${GPU_ID} python scripts/train_single.py \
    --game Breakout-v5 --algorithm dqn --steps ${TOTAL_STEPS} \
    > "${LOG_DIR}/breakout_dqn.log" 2>&1 &
else
  echo "[3/4] Breakout-v5 with DQN on CPU (no GPU detected)"
  python scripts/train_single.py \
    --game Breakout-v5 --algorithm dqn --steps ${TOTAL_STEPS} \
    > "${LOG_DIR}/breakout_dqn.log" 2>&1 &
fi
PID3=$!

# Job 4: Breakout PPO
JOB_IDX=3
BRK_PPO_EXP_DIR="outputs/single/Breakout-v5_ppo"
mkdir -p "${BRK_PPO_EXP_DIR}"
if [ "${NUM_GPUS}" -ge 1 ]; then
  GPU_ID=$((JOB_IDX % NUM_GPUS))
  echo "[4/4] Breakout-v5 with PPO on GPU ${GPU_ID}"
  CUDA_VISIBLE_DEVICES=${GPU_ID} python scripts/train_single.py \
    --game Breakout-v5 --algorithm ppo --steps ${TOTAL_STEPS} \
    > "${LOG_DIR}/breakout_ppo.log" 2>&1 &
else
  echo "[4/4] Breakout-v5 with PPO on CPU (no GPU detected)"
  python scripts/train_single.py \
    --game Breakout-v5 --algorithm ppo --steps ${TOTAL_STEPS} \
    > "${LOG_DIR}/breakout_ppo.log" 2>&1 &
fi
PID4=$!

# Wait for all 4 experiments to finish
wait "$PID1" "$PID2" "$PID3" "$PID4"

echo "All four experiments finished."

