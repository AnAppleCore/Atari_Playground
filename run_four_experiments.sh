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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

LOG_DIR="logs"
mkdir -p "${LOG_DIR}"

echo "Starting 4 experiments (2 algorithms x 2 games, 500k steps each)..."

echo "[1/4] Pong-v5 with DQN on GPU 0"
CUDA_VISIBLE_DEVICES=0 python scripts/train_single.py \
  --game Pong-v5 --algorithm dqn --steps 500000 \
  > "${LOG_DIR}/pong_dqn.log" 2>&1 &
PID1=$!

echo "[2/4] Pong-v5 with PPO on GPU 1"
CUDA_VISIBLE_DEVICES=1 python scripts/train_single.py \
  --game Pong-v5 --algorithm ppo --steps 500000 \
  > "${LOG_DIR}/pong_ppo.log" 2>&1 &
PID2=$!

echo "[3/4] Breakout-v5 with DQN on GPU 2"
CUDA_VISIBLE_DEVICES=2 python scripts/train_single.py \
  --game Breakout-v5 --algorithm dqn --steps 500000 \
  > "${LOG_DIR}/breakout_dqn.log" 2>&1 &
PID3=$!

echo "[4/4] Breakout-v5 with PPO on GPU 3"
CUDA_VISIBLE_DEVICES=3 python scripts/train_single.py \
  --game Breakout-v5 --algorithm ppo --steps 500000 \
  > "${LOG_DIR}/breakout_ppo.log" 2>&1 &
PID4=$!

# Wait for all 4 experiments to finish
wait "$PID1" "$PID2" "$PID3" "$PID4"

echo "All four experiments finished."

