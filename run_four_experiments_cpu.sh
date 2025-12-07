#!/usr/bin/env bash

# Run 4 experiments in parallel on CPU ONLY:
#   - Pong-v5 with DQN
#   - Pong-v5 with PPO
#   - Breakout-v5 with DQN
#   - Breakout-v5 with PPO
# Each experiment runs for 500,000 steps.
#
# NOTE: This script forces CPU execution by setting CUDA_VISIBLE_DEVICES=""
# This script assumes you have already activated the `atari_rl` conda environment before running, e.g.:
#   conda activate atari_rl
#   cd Atari_Playground
#   bash run_four_experiments_cpu.sh

set -e

TOTAL_STEPS=500000

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

LOG_DIR="logs"
mkdir -p "${LOG_DIR}"

# Force CPU execution by hiding all GPUs
export CUDA_VISIBLE_DEVICES=""

echo "=========================================="
echo "CPU-ONLY MODE: All experiments will run on CPU"
echo "=========================================="
echo "Starting 4 experiments (2 algorithms x 2 games, 500k steps each)..."
echo ""

# Job 1: Pong DQN
JOB_IDX=0
PONG_DQN_EXP_DIR="outputs/single/Pong-v5_dqn"
mkdir -p "${PONG_DQN_EXP_DIR}"
echo "[1/4] Pong-v5 with DQN on CPU"
python scripts/train_single.py \
  --game Pong-v5 --algorithm dqn --steps ${TOTAL_STEPS} \
  > "${LOG_DIR}/pong_dqn.log" 2>&1 &
PID1=$!

# Job 2: Pong PPO
JOB_IDX=1
PONG_PPO_EXP_DIR="outputs/single/Pong-v5_ppo"
mkdir -p "${PONG_PPO_EXP_DIR}"
echo "[2/4] Pong-v5 with PPO on CPU"
python scripts/train_single.py \
  --game Pong-v5 --algorithm ppo --steps ${TOTAL_STEPS} \
  > "${LOG_DIR}/pong_ppo.log" 2>&1 &
PID2=$!

# Job 3: Breakout DQN
JOB_IDX=2
BRK_DQN_EXP_DIR="outputs/single/Breakout-v5_dqn"
mkdir -p "${BRK_DQN_EXP_DIR}"
echo "[3/4] Breakout-v5 with DQN on CPU"
python scripts/train_single.py \
  --game Breakout-v5 --algorithm dqn --steps ${TOTAL_STEPS} \
  > "${LOG_DIR}/breakout_dqn.log" 2>&1 &
PID3=$!

# Job 4: Breakout PPO
JOB_IDX=3
BRK_PPO_EXP_DIR="outputs/single/Breakout-v5_ppo"
mkdir -p "${BRK_PPO_EXP_DIR}"
echo "[4/4] Breakout-v5 with PPO on CPU"
python scripts/train_single.py \
  --game Breakout-v5 --algorithm ppo --steps ${TOTAL_STEPS} \
  > "${LOG_DIR}/breakout_ppo.log" 2>&1 &
PID4=$!

echo ""
echo "All 4 jobs started. PIDs: ${PID1}, ${PID2}, ${PID3}, ${PID4}"
echo "Monitor progress in logs/ directory:"
echo "  - ${LOG_DIR}/pong_dqn.log"
echo "  - ${LOG_DIR}/pong_ppo.log"
echo "  - ${LOG_DIR}/breakout_dqn.log"
echo "  - ${LOG_DIR}/breakout_ppo.log"
echo ""

# Wait for all 4 experiments to finish
wait "$PID1" "$PID2" "$PID3" "$PID4"

echo "=========================================="
echo "All four experiments finished."
echo "=========================================="

