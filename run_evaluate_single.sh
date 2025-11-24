#!/usr/bin/env bash

# Evaluate the single-task experiments run by run_four_experiments.sh
# This script assumes you have already activated the `atari_rl` conda
# environment and are running from the Atari_Playground directory.
#
# The evaluation results (JSON and plots) will be saved under
#   outputs/evaluate/

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

EPISODES=10
MAX_STEPS=10000

# Experiment-specific output directories for evaluations
PONG_DQN_DIR="outputs/experiments/single/Pong-v5_dqn/eval"
PONG_PPO_DIR="outputs/experiments/single/Pong-v5_ppo/eval"
BRK_DQN_DIR="outputs/experiments/single/Breakout-v5_dqn/eval"
BRK_PPO_DIR="outputs/experiments/single/Breakout-v5_ppo/eval"

mkdir -p "${PONG_DQN_DIR}" "${PONG_PPO_DIR}" "${BRK_DQN_DIR}" "${BRK_PPO_DIR}"

# Pong-v5 DQN
python scripts/evaluate.py \
  --mode single \
  --model checkpoints/single/Pong-v5_dqn.pt \
  --algorithm dqn \
  --game Pong-v5 \
  --episodes "${EPISODES}" \
  --max-steps "${MAX_STEPS}" \
  --json-out "${PONG_DQN_DIR}/metrics.json"

# Pong-v5 PPO
python scripts/evaluate.py \
  --mode single \
  --model checkpoints/single/Pong-v5_ppo.pt \
  --algorithm ppo \
  --game Pong-v5 \
  --episodes "${EPISODES}" \
  --max-steps "${MAX_STEPS}" \
  --json-out "${PONG_PPO_DIR}/metrics.json"

# Breakout-v5 DQN
python scripts/evaluate.py \
  --mode single \
  --model checkpoints/single/Breakout-v5_dqn.pt \
  --algorithm dqn \
  --game Breakout-v5 \
  --episodes "${EPISODES}" \
  --max-steps "${MAX_STEPS}" \
  --json-out "${BRK_DQN_DIR}/metrics.json"

# Breakout-v5 PPO
python scripts/evaluate.py \
  --mode single \
  --model checkpoints/single/Breakout-v5_ppo.pt \
  --algorithm ppo \
  --game Breakout-v5 \
  --episodes "${EPISODES}" \
  --max-steps "${MAX_STEPS}" \
  --json-out "${BRK_PPO_DIR}/metrics.json"

