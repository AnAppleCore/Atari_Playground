#!/usr/bin/env bash

# Evaluate the single-task experiments run by run_four_experiments.sh
# This script assumes you have already activated the `atari_rl` conda
# environment and are running from the Atari_Playground directory.
#
# The evaluation results (JSON, plots, and videos) will be saved under
#   outputs/single/{game}_{algorithm}/eval/

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

EPISODES=10
MAX_STEPS=10000

# Evaluation results will be saved in the same directories as training outputs
# outputs/single/{game}_{algorithm}/eval/
# The evaluate.py script will automatically infer the directory from model path

# Pong-v5 DQN
python scripts/evaluate.py \
  --mode single \
  --model checkpoints/single/Pong-v5_dqn.pt \
  --algorithm dqn \
  --game Pong-v5 \
  --episodes "${EPISODES}" \
  --max-steps "${MAX_STEPS}" \
  --json-out outputs/single/Pong-v5_dqn/eval/metrics.json

# Pong-v5 PPO
python scripts/evaluate.py \
  --mode single \
  --model checkpoints/single/Pong-v5_ppo.pt \
  --algorithm ppo \
  --game Pong-v5 \
  --episodes "${EPISODES}" \
  --max-steps "${MAX_STEPS}" \
  --json-out outputs/single/Pong-v5_ppo/eval/metrics.json

# Breakout-v5 DQN
python scripts/evaluate.py \
  --mode single \
  --model checkpoints/single/Breakout-v5_dqn.pt \
  --algorithm dqn \
  --game Breakout-v5 \
  --episodes "${EPISODES}" \
  --max-steps "${MAX_STEPS}" \
  --json-out outputs/single/Breakout-v5_dqn/eval/metrics.json

# Breakout-v5 PPO
python scripts/evaluate.py \
  --mode single \
  --model checkpoints/single/Breakout-v5_ppo.pt \
  --algorithm ppo \
  --game Breakout-v5 \
  --episodes "${EPISODES}" \
  --max-steps "${MAX_STEPS}" \
  --json-out outputs/single/Breakout-v5_ppo/eval/metrics.json

