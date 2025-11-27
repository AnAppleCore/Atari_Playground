# Atari RL Playground

A comprehensive PyTorch-based reinforcement learning framework for Atari games, designed for educational purposes and research on continual learning.

## Features

- **Algorithms**: DQN, PPO, and EWC (Elastic Weight Consolidation)
- **Environment**: Gymnasium-based Atari environment with frame preprocessing and stacking
- **Training**: Single-game and multi-game continual learning support
- **Visualization**: Automatic MP4 video generation during training
- **GPU Support**: CUDA acceleration with CPU fallback

## Complete Setup Guide

### Step 1: Create Conda Environment
```bash
conda create -n atari_rl python=3.10 -y
conda activate atari_rl
```

### Step 2: Install Dependencies
```bash
# Default (direct from PyPI)
pip install -r requirements.txt

# If pip is very slow in China, you can temporarily use Tsinghua mirror:
# pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

This installs:
- PyTorch 2.0+ (with CUDA support)
- Gymnasium with Atari support
- NumPy, Matplotlib, OpenCV, imageio
- tqdm for progress bars

### Step 3: Verify Installation
```bash
python test_framework.py
```

Expected output: All tests should pass (environment test may fail without ROMs, which is normal)

### Step 4: Test Framework
```bash
python scripts/demo.py
```

This demonstrates all algorithms without needing ROMs.

## Training Guide

### Single Game Training
```bash
# Train DQN on Pong for 500,000 steps (recommended for good performance)
python scripts/train_single.py --game Pong-v5 --algorithm dqn --steps 500000

# Train PPO on Breakout for 500,000 steps
python scripts/train_single.py --game Breakout-v5 --algorithm ppo --steps 500000

# Quick test with fewer steps (for faster iteration)
python scripts/train_single.py --game Pong-v5 --algorithm dqn --steps 100000

# Enable video recording (disabled by default to save memory)
python scripts/train_single.py --game Pong-v5 --algorithm dqn --steps 100000 --save-video
```

**Output (per experiment folder):**
- MP4 video: `outputs/single/{game}_{algorithm}/training.mp4` (only if `--save-video` is used)
- Metrics plot: `outputs/single/{game}_{algorithm}/metrics.png`
- Model checkpoint: `checkpoints/single/{game}_{algorithm}.pt`

### Continual Learning (Multiple Games)
```bash
# Train DQN on 3 games sequentially WITHOUT EWC (shows catastrophic forgetting)
python scripts/train_continual.py --algorithm dqn --steps-per-game 50000

# Train DQN on 3 games sequentially WITH EWC (mitigates forgetting)
python scripts/train_continual.py --algorithm dqn --use-ewc --ewc-lambda 0.4 --steps-per-game 50000

# Train PPO on 3 games sequentially WITHOUT EWC
python scripts/train_continual.py --algorithm ppo --steps-per-game 50000

# Train PPO on 3 games sequentially WITH EWC
python scripts/train_continual.py --algorithm ppo --use-ewc --ewc-lambda 0.4 --steps-per-game 50000
```

**Note:** Both DQN and PPO support continual learning with different action spaces. The framework automatically uses multi-head architectures (MultiHeadDQNAgent and MultiHeadPPOAgent) to handle games with different action dimensions.

**Output (per experiment + per game folders):**
- Per-game training videos: `outputs/continual/{algorithm}_ewc{True/False}/{game}/training.mp4` (only if `--save-video` is used)
- Aggregated training metrics: `outputs/continual/{algorithm}_ewc{True/False}/training_metrics.png`
- Forgetting curves: `outputs/continual/{algorithm}_ewc{True/False}/forgetting_eval.png`
- Model checkpoint: `checkpoints/continual/{algorithm}_ewc{True/False}.pt`

### Available Games
The framework supports 108 Atari games. Common ones:
- Pong-v5 (6 actions)
- Breakout-v5 (4 actions)
- SpaceInvaders-v5 (6 actions)
- Atari-v5 (18 actions)
- And 104 more...

### Training Script Parameters

**Single-game training (`scripts/train_single.py`):**
```bash
--game GAME_NAME        # Game to train on (default: Pong-v5)
--algorithm {dqn,ppo}   # Algorithm to use (default: dqn)
--steps STEPS           # Total training steps (default: 500000)
--batch-size BATCH_SIZE # Batch size for training (default: 32)
--save-video            # Enable video recording (disabled by default)
```

**Continual learning training (`scripts/train_continual.py`):**
```bash
--games GAME1 GAME2 ... # List of games (default: Pong-v5 Breakout-v5 SpaceInvaders-v5)
--algorithm {dqn,ppo}   # Algorithm to use (default: dqn)
--steps-per-game STEPS  # Steps per game (default: 50000)
--batch-size BATCH_SIZE # Batch size for training (default: 32, used for DQN updates)
--save-video            # Enable video recording (disabled by default)

--use-ewc               # (Optional) Enable EWC for continual learning
--ewc-lambda LAMBDA     # EWC regularization strength (default: 0.4)
```

**Default Hyperparameters (optimized for Atari games):**

*DQN:*
- Learning rate: `1e-4`
- Batch size: `32`
- Gamma (discount factor): `0.99`
- Epsilon: `1.0` → `0.01` (linear decay over 10% of total steps)
- Learning starts: `80000` steps
- Target network update frequency: `1000` steps
- Train frequency: Every `4` steps
- Replay buffer size: `100,000`

*PPO:*
- Learning rate: `2.5e-4`
- Batch size: `32` (minibatch)
- Gamma (discount factor): `0.99`
- GAE lambda: `0.95`
- Clip coefficient: `0.1`
- Entropy coefficient: `0.01`
- Value function coefficient: `0.5`
- Max gradient norm: `0.5`
- Rollout length: `128` steps
- Update epochs: `4`
## Evaluation Guide

### Single Game Evaluation
```bash
# Evaluate a trained single-task model (e.g., Pong DQN)
python scripts/evaluate.py \
  --mode single \
  --model checkpoints/single/Pong-v5_dqn.pt \
  --algorithm dqn \
  --game Pong-v5 \
  --episodes 10 \
  --max-steps 10000 \
  --json-out outputs/single/Pong-v5_dqn/eval/metrics.json
```

**Output (per experiment folder):**
- JSON metrics: `outputs/single/{game}_{algorithm}/eval/metrics.json`
- Evaluation curve: `outputs/single/{game}_{algorithm}/eval/{game}_{algorithm}_eval_rewards.png`
- Example gameplay video: `outputs/single/{game}_{algorithm}/eval/{game}_{algorithm}_eval_gameplay.mp4`

You can also run all four default single-game evaluations via:
```bash
bash run_evaluate_single.sh
```

### Continual Learning Evaluation
```bash
# Evaluate a trained continual model (e.g., DQN without EWC)
python scripts/evaluate.py \
  --mode continual \
  --model checkpoints/continual/dqn_ewcFalse.pt \
  --algorithm dqn \
  --games Pong-v5 Breakout-v5 SpaceInvaders-v5 \
  --episodes 5 \
  --max-steps 10000 \
  --json-out outputs/continual/dqn_ewcFalse/eval/metrics.json

# Evaluate a PPO continual model with EWC
python scripts/evaluate.py \
  --mode continual \
  --model checkpoints/continual/ppo_ewcTrue.pt \
  --algorithm ppo \
  --games Pong-v5 Breakout-v5 SpaceInvaders-v5 \
  --episodes 5 \
  --max-steps 10000 \
  --ewc \
  --ewc-lambda 0.4 \
  --json-out outputs/continual/ppo_ewcTrue/eval/metrics.json
```

**Output (per continual experiment folder):**
- JSON metrics: `outputs/continual/{algorithm}_ewc{True/False}/eval/metrics.json`
- Avg reward per game bar plot: `outputs/continual/{algorithm}_ewc{True/False}/eval/continual_{algorithm}_avg_rewards.png`
- One gameplay video per game: `outputs/continual/{algorithm}_ewc{True/False}/eval/continual_{algorithm}_{game}_eval_gameplay.mp4`

You can also run all four default continual evaluations via:
```bash
bash run_evaluate_continual.sh
```



## Project Structure

```
Atari_Playground/
├── algorithms/              # Algorithm implementations
│   ├── base.py             # BaseAgent and SimpleNet CNN
│   ├── dqn.py              # DQN algorithm (includes MultiHeadDQNAgent for continual learning)
│   ├── ppo.py              # PPO algorithm (includes MultiHeadPPOAgent for continual learning)
│   └── ewc.py              # EWC wrapper (supports multi-task continual learning)
├── environments/            # Game environments
│   └── atari_env.py        # Atari environment wrapper
├── utils/                  # Utility functions
│   ├── replay_buffer.py    # Experience replay buffer (for DQN-style algorithms)
│   ├── rollout_buffer.py   # Rollout buffer (for PPO/on-policy algorithms)
│   ├── atari_wrappers.py   # Atari-specific preprocessing wrappers (NoopResetEnv, etc.)
│   └── visualization.py    # Video recording & metrics plotting utilities
├── scripts/                # Training scripts
│   ├── train_single.py     # Single game training
│   ├── train_continual.py  # Multi-game continual learning
│   ├── demo.py             # Framework demo
│   ├── full_demo.py        # Comprehensive demo
│   └── visualize_results.py # Results visualization
├── configs/                # Configuration files
├── outputs/                # Training outputs (MP4 videos)
├── checkpoints/            # Model checkpoints
├── README.md               # English documentation
├── README_CN.md            # Chinese tutorial
├── test_framework.py       # Test suite
├── requirements.txt        # Dependencies
└── LICENSE
```

## Algorithms

### DQN (Deep Q-Network)
- **Goal**: Learn optimal action-value function Q(s,a)
- **Method**: Use neural network to approximate Q function
- **Key Techniques**: Experience replay, target networks, ε-greedy exploration
- **Pros**: Stable, reliable
- **Cons**: Slower convergence
- **Continual Learning**: Uses MultiHeadDQNAgent with shared backbone and per-task output heads

### PPO (Proximal Policy Optimization)
- **Goal**: Learn optimal policy π(a|s)
- **Method**: Direct policy optimization
- **Key Techniques**: Advantage estimation, policy clipping, entropy regularization
- **Pros**: Fast learning, stable
- **Cons**: Requires more samples
- **Continual Learning**: Uses MultiHeadPPOAgent with shared backbone and per-task actor/critic heads

### EWC (Elastic Weight Consolidation)
- **Goal**: Retain knowledge of old tasks while learning new ones
- **Method**: Use Fisher Information Matrix (computed from gradient squares) to protect important weights
- **Effect**: Mitigate catastrophic forgetting
- **Application**: Continual learning, lifelong learning
- **Features**: Supports multi-task EWC by storing weights and Fisher Information for each previous task

## Understanding Catastrophic Forgetting

Imagine learning English for a month, then learning French. After a month of French, you realize your English has degraded. This is **catastrophic forgetting**.

AI faces the same problem:
1. Learn Game 1 → Performs well ✓
2. Learn Game 2 → Performs well, but Game 1 performance drops ✗

**EWC Solution**: Like taking notes while learning French to review English, EWC protects important knowledge when learning new games.

## FAQ

**Q: How long does training take?**
A: Depends on steps and hardware:
- Demo: seconds
- 10k steps: 1-2 minutes
- 100k steps: 10-20 minutes on GPU
- Continual learning (3 games, 50k each): 1-2 hours

**Q: Can I use CPU?**
A: Yes, but it will be slower. The framework auto-detects GPU.

**Q: What games are supported?**
A: 108 Atari games including Pong, Breakout, SpaceInvaders, and more.

**Q: How do I visualize results?**
A: PNG metric plots are automatically generated in `outputs/` during training. Videos are only generated if you use the `--save-video` flag (disabled by default to save memory during long training runs).

**Q: Can I modify parameters?**
A: Yes! See the "Training Parameters" section above for all available options.

**Q: How do I know if GPU is being used?**
A: Check the console output. You should see "CUDA" or "GPU" messages if GPU is available.

## Troubleshooting

### ImportError: No module named 'gymnasium'
**Solution**: Run `pip install -r requirements.txt`

### CUDA out of memory
**Solution**: Reduce batch size with `--batch-size 16` or use CPU

### Training is very slow
**Solution**:
- Check if GPU is being used
- Reduce training steps
- Use smaller batch size

### Environment test fails
**Solution**: This is normal without ROM files. Training scripts use built-in ROMs.

## References

- [DQN Paper](https://www.nature.com/articles/nature14236)
- [PPO Paper](https://arxiv.org/abs/1707.06347)
- [EWC Paper](https://arxiv.org/abs/1612.00796)
- [Gymnasium Documentation](https://gymnasium.farama.org/)

## License

MIT License - Free to use and modify

## Contributing

Issues and Pull Requests are welcome!
