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
pip install -r requirements.txt
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
# Train DQN on Pong for 200,000 steps (good for classroom demo)
python scripts/train_single.py --game Pong-v5 --algorithm dqn --steps 200000

# Train PPO on Breakout for 100,000 steps
python scripts/train_single.py --game Breakout-v5 --algorithm ppo --steps 100000
```

**Output:**
- MP4 video: `outputs/{game}_{algorithm}_training.mp4`
- Metrics plot: `outputs/{game}_{algorithm}_metrics.png`
- Model checkpoint: `checkpoints/{game}_{algorithm}.pt`

### Continual Learning (Multiple Games)
```bash
# Train on 3 games sequentially WITHOUT EWC (shows catastrophic forgetting)
python scripts/train_continual.py --algorithm dqn --steps-per-game 50000

# Train on 3 games sequentially WITH EWC (mitigates forgetting)
python scripts/train_continual.py --algorithm dqn --use-ewc --ewc-lambda 50.0 --steps-per-game 50000
```

**Output:**
- MP4 videos for each game: `outputs/{game}_{algorithm}_ewc{True/False}.mp4`
- Training metrics: `outputs/continual_{algorithm}_ewc{True/False}_metrics.png`
- Forgetting curves: `outputs/continual_{algorithm}_ewc{True/False}_eval.png`
- Model checkpoint: `checkpoints/continual_{algorithm}_ewc{True/False}.pt`

### Available Games
The framework supports 108 Atari games. Common ones:
- Pong-v5 (6 actions)
- Breakout-v5 (4 actions)
- SpaceInvaders-v5 (6 actions)
- Atari-v5 (18 actions)
- And 104 more...

### Training Parameters
```bash
# Common parameters for all training scripts:
--game GAME_NAME          # Game to train on (default: Pong-v5)
--algorithm {dqn,ppo}     # Algorithm to use (default: dqn)
--steps STEPS             # Total training steps (single game, default: 200000)
--steps-per-game STEPS    # Steps per game in continual learning (default: 50000)
--use-ewc                 # Enable EWC for continual learning
--ewc-lambda LAMBDA       # EWC regularization strength (e.g., 10.0, 50.0)
--batch-size BATCH_SIZE   # Batch size for training (default: 32)
--no-video                # Disable video recording for speed
```

## Project Structure

```
Atari_Playground/
├── algorithms/              # Algorithm implementations
│   ├── base.py             # BaseAgent and SimpleNet CNN
│   ├── dqn.py              # DQN algorithm
│   ├── ppo.py              # PPO algorithm
│   └── ewc.py              # EWC wrapper
├── environments/            # Game environments
│   └── atari_env.py        # Atari environment wrapper
├── utils/                  # Utility functions
│   ├── replay_buffer.py    # Experience replay buffer
│   └── visualization.py    # Visualization tools
├── scripts/                # Training scripts
│   ├── train_single.py     # Single game training
│   ├── train_continual.py  # Multi-game continual learning
│   ├── demo.py             # Framework demo
│   ├── full_demo.py        # Comprehensive demo
│   └── visualize_results.py # Results visualization
├── configs/                # Configuration files
├── outputs/                # Training outputs (GIFs, videos)
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

### PPO (Proximal Policy Optimization)
- **Goal**: Learn optimal policy π(a|s)
- **Method**: Direct policy optimization
- **Key Techniques**: Advantage estimation, policy clipping, entropy regularization
- **Pros**: Fast learning, stable
- **Cons**: Requires more samples

### EWC (Elastic Weight Consolidation)
- **Goal**: Retain knowledge of old tasks while learning new ones
- **Method**: Use Fisher Information Matrix to protect important weights
- **Effect**: Mitigate catastrophic forgetting
- **Application**: Continual learning, lifelong learning

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
A: MP4 videos and PNG metric plots are automatically generated in `outputs/` during training.

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
