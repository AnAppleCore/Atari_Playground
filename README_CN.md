# Atari RL Playground - 中文教程

欢迎来到 Atari 强化学习游乐场！这是一个为初学者设计的强化学习框架，用来学习和演示不同的AI算法。

## 🎮 这是什么？

这个项目让你可以用AI来学习玩经典的Atari游戏（比如Pong、Breakout等）。我们实现了3种不同的学习算法：

1. **DQN** - 一种让AI通过记住过去的经验来学习的方法
2. **PPO** - 一种更现代的学习方法，学得更快更稳定
3. **EWC** - 一种帮助AI同时学习多个游戏而不忘记之前学过的东西的方法

## 🚀 完整部署指南

### 第1步：创建 Conda 环境

打开终端，复制粘贴以下命令：

```bash
# 创建一个独立的Python环境（不会影响你的其他程序）
conda create -n atari_rl python=3.10 -y
conda activate atari_rl
```

### 第2步：安装依赖库

```bash
# 安装所有必需的库
pip install -r requirements.txt
```

这会安装：
- PyTorch 2.0+ (包含 CUDA 支持)
- Gymnasium (Atari 环境)
- NumPy, Matplotlib, OpenCV, imageio, imageio-ffmpeg
- tqdm (进度条)

### 第3步：验证安装

```bash
# 运行测试套件
python test_framework.py
```

预期输出：所有测试都应该通过（环境测试可能失败，这是正常的）

### 第4步：测试框架

```bash
# 运行演示程序（不需要 ROM 文件）
python scripts/demo.py
```

你会看到：
- ✓ DQN Agent 创建成功
- ✓ PPO Agent 创建成功
- ✓ EWC 包装器创建成功
- ✓ 模型保存/加载成功

## 📚 训练指南

### 方式1：训练单个游戏

```bash
# 让DQN学习Pong游戏（200,000步，更适合课堂演示）
python scripts/train_single.py --game Pong-v5 --algorithm dqn --steps 200000

# 或者让PPO学习Breakout游戏
python scripts/train_single.py --game Breakout-v5 --algorithm ppo --steps 100000
```

**输出文件：**
- 📹 训练过程录像: `outputs/{game}_{algorithm}_training.mp4`
- 📈 训练曲线: `outputs/{game}_{algorithm}_metrics.png`
- 💾 训练模型: `checkpoints/{game}_{algorithm}.pt`

### 方式2：连续学习（多个游戏）

让AI依次学习3个游戏：Pong → Breakout → SpaceInvaders

```bash
# 不使用EWC - AI会忘记之前学过的游戏（灾难性遗忘）
python scripts/train_continual.py --algorithm dqn --steps-per-game 50000

# 使用EWC - AI会更好地记住之前学过的游戏（但仍然很难完美平衡新旧任务）
python scripts/train_continual.py --algorithm dqn --use-ewc --ewc-lambda 50.0 --steps-per-game 50000
```

**输出文件：**
- 📹 每个游戏的训练录像: `outputs/{game}_dqn_ewc{True/False}.mp4`
- 📈 训练曲线: `outputs/continual_dqn_ewc{True/False}_metrics.png`
- 📉 遗忘曲线: `outputs/continual_dqn_ewc{True/False}_eval.png`
- 💾 训练模型: `checkpoints/continual_dqn_ewc{True/False}.pt`

## 📊 理解结果

### 什么是"灾难性遗忘"？

想象你在学习英语，学了一段时间后，你开始学习法语。结果你发现自己开始忘记英语了！这就是"灾难性遗忘"。

AI也会有同样的问题：
- 学游戏1 → 学得很好 ✓
- 学游戏2 → 学得很好，但忘记了游戏1 ✗

### EWC如何帮助？

EWC（弹性权重巩固）就像给AI的大脑做"笔记"：
- 学完游戏1后，记下哪些"知识"很重要
- 学游戏2时，保护这些重要的"知识"不被改变
- 结果：AI既能学新游戏，又不会完全忘记旧游戏

## 🎮 支持的游戏

框架支持 108 个 Atari 游戏，常见的有：
- **Pong-v5** (6 个动作) - 乒乓球
- **Breakout-v5** (4 个动作) - 打砖块
- **SpaceInvaders-v5** (6 个动作) - 太空侵略者
- **Atari-v5** (18 个动作) - 多个游戏
- 以及 104 个其他游戏...

## 📋 训练参数详解

所有训练脚本都支持以下参数：

```bash
# 游戏选择
--game GAME_NAME            # 游戏名称 (默认: Pong-v5)

# 算法选择
--algorithm {dqn,ppo}       # 算法 (默认: dqn)

# 训练参数
--steps STEPS               # 单游戏总训练步数 (默认: 200000)
--steps-per-game STEPS      # 连续学习中每个游戏的步数 (默认: 50000)
--batch-size BATCH_SIZE     # 批大小 (默认: 32)
--no-video                  # 关闭视频录制，加快训练

# EWC 参数（仅连续学习）
--use-ewc                   # 启用 EWC
--ewc-lambda LAMBDA         # EWC 强度（例如 10.0、50.0）
```

**示例：**
```bash
# 用 PPO 训练 Breakout，50,000 步
python scripts/train_single.py --game Breakout-v5 --algorithm ppo --steps 50000

# 连续学习，每个游戏 10,000 步，启用 EWC
python scripts/train_continual.py --algorithm dqn --use-ewc --steps-per-game 10000

# 改变批大小
python scripts/train_single.py --game Pong-v5 --batch-size 64
```

## 📁 项目结构

```
Atari_Playground/
├── algorithms/              # AI算法的代码
│   ├── base.py             # 基础Agent类和神经网络
│   ├── dqn.py              # DQN算法
│   ├── ppo.py              # PPO算法
│   └── ewc.py              # EWC算法
├── environments/            # 游戏环境
│   └── atari_env.py        # Atari游戏包装
├── utils/                  # 工具函数
│   ├── replay_buffer.py    # 经验存储和回放
│   └── visualization.py    # 可视化工具
├── scripts/                # 训练脚本
│   ├── train_single.py     # 训练单个游戏
│   ├── train_continual.py  # 训练多个游戏
│   ├── demo.py             # 框架演示
│   ├── full_demo.py        # 完整演示
│   └── visualize_results.py # 结果可视化
├── configs/                # 配置文件
├── outputs/                # 训练输出（MP4 视频）
├── checkpoints/            # 模型检查点
├── README.md               # 英文说明
├── README_CN.md            # 中文教程
├── test_framework.py       # 测试套件
├── requirements.txt        # 依赖列表
└── LICENSE
```

## 🎓 学习路径

### 初级（理解概念）
1. 运行 `python scripts/full_demo.py` 看演示
2. 阅读代码注释理解算法
3. 修改参数看效果变化

### 中级（自己训练）
1. 训练单个游戏：`train_single.py`
2. 对比DQN和PPO的效果
3. 调整学习率、步数等参数

### 高级（研究灾难性遗忘）
1. 运行不使用EWC的连续学习
2. 运行使用EWC的连续学习
3. 对比两者的性能差异
4. 尝试修改EWC的参数

## 💡 初学者常见问题

### Q1: 什么是 DQN？
**A:** DQN 是 Deep Q-Network 的缩写。它的工作原理是：
- AI 看到游戏画面
- 神经网络预测每个动作的"价值"（能得到多少分）
- AI 选择价值最高的动作
- 重复这个过程，不断改进预测

### Q2: PPO 和 DQN 有什么区别？
**A:**
- **DQN**: 学习"价值"（这个动作值多少分）
- **PPO**: 学习"策略"（应该做什么动作）

PPO 通常学得更快，更稳定。

### Q3: 为什么需要 EWC？
**A:** 当 AI 学习新游戏时，它会改变神经网络的权重，导致之前学过的游戏性能下降。EWC 通过"保护"重要的权重来解决这个问题。

### Q4: 训练需要多长时间？
**A:**
- 演示（demo）: 几秒钟
- 单游戏训练（10,000步）: 1-2 分钟
- 单游戏训练（100,000步）: 10-20 分钟
- 连续学习（3个游戏，各50,000步）: 1-2 小时

### Q5: 我可以修改参数吗？
**A:** 当然可以！常用参数见上面的"训练参数详解"部分。

### Q6: 如何使用 GPU？
**A:** 框架会自动检测 GPU。如果有 NVIDIA GPU，会自动使用 CUDA。如果没有，会使用 CPU（较慢）。

### Q7: 训练过程中会生成什么文件？
**A:**
- **MP4 视频文件**: `outputs/` 目录，游戏录像（15 FPS，便于观看）
- **模型文件**: `checkpoints/` 目录，训练好的模型
- **日志**: 控制台输出，包括损失值和进度

## 🎓 学习路径

### 初级（1-2小时）
1. 阅读本教程
2. 运行 `python scripts/demo.py`
3. 运行 `python test_framework.py`
4. 理解项目结构

### 中级（2-4小时）
1. 训练单个游戏：`python scripts/train_single.py --game Pong-v5 --steps 10000`
2. 查看生成的 MP4 视频文件
3. 尝试不同的游戏和算法
4. 阅读 `algorithms/` 中的代码

### 高级（4+小时）
1. 运行连续学习：`python scripts/train_continual.py --steps-per-game 10000`
2. 对比有/无 EWC 的结果
3. 修改算法参数
4. 实现自己的算法

## 📖 理论背景

### DQN (Deep Q-Network)
- **目标**: 学习最优的动作价值函数 Q(s,a)
- **方法**: 使用神经网络近似 Q 函数
- **关键技术**: 经验回放、目标网络、ε-贪心探索
- **优点**: 稳定、可靠
- **缺点**: 收敛较慢

### PPO (Proximal Policy Optimization)
- **目标**: 学习最优的策略 π(a|s)
- **方法**: 直接优化策略
- **关键技术**: 优势估计、策略裁剪、熵正则化
- **优点**: 学习快、稳定
- **缺点**: 需要更多样本

### EWC (Elastic Weight Consolidation)
- **目标**: 在学习新任务时保留旧任务的知识
- **方法**: 使用 Fisher 信息矩阵保护重要权重
- **效果**: 减轻灾难性遗忘
- **应用**: 连续学习、终身学习

## 🔧 故障排除

### 问题1: ImportError: No module named 'gymnasium'
**解决**: 运行 `pip install -r requirements.txt`

### 问题2: CUDA out of memory
**解决**: 减少批大小 `--batch-size 16` 或使用 CPU

### 问题3: 训练很慢
**解决**:
- 检查是否使用了 GPU（应该看到 CUDA 信息）
- 减少训练步数
- 使用更小的批大小

### 问题4: 环境测试失败
**解决**: 这是正常的，因为没有 ROM 文件。训练脚本会自动使用内置的 ROM。

## 📝 许可证

MIT License - 自由使用和修改

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📚 参考资源

- [DQN 论文](https://www.nature.com/articles/nature14236)
- [PPO 论文](https://arxiv.org/abs/1707.06347)
- [EWC 论文](https://arxiv.org/abs/1612.00796)
- [Gymnasium 文档](https://gymnasium.farama.org/)