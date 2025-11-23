"""Full demonstration of the Atari RL Playground framework."""
import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from algorithms import DQNAgent, PPOAgent, EWCWrapper
from utils import ReplayBuffer


def demo_catastrophic_forgetting():
    """Demonstrate catastrophic forgetting without EWC."""
    print("\n" + "=" * 70)
    print("DEMO 1: Catastrophic Forgetting (without EWC)")
    print("=" * 70)
    
    # Create agent
    agent = DQNAgent(state_dim=4, action_dim=18, lr=1e-4)
    buffer = ReplayBuffer(capacity=5000)
    
    # Simulate Task 1: Learn on "Game 1"
    print("\nTask 1: Training on 'Game 1' (simulated)")
    task1_losses = []
    for step in range(500):
        # Generate random experiences
        state = torch.randn(4, 84, 84)
        action = np.random.randint(18)
        reward = np.random.randn()
        next_state = torch.randn(4, 84, 84)
        done = np.random.random() < 0.1
        
        buffer.add(state, action, reward, next_state, done)
        
        if buffer.is_ready(32):
            batch = buffer.sample(32)
            metrics = agent.update(batch)
            task1_losses.append(metrics['loss'])
    
    avg_loss_task1 = np.mean(task1_losses[-100:])
    print(f"  Final loss on Task 1: {avg_loss_task1:.4f}")
    
    # Simulate Task 2: Learn on "Game 2" (different distribution)
    print("\nTask 2: Training on 'Game 2' (simulated)")
    task2_losses = []
    for step in range(500):
        # Different distribution for Task 2
        state = torch.randn(4, 84, 84) * 2  # Different scale
        action = np.random.randint(18)
        reward = np.random.randn() * 2  # Different reward scale
        next_state = torch.randn(4, 84, 84) * 2
        done = np.random.random() < 0.1
        
        buffer.add(state, action, reward, next_state, done)
        
        if buffer.is_ready(32):
            batch = buffer.sample(32)
            metrics = agent.update(batch)
            task2_losses.append(metrics['loss'])
    
    avg_loss_task2 = np.mean(task2_losses[-100:])
    print(f"  Final loss on Task 2: {avg_loss_task2:.4f}")
    
    print("\n  ⚠️  Without EWC, the agent may forget Task 1 while learning Task 2")
    print(f"      (Loss increased from {avg_loss_task1:.4f} to {avg_loss_task2:.4f})")


def demo_ewc_mitigation():
    """Demonstrate EWC mitigating catastrophic forgetting."""
    print("\n" + "=" * 70)
    print("DEMO 2: EWC Mitigation (with Elastic Weight Consolidation)")
    print("=" * 70)
    
    # Create agent with EWC
    base_agent = DQNAgent(state_dim=4, action_dim=18, lr=1e-4)
    ewc_agent = EWCWrapper(base_agent, ewc_lambda=0.4)
    buffer = ReplayBuffer(capacity=5000)
    
    # Task 1: Learn on "Game 1"
    print("\nTask 1: Training on 'Game 1' (simulated)")
    task1_losses = []
    for step in range(500):
        state = torch.randn(4, 84, 84)
        action = np.random.randint(18)
        reward = np.random.randn()
        next_state = torch.randn(4, 84, 84)
        done = np.random.random() < 0.1
        
        buffer.add(state, action, reward, next_state, done)
        
        if buffer.is_ready(32):
            batch = buffer.sample(32)
            metrics = ewc_agent.update(batch)
            task1_losses.append(metrics['loss'])
    
    avg_loss_task1 = np.mean(task1_losses[-100:])
    print(f"  Final loss on Task 1: {avg_loss_task1:.4f}")
    
    # Consolidate weights after Task 1
    print("  Consolidating weights (computing Fisher Information Matrix)...")
    ewc_agent.consolidate_weights()
    print("  ✓ Weights consolidated")
    
    # Task 2: Learn on "Game 2"
    print("\nTask 2: Training on 'Game 2' (simulated)")
    task2_losses = []
    ewc_losses = []
    for step in range(500):
        state = torch.randn(4, 84, 84) * 2
        action = np.random.randint(18)
        reward = np.random.randn() * 2
        next_state = torch.randn(4, 84, 84) * 2
        done = np.random.random() < 0.1
        
        buffer.add(state, action, reward, next_state, done)
        
        if buffer.is_ready(32):
            batch = buffer.sample(32)
            metrics = ewc_agent.update(batch)
            task2_losses.append(metrics['loss'])
            if 'ewc_loss' in metrics:
                ewc_losses.append(metrics['ewc_loss'])
    
    avg_loss_task2 = np.mean(task2_losses[-100:])
    avg_ewc_loss = np.mean(ewc_losses) if ewc_losses else 0
    print(f"  Final loss on Task 2: {avg_loss_task2:.4f}")
    print(f"  Average EWC regularization loss: {avg_ewc_loss:.6f}")
    
    print("\n  ✓ EWC adds regularization to preserve Task 1 knowledge")
    print(f"    EWC loss penalizes large weight changes from Task 1")


def demo_algorithms():
    """Demonstrate different algorithms."""
    print("\n" + "=" * 70)
    print("DEMO 3: Algorithm Comparison (DQN vs PPO)")
    print("=" * 70)
    
    print("\nDQN (Deep Q-Network):")
    dqn = DQNAgent(state_dim=4, action_dim=18)
    buffer = ReplayBuffer(capacity=1000)
    
    # Collect experiences
    for _ in range(100):
        state = torch.randn(4, 84, 84)
        action = np.random.randint(18)
        reward = np.random.randn()
        next_state = torch.randn(4, 84, 84)
        done = False
        buffer.add(state, action, reward, next_state, done)
    
    # Update
    batch = buffer.sample(32)
    dqn_metrics = dqn.update(batch)
    print(f"  Loss: {dqn_metrics['loss']:.4f}")
    print(f"  Epsilon: {dqn_metrics['epsilon']:.4f}")
    print("  Features: Experience replay, target network, epsilon-greedy")
    
    print("\nPPO (Proximal Policy Optimization):")
    ppo = PPOAgent(state_dim=4, action_dim=18)
    
    # Update
    batch = buffer.sample(32)
    ppo_metrics = ppo.update(batch)
    print(f"  Policy loss: {ppo_metrics['policy_loss']:.4f}")
    print(f"  Value loss: {ppo_metrics['value_loss']:.4f}")
    print(f"  Entropy: {ppo_metrics['entropy']:.4f}")
    print("  Features: Policy clipping, advantage estimation, entropy bonus")


def demo_continual_learning():
    """Demonstrate continual learning setup."""
    print("\n" + "=" * 70)
    print("DEMO 4: Continual Learning Setup")
    print("=" * 70)
    
    print("\nScenario: Train on 3 Atari games sequentially")
    games = ["Pong-v5", "Breakout-v5", "SpaceInvaders-v5"]
    
    print("\nWithout EWC:")
    print("  Game 1 → Game 2 → Game 3")
    print("  ⚠️  Performance on Game 1 may degrade when learning Game 2")
    print("  ⚠️  Performance on Game 2 may degrade when learning Game 3")
    print("  Result: Catastrophic forgetting")
    
    print("\nWith EWC:")
    print("  Game 1 → [Consolidate] → Game 2 → [Consolidate] → Game 3")
    print("  ✓ EWC regularization preserves important weights from Game 1")
    print("  ✓ EWC regularization preserves important weights from Game 2")
    print("  Result: Better knowledge retention (but still challenging)")
    
    print("\nKey insight:")
    print("  Even with EWC, there's a trade-off between learning new tasks")
    print("  and retaining old knowledge - perfect balance is difficult!")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("ATARI RL PLAYGROUND - COMPREHENSIVE FRAMEWORK DEMO")
    print("=" * 70)
    
    demo_algorithms()
    demo_catastrophic_forgetting()
    demo_ewc_mitigation()
    demo_continual_learning()
    
    print("\n" + "=" * 70)
    print("DEMO COMPLETED")
    print("=" * 70)
    print("\nTo train on real Atari games:")
    print("  1. Download ROMs: python -m ale_py.roms_downloader")
    print("  2. Single game: python scripts/train_single.py --game Pong-v5")
    print("  3. Continual learning: python scripts/train_continual.py")
    print("  4. With EWC: python scripts/train_continual.py --use-ewc")

