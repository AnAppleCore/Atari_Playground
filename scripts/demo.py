"""Demo script showing framework usage."""
import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from algorithms import DQNAgent, PPOAgent, EWCWrapper
from utils import ReplayBuffer


def demo_basic_usage():
    """Demonstrate basic framework usage."""
    print("=" * 60)
    print("Atari RL Playground - Framework Demo")
    print("=" * 60)
    
    # 1. Create agents
    print("\n1. Creating agents...")
    dqn_agent = DQNAgent(state_dim=4, action_dim=18)
    ppo_agent = PPOAgent(state_dim=4, action_dim=18)
    print("   ✓ DQN Agent created")
    print("   ✓ PPO Agent created")
    
    # 2. Create replay buffer
    print("\n2. Creating replay buffer...")
    buffer = ReplayBuffer(capacity=10000)
    print("   ✓ Replay buffer created (capacity: 10000)")
    
    # 3. Simulate experience collection
    print("\n3. Simulating experience collection...")
    for i in range(100):
        state = torch.randn(4, 84, 84)
        action = 0
        reward = 1.0
        next_state = torch.randn(4, 84, 84)
        done = False
        
        buffer.add(state, action, reward, next_state, done)
    
    print(f"   ✓ Collected 100 experiences")
    print(f"   ✓ Buffer size: {len(buffer)}")
    
    # 4. Agent action selection
    print("\n4. Testing action selection...")
    state = torch.randn(4, 84, 84)
    
    dqn_action = dqn_agent.select_action(state)
    ppo_action = ppo_agent.select_action(state)
    
    print(f"   ✓ DQN selected action: {dqn_action}")
    print(f"   ✓ PPO selected action: {ppo_action}")
    
    # 5. Agent update
    print("\n5. Testing agent update...")
    batch = buffer.sample(32)
    
    dqn_metrics = dqn_agent.update(batch)
    ppo_metrics = ppo_agent.update(batch)
    
    print(f"   ✓ DQN update metrics: {dqn_metrics}")
    print(f"   ✓ PPO update metrics: {ppo_metrics}")
    
    # 6. EWC demonstration
    print("\n6. Testing EWC (Elastic Weight Consolidation)...")
    ewc_agent = EWCWrapper(dqn_agent, ewc_lambda=0.4)
    
    print("   ✓ EWC wrapper created")
    print("   ✓ Training on first task...")
    
    for _ in range(10):
        batch = buffer.sample(32)
        ewc_agent.update(batch)
    
    print("   ✓ Consolidating weights after first task...")
    ewc_agent.consolidate_weights()
    
    print("   ✓ Training on second task...")
    for _ in range(10):
        batch = buffer.sample(32)
        metrics = ewc_agent.update(batch)
    
    print(f"   ✓ EWC metrics: {metrics}")
    
    # 7. Model saving/loading
    print("\n7. Testing model save/load...")
    Path("demo_checkpoints").mkdir(exist_ok=True)
    
    dqn_agent.save("demo_checkpoints/dqn_demo.pt")
    print("   ✓ DQN model saved")
    
    dqn_agent.load("demo_checkpoints/dqn_demo.pt")
    print("   ✓ DQN model loaded")
    
    # Cleanup
    import shutil
    shutil.rmtree("demo_checkpoints")
    
    print("\n" + "=" * 60)
    print("Demo completed successfully! ✓")
    print("=" * 60)
    
    print("\nNext steps:")
    print("1. Download Atari ROMs: python -m ale_py.roms_downloader")
    print("2. Train on single game: python scripts/train_single.py")
    print("3. Train continually: python scripts/train_continual.py")
    print("4. Check README.md or README_CN.md for more details")


if __name__ == "__main__":
    demo_basic_usage()

