"""Demo script showing framework usage."""
import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from algorithms import DQNAgent, PPOAgent, EWCWrapper
from utils import ReplayBuffer, RolloutBuffer


def demo_basic_usage():
    """Demonstrate basic framework usage."""
    print("=" * 60)
    print("Atari RL Playground - Framework Demo")
    print("=" * 60)
    
    # 1. Create agents
    print("\n1. Creating agents...")
    dqn_agent = DQNAgent(state_dim=4, action_dim=18, lr=1e-4)
    ppo_agent = PPOAgent(state_dim=4, action_dim=18, lr=2.5e-4)
    print("   ✓ DQN Agent created")
    print("   ✓ PPO Agent created")
    
    # 2. Create buffers
    print("\n2. Creating buffers...")
    dqn_buffer = ReplayBuffer(capacity=10000)
    ppo_buffer = RolloutBuffer(capacity=128)
    print("   ✓ Replay buffer created for DQN (capacity: 10000)")
    print("   ✓ Rollout buffer created for PPO (capacity: 128)")
    
    # 3. Simulate experience collection for DQN
    print("\n3. Simulating experience collection for DQN...")
    for i in range(100):
        state = torch.randn(4, 84, 84)
        action = 0
        reward = 1.0
        next_state = torch.randn(4, 84, 84)
        done = False
        
        dqn_buffer.add(state, action, reward, next_state, done)
    
    print(f"   ✓ Collected 100 experiences for DQN")
    print(f"   ✓ DQN Buffer size: {len(dqn_buffer)}")
    
    # 3b. Simulate experience collection for PPO
    print("\n3b. Simulating experience collection for PPO...")
    rollout_length = 128
    for i in range(rollout_length):
        state = torch.randn(4, 84, 84)
        state_tensor = state.unsqueeze(0).to(ppo_agent.device)
        
        with torch.no_grad():
            action_tensor, log_prob, _, value = ppo_agent.get_action_and_value(state_tensor)
            action = action_tensor.item()
            log_prob_val = log_prob.item()
            value_val = value.item()
        
        reward = 1.0
        done = False
        
        ppo_buffer.add(state, action, reward, done, log_prob_val, value_val)
    
    print(f"   ✓ Collected {rollout_length} experiences for PPO")
    print(f"   ✓ PPO Buffer size: {len(ppo_buffer)}")
    
    # 4. Agent action selection
    print("\n4. Testing action selection...")
    state = torch.randn(4, 84, 84)
    
    dqn_action = dqn_agent.select_action(state)
    ppo_action = ppo_agent.select_action(state)
    
    print(f"   ✓ DQN selected action: {dqn_action}")
    print(f"   ✓ PPO selected action: {ppo_action}")
    
    # 5. Agent update
    print("\n5. Testing agent update...")
    dqn_batch = dqn_buffer.sample(32)
    dqn_metrics = dqn_agent.update(dqn_batch)
    print(f"   ✓ DQN update metrics: {dqn_metrics}")
    
    # PPO update requires rollout data and next_value
    rollout_data = ppo_buffer.get_batch()
    # Compute next_value for PPO
    next_state = torch.randn(4, 84, 84)
    with torch.no_grad():
        next_state_tensor = next_state.unsqueeze(0).to(ppo_agent.device)
        next_value = ppo_agent.get_value(next_state_tensor).flatten()
    
    ppo_metrics = ppo_agent.update(rollout_data, next_value, update_epochs=4, minibatch_size=32)
    print(f"   ✓ PPO update metrics: {ppo_metrics}")
    
    # 6. EWC demonstration
    print("\n6. Testing EWC (Elastic Weight Consolidation)...")
    ewc_agent = EWCWrapper(dqn_agent, ewc_lambda=0.4)
    
    print("   ✓ EWC wrapper created")
    print("   ✓ Training on first task...")
    
    for _ in range(10):
        batch = dqn_buffer.sample(32)
        ewc_agent.update(batch)
    
    print("   ✓ Consolidating weights after first task...")
    # Consolidate weights with a sample batch
    consolidate_batch = dqn_buffer.sample(32)
    ewc_agent.consolidate_weights(consolidate_batch)
    
    print("   ✓ Training on second task...")
    for _ in range(10):
        batch = dqn_buffer.sample(32)
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

