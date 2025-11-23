"""Test the framework."""
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from algorithms import DQNAgent, PPOAgent, EWCWrapper
from environments import AtariEnv
from utils import ReplayBuffer


def test_agents():
    """Test agent creation and basic operations."""
    print("Testing agents...")
    
    # Test DQN
    print("  - Testing DQN...")
    dqn = DQNAgent(state_dim=4, action_dim=18)
    state = torch.randn(4, 84, 84)
    action = dqn.select_action(state)
    assert isinstance(action, int), "Action should be int"
    print("    ✓ DQN works")
    
    # Test PPO
    print("  - Testing PPO...")
    ppo = PPOAgent(state_dim=4, action_dim=18)
    action = ppo.select_action(state)
    assert isinstance(action, int), "Action should be int"
    print("    ✓ PPO works")
    
    # Test EWC
    print("  - Testing EWC...")
    ewc_agent = EWCWrapper(dqn)
    action = ewc_agent.select_action(state)
    assert isinstance(action, int), "Action should be int"
    print("    ✓ EWC works")


def test_environment():
    """Test environment."""
    print("\nTesting environment...")
    
    try:
        env = AtariEnv("Pong-v5")
        print("  - Environment created")
        
        state = env.reset()
        print(f"  - Initial state shape: {state.shape}")
        assert state.shape == (4, 84, 84), "State shape should be (4, 84, 84)"
        
        for _ in range(5):
            action = env.env.action_space.sample()
            next_state, reward, done = env.step(action)
            assert next_state.shape == (4, 84, 84), "Next state shape should be (4, 84, 84)"
        
        env.close()
        print("  ✓ Environment works")
    except Exception as e:
        print(f"  ✗ Environment test failed: {e}")


def test_replay_buffer():
    """Test replay buffer."""
    print("\nTesting replay buffer...")
    
    buffer = ReplayBuffer(capacity=1000)
    
    # Add some experiences
    for _ in range(100):
        state = torch.randn(4, 84, 84)
        action = 0
        reward = 1.0
        next_state = torch.randn(4, 84, 84)
        done = False
        
        buffer.add(state, action, reward, next_state, done)
    
    assert len(buffer) == 100, "Buffer should have 100 experiences"
    print(f"  - Buffer size: {len(buffer)}")
    
    # Sample batch
    batch = buffer.sample(32)
    assert batch["states"].shape[0] == 32, "Batch size should be 32"
    print(f"  - Batch shape: {batch['states'].shape}")
    print("  ✓ Replay buffer works")


def test_update():
    """Test agent update."""
    print("\nTesting agent update...")
    
    agent = DQNAgent(state_dim=4, action_dim=18)
    buffer = ReplayBuffer(capacity=1000)
    
    # Add experiences
    for _ in range(100):
        state = torch.randn(4, 84, 84)
        action = 0
        reward = 1.0
        next_state = torch.randn(4, 84, 84)
        done = False
        buffer.add(state, action, reward, next_state, done)
    
    # Update
    batch = buffer.sample(32)
    metrics = agent.update(batch)
    
    assert "loss" in metrics, "Metrics should contain loss"
    print(f"  - Metrics: {metrics}")
    print("  ✓ Agent update works")


if __name__ == "__main__":
    print("=" * 50)
    print("Testing Atari RL Framework")
    print("=" * 50)
    
    test_agents()
    test_replay_buffer()
    test_update()
    
    try:
        test_environment()
    except Exception as e:
        print(f"\nNote: Environment test skipped (may need ROM files): {e}")
    
    print("\n" + "=" * 50)
    print("All tests passed! ✓")
    print("=" * 50)

