"""Experience replay buffer."""
import numpy as np
import torch
from typing import Dict
from collections import deque


class ReplayBuffer:
    """Experience replay buffer for RL."""
    
    def __init__(self, capacity: int = 100000):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of experiences to store
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def add(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
    ):
        """Add experience to buffer."""
        self.buffer.append({
            "state": state.cpu().numpy(),
            "action": action,
            "reward": reward,
            "next_state": next_state.cpu().numpy(),
            "done": done,
        })
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample a batch of experiences."""
        actual_batch_size = min(batch_size, len(self.buffer))
        replace = len(self.buffer) < batch_size

        indices = np.random.choice(len(self.buffer), actual_batch_size, replace=replace)
        batch = [self.buffer[i] for i in indices]
        
        states = torch.from_numpy(np.array([b["state"] for b in batch])).float()
        actions = torch.from_numpy(np.array([b["action"] for b in batch])).long()
        rewards = torch.from_numpy(np.array([b["reward"] for b in batch])).float()
        next_states = torch.from_numpy(np.array([b["next_state"] for b in batch])).float()
        dones = torch.from_numpy(np.array([b["done"] for b in batch])).float()
        
        return {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "next_states": next_states,
            "dones": dones,
        }
    
    def __len__(self) -> int:
        """Return buffer size."""
        return len(self.buffer)
    
    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough samples."""
        return len(self.buffer) >= batch_size

