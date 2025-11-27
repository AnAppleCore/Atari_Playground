"""Rollout buffer for on-policy algorithms like PPO."""
import numpy as np
import torch
from typing import Dict, List
from collections import deque


class RolloutBuffer:
    """Rollout buffer for on-policy algorithms like PPO."""
    
    def __init__(self, capacity: int = 128):
        """
        Initialize rollout buffer.
        
        Args:
            capacity: Number of steps to collect before updating (rollout length)
        """
        self.capacity = capacity
        self.reset()
    
    def reset(self):
        """Reset the buffer."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        self.pos = 0
    
    def add(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        done: bool,
        log_prob: float,
        value: float,
    ):
        """Add a transition to the buffer."""
        self.states.append(state.cpu().numpy())
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.pos += 1
    
    def is_full(self) -> bool:
        """Check if buffer is full."""
        return self.pos >= self.capacity
    
    def get_batch(self) -> Dict[str, torch.Tensor]:
        """Get all collected data as a batch."""
        return {
            "states": torch.from_numpy(np.array(self.states)).float(),
            "actions": torch.from_numpy(np.array(self.actions)).long(),
            "rewards": torch.from_numpy(np.array(self.rewards)).float(),
            "dones": torch.from_numpy(np.array(self.dones)).float(),
            "log_probs": torch.from_numpy(np.array(self.log_probs)).float(),
            "values": torch.from_numpy(np.array(self.values)).float(),
        }
    
    def __len__(self) -> int:
        """Return current buffer size."""
        return self.pos

