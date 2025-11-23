"""PPO (Proximal Policy Optimization) implementation."""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict
from .base import BaseAgent, SimpleNet


class PPOAgent(BaseAgent):
    """PPO Agent for Atari games."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        entropy_coef: float = 0.01,
        device: str = "cuda",
    ):
        super().__init__(state_dim, action_dim, device)
        
        self.policy_net = SimpleNet(state_dim, action_dim).to(self.device)
        self.value_net = nn.Sequential(
            nn.Conv2d(state_dim, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        ).to(self.device)
        
        self.network = self.policy_net  # For compatibility
        
        self.optimizer = optim.Adam(
            list(self.policy_net.parameters()) + list(self.value_net.parameters()),
            lr=lr
        )
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
    
    def select_action(self, state: torch.Tensor) -> int:
        """Select action using policy network."""
        with torch.no_grad():
            state = state.unsqueeze(0).to(self.device)
            logits = self.policy_net(state)
            probs = torch.softmax(logits, dim=1)
            action = torch.multinomial(probs, 1).item()
        return action
    
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Update PPO with a batch of experiences."""
        states = batch["states"].to(self.device)
        actions = batch["actions"].to(self.device)
        rewards = batch["rewards"].to(self.device)
        next_states = batch["next_states"].to(self.device)
        dones = batch["dones"].to(self.device)
        
        # Compute advantages
        with torch.no_grad():
            values = self.value_net(states).squeeze(1)
            next_values = self.value_net(next_states).squeeze(1)
            td_targets = rewards + self.gamma * next_values * (1 - dones)
            advantages = td_targets - values
        
        # Policy update
        logits = self.policy_net(states)
        log_probs = torch.log_softmax(logits, dim=1)
        action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Entropy bonus
        probs = torch.softmax(logits, dim=1)
        entropy = -(probs * log_probs).sum(dim=1).mean()
        
        # Policy loss
        policy_loss = -(action_log_probs * advantages).mean()
        
        # Value loss
        value_loss = nn.MSELoss()(self.value_net(states).squeeze(1), td_targets)
        
        total_loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy
        
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.policy_net.parameters()) + list(self.value_net.parameters()),
            1.0
        )
        self.optimizer.step()
        
        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
        }

