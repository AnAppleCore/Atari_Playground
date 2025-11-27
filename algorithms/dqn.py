"""DQN (Deep Q-Network) implementation."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict
from .base import BaseAgent, SimpleNet, AtariBackbone, safe_torch_load


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    """Linear epsilon schedule."""
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


class DQNAgent(BaseAgent):
    """DQN Agent for Atari games."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_fraction: float = 0.10,
        total_timesteps: int = 10000000,
        target_update_freq: int = 1000,
        tau: float = 1.0,
        device: str = "cuda",
    ):
        super().__init__(state_dim, action_dim, device)

        self.network = SimpleNet(state_dim, action_dim).to(self.device)
        self.target_network = SimpleNet(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())

        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_fraction = epsilon_fraction
        self.total_timesteps = total_timesteps
        self.target_update_freq = target_update_freq
        self.tau = tau
        self.update_count = 0
        self.global_step = 0
        self.q_value_history = []

    def select_action(self, state: torch.Tensor, training: bool = True) -> int:
        """Select action using epsilon-greedy policy."""
        epsilon = linear_schedule(
            self.epsilon_start,
            self.epsilon_end,
            int(self.epsilon_fraction * self.total_timesteps),
            self.global_step
        )
        
        if training and np.random.random() < epsilon:
            return np.random.randint(self.action_dim)

        with torch.no_grad():
            state = state.unsqueeze(0).to(self.device)
            q_values = self.network(state)
            return q_values.argmax(dim=1).item()

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Update DQN with a batch of experiences."""
        states = batch["states"].to(self.device)
        actions = batch["actions"].to(self.device)
        rewards = batch["rewards"].to(self.device)
        next_states = batch["next_states"].to(self.device)
        dones = batch["dones"].to(self.device)

        q_values = self.network(states)
        q_values_selected = q_values.gather(1, actions.unsqueeze(1)).squeeze()

        with torch.no_grad():
            target_max, _ = self.target_network(next_states).max(dim=1)
            td_target = rewards.flatten() + self.gamma * target_max * (1 - dones.flatten())

        loss = F.mse_loss(td_target, q_values_selected)
        
        with torch.no_grad():
            self.q_value_history.append(q_values.mean().item())
            if len(self.q_value_history) > 1000:
                self.q_value_history.pop(0)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 10.0)
        self.optimizer.step()

        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            for target_param, q_network_param in zip(self.target_network.parameters(), self.network.parameters()):
                target_param.data.copy_(
                    self.tau * q_network_param.data + (1.0 - self.tau) * target_param.data
                )

        epsilon = linear_schedule(
            self.epsilon_start,
            self.epsilon_end,
            int(self.epsilon_fraction * self.total_timesteps),
            self.global_step
        )
        
        avg_q = sum(self.q_value_history) / len(self.q_value_history) if self.q_value_history else 0.0

        return {"loss": loss.item(), "epsilon": epsilon, "q_value": avg_q}
    
    def save(self, path: str):
        """Save the agent's networks."""
        torch.save({
            'network': self.network.state_dict(),
            'target_network': self.target_network.state_dict(),
        }, path)
    
    def load(self, path: str):
        """Load the agent's networks."""
        checkpoint = safe_torch_load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network'])
        self.target_network.load_state_dict(checkpoint['target_network'])




class MultiHeadDQNAgent(BaseAgent):
    """DQN agent with shared AtariBackbone and per-task output heads.

    Used for continual learning where different games may have different
    action spaces but share the same convolutional backbone.
    """

    def __init__(
        self,
        state_dim: int,
        lr: float = 1e-4,  # Same as DQNAgent for consistency
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.99995,
        epsilon_min: float = 0.1,
        device: str = "cuda",
    ):
        super().__init__(state_dim, action_dim=1, device=device)

        self.backbone = AtariBackbone(input_channels=state_dim).to(self.device)
        self.target_backbone = AtariBackbone(input_channels=state_dim).to(self.device)
        self.target_backbone.load_state_dict(self.backbone.state_dict())
        self.network = self.backbone

        self.heads = nn.ModuleDict()
        self.target_heads = nn.ModuleDict()

        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.update_count = 0
        self.target_update_freq = 1000
        self.loss_fn = nn.MSELoss()
        self.optimizer = None
        self.current_task = None

    def _rebuild_optimizer(self):
        """Recreate optimizer over backbone and all heads."""
        params = list(self.backbone.parameters())
        for head in self.heads.values():
            params += list(head.parameters())
        self.optimizer = optim.Adam(params, lr=self.lr)

    def register_task(self, task_id: str, action_dim: int):
        """Create a new output head for a task if it does not exist."""
        if task_id in self.heads:
            return

        head = nn.Linear(self.backbone.feature_dim, action_dim).to(self.device)
        target_head = nn.Linear(self.backbone.feature_dim, action_dim).to(self.device)
        target_head.load_state_dict(head.state_dict())

        self.heads[task_id] = head
        self.target_heads[task_id] = target_head
        self._rebuild_optimizer()

    def set_task(self, task_id: str):
        """Select which task/head to use for subsequent calls."""
        if task_id not in self.heads:
            raise ValueError(f"Task '{task_id}' not registered in MultiHeadDQNAgent.")
        self.current_task = task_id
        self.action_dim = self.heads[task_id].out_features

    def _current_heads(self):
        if self.current_task is None:
            raise RuntimeError("Current task is not set for MultiHeadDQNAgent.")
        return self.heads[self.current_task], self.target_heads[self.current_task]

    def select_action(self, state: torch.Tensor) -> int:
        """Epsilon-greedy action selection for the current task."""
        if self.current_task is None:
            raise RuntimeError("Current task is not set before select_action().")

        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)

        with torch.no_grad():
            state = state.unsqueeze(0).to(self.device)
            features = self.backbone(state)
            head, _ = self._current_heads()
            q_values = head(features)
            return q_values.argmax(dim=1).item()

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """DQN update using shared backbone and task-specific head."""
        if self.current_task is None:
            raise RuntimeError("Current task is not set before update().")
        if self.optimizer is None:
            raise RuntimeError("Optimizer has not been initialized; call register_task() first.")

        states = batch["states"].to(self.device)
        actions = batch["actions"].to(self.device)
        rewards = batch["rewards"].to(self.device)
        next_states = batch["next_states"].to(self.device)
        dones = batch["dones"].to(self.device)

        head, target_head = self._current_heads()

        features = self.backbone(states)
        q_values = head(features)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_features = self.target_backbone(next_states)
            next_q_values = target_head(next_features).max(dim=1)[0]
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        loss = self.loss_fn(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.backbone.parameters(), 1.0)
        for head in self.heads.values():
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        self.optimizer.step()

        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_backbone.load_state_dict(self.backbone.state_dict())
            for name, src_head in self.heads.items():
                self.target_heads[name].load_state_dict(src_head.state_dict())

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return {"loss": loss.item(), "epsilon": self.epsilon}
