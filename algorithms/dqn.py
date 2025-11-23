"""DQN (Deep Q-Network) implementation."""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict
from .base import BaseAgent, SimpleNet, AtariBackbone


class DQNAgent(BaseAgent):
    """DQN Agent for Atari games."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 2.5e-4,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.99995,
        epsilon_min: float = 0.1,
        device: str = "cuda",
    ):
        super().__init__(state_dim, action_dim, device)

        self.network = SimpleNet(state_dim, action_dim).to(self.device)
        self.target_network = SimpleNet(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())

        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.update_count = 0
        self.target_update_freq = 1000

    def select_action(self, state: torch.Tensor) -> int:
        """Select action using epsilon-greedy policy."""
        if np.random.random() < self.epsilon:
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

        # Current Q-values
        q_values = self.network(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(dim=1)[0]
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        loss = self.loss_fn(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
        self.optimizer.step()

        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.network.state_dict())

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return {"loss": loss.item(), "epsilon": self.epsilon}




class MultiHeadDQNAgent(BaseAgent):
    """DQN agent with shared AtariBackbone and per-task output heads.

    Used for continual learning where different games may have different
    action spaces but share the same convolutional backbone.
    """

    def __init__(
        self,
        state_dim: int,
        lr: float = 2.5e-4,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.99995,
        epsilon_min: float = 0.1,
        device: str = "cuda",
    ):
        # action_dim is a placeholder; real per-task dims are handled via heads
        super().__init__(state_dim, action_dim=1, device=device)

        self.backbone = AtariBackbone(input_channels=state_dim).to(self.device)
        self.target_backbone = AtariBackbone(input_channels=state_dim).to(self.device)
        self.target_backbone.load_state_dict(self.backbone.state_dict())

        # Expose backbone as "network" so EWC operates on shared features only
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
        """(Re)create optimizer over backbone and all heads."""
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
        # Keep BaseAgent.action_dim consistent with the active head
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

        # Current Q-values
        features = self.backbone(states)
        q_values = head(features)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values
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
