"""Elastic Weight Consolidation (EWC) for continual learning."""
import torch
import torch.nn as nn
from typing import Dict, Optional
from .base import BaseAgent


class EWCWrapper:
    """Wrapper to add EWC regularization to any agent."""

    def __init__(self, agent: BaseAgent, ewc_lambda: float = 0.4):
        """
        Initialize EWC wrapper.

        Args:
            agent: The base agent to wrap
            ewc_lambda: Strength of EWC regularization
        """
        self.agent = agent
        self.ewc_lambda = ewc_lambda
        self.fisher_information = {}
        self.previous_weights = {}
        self.is_first_task = True

    def consolidate_weights(self):
        """Consolidate weights after learning a task."""
        if self.agent.network is None:
            return

        # Store current weights
        self.previous_weights = {
            name: param.clone().detach()
            for name, param in self.agent.network.named_parameters()
        }

        # Compute Fisher Information Matrix (simplified: use parameter gradients)
        self.fisher_information = {
            name: torch.ones_like(param) * 0.1  # Simplified: uniform importance
            for name, param in self.agent.network.named_parameters()
        }

        self.is_first_task = False

    def compute_ewc_loss(self) -> torch.Tensor:
        """Compute EWC regularization loss."""
        if self.is_first_task or self.agent.network is None:
            return torch.tensor(0.0, device=self.agent.device)

        ewc_loss = torch.tensor(0.0, device=self.agent.device)

        for name, param in self.agent.network.named_parameters():
            if name in self.previous_weights and name in self.fisher_information:
                fisher = self.fisher_information[name]
                prev_weight = self.previous_weights[name]

                # EWC loss: sum of Fisher * (weight - prev_weight)^2
                ewc_loss += (fisher * (param - prev_weight) ** 2).sum()

        return self.ewc_lambda * ewc_loss

    def select_action(self, state: torch.Tensor) -> int:
        """Delegate to wrapped agent."""
        return self.agent.select_action(state)

    def register_task(self, task_id: str, action_dim: int):
        """Optionally register a new task on the wrapped agent if supported."""
        if hasattr(self.agent, "register_task"):
            self.agent.register_task(task_id, action_dim)

    def set_task(self, task_id: str):
        """Optionally set the current task on the wrapped agent if supported."""
        if hasattr(self.agent, "set_task"):
            self.agent.set_task(task_id)


    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Update with EWC regularization."""
        # For DQN and PPO, we need to add EWC loss to the optimization
        # This is a simplified approach that works with the current agent implementations

        # First, get the base metrics from the agent
        metrics = self.agent.update(batch)

        # Compute and add EWC regularization loss
        ewc_loss = self.compute_ewc_loss()
        if ewc_loss.item() > 0:
            # Add EWC loss to metrics for monitoring
            metrics["ewc_loss"] = ewc_loss.item()

            # For proper EWC, we need to do an additional backward pass
            # This ensures EWC regularization is actually applied
            if self.agent.network is not None:
                self.agent.optimizer.zero_grad()
                ewc_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.agent.network.parameters(), 1.0)
                self.agent.optimizer.step()

        return metrics

    def save(self, path: str):
        """Save the wrapped agent."""
        self.agent.save(path)

    def load(self, path: str):
        """Load the wrapped agent."""
        self.agent.load(path)

