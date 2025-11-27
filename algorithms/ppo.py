"""PPO (Proximal Policy Optimization) implementation."""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple
from torch.distributions.categorical import Categorical
from .base import BaseAgent, layer_init, safe_torch_load


class PPOAgent(BaseAgent):
    """PPO Agent for Atari games."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 2.5e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_coef: float = 0.1,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        device: str = "cuda",
    ):
        super().__init__(state_dim, action_dim, device)
        
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(state_dim, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        ).to(self.device)
        
        self.actor = layer_init(nn.Linear(512, action_dim), std=0.01).to(self.device)
        self.critic = layer_init(nn.Linear(512, 1), std=1).to(self.device)
        
        self.optimizer = optim.Adam(
            list(self.network.parameters()) + list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=lr,
            eps=1e-5
        )
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
    
    def save(self, path: str):
        """Save the agent's networks."""
        torch.save({
            'network': self.network.state_dict(),
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
        }, path)
    
    def load(self, path: str):
        """Load the agent's networks."""
        checkpoint = safe_torch_load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network'])
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
    
    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """Get value estimate."""
        return self.critic(self.network(x / 255.0))
    
    def get_action_and_value(self, x: torch.Tensor, action: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action, log_prob, entropy, and value."""
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)
    
    def select_action(self, state: torch.Tensor) -> int:
        """Select action using policy network."""
        with torch.no_grad():
            state = state.unsqueeze(0).to(self.device)
            action, _, _, _ = self.get_action_and_value(state)
            return action.item()
    
    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        next_value: torch.Tensor,
    ):
        """Compute GAE advantages and returns."""
        advantages = torch.zeros_like(rewards)
        lastgaelam = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                nextnonterminal = 1.0 - dones[t]
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]
            delta = rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
        
        returns = advantages + values
        return advantages, returns
    
    def update(
        self,
        rollout_data: Dict[str, torch.Tensor],
        next_value: torch.Tensor,
        update_epochs: int = 4,
        minibatch_size: int = 32,
    ) -> Dict[str, float]:
        """Update PPO with rollout data."""
        states = rollout_data["states"].to(self.device)
        actions = rollout_data["actions"].to(self.device)
        old_log_probs = rollout_data["log_probs"].to(self.device)
        rewards = rollout_data["rewards"].to(self.device)
        dones = rollout_data["dones"].to(self.device)
        old_values = rollout_data["values"].to(self.device)
        
        with torch.no_grad():
            advantages, returns = self.compute_gae(rewards, old_values, dones, next_value)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        b_obs = states
        b_actions = actions.flatten()
        b_log_probs = old_log_probs.flatten()
        b_advantages = advantages.flatten()
        b_returns = returns.flatten()
        b_values = old_values.flatten()
        
        clipfracs = []
        b_inds = np.arange(len(b_obs))
        
        for epoch in range(update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, len(b_obs), minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]
                
                _, new_log_probs, entropy, new_values = self.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                new_log_probs = new_log_probs.flatten()
                new_values = new_values.flatten()
                
                logratio = new_log_probs - b_log_probs[mb_inds]
                ratio = logratio.exp()
                
                mb_advantages = b_advantages[mb_inds]
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                v_loss_unclipped = (new_values - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(
                    new_values - b_values[mb_inds],
                    -self.clip_coef,
                    self.clip_coef,
                )
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
                
                entropy_loss = entropy.mean()
                loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.network.parameters()) + list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.max_grad_norm
                )
                self.optimizer.step()
                
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > self.clip_coef).float().mean().item())
        
        return {
            "policy_loss": pg_loss.item(),
            "value_loss": v_loss.item(),
            "entropy": entropy_loss.item(),
            "approx_kl": approx_kl.item(),
            "clipfrac": np.mean(clipfracs),
        }


class MultiHeadPPOAgent(BaseAgent):
    """PPO agent with shared AtariBackbone and per-task actor/critic heads.

    Used for continual learning where different games may have different
    action spaces but share the same convolutional backbone.
    """

    def __init__(
        self,
        state_dim: int,
        lr: float = 2.5e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_coef: float = 0.1,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        device: str = "cuda",
    ):
        super().__init__(state_dim, action_dim=1, device=device)

        # Shared backbone (same as PPOAgent's network)
        self.backbone = nn.Sequential(
            layer_init(nn.Conv2d(state_dim, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        ).to(self.device)
        self.network = self.backbone

        # Per-task actor and critic heads
        self.actors = nn.ModuleDict()
        self.critics = nn.ModuleDict()

        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.optimizer = None
        self.current_task = None

    def _rebuild_optimizer(self):
        """Recreate optimizer over backbone and all heads."""
        params = list(self.backbone.parameters())
        for actor in self.actors.values():
            params += list(actor.parameters())
        for critic in self.critics.values():
            params += list(critic.parameters())
        self.optimizer = optim.Adam(params, lr=self.lr, eps=1e-5)

    def register_task(self, task_id: str, action_dim: int):
        """Create new actor and critic heads for a task if they don't exist."""
        if task_id in self.actors:
            return

        actor = layer_init(nn.Linear(512, action_dim), std=0.01).to(self.device)
        critic = layer_init(nn.Linear(512, 1), std=1).to(self.device)

        self.actors[task_id] = actor
        self.critics[task_id] = critic
        self._rebuild_optimizer()

    def set_task(self, task_id: str):
        """Select which task/heads to use for subsequent calls."""
        if task_id not in self.actors:
            raise ValueError(f"Task '{task_id}' not registered in MultiHeadPPOAgent.")
        self.current_task = task_id
        self.action_dim = self.actors[task_id].out_features

    def _current_heads(self):
        """Get current task's actor and critic heads."""
        if self.current_task is None:
            raise RuntimeError("Current task is not set for MultiHeadPPOAgent.")
        return self.actors[self.current_task], self.critics[self.current_task]

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """Get value estimate for current task."""
        if self.current_task is None:
            raise RuntimeError("Current task is not set before get_value().")
        hidden = self.backbone(x / 255.0)
        _, critic = self._current_heads()
        return critic(hidden)

    def get_action_and_value(
        self, x: torch.Tensor, action: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action, log_prob, entropy, and value for current task."""
        if self.current_task is None:
            raise RuntimeError("Current task is not set before get_action_and_value().")
        hidden = self.backbone(x / 255.0)
        actor, critic = self._current_heads()
        logits = actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), critic(hidden)

    def select_action(self, state: torch.Tensor) -> int:
        """Select action using policy network for current task."""
        if self.current_task is None:
            raise RuntimeError("Current task is not set before select_action().")
        with torch.no_grad():
            state = state.unsqueeze(0).to(self.device)
            action, _, _, _ = self.get_action_and_value(state)
            return action.item()

    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        next_value: torch.Tensor,
    ):
        """Compute GAE advantages and returns."""
        advantages = torch.zeros_like(rewards)
        lastgaelam = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                nextnonterminal = 1.0 - dones[t]
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]
            delta = rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = (
                delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
            )

        returns = advantages + values
        return advantages, returns

    def update(
        self,
        rollout_data: Dict[str, torch.Tensor],
        next_value: torch.Tensor,
        update_epochs: int = 4,
        minibatch_size: int = 32,
    ) -> Dict[str, float]:
        """Update PPO with rollout data using shared backbone and task-specific heads."""
        if self.current_task is None:
            raise RuntimeError("Current task is not set before update().")
        if self.optimizer is None:
            raise RuntimeError(
                "Optimizer has not been initialized; call register_task() first."
            )

        states = rollout_data["states"].to(self.device)
        actions = rollout_data["actions"].to(self.device)
        old_log_probs = rollout_data["log_probs"].to(self.device)
        rewards = rollout_data["rewards"].to(self.device)
        dones = rollout_data["dones"].to(self.device)
        old_values = rollout_data["values"].to(self.device)

        with torch.no_grad():
            advantages, returns = self.compute_gae(rewards, old_values, dones, next_value)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        b_obs = states
        b_actions = actions.flatten()
        b_log_probs = old_log_probs.flatten()
        b_advantages = advantages.flatten()
        b_returns = returns.flatten()
        b_values = old_values.flatten()

        clipfracs = []
        b_inds = np.arange(len(b_obs))

        for epoch in range(update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, len(b_obs), minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, new_log_probs, entropy, new_values = self.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                new_log_probs = new_log_probs.flatten()
                new_values = new_values.flatten()

                logratio = new_log_probs - b_log_probs[mb_inds]
                ratio = logratio.exp()

                mb_advantages = b_advantages[mb_inds]
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - self.clip_coef, 1 + self.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss_unclipped = (new_values - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(
                    new_values - b_values[mb_inds],
                    -self.clip_coef,
                    self.clip_coef,
                )
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef

                self.optimizer.zero_grad()
                loss.backward()
                # Collect all parameters for gradient clipping
                all_params = list(self.backbone.parameters())
                for actor in self.actors.values():
                    all_params.extend(list(actor.parameters()))
                for critic in self.critics.values():
                    all_params.extend(list(critic.parameters()))
                nn.utils.clip_grad_norm_(all_params, self.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(
                        ((ratio - 1.0).abs() > self.clip_coef).float().mean().item()
                    )

        return {
            "policy_loss": pg_loss.item(),
            "value_loss": v_loss.item(),
            "entropy": entropy_loss.item(),
            "approx_kl": approx_kl.item(),
            "clipfrac": np.mean(clipfracs),
        }

    def save(self, path: str):
        """Save the agent's networks."""
        torch.save(
            {
                "backbone": self.backbone.state_dict(),
                "actors": {k: v.state_dict() for k, v in self.actors.items()},
                "critics": {k: v.state_dict() for k, v in self.critics.items()},
                "task_action_dims": {k: v.out_features for k, v in self.actors.items()},
            },
            path,
        )

    def load(self, path: str):
        """Load the agent's networks.
        
        Note: Tasks should be registered before loading. If a task in the checkpoint
        is not registered, it will be skipped.
        """
        checkpoint = safe_torch_load(path, map_location=self.device)
        self.backbone.load_state_dict(checkpoint["backbone"])
        
        # Load actors
        for task_id, state_dict in checkpoint["actors"].items():
            if task_id not in self.actors:
                # Try to auto-register if we have action_dim info
                if "task_action_dims" in checkpoint and task_id in checkpoint["task_action_dims"]:
                    action_dim = checkpoint["task_action_dims"][task_id]
                    self.register_task(task_id, action_dim)
                else:
                    # Skip if we can't determine action_dim
                    continue
            self.actors[task_id].load_state_dict(state_dict)
        
        # Load critics
        for task_id, state_dict in checkpoint["critics"].items():
            if task_id not in self.critics:
                # Should have been registered above, but check anyway
                if "task_action_dims" in checkpoint and task_id in checkpoint["task_action_dims"]:
                    action_dim = checkpoint["task_action_dims"][task_id]
                    if task_id not in self.actors:
                        self.register_task(task_id, action_dim)
                else:
                    continue
            self.critics[task_id].load_state_dict(state_dict)
        
        # Rebuild optimizer after loading
        self._rebuild_optimizer()

