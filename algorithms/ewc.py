"""Elastic Weight Consolidation (EWC) for continual learning."""
import torch
import torch.nn as nn
from typing import Dict, Optional, List
from .base import BaseAgent


class EWCWrapper:
    """Wrapper to add EWC regularization to any agent.
    
    Supports multi-task EWC by storing weights and Fisher Information
    for each previous task.
    """

    def __init__(self, agent: BaseAgent, ewc_lambda: float = 0.4):
        """
        Initialize EWC wrapper.

        Args:
            agent: The base agent to wrap
            ewc_lambda: Strength of EWC regularization
        """
        self.agent = agent
        self.ewc_lambda = ewc_lambda
        # Store weights and Fisher Information for each task
        self.task_weights: Dict[int, Dict[str, torch.Tensor]] = {}
        self.task_fisher: Dict[int, Dict[str, torch.Tensor]] = {}
        self.current_task_id = 0
        self.is_first_task = True

    def _collect_all_params(self) -> Dict[str, torch.Tensor]:
        """Collect all trainable parameters from the agent."""
        all_params = {}
        
        # For DQN/MultiHeadDQN: network or backbone
        if hasattr(self.agent, 'network') and self.agent.network is not None:
            for name, param in self.agent.network.named_parameters():
                if param.requires_grad:
                    all_params[f"network.{name}"] = param
        
        # For PPO: network/backbone, actor, and critic
        if hasattr(self.agent, 'backbone') and self.agent.backbone is not None:
            for name, param in self.agent.backbone.named_parameters():
                if param.requires_grad:
                    all_params[f"backbone.{name}"] = param
        
        if hasattr(self.agent, 'actor') and self.agent.actor is not None:
            for name, param in self.agent.actor.named_parameters():
                if param.requires_grad:
                    all_params[f"actor.{name}"] = param
        
        if hasattr(self.agent, 'critic') and self.agent.critic is not None:
            for name, param in self.agent.critic.named_parameters():
                if param.requires_grad:
                    all_params[f"critic.{name}"] = param
        
        # For MultiHeadDQN: also include heads
        if hasattr(self.agent, 'heads'):
            for head_name, head in self.agent.heads.items():
                for name, param in head.named_parameters():
                    if param.requires_grad:
                        all_params[f"heads.{head_name}.{name}"] = param
        
        # For MultiHeadPPO: include actors and critics
        if hasattr(self.agent, 'actors'):
            for task_id, actor in self.agent.actors.items():
                for name, param in actor.named_parameters():
                    if param.requires_grad:
                        all_params[f"actors.{task_id}.{name}"] = param
        
        if hasattr(self.agent, 'critics'):
            for task_id, critic in self.agent.critics.items():
                for name, param in critic.named_parameters():
                    if param.requires_grad:
                        all_params[f"critics.{task_id}.{name}"] = param
        
        return all_params

    def compute_fisher_information(
        self, 
        batch: Dict[str, torch.Tensor],
        num_samples: int = 100
    ) -> Dict[str, torch.Tensor]:
        """
        Compute Fisher Information Matrix by estimating E[(∂L/∂θ)²].
        
        Args:
            batch: Training batch
            num_samples: Number of samples to use for estimation
            
        Returns:
            Dictionary mapping parameter names to Fisher Information values
        """
        all_params = self._collect_all_params()
        fisher = {name: torch.zeros_like(param) for name, param in all_params.items()}
        
        # Get optimizer to compute gradients
        if not hasattr(self.agent, 'optimizer') or self.agent.optimizer is None:
            return fisher
        
        # Sample a subset of the batch for Fisher estimation
        batch_size = min(num_samples, len(batch.get("states", batch.get("rewards", torch.tensor([])))))
        if batch_size == 0:
            return fisher
        
        # For DQN-style update
        if "states" in batch and "actions" in batch:
            indices = torch.randperm(len(batch["states"]))[:batch_size]
            sample_batch = {
                k: v[indices] if isinstance(v, torch.Tensor) and len(v) == len(batch["states"]) else v
                for k, v in batch.items()
            }
        else:
            sample_batch = batch
        
        # Compute gradients for each sample and accumulate Fisher Information
        self.agent.optimizer.zero_grad()
        
        # Compute loss (this depends on agent type)
        if hasattr(self.agent, 'update'):
            # We need to compute the loss without updating
            # For DQN: compute Q-loss
            if hasattr(self.agent, 'backbone') or (hasattr(self.agent, 'network') and not hasattr(self.agent, 'actor')):
                # DQN-style
                states = sample_batch["states"].to(self.agent.device)
                actions = sample_batch["actions"].to(self.agent.device)
                rewards = sample_batch["rewards"].to(self.agent.device)
                next_states = sample_batch["next_states"].to(self.agent.device)
                dones = sample_batch["dones"].to(self.agent.device)
                
                if hasattr(self.agent, 'backbone') and hasattr(self.agent, 'heads'):
                    # MultiHeadDQN
                    features = self.agent.backbone(states)
                    head, _ = self.agent._current_heads()
                    q_values = head(features)
                    q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
                    
                    with torch.no_grad():
                        next_features = self.agent.target_backbone(next_states)
                        next_q_values = self.agent.target_heads[self.agent.current_task](next_features).max(dim=1)[0]
                        target_q_values = rewards + self.agent.gamma * next_q_values * (1 - dones)
                    
                    loss = self.agent.loss_fn(q_values, target_q_values)
                else:
                    # Regular DQN
                    q_values = self.agent.network(states)
                    q_values_selected = q_values.gather(1, actions.unsqueeze(1)).squeeze()
                    
                    with torch.no_grad():
                        target_max, _ = self.agent.target_network(next_states).max(dim=1)
                        td_target = rewards.flatten() + self.agent.gamma * target_max * (1 - dones.flatten())
                    
                    loss = nn.functional.mse_loss(td_target, q_values_selected)
            else:
                # PPO-style: use a simplified loss computation
                # For Fisher estimation, we use policy loss
                states = sample_batch["states"].to(self.agent.device)
                actions = sample_batch["actions"].to(self.agent.device)
                old_log_probs = sample_batch["log_probs"].to(self.agent.device)
                
                if hasattr(self.agent, 'backbone') and hasattr(self.agent, 'actors'):
                    # MultiHeadPPO
                    hidden = self.agent.backbone(states / 255.0)
                    actor, _ = self.agent._current_heads()
                    logits = actor(hidden)
                    probs = torch.distributions.Categorical(logits=logits)
                    new_log_probs = probs.log_prob(actions.flatten())
                    loss = -new_log_probs.mean()  # Negative log likelihood
                else:
                    # Regular PPO
                    hidden = self.agent.network(states / 255.0)
                    logits = self.agent.actor(hidden)
                    probs = torch.distributions.Categorical(logits=logits)
                    new_log_probs = probs.log_prob(actions.flatten())
                    loss = -new_log_probs.mean()
        else:
            return fisher
        
        # Compute gradients
        loss.backward()
        
        # Accumulate Fisher Information: F_i = E[(∂L/∂θ_i)²]
        for name, param in all_params.items():
            if param.grad is not None:
                fisher[name] += (param.grad ** 2) / batch_size
        
        self.agent.optimizer.zero_grad()
        
        return fisher

    def consolidate_weights(self, batch: Optional[Dict[str, torch.Tensor]] = None):
        """
        Consolidate weights after learning a task.
        
        Args:
            batch: Optional batch for computing Fisher Information.
                  If None, uses a simple approximation.
        """
        all_params = self._collect_all_params()
        
        if not all_params:
            return
        
        # Save current weights
        task_weights = {
            name: param.clone().detach()
            for name, param in all_params.items()
        }
        
        # Compute Fisher Information Matrix
        if batch is not None:
            fisher = self.compute_fisher_information(batch)
        else:
            # Fallback: use a simple approximation (diagonal with small values)
            fisher = {
                name: torch.ones_like(param) * 0.1
                for name, param in all_params.items()
            }
        
        # Store for this task
        self.task_weights[self.current_task_id] = task_weights
        self.task_fisher[self.current_task_id] = fisher
        
        # Move to next task
        self.current_task_id += 1
        self.is_first_task = False

    def compute_ewc_loss(self) -> torch.Tensor:
        """Compute EWC regularization loss across all previous tasks."""
        if self.is_first_task or len(self.task_weights) == 0:
            return torch.tensor(0.0, device=self.agent.device)

        current_params = self._collect_all_params()
        ewc_loss = torch.tensor(0.0, device=self.agent.device)

        # Sum EWC loss over all previous tasks
        for task_id in range(self.current_task_id):
            if task_id not in self.task_weights or task_id not in self.task_fisher:
                continue
            
            prev_weights = self.task_weights[task_id]
            fisher = self.task_fisher[task_id]
            
            for name, param in current_params.items():
                if name in prev_weights and name in fisher:
                    fisher_val = fisher[name]
                    prev_weight = prev_weights[name]
                    ewc_loss += (fisher_val * (param - prev_weight) ** 2).sum()

        return self.ewc_lambda * ewc_loss

    def select_action(self, state: torch.Tensor) -> int:
        """Delegate to wrapped agent."""
        return self.agent.select_action(state)

    def get_action_and_value(self, state: torch.Tensor, action: torch.Tensor = None):
        """Delegate to wrapped agent (for PPO)."""
        if hasattr(self.agent, "get_action_and_value"):
            return self.agent.get_action_and_value(state, action)
        raise AttributeError(f"Agent {type(self.agent)} does not support get_action_and_value")

    def get_value(self, state: torch.Tensor):
        """Delegate to wrapped agent (for PPO)."""
        if hasattr(self.agent, "get_value"):
            return self.agent.get_value(state)
        raise AttributeError(f"Agent {type(self.agent)} does not support get_value")

    @property
    def device(self):
        """Get device from wrapped agent."""
        return self.agent.device

    def register_task(self, task_id: str, action_dim: int):
        """Register a new task on the wrapped agent if supported."""
        if hasattr(self.agent, "register_task"):
            self.agent.register_task(task_id, action_dim)

    def set_task(self, task_id: str):
        """Set the current task on the wrapped agent if supported."""
        if hasattr(self.agent, "set_task"):
            self.agent.set_task(task_id)

    def update(self, batch: Dict[str, torch.Tensor], *args, **kwargs) -> Dict[str, float]:
        """
        Update with EWC regularization.
        
        Supports both DQN-style (single batch arg) and PPO-style 
        (batch + next_value + update_epochs + minibatch_size) updates.
        
        The EWC loss is integrated into the main loss by modifying
        the agent's update method to include EWC regularization.
        """
        # Compute EWC loss before the update
        ewc_loss = self.compute_ewc_loss()
        
        # Detect agent type by checking batch structure
        # DQN batches have 'next_states', PPO batches have 'log_probs' and 'values'
        is_dqn = "next_states" in batch
        is_ppo = "log_probs" in batch and "values" in batch
        
        if is_dqn:
            # DQN-style: modify update to include EWC loss
            metrics = self._update_dqn_with_ewc(batch, ewc_loss)
        elif is_ppo:
            # PPO-style: modify update to include EWC loss
            metrics = self._update_ppo_with_ewc(batch, ewc_loss, *args, **kwargs)
        else:
            # Fallback: try to detect by agent attributes
            if hasattr(self.agent, 'backbone') or (hasattr(self.agent, 'network') and not hasattr(self.agent, 'actor')):
                metrics = self._update_dqn_with_ewc(batch, ewc_loss)
            else:
                metrics = self._update_ppo_with_ewc(batch, ewc_loss, *args, **kwargs)
        
        if ewc_loss.item() > 0:
            metrics["ewc_loss"] = ewc_loss.item()
        
        return metrics

    def _update_dqn_with_ewc(self, batch: Dict[str, torch.Tensor], ewc_loss: torch.Tensor) -> Dict[str, float]:
        """Update DQN agent with EWC loss integrated."""
        # For DQN, we need to integrate EWC loss into the main loss before backward
        states = batch["states"].to(self.agent.device)
        actions = batch["actions"].to(self.agent.device)
        rewards = batch["rewards"].to(self.agent.device)
        next_states = batch["next_states"].to(self.agent.device)
        dones = batch["dones"].to(self.agent.device)
        
        if hasattr(self.agent, 'backbone') and hasattr(self.agent, 'heads'):
            # MultiHeadDQN
            head, target_head = self.agent._current_heads()
            
            features = self.agent.backbone(states)
            q_values = head(features)
            q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
            
            with torch.no_grad():
                next_features = self.agent.target_backbone(next_states)
                next_q_values = target_head(next_features).max(dim=1)[0]
                target_q_values = rewards + self.agent.gamma * next_q_values * (1 - dones)
            
            loss = self.agent.loss_fn(q_values, target_q_values)
            
            # Add EWC loss
            total_loss = loss + ewc_loss
            
            self.agent.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.agent.backbone.parameters(), 1.0)
            for h in self.agent.heads.values():
                torch.nn.utils.clip_grad_norm_(h.parameters(), 1.0)
            self.agent.optimizer.step()
            
            self.agent.update_count += 1
            if self.agent.update_count % self.agent.target_update_freq == 0:
                self.agent.target_backbone.load_state_dict(self.agent.backbone.state_dict())
                for name, src_head in self.agent.heads.items():
                    self.agent.target_heads[name].load_state_dict(src_head.state_dict())
            
            self.agent.epsilon = max(self.agent.epsilon_min, self.agent.epsilon * self.agent.epsilon_decay)
            
            return {"loss": loss.item(), "epsilon": self.agent.epsilon}
        else:
            # Regular DQN
            q_values = self.agent.network(states)
            q_values_selected = q_values.gather(1, actions.unsqueeze(1)).squeeze()
            
            with torch.no_grad():
                target_max, _ = self.agent.target_network(next_states).max(dim=1)
                td_target = rewards.flatten() + self.agent.gamma * target_max * (1 - dones.flatten())
            
            loss = nn.functional.mse_loss(td_target, q_values_selected)
            
            # Add EWC loss
            total_loss = loss + ewc_loss
            
            self.agent.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.agent.network.parameters(), 10.0)
            self.agent.optimizer.step()
            
            self.agent.update_count += 1
            if self.agent.update_count % self.agent.target_update_freq == 0:
                for target_param, q_network_param in zip(self.agent.target_network.parameters(), self.agent.network.parameters()):
                    target_param.data.copy_(
                        self.agent.tau * q_network_param.data + (1.0 - self.agent.tau) * target_param.data
                    )
            
            epsilon = self.agent.epsilon_start  # Use linear schedule if available
            if hasattr(self.agent, 'global_step'):
                from .dqn import linear_schedule
                epsilon = linear_schedule(
                    self.agent.epsilon_start,
                    self.agent.epsilon_end,
                    int(self.agent.epsilon_fraction * self.agent.total_timesteps),
                    self.agent.global_step
                )
            
            avg_q = 0.0
            if hasattr(self.agent, 'q_value_history') and self.agent.q_value_history:
                avg_q = sum(self.agent.q_value_history) / len(self.agent.q_value_history)
            
            return {"loss": loss.item(), "epsilon": epsilon, "q_value": avg_q}

    def _update_ppo_with_ewc(
        self, 
        batch: Dict[str, torch.Tensor], 
        ewc_loss: torch.Tensor,
        next_value: torch.Tensor,
        update_epochs: int = 4,
        minibatch_size: int = 32
    ) -> Dict[str, float]:
        """Update PPO agent with EWC loss integrated."""
        # For PPO, we need to modify the update loop to include EWC loss
        # This is more complex because PPO has multiple update epochs
        
        states = batch["states"].to(self.agent.device)
        actions = batch["actions"].to(self.agent.device)
        old_log_probs = batch["log_probs"].to(self.agent.device)
        rewards = batch["rewards"].to(self.agent.device)
        dones = batch["dones"].to(self.agent.device)
        old_values = batch["values"].to(self.agent.device)
        
        with torch.no_grad():
            advantages, returns = self.agent.compute_gae(rewards, old_values, dones, next_value)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        b_obs = states
        b_actions = actions.flatten()
        b_log_probs = old_log_probs.flatten()
        b_advantages = advantages.flatten()
        b_returns = returns.flatten()
        b_values = old_values.flatten()
        
        clipfracs = []
        b_inds = torch.arange(len(b_obs))
        
        for epoch in range(update_epochs):
            b_inds = b_inds[torch.randperm(len(b_inds))]
            for start in range(0, len(b_obs), minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]
                
                _, new_log_probs, entropy, new_values = self.agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                new_log_probs = new_log_probs.flatten()
                new_values = new_values.flatten()
                
                logratio = new_log_probs - b_log_probs[mb_inds]
                ratio = logratio.exp()
                
                mb_advantages = b_advantages[mb_inds]
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - self.agent.clip_coef, 1 + self.agent.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                v_loss_unclipped = (new_values - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(
                    new_values - b_values[mb_inds],
                    -self.agent.clip_coef,
                    self.agent.clip_coef,
                )
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
                
                entropy_loss = entropy.mean()
                
                # Compute EWC loss for this iteration (need to recompute as parameters change)
                current_ewc_loss = self.compute_ewc_loss()
                
                # Add EWC loss to the main loss
                # Normalize by number of update epochs and minibatches
                num_minibatches = (len(b_obs) + minibatch_size - 1) // minibatch_size
                total_loss = (
                    pg_loss 
                    - self.agent.ent_coef * entropy_loss 
                    + v_loss * self.agent.vf_coef
                    + current_ewc_loss / (update_epochs * num_minibatches)
                )
                
                self.agent.optimizer.zero_grad()
                total_loss.backward()
                # Collect all parameters for gradient clipping
                all_params = []
                if hasattr(self.agent, 'backbone'):
                    all_params.extend(list(self.agent.backbone.parameters()))
                elif hasattr(self.agent, 'network'):
                    all_params.extend(list(self.agent.network.parameters()))
                if hasattr(self.agent, 'actors'):
                    for actor in self.agent.actors.values():
                        all_params.extend(list(actor.parameters()))
                if hasattr(self.agent, 'critics'):
                    for critic in self.agent.critics.values():
                        all_params.extend(list(critic.parameters()))
                if hasattr(self.agent, 'actor') and not hasattr(self.agent, 'actors'):
                    all_params.extend(list(self.agent.actor.parameters()))
                if hasattr(self.agent, 'critic') and not hasattr(self.agent, 'critics'):
                    all_params.extend(list(self.agent.critic.parameters()))
                nn.utils.clip_grad_norm_(all_params, self.agent.max_grad_norm)
                self.agent.optimizer.step()
                
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(
                        ((ratio - 1.0).abs() > self.agent.clip_coef).float().mean().item()
                    )
        
        return {
            "policy_loss": pg_loss.item(),
            "value_loss": v_loss.item(),
            "entropy": entropy_loss.item(),
            "approx_kl": approx_kl.item(),
            "clipfrac": sum(clipfracs) / len(clipfracs) if clipfracs else 0.0,
        }

    def save(self, path: str):
        """Save the wrapped agent."""
        self.agent.save(path)

    def load(self, path: str):
        """Load the wrapped agent."""
        self.agent.load(path)
