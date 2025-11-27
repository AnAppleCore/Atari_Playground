"""Base Agent class for RL algorithms."""
import inspect
import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any


_HAS_WEIGHTS_ONLY_ARG = "weights_only" in inspect.signature(torch.load).parameters


def safe_torch_load(path: str, map_location=None):
    """Load checkpoints while explicitly disabling weights_only when available."""
    load_kwargs = {}
    if map_location is not None:
        load_kwargs["map_location"] = map_location
    if _HAS_WEIGHTS_ONLY_ARG:
        load_kwargs["weights_only"] = False
    return torch.load(path, **load_kwargs)


class BaseAgent(ABC):
    """Abstract base class for RL agents."""
    
    def __init__(self, state_dim: int, action_dim: int, device: str = "cuda"):
        """
        Initialize the base agent.

        Args:
            state_dim: Dimension of the state space
            action_dim: Number of actions
            device: Device to use (cuda or cpu)
        """
        self.state_dim = state_dim
        self.action_dim = action_dim

        if device == "cuda" and not torch.cuda.is_available():
            import warnings
            warnings.warn("CUDA requested but not available, falling back to CPU")
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.network = None
        
    @abstractmethod
    def select_action(self, state: torch.Tensor) -> int:
        """Select an action given a state."""
        pass
    
    @abstractmethod
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Update the agent with a batch of experiences."""
        pass
    
    def save(self, path: str):
        """Save the agent's network."""
        if self.network is not None:
            torch.save(self.network.state_dict(), path)
    
    def load(self, path: str):
        """Load the agent's network."""
        if self.network is not None:
            state_dict = safe_torch_load(path, map_location=self.device)
            self.network.load_state_dict(state_dict)
    
    def get_weights(self) -> Dict[str, torch.Tensor]:
        """Get network weights for EWC."""
        if self.network is None:
            return {}
        return {name: param.clone().detach() for name, param in self.network.named_parameters()}
    
    def set_weights(self, weights: Dict[str, torch.Tensor]):
        """Set network weights."""
        if self.network is None:
            return
        for name, param in self.network.named_parameters():
            if name in weights:
                param.data = weights[name].clone()


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Initialize layer with orthogonal weights."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class SimpleNet(nn.Module):
    """Simple neural network for Atari."""

    def __init__(self, input_channels: int, action_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(input_channels, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x / 255.0)


class AtariBackbone(nn.Module):
    """Shared convolutional backbone for Atari agents.

    Produces a feature vector (default 512-dim) from stacked frames.
    """

    def __init__(self, input_channels: int, feature_dim: int = 512):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(input_channels, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, feature_dim),
            nn.ReLU(),
        )

        self.feature_dim = feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x / 255.0)

