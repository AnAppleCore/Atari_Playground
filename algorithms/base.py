"""Base Agent class for RL algorithms."""
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any


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

        # Handle device selection with warning if GPU not available
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
            self.network.load_state_dict(torch.load(path, map_location=self.device))
    
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


class SimpleNet(nn.Module):
    """Simple neural network for Atari."""

    def __init__(self, input_channels: int, action_dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        # Calculate flattened size
        self.fc_input_size = 64 * 7 * 7

        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_size, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class AtariBackbone(nn.Module):
    """Shared convolutional backbone for Atari agents.

    Produces a feature vector (default 512-dim) from stacked frames.
    """

    def __init__(self, input_channels: int, feature_dim: int = 512):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        # Flattened size after conv layers for 84x84 input
        self.fc_input_size = 64 * 7 * 7

        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_size, feature_dim),
            nn.ReLU(),
        )

        self.feature_dim = feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

