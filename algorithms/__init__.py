"""RL Algorithms module."""
from .base import BaseAgent, SimpleNet
from .dqn import DQNAgent, MultiHeadDQNAgent
from .ppo import PPOAgent, MultiHeadPPOAgent
from .ewc import EWCWrapper

__all__ = [
    "BaseAgent",
    "SimpleNet",
    "DQNAgent",
    "MultiHeadDQNAgent",
    "PPOAgent",
    "MultiHeadPPOAgent",
    "EWCWrapper",
]

