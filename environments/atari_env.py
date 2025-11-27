"""Atari environment wrapper."""
import gymnasium as gym
import numpy as np
import torch
from typing import Tuple
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

_ATARI_ENVS_REGISTERED = False

def _register_atari_envs():
    """Register Atari environments (only once)."""
    global _ATARI_ENVS_REGISTERED

    if _ATARI_ENVS_REGISTERED:
        return

    try:
        if "ALE/Pong-v5" not in gym.registry:
            from ale_py import register_v5_envs
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*Overriding environment.*")
                register_v5_envs()
        _ATARI_ENVS_REGISTERED = True
    except ImportError:
        pass
    except Exception as e:
        import warnings
        warnings.warn(f"Failed to register Atari environments: {e}")


class AtariEnv:
    """Wrapper for Atari environments with cleanrl-style preprocessing."""

    def __init__(self, game_name: str, frame_stack: int = 4, render_mode: str = None, seed: int = None, use_skip: bool = True):
        """
        Initialize Atari environment with cleanrl-style wrappers.

        Args:
            game_name: Name of the Atari game (e.g., 'Pong-v5' or 'ALE/Pong-v5')
            frame_stack: Number of frames to stack (should be 4)
            render_mode: Render mode ('rgb_array' for visualization, None for training)
            seed: Random seed for environment
            use_skip: Whether to use frame skipping (default: True for training, False for testing)
        """
        _register_atari_envs()

        self.game_name = game_name
        self.frame_stack = frame_stack

        if not game_name.startswith("ALE/"):
            game_name = f"ALE/{game_name}"

        if render_mode:
            self.env = gym.make(game_name, render_mode=render_mode)
        else:
            self.env = gym.make(game_name)
        
        self.env = gym.wrappers.RecordEpisodeStatistics(self.env)
        self.env = NoopResetEnv(self.env, noop_max=30)
        if use_skip:
            self.env = MaxAndSkipEnv(self.env, skip=4)
        self.env = EpisodicLifeEnv(self.env)
        if "FIRE" in self.env.unwrapped.get_action_meanings():
            self.env = FireResetEnv(self.env)
        self.env = ClipRewardEnv(self.env)
        self.env = gym.wrappers.ResizeObservation(self.env, (84, 84))
        self.env = gym.wrappers.GrayscaleObservation(self.env)
        self.env = gym.wrappers.FrameStackObservation(self.env, frame_stack)
        
        if seed is not None:
            self.env.action_space.seed(seed)
        
        self.action_space = self.env.action_space.n
    
    def reset(self) -> torch.Tensor:
        """Reset environment and return initial state."""
        obs, _ = self.env.reset()
        obs = torch.from_numpy(obs).float()
        return obs
    
    def step(self, action: int) -> Tuple[torch.Tensor, float, bool]:
        """Take a step in the environment.

        Args:
            action: Discrete action index (clipped to valid range if needed)

        Returns:
            state: Current state
            reward: Reward
            done: Whether episode is done
        """
        valid_action = int(np.clip(action, 0, self.env.action_space.n - 1))
        obs, reward, terminated, truncated, _ = self.env.step(valid_action)
        obs = torch.from_numpy(obs).float()

        done = terminated or truncated

        return obs, float(reward), done
    
    def close(self):
        """Close environment."""
        self.env.close()

