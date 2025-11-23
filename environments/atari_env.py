"""Atari environment wrapper."""
import gymnasium as gym
import numpy as np
import torch
from typing import Tuple

# Register Atari environments (only once)
_ATARI_ENVS_REGISTERED = False

def _register_atari_envs():
    """Register Atari environments (only once)."""
    global _ATARI_ENVS_REGISTERED

    if _ATARI_ENVS_REGISTERED:
        return

    try:
        # Check if Atari environments are already registered
        if "ALE/Pong-v5" not in gym.registry:
            from ale_py import register_v5_envs
            # Suppress the override warnings from ale_py
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*Overriding environment.*")
                register_v5_envs()
        _ATARI_ENVS_REGISTERED = True
    except ImportError:
        # ale_py not installed, will fail when trying to create environment
        pass
    except Exception as e:
        # Other errors during registration
        import warnings
        warnings.warn(f"Failed to register Atari environments: {e}")


class AtariEnv:
    """Wrapper for Atari environments."""

    def __init__(self, game_name: str, frame_stack: int = 4, render_mode: str = None):
        """
        Initialize Atari environment.

        Args:
            game_name: Name of the Atari game (e.g., 'Pong-v5' or 'ALE/Pong-v5')
            frame_stack: Number of frames to stack
            render_mode: Render mode ('rgb_array' for visualization, None for training)
        """
        # Register Atari environments (only once)
        _register_atari_envs()

        self.game_name = game_name
        self.frame_stack = frame_stack

        # Convert game name format if needed
        if not game_name.startswith("ALE/"):
            game_name = f"ALE/{game_name}"

        # Create environment with render mode for visualization
        self.env = gym.make(game_name, render_mode=render_mode)
        self.action_space = self.env.action_space.n
        
        # Frame stacking
        self.frames = None
        self.reset()
    
    def reset(self) -> torch.Tensor:
        """Reset environment and return initial state."""
        obs, _ = self.env.reset()
        obs = self._preprocess(obs)
        
        # Initialize frame stack
        self.frames = np.zeros((self.frame_stack, 84, 84), dtype=np.uint8)
        for i in range(self.frame_stack):
            self.frames[i] = obs
        
        return self._get_state()
    
    def step(self, action: int) -> Tuple[torch.Tensor, float, bool]:
        """Take a step in the environment.

        Args:
            action: Discrete action index. If it falls outside the valid
                range ``[0, self.env.action_space.n - 1]``, it will be
                clipped. This allows sharing a single agent with a larger
                action space across multiple games that may have fewer
                primitive actions.

        Returns:
            state: Current state
            reward: Reward
            done: Whether episode is done (terminated or truncated)
        """
        # Clip action to the valid range of the underlying environment
        valid_action = int(np.clip(action, 0, self.env.action_space.n - 1))

        obs, reward, terminated, truncated, _ = self.env.step(valid_action)
        obs = self._preprocess(obs)

        # Update frame stack
        self.frames = np.roll(self.frames, shift=1, axis=0)
        self.frames[0] = obs

        done = terminated or truncated

        return self._get_state(), float(reward), done

    def _preprocess(self, obs: np.ndarray) -> np.ndarray:
        """Preprocess observation."""
        # Convert to grayscale
        obs = np.dot(obs[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
        # Resize to 84x84
        obs = self._resize(obs, (84, 84))
        return obs
    
    def _resize(self, img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        """Resize image."""
        from PIL import Image
        return np.array(Image.fromarray(img).resize(size, Image.BILINEAR))
    
    def _get_state(self) -> torch.Tensor:
        """Get current state as tensor."""
        return torch.from_numpy(self.frames.astype(np.float32) / 255.0)
    
    def close(self):
        """Close environment."""
        self.env.close()

