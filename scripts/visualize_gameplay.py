"""Visualize trained agent gameplay by loading saved models."""
import sys
import torch
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from environments import AtariEnv
from algorithms import DQNAgent, PPOAgent
from utils import VideoRecorder


def visualize_agent(
    model_path: str,
    game_name: str = "Pong-v5",
    algorithm: str = "dqn",
    num_episodes: int = 3,
    output_path: str = None,
    fps: int = 30,
    video_format: str = "mp4",
    max_steps: int = 10000,
):
    """
    Visualize a trained agent playing a game.
    
    Args:
        model_path: Path to saved model checkpoint
        game_name: Name of the Atari game
        algorithm: Algorithm used (dqn or ppo)
        num_episodes: Number of episodes to record
        output_path: Path to save video (default: outputs/{game}_{algorithm}.{format})
        fps: Frames per second (default: 30)
        video_format: Video format ('mp4' or 'gif')
        max_steps: Maximum steps per episode
    """
    
    # Set default output path
    if output_path is None:
        output_path = f"outputs/{game_name}_{algorithm}_gameplay.{video_format}"
    
    print(f"Loading model from {model_path}...")

    env = AtariEnv(game_name, render_mode="rgb_array", use_skip=False)
    
    # Create agent
    if algorithm == "dqn":
        agent = DQNAgent(state_dim=4, action_dim=env.action_space)
    else:  # ppo
        agent = PPOAgent(state_dim=4, action_dim=env.action_space)
    
    # Load model
    agent.load(model_path)
    # Set to eval mode
    if hasattr(agent, 'network'):
        agent.network.eval()
    if hasattr(agent, 'actor'):
        agent.actor.eval()
    if hasattr(agent, 'critic'):
        agent.critic.eval()
    print(f"✓ Model loaded successfully")
    
    video_recorder = VideoRecorder(output_path, fps=fps)
    print(f"Recording {num_episodes} episodes with {fps} FPS...")
    
    total_reward = 0
    episode_rewards = []
    
    with torch.no_grad():
        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0
            
            for step in range(max_steps):
                frame = None
                try:
                    if hasattr(env.env, 'render'):
                        frame = env.env.render()
                    if frame is None and hasattr(env.env, 'unwrapped'):
                        unwrapped = env.env.unwrapped
                        if hasattr(unwrapped, 'render'):
                            frame = unwrapped.render()
                    if frame is None:
                        if isinstance(state, torch.Tensor):
                            frame = state[0].cpu().numpy()
                        else:
                            frame = state[0] if len(state.shape) > 2 else state
                except Exception:
                    if isinstance(state, torch.Tensor):
                        frame = state[0].cpu().numpy()
                    else:
                        frame = state[0] if len(state.shape) > 2 else state
                
                if frame is not None:
                    video_recorder.add_frame(frame)
                
                # Select action
                action = agent.select_action(state)
                
                # Take step
                next_state, reward, done = env.step(action)
                episode_reward += reward
                state = next_state
                
                if done:
                    break
            
            episode_rewards.append(episode_reward)
            total_reward += episode_reward
            print(f"  Episode {episode + 1}/{num_episodes}: Reward = {episode_reward:.1f}")
    
    # Save video
    video_recorder.save(format=video_format)
    print(f"✓ Video saved to {output_path}")
    
    # Print statistics
    avg_reward = total_reward / num_episodes
    print(f"\nStatistics:")
    print(f"  Average reward: {avg_reward:.1f}")
    print(f"  Min reward: {min(episode_rewards):.1f}")
    print(f"  Max reward: {max(episode_rewards):.1f}")
    
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize trained agent gameplay")
    parser.add_argument("--model", required=True, help="Path to saved model checkpoint")
    parser.add_argument("--game", default="Pong-v5", help="Game name (default: Pong-v5)")
    parser.add_argument("--algorithm", choices=["dqn", "ppo"], default="dqn", 
                       help="Algorithm used (default: dqn)")
    parser.add_argument("--episodes", type=int, default=3, 
                       help="Number of episodes to record (default: 3)")
    parser.add_argument("--output", help="Output video path")
    parser.add_argument("--fps", type=int, default=30, 
                       help="Frames per second (default: 30)")
    parser.add_argument("--format", choices=["mp4", "gif"], default="mp4",
                       help="Video format (default: mp4)")
    parser.add_argument("--max-steps", type=int, default=10000,
                       help="Maximum steps per episode (default: 10000)")
    
    args = parser.parse_args()
    
    visualize_agent(
        model_path=args.model,
        game_name=args.game,
        algorithm=args.algorithm,
        num_episodes=args.episodes,
        output_path=args.output,
        fps=args.fps,
        video_format=args.format,
        max_steps=args.max_steps,
    )

