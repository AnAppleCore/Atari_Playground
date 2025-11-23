"""Train a single agent on a single Atari game."""
import sys
import torch
import argparse
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from algorithms import DQNAgent, PPOAgent
from environments import AtariEnv
from utils import ReplayBuffer, VideoRecorder, MetricsPlotter


def train_single_game(
    game_name: str = "Pong-v5",
    algorithm: str = "dqn",
    num_steps: int = 200000,
    batch_size: int = 32,
    save_video: bool = True,
):
    """Train agent on a single game.

    This version matches the earlier, empirically better-performing
    configuration you used for ~200k-step Pong runs: no explicit
    learning warmup, straightforward per-step updates, and simple
    logging.
    """

    print(f"Training {algorithm.upper()} on {game_name}")

    # Create environment (no rendering during training for performance)
    env = AtariEnv(game_name, render_mode=None)

    # Create agent
    if algorithm == "dqn":
        agent = DQNAgent(
            state_dim=4,
            action_dim=env.action_space,
            lr=2.5e-4,
        )
    else:  # ppo
        agent = PPOAgent(
            state_dim=4,
            action_dim=env.action_space,
            lr=3e-4,
        )

    # Create replay buffer
    replay_buffer = ReplayBuffer(capacity=100000)

    # Video recorder (MP4 format with 15 FPS for better playback)
    video_recorder = None
    if save_video:
        video_recorder = VideoRecorder(
            f"outputs/{game_name}_{algorithm}_training.mp4",
            fps=15,
        )

    # Metrics recorder
    metrics_plotter = MetricsPlotter()
    episode_rewards = []

    # Training loop
    state = env.reset()
    episode_reward = 0

    pbar = tqdm(range(num_steps), desc="Training")
    for step in pbar:
        # Select action
        action = agent.select_action(state)

        # Take step
        next_state, reward, done = env.step(action)
        episode_reward += reward

        # Store experience
        replay_buffer.add(state, action, reward, next_state, done)

        # Record frame at every step for smoother video
        if video_recorder is not None:
            frame = (state[0].numpy() * 255).astype("uint8")
            video_recorder.add_frame(frame)

        # Update agent as soon as buffer has enough samples
        if replay_buffer.is_ready(batch_size):
            batch = replay_buffer.sample(batch_size)
            metrics = agent.update(batch)
            pbar.set_postfix(metrics)

            # Log losses (DQN and PPO use different keys)
            if "loss" in metrics:
                metrics_plotter.add_metric("loss", metrics["loss"])
            if "policy_loss" in metrics:
                metrics_plotter.add_metric("policy_loss", metrics["policy_loss"])
            if "value_loss" in metrics:
                metrics_plotter.add_metric("value_loss", metrics["value_loss"])

        state = next_state

        if done:
            episode_rewards.append(episode_reward)
            metrics_plotter.add_metric("episode_reward", episode_reward)
            episode_reward = 0
            state = env.reset()

    # Save video
    if video_recorder is not None:
        video_recorder.save(format="mp4")
        print(f"Video saved to outputs/{game_name}_{algorithm}.mp4")

    # Simple reward statistics for inspection
    if episode_rewards:
        avg_reward = sum(episode_rewards) / len(episode_rewards)
        last_n = min(10, len(episode_rewards))
        avg_last_n = sum(episode_rewards[-last_n:]) / last_n
        print(f"Total episodes: {len(episode_rewards)}")
        print(f"Average episode reward: {avg_reward:.2f}")
        print(f"Average reward over last {last_n} episodes: {avg_last_n:.2f}")

    # Save metrics plot
    metrics_output_path = f"outputs/{game_name}_{algorithm}_metrics.png"
    metrics_plotter.plot(metrics_output_path)
    print(f"Metrics plot saved to {metrics_output_path}")

    # Save agent
    Path("checkpoints").mkdir(exist_ok=True)
    agent.save(f"checkpoints/{game_name}_{algorithm}.pt")
    print(f"Agent saved to checkpoints/{game_name}_{algorithm}.pt")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="Pong-v5", help="Game name")
    parser.add_argument("--algorithm", default="dqn", choices=["dqn", "ppo"])
    parser.add_argument("--steps", type=int, default=200000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--no-video", action="store_true")

    args = parser.parse_args()

    train_single_game(
        game_name=args.game,
        algorithm=args.algorithm,
        num_steps=args.steps,
        batch_size=args.batch_size,
        save_video=not args.no_video,
    )

