"""Train a single agent on a single Atari game."""
import sys
import torch
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from algorithms import DQNAgent, PPOAgent
from environments import AtariEnv
from utils import ReplayBuffer, RolloutBuffer, VideoRecorder, MetricsPlotter


def train_single_game(
    game_name: str = "Pong-v5",
    algorithm: str = "dqn",
    num_steps: int = 500000,
    batch_size: int = 32,
    save_video: bool = False,
):
    """Train agent on a single game."""
    print(f"Training {algorithm.upper()} on {game_name}")

    env = AtariEnv(game_name, render_mode=None)

    if algorithm == "dqn":
        agent = DQNAgent(
            state_dim=4,
            action_dim=env.action_space,
            lr=1e-4,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_fraction=0.10,
            total_timesteps=num_steps,
            target_update_freq=1000,
            tau=1.0,
        )
        learning_starts = 80000
        train_frequency = 4
        buffer = ReplayBuffer(capacity=100000)
    else:
        agent = PPOAgent(
            state_dim=4,
            action_dim=env.action_space,
            lr=2.5e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_coef=0.1,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
        )
        learning_starts = 0
        train_frequency = 1
        rollout_length = 128
        update_epochs = 4
        minibatch_size = 32
        buffer = RolloutBuffer(capacity=rollout_length)

    exp_dir = Path("outputs") / "single" / f"{game_name}_{algorithm}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    video_recorder = None
    if save_video:
        video_path = exp_dir / "training.mp4"
        video_recorder = VideoRecorder(
            str(video_path),
            fps=30,
        )

    metrics_plotter = MetricsPlotter()
    episode_rewards = []

    state = env.reset()
    episode_reward = 0
    episode_count = 0

    pbar = tqdm(total=num_steps, desc="Training")
    step = 0
    
    while step < num_steps:
        if algorithm == "dqn":
            agent.global_step = step
            action = agent.select_action(state, training=True)
            next_state, reward, done = env.step(action)
            episode_reward += reward
            buffer.add(state, action, reward, next_state, done)

            if video_recorder is not None and step % 2 == 0:
                frame = state[0].cpu().numpy() if isinstance(state, torch.Tensor) else state[0]
                video_recorder.add_frame(frame)

            if step > learning_starts and step % train_frequency == 0:
                if buffer.is_ready(batch_size):
                    batch = buffer.sample(batch_size)
                    metrics = agent.update(batch)
                    pbar.set_postfix(metrics)

                    if "loss" in metrics:
                        metrics_plotter.add_metric("loss", metrics["loss"])
                    if "epsilon" in metrics:
                        metrics_plotter.add_metric("epsilon", metrics["epsilon"])
                    if "q_value" in metrics:
                        metrics_plotter.add_metric("q_value", metrics["q_value"])

            state = next_state
            step += 1
            pbar.update(1)

            if done:
                episode_rewards.append(episode_reward)
                metrics_plotter.add_metric("episode_reward", episode_reward)
                episode_reward = 0
                state = env.reset()
        
        else:
            for rollout_step in range(rollout_length):
                if step >= num_steps:
                    break
                
                with torch.no_grad():
                    state_tensor = state.unsqueeze(0).to(agent.device)
                    action_tensor, log_prob, _, value = agent.get_action_and_value(state_tensor)
                    action = action_tensor.item()
                    log_prob_val = log_prob.item()
                    value_val = value.item()

                next_state, reward, done = env.step(action)
                episode_reward += reward
                buffer.add(state, action, reward, done, log_prob_val, value_val)

                if video_recorder is not None and step % 2 == 0:
                    frame = state[0].cpu().numpy() if isinstance(state, torch.Tensor) else state[0]
                    video_recorder.add_frame(frame)

                state = next_state
                step += 1

                if done:
                    episode_rewards.append(episode_reward)
                    metrics_plotter.add_metric("episode_reward", episode_reward)
                    episode_reward = 0
                    state = env.reset()
                    episode_count += 1

            if buffer.is_full() and step >= rollout_length:
                with torch.no_grad():
                    next_state_tensor = state.unsqueeze(0).to(agent.device)
                    next_value = agent.get_value(next_state_tensor).flatten()

                rollout_data = buffer.get_batch()
                metrics = agent.update(rollout_data, next_value, update_epochs, minibatch_size)
                pbar.set_postfix({**metrics, "episodes": episode_count})

                if "policy_loss" in metrics:
                    metrics_plotter.add_metric("policy_loss", metrics["policy_loss"])
                if "value_loss" in metrics:
                    metrics_plotter.add_metric("value_loss", metrics["value_loss"])
                if "entropy" in metrics:
                    metrics_plotter.add_metric("entropy", metrics["entropy"])

                buffer.reset()
            
            pbar.update(min(rollout_length, num_steps - step))

    if video_recorder is not None:
        video_recorder.save(format="mp4")
        print(f"Video saved to {exp_dir / 'training.mp4'}")

    if episode_rewards:
        avg_reward = sum(episode_rewards) / len(episode_rewards)
        last_n = min(20, len(episode_rewards))
        avg_last_n = sum(episode_rewards[-last_n:]) / last_n
        print(f"\n{'='*60}")
        print(f"Training Summary")
        print(f"{'='*60}")
        print(f"Total episodes: {len(episode_rewards)}")
        print(f"Average episode reward: {avg_reward:.2f}")
        print(f"Average reward over last {last_n} episodes: {avg_last_n:.2f}")
        
        if len(episode_rewards) >= 20:
            first_10 = sum(episode_rewards[:10]) / 10
            last_10 = sum(episode_rewards[-10:]) / 10
            print(f"\nReward Progression:")
            print(f"  First 10 episodes: {first_10:.2f}")
            print(f"  Last 10 episodes: {last_10:.2f}")
            improvement = last_10 - first_10
            print(f"  Improvement: {improvement:+.2f}")
            if improvement > 0:
                print(f"  ✓ Agent is learning!")
            elif improvement < -1:
                print(f"  ⚠ Agent performance decreased")
            else:
                print(f"  → Agent performance stable")
        
        best_ep = max(episode_rewards)
        worst_ep = min(episode_rewards)
        print(f"\nBest episode reward: {best_ep:.2f}")
        print(f"Worst episode reward: {worst_ep:.2f}")
        print(f"{'='*60}")

    metrics_output_path = exp_dir / "metrics.png"
    metrics_plotter.plot(str(metrics_output_path))
    print(f"Metrics plot saved to {metrics_output_path}")

    Path("checkpoints").mkdir(exist_ok=True)
    checkpoint_path = Path("checkpoints") / "single" / f"{game_name}_{algorithm}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    agent.save(str(checkpoint_path))
    print(f"Agent saved to {checkpoint_path}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="Pong-v5", help="Game name")
    parser.add_argument("--algorithm", default="dqn", choices=["dqn", "ppo"])
    parser.add_argument("--steps", type=int, default=500000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--save-video", action="store_true", help="Enable video recording (disabled by default)")

    args = parser.parse_args()

    train_single_game(
        game_name=args.game,
        algorithm=args.algorithm,
        num_steps=args.steps,
        batch_size=args.batch_size,
        save_video=args.save_video,
    )
