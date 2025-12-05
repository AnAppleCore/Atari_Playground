"""Multi-task joint training on multiple Atari games."""
import sys
import torch
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from algorithms import MultiHeadDQNAgent, MultiHeadPPOAgent
from environments import AtariEnv
from utils import ReplayBuffer, RolloutBuffer, VideoRecorder, MetricsPlotter


def train_multitask(
    games: list = None,
    algorithm: str = "dqn",
    total_steps: int = 150000,
    batch_size: int = 32,
    save_video: bool = False,
):
    """Train agent on multiple games jointly (random task sampling per iteration)."""
    if games is None:
        games = ["Pong-v5", "Breakout-v5", "SpaceInvaders-v5"]

    print(f"Multi-task Joint Training: {algorithm.upper()} on {games}")
    print(f"Total steps: {total_steps}")

    # Initialize environments and get action dimensions
    envs = {}
    action_dims = {}
    for game_name in games:
        env = AtariEnv(game_name, render_mode=None)
        envs[game_name] = env
        action_dims[game_name] = env.action_space

    # Initialize agent with multi-head architecture
    if algorithm == "dqn":
        agent = MultiHeadDQNAgent(
            state_dim=4,
            lr=1e-4,
            gamma=0.99,
        )
        learning_starts = 80000
        train_frequency = 4
        buffers = {game: ReplayBuffer(capacity=100000) for game in games}
    else:  # ppo
        agent = MultiHeadPPOAgent(
            state_dim=4,
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
        buffers = {game: RolloutBuffer(capacity=rollout_length) for game in games}

    # Register all tasks
    for game_name in games:
        agent.register_task(game_name, action_dims[game_name])

    exp_dir = Path("outputs") / "multitask" / f"{algorithm}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Per-game video recorders (optional)
    video_recorders = {}
    if save_video:
        for game_name in games:
            game_dir = exp_dir / game_name
            game_dir.mkdir(parents=True, exist_ok=True)
            video_recorders[game_name] = VideoRecorder(
                str(game_dir / "training.mp4"),
                fps=30
            )

    metrics_plotter = MetricsPlotter()
    episode_rewards = defaultdict(list)
    
    # Initialize states for all environments
    states = {game: envs[game].reset() for game in games}
    episode_reward = {game: 0.0 for game in games}
    episode_count = {game: 0 for game in games}

    # Track which game we're currently collecting data for (for PPO)
    current_collect_game = None
    rollout_step = 0

    pbar = tqdm(total=total_steps, desc="Multi-task Training")
    step = 0

    while step < total_steps:
        # Randomly sample a task for this iteration
        current_game = np.random.choice(games)
        task_id = current_game
        agent.set_task(task_id)
        env = envs[current_game]
        state = states[current_game]
        buffer = buffers[current_game]

        if algorithm == "dqn":
            # DQN: collect one step, then potentially train
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            episode_reward[current_game] += reward
            buffer.add(state, action, reward, next_state, done)

            if save_video and current_game in video_recorders and step % 2 == 0:
                frame = state[0].cpu().numpy() if isinstance(state, torch.Tensor) else state[0]
                video_recorders[current_game].add_frame(frame)

            states[current_game] = next_state
            step += 1
            pbar.update(1)

            if done:
                episode_rewards[current_game].append(episode_reward[current_game])
                metrics_plotter.add_metric(f"{current_game}_episode_reward", episode_reward[current_game])
                episode_reward[current_game] = 0.0
                states[current_game] = env.reset()
                episode_count[current_game] += 1

            # Training step
            if step > learning_starts and step % train_frequency == 0:
                if buffer.is_ready(batch_size):
                    batch = buffer.sample(batch_size)
                    metrics = agent.update(batch)
                    pbar.set_postfix({**metrics, "game": current_game[:8]})

                    if "loss" in metrics:
                        metrics_plotter.add_metric(f"{current_game}_loss", metrics["loss"])
                    if "epsilon" in metrics:
                        metrics_plotter.add_metric(f"{current_game}_epsilon", metrics["epsilon"])
                    if "q_value" in metrics:
                        metrics_plotter.add_metric(f"{current_game}_q_value", metrics["q_value"])

        else:  # ppo
            # PPO: collect rollout_length steps before updating
            # We need to collect a full rollout for the current game before switching
            if current_collect_game is None:
                # Start collecting for a new game
                current_collect_game = current_game
                rollout_step = 0
            elif current_collect_game != current_game:
                # If we switched games but haven't finished the previous rollout, continue with previous game
                current_game = current_collect_game
                env = envs[current_game]
                state = states[current_game]
                buffer = buffers[current_game]
                agent.set_task(current_game)

            if rollout_step < rollout_length:
                # Collect one step
                with torch.no_grad():
                    state_tensor = state.unsqueeze(0).to(agent.device)
                    action_tensor, log_prob, _, value = agent.get_action_and_value(state_tensor)
                    action = action_tensor.item()
                    log_prob_val = log_prob.item()
                    value_val = value.item()

                next_state, reward, done = env.step(action)
                episode_reward[current_game] += reward
                buffer.add(state, action, reward, done, log_prob_val, value_val)

                if save_video and current_game in video_recorders and step % 2 == 0:
                    frame = state[0].cpu().numpy() if isinstance(state, torch.Tensor) else state[0]
                    video_recorders[current_game].add_frame(frame)

                states[current_game] = next_state
                step += 1
                rollout_step += 1
                pbar.update(1)

                if done:
                    episode_rewards[current_game].append(episode_reward[current_game])
                    metrics_plotter.add_metric(f"{current_game}_episode_reward", episode_reward[current_game])
                    episode_reward[current_game] = 0.0
                    states[current_game] = env.reset()
                    episode_count[current_game] += 1

            # Update when buffer is full
            if buffer.is_full() and rollout_step >= rollout_length:
                with torch.no_grad():
                    next_state_tensor = states[current_game].unsqueeze(0).to(agent.device)
                    next_value = agent.get_value(next_state_tensor).flatten()

                rollout_data = buffer.get_batch()
                metrics = agent.update(rollout_data, next_value, update_epochs, minibatch_size)
                pbar.set_postfix({**metrics, "game": current_game[:8], "episodes": episode_count[current_game]})

                if "policy_loss" in metrics:
                    metrics_plotter.add_metric(f"{current_game}_policy_loss", metrics["policy_loss"])
                if "value_loss" in metrics:
                    metrics_plotter.add_metric(f"{current_game}_value_loss", metrics["value_loss"])
                if "entropy" in metrics:
                    metrics_plotter.add_metric(f"{current_game}_entropy", metrics["entropy"])

                buffer.reset()
                current_collect_game = None
                rollout_step = 0

    # Save videos
    if save_video:
        for game_name, recorder in video_recorders.items():
            recorder.save(format="mp4")
            print(f"Video saved: {exp_dir / game_name / 'training.mp4'}")

    # Print training summary
    print(f"\n{'='*60}")
    print(f"Multi-task Training Summary")
    print(f"{'='*60}")
    for game_name in games:
        if episode_rewards[game_name]:
            avg_reward = sum(episode_rewards[game_name]) / len(episode_rewards[game_name])
            last_n = min(20, len(episode_rewards[game_name]))
            avg_last_n = sum(episode_rewards[game_name][-last_n:]) / last_n if last_n > 0 else 0.0
            print(f"\n{game_name}:")
            print(f"  Total episodes: {len(episode_rewards[game_name])}")
            print(f"  Average episode reward: {avg_reward:.2f}")
            print(f"  Average reward over last {last_n} episodes: {avg_last_n:.2f}")

    # Save metrics plot
    metrics_output_path = exp_dir / "training_metrics.png"
    metrics_plotter.plot(str(metrics_output_path))
    print(f"\nMetrics plot saved: {metrics_output_path}")

    # Save model checkpoint
    ckpt_root = Path("checkpoints") / "multitask"
    ckpt_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = ckpt_root / f"{algorithm}.pt"
    agent.save(str(checkpoint_path))
    print(f"Agent saved: {checkpoint_path}")

    # Close all environments
    for env in envs.values():
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", nargs="+", default=["Pong-v5", "Breakout-v5", "SpaceInvaders-v5"])
    parser.add_argument("--algorithm", default="dqn", choices=["dqn", "ppo"])
    parser.add_argument("--steps", type=int, default=150000, help="Total training steps across all games")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--save-video", action="store_true", help="Enable video recording (disabled by default)")

    args = parser.parse_args()

    train_multitask(
        games=args.games,
        algorithm=args.algorithm,
        total_steps=args.steps,
        batch_size=args.batch_size,
        save_video=args.save_video,
    )
