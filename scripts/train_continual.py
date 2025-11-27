"""Continual learning on multiple Atari games."""
import sys
import torch
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from algorithms import DQNAgent, PPOAgent, EWCWrapper, MultiHeadDQNAgent, MultiHeadPPOAgent
from environments import AtariEnv
from utils import ReplayBuffer, RolloutBuffer, VideoRecorder, MetricsPlotter


def plot_forgetting_curves(eval_history: dict, output_path: Path) -> bool:
    """Plot forgetting curves using evaluation history."""
    has_data = False
    max_stage = 0

    plt.figure(figsize=(8, 5))
    for game_name, history in eval_history.items():
        if not history:
            continue
        history = sorted(history, key=lambda item: item[0])
        stages, rewards = zip(*history)
        plt.plot(stages, rewards, marker="o", label=game_name)
        max_stage = max(max_stage, stages[-1])
        has_data = True

    if not has_data:
        print("No evaluation history collected; skipping forgetting curve plot.")
        plt.close()
        return False

    plt.xlabel("Training Stage")
    plt.ylabel("Average Reward")
    plt.title("Continual Evaluation (Forgetting Curves)")
    plt.xticks(range(1, max_stage + 1))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Continual evaluation (forgetting) plot saved: {output_path}")
    return True


def train_continual(
    games: list = None,
    algorithm: str = "dqn",
    use_ewc: bool = False,
    ewc_lambda: float = 0.4,
    steps_per_game: int = 50000,
    batch_size: int = 32,
    save_video: bool = False,
):
    """Train agent on multiple games sequentially."""
    if games is None:
        games = ["Pong-v5", "Breakout-v5", "SpaceInvaders-v5"]

    print(f"Continual Learning: {algorithm.upper()} on {games}")
    print(f"EWC: {use_ewc}")
    if use_ewc:
        print(f"EWC Lambda: {ewc_lambda}")

    continual_metrics = {}
    eval_history = {game: [] for game in games}
    agent = None
    base_action_dim = None

    for game_idx, game_name in enumerate(games):
        print(f"\n=== Task {game_idx + 1}/{len(games)}: {game_name} ===")

        env = AtariEnv(game_name, render_mode=None)
        action_dim = env.action_space
        task_id = game_name

        if agent is None:
            base_action_dim = action_dim
            if algorithm == "dqn":
                base_agent = MultiHeadDQNAgent(
                    state_dim=4,
                    lr=1e-4,
                    gamma=0.99,
                )
                base_agent.register_task(task_id, action_dim)
                base_agent.set_task(task_id)
            else:  # ppo
                base_agent = MultiHeadPPOAgent(
                    state_dim=4,
                    lr=2.5e-4,
                    gamma=0.99,
                    gae_lambda=0.95,
                    clip_coef=0.1,
                    ent_coef=0.01,
                    vf_coef=0.5,
                    max_grad_norm=0.5,
                )
                base_agent.register_task(task_id, action_dim)
                base_agent.set_task(task_id)

            agent = EWCWrapper(base_agent, ewc_lambda=ewc_lambda) if use_ewc else base_agent
        else:
            if isinstance(agent, EWCWrapper):
                if isinstance(agent.agent, (MultiHeadDQNAgent, MultiHeadPPOAgent)):
                    agent.register_task(task_id, action_dim)
                    agent.set_task(task_id)
            elif isinstance(agent, (MultiHeadDQNAgent, MultiHeadPPOAgent)):
                agent.register_task(task_id, action_dim)
                agent.set_task(task_id)

        # Initialize buffers and training parameters based on algorithm
        if algorithm == "dqn":
            learning_starts = 80000
            train_frequency = 4
            buffer = ReplayBuffer(capacity=100000)
        else:  # ppo
            learning_starts = 0
            train_frequency = 1
            rollout_length = 128
            update_epochs = 4
            minibatch_size = 32
            buffer = RolloutBuffer(capacity=rollout_length)

        exp_dir = Path("outputs") / "continual" / f"{algorithm}_ewc{use_ewc}" / game_name
        exp_dir.mkdir(parents=True, exist_ok=True)

        video_recorder = None
        if save_video:
            video_recorder = VideoRecorder(
                str(exp_dir / "training.mp4"),
                fps=30
            )

        state = env.reset()
        episode_reward = 0
        episode_rewards = []
        episode_count = 0
        step = 0

        pbar = tqdm(total=steps_per_game, desc=f"Training on {game_name}")
        
        while step < steps_per_game:
            if algorithm == "dqn":
                # DQN training logic (same as train_single.py)
                if isinstance(agent, EWCWrapper) and isinstance(agent.agent, MultiHeadDQNAgent):
                    # MultiHeadDQNAgent uses its own epsilon management
                    action = agent.select_action(state)
                elif isinstance(agent, MultiHeadDQNAgent):
                    action = agent.select_action(state)
                else:
                    # Regular DQNAgent
                    if hasattr(agent, 'global_step'):
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
                            continual_metrics.setdefault(game_name + "_loss", []).append(metrics["loss"])
                        if "epsilon" in metrics:
                            continual_metrics.setdefault(game_name + "_epsilon", []).append(metrics["epsilon"])
                        if "q_value" in metrics:
                            continual_metrics.setdefault(game_name + "_q_value", []).append(metrics["q_value"])
                        if "ewc_loss" in metrics:
                            continual_metrics.setdefault(game_name + "_ewc_loss", []).append(metrics["ewc_loss"])

                state = next_state
                step += 1
                pbar.update(1)

                if done:
                    episode_rewards.append(episode_reward)
                    episode_reward = 0
                    state = env.reset()
            
            else:  # ppo
                # PPO training logic (same as train_single.py)
                # MultiHeadPPOAgent uses the same interface as MultiHeadDQNAgent
                for rollout_step in range(rollout_length):
                    if step >= steps_per_game:
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
                        continual_metrics.setdefault(game_name + "_policy_loss", []).append(metrics["policy_loss"])
                    if "value_loss" in metrics:
                        continual_metrics.setdefault(game_name + "_value_loss", []).append(metrics["value_loss"])
                    if "entropy" in metrics:
                        continual_metrics.setdefault(game_name + "_entropy", []).append(metrics["entropy"])
                    if "ewc_loss" in metrics:
                        continual_metrics.setdefault(game_name + "_ewc_loss", []).append(metrics["ewc_loss"])

                    buffer.reset()
                
                pbar.update(min(rollout_length, steps_per_game - step))

        if episode_rewards:
            continual_metrics.setdefault(game_name + "_episode_reward", []).extend(episode_rewards)

        for eval_task_idx, eval_game in enumerate(games[: game_idx + 1]):
            eval_env = AtariEnv(eval_game, render_mode=None)
            eval_task_id = eval_game
            if isinstance(agent, EWCWrapper) and hasattr(agent.agent, "set_task"):
                try:
                    agent.set_task(eval_task_id)
                except Exception:
                    pass
            elif hasattr(agent, "set_task"):
                try:
                    agent.set_task(eval_task_id)
                except Exception:
                    pass

            num_eval_episodes = 5
            total_eval_reward = 0.0
            for _ in range(num_eval_episodes):
                s = eval_env.reset()
                done_eval = False
                ep_r = 0.0
                while not done_eval:
                    with torch.no_grad():
                        a = agent.select_action(s)
                    s, r, done_eval = eval_env.step(a)
                    ep_r += r
                total_eval_reward += ep_r
            avg_eval_reward = total_eval_reward / num_eval_episodes
            eval_env.close()

            eval_history[eval_game].append((game_idx + 1, avg_eval_reward))
            print(f"[Eval] After task {game_name}, on {eval_game}: avg reward {avg_eval_reward:.2f}")

        if video_recorder is not None:
            video_recorder.save(format="mp4")
            print(f"Video saved: {exp_dir / 'training.mp4'}")

        if use_ewc:
            # For Fisher Information computation, we need a sample batch
            # Use the last batch from the buffer if available
            sample_batch = None
            if algorithm == "dqn" and buffer.is_ready(batch_size):
                sample_batch = buffer.sample(min(batch_size, 100))
            elif algorithm == "ppo" and buffer.is_full():
                sample_batch = buffer.get_batch()
                # Convert to dict format if needed
                if not isinstance(sample_batch, dict):
                    sample_batch = {
                        "states": sample_batch[0],
                        "actions": sample_batch[1],
                        "rewards": sample_batch[2],
                        "dones": sample_batch[3],
                        "log_probs": sample_batch[4],
                        "values": sample_batch[5],
                    }
            
            agent.consolidate_weights(sample_batch)
            print("Weights consolidated for EWC")

        env.close()

    metrics_plotter = MetricsPlotter()
    for name, values in continual_metrics.items():
        for v in values:
            metrics_plotter.add_metric(name, v)

    exp_root = Path("outputs") / "continual" / f"{algorithm}_ewc{use_ewc}"
    exp_root.mkdir(parents=True, exist_ok=True)

    metrics_output_path = exp_root / "training_metrics.png"
    metrics_plotter.plot(str(metrics_output_path))
    print(f"Continual metrics plot saved: {metrics_output_path}")

    eval_metrics_path = exp_root / "forgetting_eval.png"
    plot_forgetting_curves(eval_history, eval_metrics_path)

    # Save final agent checkpoint in a structured location
    ckpt_root = Path("checkpoints") / "continual"
    ckpt_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = ckpt_root / f"{algorithm}_ewc{use_ewc}.pt"
    agent.save(str(checkpoint_path))
    print(f"\nAgent saved: {checkpoint_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", nargs="+", default=["Pong-v5", "Breakout-v5", "SpaceInvaders-v5"])
    parser.add_argument("--algorithm", default="dqn", choices=["dqn", "ppo"])
    parser.add_argument("--use-ewc", action="store_true")
    parser.add_argument("--ewc-lambda", type=float, default=0.4, help="EWC regularization strength")
    parser.add_argument("--steps-per-game", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--save-video", action="store_true", help="Enable video recording (disabled by default)")

    args = parser.parse_args()

    train_continual(
        games=args.games,
        algorithm=args.algorithm,
        use_ewc=args.use_ewc,
        ewc_lambda=args.ewc_lambda,
        steps_per_game=args.steps_per_game,
        batch_size=args.batch_size,
        save_video=args.save_video,
    )

