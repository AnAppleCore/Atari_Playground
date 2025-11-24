"""Continual learning on multiple Atari games."""
import sys
import torch
import argparse
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from algorithms import DQNAgent, PPOAgent, EWCWrapper, MultiHeadDQNAgent
from environments import AtariEnv
from utils import ReplayBuffer, VideoRecorder, MetricsPlotter


def train_continual(
    games: list = None,
    algorithm: str = "dqn",
    use_ewc: bool = False,
    ewc_lambda: float = 0.4,
    steps_per_game: int = 50000,
    batch_size: int = 32,
):
    """Train agent on multiple games sequentially."""

    if games is None:
        games = ["Pong-v5", "Breakout-v5", "SpaceInvaders-v5"]

    print(f"Continual Learning: {algorithm.upper()} on {games}")
    print(f"EWC: {use_ewc}")
    if use_ewc:
        print(f"EWC Lambda: {ewc_lambda}")

    # Metrics recorder for continual run (per-task averages)
    continual_metrics = {}

    # Evaluation metrics for catastrophic forgetting (per task, per stage)
    eval_history = {game: [] for game in games}

    # Shared agent across all tasks
    agent = None
    base_action_dim = None

    # Train on each game sequentially using the same agent
    for game_idx, game_name in enumerate(games):
        print(f"\n=== Task {game_idx + 1}/{len(games)}: {game_name} ===")

        # Create environment (no rendering during training for performance)
        env = AtariEnv(game_name, render_mode=None)
        action_dim = env.action_space

        task_id = game_name  # use game name as task identifier

        # Initialize shared agent on the first task
        if agent is None:
            base_action_dim = action_dim
            if algorithm == "dqn":
                base_agent = MultiHeadDQNAgent(state_dim=4, lr=1e-4)
                base_agent.register_task(task_id, action_dim)
                base_agent.set_task(task_id)
            else:  # ppo (single-head PPO for now)
                base_agent = PPOAgent(state_dim=4, action_dim=action_dim, lr=3e-4)

            # Wrap with EWC if needed
            agent = EWCWrapper(base_agent, ewc_lambda=ewc_lambda) if use_ewc else base_agent
        else:
            # Register/select the appropriate head for this task when using
            # the multi-head DQN agent. For standard DQN/PPO this is a no-op.
            if isinstance(agent, EWCWrapper):
                # Agent is wrapped; access the underlying base agent
                if isinstance(agent.agent, MultiHeadDQNAgent):
                    agent.register_task(task_id, action_dim)
                    agent.set_task(task_id)
            elif isinstance(agent, MultiHeadDQNAgent):
                agent.register_task(task_id, action_dim)
                agent.set_task(task_id)

        replay_buffer = ReplayBuffer(capacity=100000)

        # Per-task experiment directory for this continual run
        exp_dir = Path("outputs") / "continual" / f"{algorithm}_ewc{use_ewc}" / game_name
        exp_dir.mkdir(parents=True, exist_ok=True)

        video_recorder = VideoRecorder(
            str(exp_dir / "training.mp4"),
            fps=15  # Lower FPS for better viewing
        )

        state = env.reset()
        episode_reward = 0
        episode_rewards = []

        pbar = tqdm(range(steps_per_game), desc=f"Training on {game_name}")
        for step in pbar:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            episode_reward += reward

            replay_buffer.add(state, action, reward, next_state, done)

            if step % 10 == 0:
                frame = (state[0].numpy() * 255).astype('uint8')
                video_recorder.add_frame(frame)

            if replay_buffer.is_ready(batch_size):
                batch = replay_buffer.sample(batch_size)
                metrics = agent.update(batch)
                pbar.set_postfix(metrics)

                # Log step-wise losses if present
                if "loss" in metrics:
                    continual_metrics.setdefault(game_name + "_loss", []).append(metrics["loss"])
                if "policy_loss" in metrics:
                    continual_metrics.setdefault(game_name + "_policy_loss", []).append(metrics["policy_loss"])
                if "value_loss" in metrics:
                    continual_metrics.setdefault(game_name + "_value_loss", []).append(metrics["value_loss"])
                if "ewc_loss" in metrics:
                    continual_metrics.setdefault(game_name + "_ewc_loss", []).append(metrics["ewc_loss"])

            state = next_state

            if done:
                episode_rewards.append(episode_reward)
                episode_reward = 0
                state = env.reset()

        # Per-task stats
        if episode_rewards:
            continual_metrics.setdefault(game_name + "_episode_reward", []).extend(episode_rewards)

        # ---- Evaluation on all seen tasks so far (catastrophic forgetting) ----
        for eval_task_idx, eval_game in enumerate(games[: game_idx + 1]):
            eval_env = AtariEnv(eval_game, render_mode=None)
            # For multi-head DQN, switch head; for PPO, noop
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

            # Run a few greedy evaluation episodes (no learning)
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

        video_recorder.save(format="mp4")
        print(f"Video saved: outputs/{game_name}_{algorithm}_ewc{use_ewc}.mp4")

        # Consolidate weights after task
        if use_ewc:
            agent.consolidate_weights()
            print("Weights consolidated for EWC")

        env.close()

    # Save aggregated continual metrics plot (training dynamics)
    metrics_plotter = MetricsPlotter()
    for name, values in continual_metrics.items():
        for v in values:
            metrics_plotter.add_metric(name, v)

    # Experiment-level directory (not per-task) for aggregated continual results
    exp_root = Path("outputs") / "continual" / f"{algorithm}_ewc{use_ewc}"
    exp_root.mkdir(parents=True, exist_ok=True)

    metrics_output_path = exp_root / "training_metrics.png"
    metrics_plotter.plot(str(metrics_output_path))
    print(f"Continual metrics plot saved: {metrics_output_path}")

    # Plot catastrophic forgetting curves: each game's eval reward over stages
    eval_plotter = MetricsPlotter()
    for game_name, history in eval_history.items():
        for stage_idx, avg_reward in history:
            eval_plotter.add_metric(f"{game_name}_stage{stage_idx}", avg_reward)
    eval_metrics_path = exp_root / "forgetting_eval.png"
    eval_plotter.plot(str(eval_metrics_path))
    print(f"Continual evaluation (forgetting) plot saved: {eval_metrics_path}")

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

    args = parser.parse_args()

    train_continual(
        games=args.games,
        algorithm=args.algorithm,
        use_ewc=args.use_ewc,
        ewc_lambda=args.ewc_lambda,
        steps_per_game=args.steps_per_game,
        batch_size=args.batch_size,
    )

