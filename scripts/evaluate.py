"""Evaluate trained Atari agents (single-task or continual)."""
import sys
import json
import argparse
from pathlib import Path
from typing import Optional

import torch
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from environments import AtariEnv
from algorithms import DQNAgent, PPOAgent, EWCWrapper, MultiHeadDQNAgent
from utils import VideoRecorder

# Base directory for evaluation figures and videos.
# For better organization, we create one folder per experiment; by
# default this is derived from either --json-out (if provided) or
# falls back to a generic "outputs/evaluate" directory.
EVAL_DIR = Path("outputs/evaluate")


def _set_eval_dir_from_json_out(json_out: Optional[str]) -> None:
    """Configure EVAL_DIR based on the JSON output path (if any).

    If the user passes --json-out PATH, we treat PATH's parent directory
    as the experiment-specific evaluation directory, so that plots and
    videos for that run live alongside its JSON. If not given, we keep
    using the default EVAL_DIR.
    """
    global EVAL_DIR
    if json_out:
        EVAL_DIR = Path(json_out).parent
    EVAL_DIR.mkdir(parents=True, exist_ok=True)


def _get_eval_dir() -> Path:
    """Ensure and return the evaluation output directory."""
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    return EVAL_DIR


def _plot_single_results(result: dict) -> Path:
    """Plot episode rewards for a single-task evaluation.

    Saves a line plot ``episode -> reward`` and returns its path.
    """
    out_dir = _get_eval_dir()
    game = result["game"]
    algorithm = result["algorithm"]
    rewards = result["rewards"]

    plot_path = out_dir / f"{game}_{algorithm}_eval_rewards.png"

    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(rewards) + 1), rewards, marker="o")
    plt.title(f"{game} ({algorithm}) Evaluation Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

    return plot_path


def _plot_continual_results(result: dict) -> Path:
    """Plot average reward per game for continual evaluation."""
    out_dir = _get_eval_dir()
    algorithm = result["algorithm"]
    games = sorted(result["games"].keys())
    avg_rewards = [result["games"][g]["avg_reward"] for g in games]

    plot_path = out_dir / f"continual_{algorithm}_avg_rewards.png"

    plt.figure(figsize=(6, 4))
    plt.bar(games, avg_rewards)
    plt.title(f"Continual {algorithm.upper()} Avg Reward per Game")
    plt.xlabel("Game")
    plt.ylabel("Average Reward")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

    return plot_path


def _record_example_video(agent, game: str, algorithm: str, output_path: Path, max_steps: int = 10000):
    """Record a single-episode gameplay video for the given agent and game.

    The agent is assumed to be already loaded and in eval mode.
    """
    env = AtariEnv(game, render_mode="rgb_array")
    video_recorder = VideoRecorder(str(output_path), fps=15)

    # For continual DQN agents, make sure the correct head is active
    if hasattr(agent, "set_task"):
        try:
            agent.set_task(game)
        except Exception:
            pass

    with torch.no_grad():
        state = env.reset()
        for _ in range(max_steps):
            frame = env.env.render()
            if frame is not None:
                video_recorder.add_frame(frame)

            action = agent.select_action(state)
            state, _, done = env.step(action)
            if done:
                break

    video_recorder.save(format="mp4")
    env.close()


def evaluate_single(model_path: str, game: str, algorithm: str, episodes: int, max_steps: int):
    """Evaluate a single-task agent checkpoint on one game.

    In addition to printing statistics (and optional JSON via CLI), this will:
    - Save a reward curve plot under ``outputs/evaluate``
    - Save one example gameplay video under ``outputs/evaluate``
    """
    env = AtariEnv(game, render_mode=None)

    if algorithm == "dqn":
        agent = DQNAgent(state_dim=4, action_dim=env.action_space)
    else:
        agent = PPOAgent(state_dim=4, action_dim=env.action_space)

    agent.load(model_path)
    agent.network.eval()

    episode_rewards = []
    with torch.no_grad():
        for _ in range(episodes):
            state = env.reset()
            done = False
            ep_r = 0.0
            steps = 0
            while not done and steps < max_steps:
                action = agent.select_action(state)
                state, reward, done = env.step(action)
                ep_r += reward
                steps += 1
            episode_rewards.append(ep_r)

    env.close()

    avg_r = sum(episode_rewards) / len(episode_rewards)
    print(f"[Single] {game} ({algorithm}) - Episodes: {episodes}")
    print(f"  Avg reward: {avg_r:.2f}, Min: {min(episode_rewards):.2f}, Max: {max(episode_rewards):.2f}")

    # Pack result and generate visualization artifacts
    result = {
        "mode": "single",
        "game": game,
        "algorithm": algorithm,
        "episodes": episodes,
        "rewards": episode_rewards,
        "avg_reward": avg_r,
    }

    # 1) Reward curve plot
    plot_path = _plot_single_results(result)
    print(f"  Reward plot saved to: {plot_path}")

    # 2) Example gameplay video
    video_path = _get_eval_dir() / f"{game}_{algorithm}_eval_gameplay.mp4"
    _record_example_video(agent, game, algorithm, video_path, max_steps=max_steps)
    print(f"  Example gameplay video saved to: {video_path}")

    return result


def _build_continual_agent(algorithm: str, games: list, use_ewc: bool, ewc_lambda: float):
    """Rebuild a continual agent architecture for evaluation.

    For DQN, we use MultiHeadDQNAgent with one head per game.
    For PPO, a single-head PPOAgent is reused across games.
    """
    if algorithm == "dqn":
        base_agent = MultiHeadDQNAgent(state_dim=4)
        # Dummy envs just to get action dims
        for game in games:
            env = AtariEnv(game, render_mode=None)
            base_agent.register_task(game, env.action_space)
            env.close()
    else:
        # PPO: assume same action space for all games in list; use first game's
        env0 = AtariEnv(games[0], render_mode=None)
        base_agent = PPOAgent(state_dim=4, action_dim=env0.action_space)
        env0.close()

    if use_ewc:
        agent = EWCWrapper(base_agent, ewc_lambda=ewc_lambda)
    else:
        agent = base_agent

    return agent


def evaluate_continual(model_path: str, games: list, algorithm: str, episodes: int, max_steps: int, use_ewc: bool, ewc_lambda: float):
    """Evaluate a continual-learning agent on a list of games.

    In addition to printing statistics (and optional JSON via CLI), this will:
    - Save a bar plot of average reward per game under the experiment's eval dir
    - Save one example gameplay video **per game** under the same directory
    """
    print(f"Loading continual agent from {model_path}...")
    agent = _build_continual_agent(algorithm, games, use_ewc=use_ewc, ewc_lambda=ewc_lambda)
    agent.load(model_path)
    if hasattr(agent, "network") and agent.network is not None:
        agent.network.eval()

    results = {"mode": "continual", "algorithm": algorithm, "games": {}, "episodes": episodes}

    with torch.no_grad():
        for game in games:
            print(f"\n[Continual] Evaluating on {game} ...")
            env = AtariEnv(game, render_mode=None)

            # For MultiHeadDQN, select the appropriate head
            if hasattr(agent, "set_task"):
                try:
                    agent.set_task(game)
                except Exception:
                    pass

            episode_rewards = []
            for _ in range(episodes):
                state = env.reset()
                done = False
                ep_r = 0.0
                steps = 0
                while not done and steps < max_steps:
                    action = agent.select_action(state)
                    state, reward, done = env.step(action)
                    ep_r += reward
                    steps += 1
                episode_rewards.append(ep_r)

            env.close()

            avg_r = sum(episode_rewards) / len(episode_rewards)
            print(f"  Avg reward: {avg_r:.2f}, Min: {min(episode_rewards):.2f}, Max: {max(episode_rewards):.2f}")

            results["games"][game] = {
                "rewards": episode_rewards,
                "avg_reward": avg_r,
            }

    # 1) Bar plot of average reward per game
    plot_path = _plot_continual_results(results)
    print(f"\n[Continual] Avg reward plot saved to: {plot_path}")

    # 2) Example gameplay video for **each** game
    for game in games:
        video_path = _get_eval_dir() / f"continual_{algorithm}_{game}_eval_gameplay.mp4"
        _record_example_video(agent, game, algorithm, video_path, max_steps=max_steps)
        print(f"[Continual] Example gameplay video for {game} saved to: {video_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained Atari agents")
    parser.add_argument("--mode", choices=["single", "continual"], default="single")
    parser.add_argument("--model", required=True, help="Path to model checkpoint")
    parser.add_argument("--algorithm", choices=["dqn", "ppo"], default="dqn")
    parser.add_argument("--game", help="Game name for single mode (e.g., Pong-v5)")
    parser.add_argument("--games", nargs="*", help="List of games for continual mode")
    parser.add_argument("--episodes", type=int, default=5, help="Episodes per game")
    parser.add_argument("--max-steps", type=int, default=10000, help="Max steps per episode")
    parser.add_argument("--ewc", action="store_true", help="Use EWC wrapper for continual DQN/PPO")
    parser.add_argument("--ewc-lambda", type=float, default=0.4, help="EWC lambda (if --ewc)")
    parser.add_argument("--json-out", help="Optional path to write JSON results")

    args = parser.parse_args()

    # Configure the eval directory based on --json-out (if provided)
    _set_eval_dir_from_json_out(args.json_out)

    if args.mode == "single":
        if not args.game:
            raise SystemExit("--game is required for single mode")
        result = evaluate_single(
            model_path=args.model,
            game=args.game,
            algorithm=args.algorithm,
            episodes=args.episodes,
            max_steps=args.max_steps,
        )
    else:
        games = args.games
        if not games:
            raise SystemExit("--games is required for continual mode")
        result = evaluate_continual(
            model_path=args.model,
            games=games,
            algorithm=args.algorithm,
            episodes=args.episodes,
            max_steps=args.max_steps,
            use_ewc=args.ewc,
            ewc_lambda=args.ewc_lambda,
        )

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Results written to {args.json_out}")

