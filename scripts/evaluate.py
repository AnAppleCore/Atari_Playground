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
from algorithms import DQNAgent, PPOAgent, EWCWrapper, MultiHeadDQNAgent, MultiHeadPPOAgent
from utils import VideoRecorder

EVAL_DIR = None


def _infer_eval_dir_from_model_path(model_path: str, mode: str) -> Path:
    """Infer evaluation directory from model path to match training structure."""
    model_path = Path(model_path)
    if mode == "single":
        checkpoint_name = model_path.stem
        eval_dir = Path("outputs") / "single" / checkpoint_name / "eval"
    else:
        checkpoint_name = model_path.stem
        eval_dir = Path("outputs") / "continual" / checkpoint_name / "eval"
    return eval_dir


def _set_eval_dir_from_json_out(json_out: Optional[str], model_path: str, mode: str) -> None:
    """Configure EVAL_DIR from json_out or infer from model path."""
    global EVAL_DIR
    if json_out:
        EVAL_DIR = Path(json_out).parent
    else:
        EVAL_DIR = _infer_eval_dir_from_model_path(model_path, mode)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)


def _get_eval_dir() -> Path:
    """Ensure and return the evaluation output directory."""
    if EVAL_DIR is None:
        raise RuntimeError("EVAL_DIR not set. Call _set_eval_dir_from_json_out first.")
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    return EVAL_DIR


def _plot_single_results(result: dict) -> Path:
    """Plot episode rewards for a single-task evaluation."""
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
    """Record a single-episode gameplay video."""
    env = AtariEnv(game, render_mode="rgb_array", use_skip=False)
    video_recorder = VideoRecorder(str(output_path), fps=30)

    if hasattr(agent, "set_task"):
        try:
            agent.set_task(game)
        except Exception:
            pass

    # Find EpisodicLifeEnv wrapper to check was_real_done
    episodic_life_wrapper = None
    wrapper = env.env
    while wrapper is not None:
        if hasattr(wrapper, 'was_real_done'):
            episodic_life_wrapper = wrapper
            break
        if hasattr(wrapper, 'env'):
            wrapper = wrapper.env
        else:
            break

    with torch.no_grad():
        state = env.reset()
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

            action = agent.select_action(state)
            state, _, done = env.step(action)
            
            # Check if this is a real game over (not just loss of life)
            # If EpisodicLifeEnv wrapper exists, check was_real_done
            # Otherwise, check if lives are exhausted
            real_done = done
            if episodic_life_wrapper is not None:
                real_done = episodic_life_wrapper.was_real_done
            elif hasattr(env.env, 'unwrapped') and hasattr(env.env.unwrapped, 'ale'):
                try:
                    lives = env.env.unwrapped.ale.lives()
                    real_done = done and lives == 0
                except:
                    real_done = done
            
            if real_done:
                break

    video_recorder.save(format="mp4")
    env.close()


def evaluate_single(model_path: str, game: str, algorithm: str, episodes: int, max_steps: int):
    """Evaluate a single-task agent checkpoint on one game."""
    env = AtariEnv(game, render_mode=None)

    if algorithm == "dqn":
        agent = DQNAgent(state_dim=4, action_dim=env.action_space)
    else:
        agent = PPOAgent(state_dim=4, action_dim=env.action_space)

    agent.load(model_path)
    if hasattr(agent, 'network'):
        agent.network.eval()
    if hasattr(agent, 'actor'):
        agent.actor.eval()
    if hasattr(agent, 'critic'):
        agent.critic.eval()

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

    result = {
        "mode": "single",
        "game": game,
        "algorithm": algorithm,
        "episodes": episodes,
        "rewards": episode_rewards,
        "avg_reward": avg_r,
    }

    plot_path = _plot_single_results(result)
    print(f"  Reward plot saved to: {plot_path}")

    video_path = _get_eval_dir() / f"{game}_{algorithm}_eval_gameplay.mp4"
    _record_example_video(agent, game, algorithm, video_path, max_steps=max_steps)
    print(f"  Example gameplay video saved to: {video_path}")

    return result


def _build_continual_agent(algorithm: str, games: list, use_ewc: bool, ewc_lambda: float):
    """Rebuild a continual agent architecture for evaluation."""
    if algorithm == "dqn":
        base_agent = MultiHeadDQNAgent(state_dim=4)
        for game in games:
            env = AtariEnv(game, render_mode=None)
            base_agent.register_task(game, env.action_space)
            env.close()
    else:  # ppo
        base_agent = MultiHeadPPOAgent(state_dim=4)
        for game in games:
            env = AtariEnv(game, render_mode=None)
            base_agent.register_task(game, env.action_space)
            env.close()

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
    if hasattr(agent, 'actor'):
        agent.actor.eval()
    if hasattr(agent, 'critic'):
        agent.critic.eval()

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

    plot_path = _plot_continual_results(results)
    print(f"\n[Continual] Avg reward plot saved to: {plot_path}")

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

    _set_eval_dir_from_json_out(args.json_out, args.model, args.mode)

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

