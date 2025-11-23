"""Visualize training results from experiments."""
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))


def plot_training_curves(results_dir: str, output_path: str = "outputs/training_curves.png"):
    """
    Plot training curves from experiment results.
    
    Args:
        results_dir: Directory containing experiment results
        output_path: Path to save the plot
    """
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"Results directory not found: {results_dir}")
        return
    
    # Find all metrics files
    metrics_files = list(results_path.glob("**/metrics.json"))
    
    if not metrics_files:
        print(f"No metrics.json files found in {results_dir}")
        return
    
    print(f"Found {len(metrics_files)} experiment(s)")
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Training Results", fontsize=16)
    
    for metrics_file in metrics_files:
        with open(metrics_file, 'r') as f:
            data = json.load(f)
        
        exp_name = metrics_file.parent.name
        
        # Plot 1: Episode Rewards
        if 'episode_rewards' in data:
            axes[0, 0].plot(data['episode_rewards'], label=exp_name, alpha=0.7)
            axes[0, 0].set_title("Episode Rewards")
            axes[0, 0].set_xlabel("Episode")
            axes[0, 0].set_ylabel("Reward")
            axes[0, 0].legend()
            axes[0, 0].grid(True)
        
        # Plot 2: Loss
        if 'losses' in data:
            axes[0, 1].plot(data['losses'], label=exp_name, alpha=0.7)
            axes[0, 1].set_title("Training Loss")
            axes[0, 1].set_xlabel("Step")
            axes[0, 1].set_ylabel("Loss")
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # Plot 3: Epsilon (for DQN)
        if 'epsilon' in data:
            axes[1, 0].plot(data['epsilon'], label=exp_name, alpha=0.7)
            axes[1, 0].set_title("Exploration Rate (Epsilon)")
            axes[1, 0].set_xlabel("Step")
            axes[1, 0].set_ylabel("Epsilon")
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Plot 4: EWC Loss (if available)
        if 'ewc_loss' in data:
            axes[1, 1].plot(data['ewc_loss'], label=exp_name, alpha=0.7)
            axes[1, 1].set_title("EWC Regularization Loss")
            axes[1, 1].set_xlabel("Step")
            axes[1, 1].set_ylabel("EWC Loss")
            axes[1, 1].legend()
            axes[1, 1].grid(True)
    
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to {output_path}")
    plt.close()


def compare_algorithms(results_dir: str, output_path: str = "outputs/algorithm_comparison.png"):
    """
    Compare different algorithms on the same game.
    
    Args:
        results_dir: Directory containing experiment results
        output_path: Path to save the comparison plot
    """
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"Results directory not found: {results_dir}")
        return
    
    # Group results by game
    games_data = {}
    
    for metrics_file in results_path.glob("**/metrics.json"):
        with open(metrics_file, 'r') as f:
            data = json.load(f)
        
        exp_name = metrics_file.parent.name
        # Parse experiment name: game_algorithm
        parts = exp_name.rsplit('_', 1)
        if len(parts) == 2:
            game, algo = parts
            if game not in games_data:
                games_data[game] = {}
            games_data[game][algo] = data
    
    # Create comparison plots
    num_games = len(games_data)
    fig, axes = plt.subplots(num_games, 2, figsize=(14, 5 * num_games))
    
    if num_games == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle("Algorithm Comparison", fontsize=16)
    
    for idx, (game, algos) in enumerate(games_data.items()):
        # Plot rewards
        for algo, data in algos.items():
            if 'episode_rewards' in data:
                axes[idx, 0].plot(data['episode_rewards'], label=algo, alpha=0.7)
        
        axes[idx, 0].set_title(f"{game} - Episode Rewards")
        axes[idx, 0].set_xlabel("Episode")
        axes[idx, 0].set_ylabel("Reward")
        axes[idx, 0].legend()
        axes[idx, 0].grid(True)
        
        # Plot loss
        for algo, data in algos.items():
            if 'losses' in data:
                axes[idx, 1].plot(data['losses'], label=algo, alpha=0.7)
        
        axes[idx, 1].set_title(f"{game} - Training Loss")
        axes[idx, 1].set_xlabel("Step")
        axes[idx, 1].set_ylabel("Loss")
        axes[idx, 1].legend()
        axes[idx, 1].grid(True)
    
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"Comparison plot saved to {output_path}")
    plt.close()


def compare_ewc_effect(results_dir: str, output_path: str = "outputs/ewc_comparison.png"):
    """
    Compare training with and without EWC.
    
    Args:
        results_dir: Directory containing experiment results
        output_path: Path to save the comparison plot
    """
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"Results directory not found: {results_dir}")
        return
    
    # Find experiments with and without EWC
    with_ewc = {}
    without_ewc = {}
    
    for metrics_file in results_path.glob("**/metrics.json"):
        with open(metrics_file, 'r') as f:
            data = json.load(f)
        
        exp_name = metrics_file.parent.name
        
        if 'ewc' in exp_name.lower():
            with_ewc[exp_name] = data
        else:
            without_ewc[exp_name] = data
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("EWC Effect on Continual Learning", fontsize=16)
    
    # Plot without EWC
    for exp_name, data in without_ewc.items():
        if 'episode_rewards' in data:
            axes[0].plot(data['episode_rewards'], label=exp_name, alpha=0.7)
    
    axes[0].set_title("Without EWC (Catastrophic Forgetting)")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Reward")
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot with EWC
    for exp_name, data in with_ewc.items():
        if 'episode_rewards' in data:
            axes[1].plot(data['episode_rewards'], label=exp_name, alpha=0.7)
    
    axes[1].set_title("With EWC (Mitigated Forgetting)")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Reward")
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"EWC comparison plot saved to {output_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize training results")
    parser.add_argument("--results-dir", default="results", help="Directory with experiment results")
    parser.add_argument("--type", choices=["curves", "comparison", "ewc"], default="curves",
                       help="Type of visualization")
    parser.add_argument("--output", help="Output path for the plot")
    
    args = parser.parse_args()
    
    if args.type == "curves":
        output = args.output or "outputs/training_curves.png"
        plot_training_curves(args.results_dir, output)
    elif args.type == "comparison":
        output = args.output or "outputs/algorithm_comparison.png"
        compare_algorithms(args.results_dir, output)
    elif args.type == "ewc":
        output = args.output or "outputs/ewc_comparison.png"
        compare_ewc_effect(args.results_dir, output)

