import os
import argparse
import torch

from PPO import PPO, evaluate_ppo # evaluate_ppo handles conditional visualization import
from configs import CONFIGS, DefaultConfig
# from visualization import reset_trajectories # Moved inside evaluate_ppo

def run_experiment(config_name: str, model_path: str, num_episodes: int, max_steps: int, render: bool):
    """
    Load a trained PPO model and run an evaluation experiment for oil spill mapping.
    """
    if config_name not in CONFIGS:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(CONFIGS.keys())}")

    config: DefaultConfig = CONFIGS[config_name]
    print(f"Using configuration: '{config_name}' for PPO experiment")

    # --- Ensure Algorithm is PPO in Config ---
    if config.algorithm != 'ppo':
        print(f"Warning: Config '{config_name}' has algorithm '{config.algorithm}'. Forcing to 'ppo' for this script.")
        config.algorithm = 'ppo' # Updates model paths etc.

    # Override evaluation parameters if provided
    if num_episodes is not None: config.evaluation.num_episodes = num_episodes; print(f"Overriding num_episodes: {num_episodes}")
    if max_steps is not None: config.evaluation.max_steps = max_steps; print(f"Overriding max_steps: {max_steps}")
    if render is not None: config.evaluation.render = render; print(f"Setting render: {render}")

    device = torch.device(config.cuda_device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Instantiate Agent ---
    # PPO needs ppo_config and device
    agent = PPO(config=config.ppo, device=device)

    # --- Load Model ---
    models_dir = config.training.models_dir
    if not os.path.isabs(model_path) and not os.path.exists(model_path):
        check_path = os.path.join(models_dir, model_path)
        if os.path.exists(check_path):
             model_path = check_path
             print(f"Found model in default directory: {model_path}")
        else:
             raise FileNotFoundError(f"Model file not found: {model_path} or {check_path}")
    elif not os.path.exists(model_path):
         raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"Loading PPO model from {model_path}...")
    agent.load_model(model_path)

    print(f"\nRunning experiment with PPO model {os.path.basename(model_path)}...")
    # evaluate_ppo handles rendering internally based on config
    evaluate_ppo(agent=agent, config=config)

    print(f"\nPPO Experiment complete.")
    if config.evaluation.render:
         print(f"Visualizations potentially saved to '{config.visualization.save_dir}' directory (if libraries were available).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation experiment with a trained PPO model for oil spill mapping.")
    parser.add_argument(
        "--config", "-c", type=str, default="ppo_mapping", # Default PPO mapping config
        help=f"Configuration name. Available: {list(CONFIGS.keys())}"
    )
    parser.add_argument(
        "--model", "-m", type=str, default="ppo_mapping_final.pt", # Default mapping filename
        help="Path to trained PPO model checkpoint (relative to config's models_dir or absolute)."
    )
    parser.add_argument(
        "--episodes", "-e", type=int, default=None, help="Number of episodes (overrides config)."
    )
    parser.add_argument(
        "--steps", "-s", type=int, default=None, help="Max steps per episode (overrides config)."
    )
    render_group = parser.add_mutually_exclusive_group()
    render_group.add_argument(
        "--render", action="store_true", default=None, help="Enable rendering (overrides config)."
    )
    render_group.add_argument(
        "--no-render", dest="render", action="store_false", help="Disable rendering (overrides config)."
    )

    args = parser.parse_args()

    run_experiment(
        config_name=args.config,
        model_path=args.model,
        num_episodes=args.episodes,
        max_steps=args.steps,
        render=args.render
    )