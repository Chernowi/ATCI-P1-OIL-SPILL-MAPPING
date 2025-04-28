import argparse
import os
import torch

# Import training functions for available algorithms
from SAC import train_sac, evaluate_sac
from PPO import train_ppo, evaluate_ppo
from TSAC import train_tsac, evaluate_tsac

from configs import CONFIGS, DefaultConfig

# Note: Visualization imports are now conditional within evaluate_* functions

def main(config_name: str, cuda_device: str = None, algorithm: str = None, run_evaluation: bool = True):
    """Main function to train and optionally evaluate the oil spill mapping RL agent."""
    if config_name not in CONFIGS:
        raise ValueError(f"Unknown configuration name: {config_name}. Available: {list(CONFIGS.keys())}")

    config: DefaultConfig = CONFIGS[config_name]
    print(f"Using configuration: '{config_name}'")

    # --- Override settings from command line ---
    if cuda_device:
        config.cuda_device = cuda_device
        print(f"Overriding CUDA device: {cuda_device}")

    # Determine effective algorithm (CLI > config file > default)
    effective_algorithm = algorithm if algorithm else config.algorithm
    if algorithm and algorithm != config.algorithm:
         print(f"Overriding config algorithm '{config.algorithm}' with command line argument: '{algorithm}'")
         config.algorithm = algorithm # This triggers model_post_setattr to update paths
    elif not algorithm and config_name == "default_mapping":
         print(f"Using algorithm specified in '{config_name}' config: '{config.algorithm}'")
    else:
         print(f"Using algorithm: '{effective_algorithm}'")
         # Ensure config reflects the effective algorithm if it differs initially
         if config.algorithm != effective_algorithm:
              config.algorithm = effective_algorithm


    use_multi_gpu = torch.cuda.is_available() and torch.cuda.device_count() > 1
    if use_multi_gpu:
         print(f"Detected {torch.cuda.device_count()} GPUs. Note: Multi-GPU support varies by algorithm.")

    # Use the potentially updated models_dir from the config
    model_dir = config.training.models_dir
    os.makedirs(model_dir, exist_ok=True)
    print(f"Models will be saved in: {os.path.abspath(model_dir)}")

    agent = None
    episode_rewards = []
    final_model_path = None

    # --- Training Phase ---
    if effective_algorithm.lower() == "ppo":
        print("Training PPO agent...")
        # Pass run_evaluation=False to separate training and evaluation phases
        agent, episode_rewards = train_ppo(config=config, use_multi_gpu=use_multi_gpu, run_evaluation=False)
        final_model_path = os.path.join(model_dir, f"ppo_mapping_final_ep{config.training.num_episodes}.pt")
        if agent: agent.save_model(final_model_path)
        print(f"Final PPO model saved to {final_model_path}")

    elif effective_algorithm.lower() == "tsac":
        print("Training T-SAC agent...")
        agent, episode_rewards = train_tsac(config=config, use_multi_gpu=use_multi_gpu, run_evaluation=False)
        final_model_path = os.path.join(model_dir, f"tsac_mapping_final_ep{config.training.num_episodes}.pt")
        if agent: agent.save_model(final_model_path)
        print(f"Final T-SAC model saved to {final_model_path}")

    elif effective_algorithm.lower() == "sac":
        print("Training SAC agent...")
        agent, episode_rewards = train_sac(config=config, use_multi_gpu=use_multi_gpu, run_evaluation=False)
        final_model_path = os.path.join(model_dir, f"sac_mapping_final_ep{config.training.num_episodes}.pt")
        if agent: agent.save_model(final_model_path)
        print(f"Final SAC model saved to {final_model_path}")

    else:
        raise ValueError(f"Unknown algorithm specified: {effective_algorithm}. Choose 'sac', 'ppo', or 'tsac'.")

    # --- Evaluation Phase (Optional) ---
    if run_evaluation and agent is not None:
        print(f"\nEvaluating {effective_algorithm.upper()} agent...")
        # The evaluate_* functions handle conditional visualization imports
        if effective_algorithm.lower() == "ppo":
            evaluate_ppo(agent=agent, config=config)
        elif effective_algorithm.lower() == "tsac":
            evaluate_tsac(agent=agent, config=config)
        elif effective_algorithm.lower() == "sac":
            evaluate_sac(agent=agent, config=config)
    elif not run_evaluation:
         print("\nSkipping evaluation phase.")

    print(f"\nTraining {'and evaluation ' if run_evaluation else ''}complete.")
    if run_evaluation and config.evaluation.render:
         print(f"Find potential visualizations in the '{config.visualization.save_dir}' directory (if libraries were available).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate RL agent for oil spill mapping.")
    parser.add_argument(
        "--config", "-c", type=str, default="default_mapping", # Updated default
        help=f"Configuration name to use. Available: {list(CONFIGS.keys())}"
    )
    parser.add_argument(
        "--device", "-d", type=str, default=None,
        help="CUDA device to use (e.g., 'cuda:0', 'cuda:1', 'cpu')"
    )
    parser.add_argument(
        "--algorithm", "-a", type=str, default=None,
        choices=["sac", "ppo", "tsac"],
        help="RL algorithm to use ('sac', 'ppo', 'tsac'). Overrides config."
    )
    # Add option to skip evaluation
    eval_group = parser.add_mutually_exclusive_group()
    eval_group.add_argument(
        "--evaluate", action="store_true", default=True,
        help="Run evaluation after training (default)."
    )
    eval_group.add_argument(
        "--no-evaluate", dest="evaluate", action="store_false",
        help="Skip evaluation after training."
    )

    args = parser.parse_args()
    main(config_name=args.config, cuda_device=args.device, algorithm=args.algorithm, run_evaluation=args.evaluate)