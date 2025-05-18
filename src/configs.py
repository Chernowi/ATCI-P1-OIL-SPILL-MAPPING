from typing import Dict, Literal, Tuple, List, Any, Optional
from pydantic import BaseModel, Field
import math
import numpy as np

# Core dimensions for Oil Spill Mapping (Module-level constants)
CORE_STATE_DIM = 8
CORE_ACTION_DIM = 1
TRAJECTORY_REWARD_DIM = 1

# Pre-calculate the default trajectory feature dimension using the global constants
DEFAULT_TRAJECTORY_FEATURE_DIM = CORE_STATE_DIM + CORE_ACTION_DIM + TRAJECTORY_REWARD_DIM

# --- SAC / PPO Configs ---
class SACConfig(BaseModel):
    """Configuration for the SAC agent"""
    state_dim: int = Field(CORE_STATE_DIM, description="Dimension of the basic state tuple (sensors + norm_coords + norm_heading)") # Use global
    action_dim: int = Field(CORE_ACTION_DIM, description="Action dimension (yaw_change)") # Use global
    hidden_dims: List[int] = Field([128, 128], description="List of hidden layer dimensions for MLP part")
    log_std_min: int = Field(-20, description="Minimum log std for action distribution")
    log_std_max: int = Field(1, description="Maximum log std for action distribution")
    actor_lr: float = Field(5e-5, description="Actor learning rate")
    critic_lr: float = Field(5e-5, description="Critic learning rate")
    gamma: float = Field(0.99, description="Discount factor")
    tau: float = Field(0.005, description="Target network update rate")
    alpha: float = Field(0.2, description="Temperature parameter (Initial value if auto-tuning)")
    auto_tune_alpha: bool = Field(True, description="Whether to auto-tune the alpha parameter")
    use_rnn: bool = Field(False, description="Whether to use RNN layers in Actor/Critic")
    rnn_type: Literal['lstm', 'gru'] = Field('lstm', description="Type of RNN cell (Only used if use_rnn is True)")
    rnn_hidden_size: int = Field(68, description="Hidden size of RNN layers (Only used if use_rnn is True)")
    rnn_num_layers: int = Field(1, description="Number of RNN layers (Only used if use_rnn is True)")
    use_state_normalization: bool = Field(False, description="Enable/disable state normalization using RunningMeanStd")
    use_reward_normalization: bool = Field(True, description="Enable/disable reward normalization by batch std dev")
    use_per: bool = Field(False, description="Enable Prioritized Experience Replay")
    per_alpha: float = Field(0.6, description="PER alpha (prioritization exponent)")
    per_beta_start: float = Field(0.4, description="PER beta initial value (importance sampling exponent)")
    per_beta_frames: int = Field(100000, description="PER beta annealing frames")
    per_epsilon: float = Field(1e-5, description="PER epsilon (small value added to priorities)")

class PPOConfig(BaseModel):
    """Configuration for the PPO agent"""
    state_dim: int = Field(CORE_STATE_DIM, description="Dimension of the basic state tuple (sensors + norm_coords + norm_heading)") # Use global
    action_dim: int = Field(CORE_ACTION_DIM, description="Action dimension (yaw_change)") # Use global
    hidden_dim: int = Field(256, description="Hidden layer dimension (for MLP part)")
    log_std_min: int = Field(-20, description="Minimum log std for action distribution")
    log_std_max: int = Field(1, description="Maximum log std for action distribution")
    actor_lr: float = Field(5e-5, description="Actor learning rate")
    critic_lr: float = Field(5e-5, description="Critic learning rate")
    gamma: float = Field(0.99, description="Discount factor")
    gae_lambda: float = Field(0.95, description="GAE lambda parameter")
    policy_clip: float = Field(0.2, description="PPO clipping parameter")
    n_epochs: int = Field(10, description="Number of optimization epochs per update")
    entropy_coef: float = Field(0.25, description="Entropy coefficient for exploration")
    value_coef: float = Field(0.5, description="Value loss coefficient")
    batch_size: int = Field(64, description="Batch size for training (rollouts for RNN PPO)")
    steps_per_update: int = Field(256, description="Environment steps between PPO updates (rollout length for RNN PPO)")
    use_state_normalization: bool = Field(False, description="Enable/disable state normalization")
    use_reward_normalization: bool = Field(False, description="Enable/disable reward normalization")
    use_rnn: bool = Field(False, description="Whether to use RNN layers in Actor/Critic")
    rnn_type: Literal['lstm', 'gru'] = Field('gru', description="Type of RNN cell (Only used if use_rnn is True)")
    rnn_hidden_size: int = Field(64, description="Hidden size of RNN layers (Only used if use_rnn is True)")
    rnn_num_layers: int = Field(1, description="Number of RNN layers (Only used if use_rnn is True)")

# --- Replay Buffer Config ---
class ReplayBufferConfig(BaseModel):
    capacity: int = Field(3000000, description="Maximum capacity of replay buffer")
    gamma: float = Field(0.99, description="Discount factor for returns")

# --- Mapper Config ---
class MapperConfig(BaseModel):
    min_oil_points_for_estimate: int = Field(3, description="Minimum oil-detecting sensor locations for Convex Hull")

# --- Training Config ---
class TrainingConfig(BaseModel):
    num_episodes: int = Field(30000, description="Number of episodes to train")
    max_steps: int = Field(350, description="Maximum steps per episode")
    batch_size: int = Field(512, description="Batch size for training")
    save_interval: int = Field(200, description="Interval (episodes) for saving models")
    log_frequency: int = Field(10, description="Frequency (episodes) for TensorBoard logging")
    learning_starts: int = Field(8000, description="Steps before SAC training updates start")
    train_freq: int = Field(4, description="Update policy every n env steps (SAC)")
    gradient_steps: int = Field(1, description="Gradient steps per training frequency (SAC)")
    enable_early_stopping: bool = Field(False, description="Enable early stopping")
    early_stopping_threshold: float = Field(50, description="Avg reward threshold for early stopping")
    early_stopping_window: int = Field(50, description="Window for avg reward in early stopping")

# --- Evaluation Config ---
class EvaluationConfig(BaseModel):
    num_episodes: int = Field(6, description="Number of episodes for evaluation")
    max_steps: int = Field(200, description="Maximum steps per evaluation episode")
    render: bool = Field(True, description="Whether to render evaluation")
    use_stochastic_policy_eval: bool = Field(False, description="Use stochastic policy for eval")

# --- Pos/Vel/Randomization ---
class Position(BaseModel):
    x: float = 0.0
    y: float = 0.0

class Velocity(BaseModel):
    x: float = 0.0
    y: float = 0.0

class RandomizationRange(BaseModel):
    x_range: Tuple[float, float] = Field((10.0, 90.0), description="Min/Max X for randomization (unnormalized)")
    y_range: Tuple[float, float] = Field((10.0, 90.0), description="Min/Max Y for randomization (unnormalized)")

# --- Visualization Config ---
class VisualizationConfig(BaseModel):
    save_dir: str = Field("mapping_snapshots", description="Directory for saving visualizations")
    figure_size: tuple = Field((10, 10), description="Figure size for visualizations")
    max_trajectory_points: int = Field(5, description="Max trajectory points to display")
    output_format: Literal['gif', 'mp4'] = Field('gif', description="Output format for rendered episodes")
    video_fps: int = Field(15, description="FPS for video/GIF.")
    delete_png_frames: bool = Field(True, description="Delete PNG frames after GIF creation.")
    sensor_marker_size: int = Field(10, description="Marker size for sensors")
    sensor_color_oil: str = Field("red", description="Color for sensors detecting oil")
    sensor_color_water: str = Field("blue", description="Color for sensors detecting water")
    plot_oil_points: bool = Field(True, description="Whether to plot true oil points")
    plot_water_points: bool = Field(False, description="Whether to plot true water points")
    point_marker_size: int = Field(2, description="Marker size for oil/water points")

# --- World Config ---
class WorldConfig(BaseModel):
    """Configuration for the world"""
    # These fields allow overriding the global constants for a specific world if ever needed.
    # Their defaults come from the global constants.
    CORE_STATE_DIM: int = Field(CORE_STATE_DIM, description="Dimension of core state part")
    CORE_ACTION_DIM: int = Field(CORE_ACTION_DIM, description="Dimension of core action part")
    TRAJECTORY_REWARD_DIM: int = Field(TRAJECTORY_REWARD_DIM, description="Dimension of reward part in trajectory")

    dt: float = Field(1.0, description="Time step")
    world_size: Tuple[float, float] = Field((125.0, 125.0), description="Dimensions (X, Y) of the world")
    normalize_coords: bool = Field(True, description="Normalize agent coordinates in state")
    agent_speed: float = Field(3, description="Agent speed (unnormalized units/dt)")
    yaw_angle_range: Tuple[float, float] = Field((-math.pi / 6, math.pi / 6), description="Yaw angle change range/step")
    num_sensors: int = Field(5, description="Number of sensors")
    sensor_distance: float = Field(2.5, description="Sensor distance from agent (unnormalized)")
    sensor_radius: float = Field(4.0, description="Sensor detection radius (unnormalized)")
    agent_initial_location: Position = Field(default_factory=lambda: Position(x=50, y=10))
    randomize_agent_initial_location: bool = Field(True)
    agent_randomization_ranges: RandomizationRange = Field(default_factory=lambda: RandomizationRange(x_range=(25.0, 100.0), y_range=(25.0, 100.0)))
    num_oil_points: int = Field(200, description="Number of true oil spill points")
    num_water_points: int = Field(400, description="Number of non-spill area points")
    oil_cluster_std_dev_range: Tuple[float, float] = Field((8.0, 10.0), description="Std dev range for oil cluster")
    randomize_oil_cluster: bool = Field(True)
    oil_center_randomization_range: RandomizationRange = Field(default_factory=lambda: RandomizationRange(x_range=(25.0, 100.0), y_range=(25.0, 100.0)))
    initial_oil_center: Position = Field(default_factory=lambda: Position(x=50, y=50))
    initial_oil_std_dev: float = Field(10.0)
    min_initial_separation_distance: float = Field(40.0, description="Min dist: agent start & oil center")
    trajectory_length: int = Field(10, description="Steps (N) in trajectory state")
    
    # CORRECTED LINE: Use the pre-calculated module-level constant for the default value
    trajectory_feature_dim: int = Field(DEFAULT_TRAJECTORY_FEATURE_DIM, description="Dimension of features per step in trajectory state (normalized_state_incl_heading + prev_action + prev_reward)")
    
    max_steps: int = Field(350, description="Max steps per episode for world termination")
    success_metric_threshold: float = Field(0.95, description="Point inclusion % for success")
    terminate_on_success: bool = Field(True)
    terminate_out_of_bounds: bool = Field(True)
    metric_improvement_scale: float = Field(50.0, description="Scaling for metric improvement reward")
    step_penalty: float = Field(0, description="Penalty per step")
    new_oil_detection_bonus: float = Field(0.0, description="Bonus for new oil detection")
    out_of_bounds_penalty: float = Field(20.0, description="Penalty for going out of bounds")
    success_bonus: float = Field(55.0, description="Bonus for success")
    uninitialized_mapper_penalty: float = Field(0, description="Penalty if mapper uninitialized")
    mapper_config: MapperConfig = Field(default_factory=MapperConfig)
    seeds: List[int] = Field([], description="Seeds for env generation during eval/specific resets.")

class DefaultConfig(BaseModel):
    """Default configuration for the entire oil spill mapping application"""
    sac: SACConfig = Field(default_factory=SACConfig)
    ppo: PPOConfig = Field(default_factory=PPOConfig)
    replay_buffer: ReplayBufferConfig = Field(default_factory=ReplayBufferConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    world: WorldConfig = Field(default_factory=WorldConfig)
    mapper: MapperConfig = Field(default_factory=MapperConfig)
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)
    cuda_device: str = Field("cuda:0", description="CUDA device to use")
    algorithm: str = Field("sac", description="RL algorithm ('sac', 'ppo')")

    def model_post_init(self, __context: Any) -> None:
        self.world.mapper_config = self.mapper
        # Ensure agent configs use the world's effective core dimensions
        # These are instance values, so direct addition is fine here.
        self.sac.state_dim = self.world.CORE_STATE_DIM
        self.sac.action_dim = self.world.CORE_ACTION_DIM
        self.ppo.state_dim = self.world.CORE_STATE_DIM
        self.ppo.action_dim = self.world.CORE_ACTION_DIM

        # Ensure the world's trajectory_feature_dim is consistent with its own core dimensions
        # This will use the integer values from the world instance
        self.world.trajectory_feature_dim = self.world.CORE_STATE_DIM + \
                                            self.world.CORE_ACTION_DIM + \
                                            self.world.TRAJECTORY_REWARD_DIM
        # Also ensure DEFAULT_TRAJECTORY_FEATURE_DIM (used for Field default) is consistent
        # This is a bit redundant if model_post_init always runs, but safe.
        if self.world.trajectory_feature_dim != DEFAULT_TRAJECTORY_FEATURE_DIM:
             # This case should ideally not happen if WorldConfig.CORE_STATE_DIM etc.
             # always default to the  constants.
             # If they were somehow changed before model_post_init, this would log.
             print(f"Warning: Recalculated trajectory_feature_dim ({self.world.trajectory_feature_dim}) "
                   f"differs from initial default ({DEFAULT_TRAJECTORY_FEATURE_DIM}). "
                   f"This implies WorldConfig's CORE dimensions were modified post-init of WorldConfig "
                   f"but before DefaultConfig.model_post_init, or an inconsistency exists.")


# --- Define Base Default Configurations ---
default_sac_mlp_config = DefaultConfig()
default_sac_mlp_config.algorithm = "sac"
default_sac_mlp_config.sac.use_rnn = False

default_ppo_mlp_config = DefaultConfig()
default_ppo_mlp_config.algorithm = "ppo"
default_ppo_mlp_config.ppo.use_rnn = False

default_sac_rnn_config = DefaultConfig()
default_sac_rnn_config.algorithm = "sac"
default_sac_rnn_config.sac.use_rnn = True

default_ppo_rnn_config = DefaultConfig()
default_ppo_rnn_config.algorithm = "ppo"
default_ppo_rnn_config.sac.use_rnn = True # This was likely a typo, should be ppo.use_rnn
# Correcting the above potential typo for default_ppo_rnn_config
default_ppo_rnn_config.ppo.use_rnn = True # Corrected: PPO config should modify ppo.use_rnn
default_ppo_rnn_config.sac.use_rnn = False # And ensure SAC RNN is false for this PPO specific config


# Initialize CONFIGS dictionary
CONFIGS: Dict[str, DefaultConfig] = {
    "default_sac_mlp": default_sac_mlp_config,
    "default_ppo_mlp": default_ppo_mlp_config,
    "default_sac_rnn": default_sac_rnn_config,
    "default_ppo_rnn": default_ppo_rnn_config
}

# --- SAC MLP Hyperparameter Variations ---
sac_mlp_variations_list = [
    ("actor_lr_low", "sac.actor_lr", 1e-5), ("actor_lr_high", "sac.actor_lr", 1e-4),
    ("critic_lr_low", "sac.critic_lr", 1e-5), ("critic_lr_high", "sac.critic_lr", 1e-4),
    ("gamma_low", "sac.gamma", 0.95), ("gamma_high", "sac.gamma", 0.999),
    ("tau_low", "sac.tau", 0.001), ("tau_high", "sac.tau", 0.01),
    ("hidden_dims_small", "sac.hidden_dims", [64, 64]),
    ("hidden_dims_large", "sac.hidden_dims", [256, 256]),
]
for name_suffix, param_path, value in sac_mlp_variations_list:
    config_name = f"sac_mlp_{name_suffix}"; new_config = default_sac_mlp_config.model_copy(deep=True)
    parts = param_path.split("."); attr = new_config
    for part in parts[:-1]: attr = getattr(attr, part)
    setattr(attr, parts[-1], value); CONFIGS[config_name] = new_config

# --- PPO MLP Hyperparameter Variations ---
ppo_mlp_variations_list = [
    ("actor_lr_low", "ppo.actor_lr", 1e-5), ("actor_lr_high", "ppo.actor_lr", 1e-4),
    ("gae_lambda_low", "ppo.gae_lambda", 0.90), ("gae_lambda_high", "ppo.gae_lambda", 0.99),
    ("policy_clip_low", "ppo.policy_clip", 0.1), ("policy_clip_high", "ppo.policy_clip", 0.3),
    ("entropy_coef_low", "ppo.entropy_coef", 0.005), ("entropy_coef_high", "ppo.entropy_coef", 0.5),
    ("hidden_dim_small", "ppo.hidden_dim", 128), ("hidden_dim_large", "ppo.hidden_dim", 512),
]
for name_suffix, param_path, value in ppo_mlp_variations_list:
    config_name = f"ppo_mlp_{name_suffix}"; new_config = default_ppo_mlp_config.model_copy(deep=True)
    parts = param_path.split("."); attr = new_config
    for part in parts[:-1]: attr = getattr(attr, part)
    setattr(attr, parts[-1], value); CONFIGS[config_name] = new_config

# --- SAC RNN Hyperparameter Variations ---
sac_rnn_variations_list = [
    ("rnn_hidden_size_small", "sac.rnn_hidden_size", 32),
    ("rnn_hidden_size_big", "sac.rnn_hidden_size", 128),
]
for name_suffix, param_path, value in sac_rnn_variations_list:
    config_name = f"sac_rnn_{name_suffix}"; new_config = default_sac_rnn_config.model_copy(deep=True)
    new_config.sac.use_rnn = True # Ensure RNN is enabled
    parts = param_path.split("."); attr = new_config
    for part in parts[:-1]: attr = getattr(attr, part)
    setattr(attr, parts[-1], value); CONFIGS[config_name] = new_config

# --- SAC MLP with Prioritized Experience Replay (PER) ---
sac_mlp_per_config = default_sac_mlp_config.model_copy(deep=True)
sac_mlp_per_config.algorithm = "sac" # Redundant as it's copied from default_sac_mlp_config, but explicit
sac_mlp_per_config.sac.use_rnn = False # Ensure MLP
sac_mlp_per_config.sac.use_per = True # Enable PER
# Other PER parameters (per_alpha, per_beta_start, etc.) will use their defaults from SACConfig
CONFIGS["sac_mlp_per"] = sac_mlp_per_config


CONFIGS["default_mapping"] = default_sac_mlp_config

# List all configuration names:
# default_sac_mlp
# default_ppo_mlp
# default_sac_rnn
# default_ppo_rnn
# sac_mlp_actor_lr_low
# sac_mlp_actor_lr_high
# sac_mlp_critic_lr_low
# sac_mlp_critic_lr_high
# sac_mlp_gamma_low
# sac_mlp_gamma_high
# sac_mlp_tau_low
# sac_mlp_tau_high
# sac_mlp_hidden_dims_small
# sac_mlp_hidden_dims_large
# ppo_mlp_actor_lr_low
# ppo_mlp_actor_lr_high
# ppo_mlp_gae_lambda_low
# ppo_mlp_gae_lambda_high
# ppo_mlp_policy_clip_low
# ppo_mlp_policy_clip_high
# ppo_mlp_entropy_coef_low
# ppo_mlp_entropy_coef_high
# ppo_mlp_hidden_dim_small
# ppo_mlp_hidden_dim_large
# sac_rnn_rnn_hidden_size_small
# sac_rnn_rnn_hidden_size_big
# sac_mlp_per
# default_mapping (alias to default_sac_mlp)
