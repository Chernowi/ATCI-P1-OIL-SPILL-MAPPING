from typing import Dict, Literal, Tuple, List, Any, Optional
from pydantic import BaseModel, Field
import math
import numpy as np # Added

# Core dimensions for Oil Spill Mapping
CORE_STATE_DIM = 7  # 5 sensor booleans (0/1) + agent_x_normalized + agent_y_normalized
CORE_ACTION_DIM = 1 # yaw_change_normalized
TRAJECTORY_REWARD_DIM = 1 # Reward stored per step

# --- SAC / TSAC / PPO Configs (Mostly unchanged dimensions, but note coord meaning) ---
class SACConfig(BaseModel):
    """Configuration for the SAC agent"""
    state_dim: int = Field(CORE_STATE_DIM, description="Dimension of the basic state tuple (sensors + norm_coords)")
    action_dim: int = Field(CORE_ACTION_DIM, description="Action dimension (yaw_change)")
    hidden_dims: List[int] = Field([256, 256], description="List of hidden layer dimensions for MLP part")
    log_std_min: int = Field(-20, description="Minimum log std for action distribution")
    log_std_max: int = Field(1, description="Maximum log std for action distribution")
    # lr: float = Field(3e-5, description="Learning rate") # Removed single LR
    actor_lr: float = Field(3e-5, description="Actor learning rate") # Added actor LR
    critic_lr: float = Field(3e-5, description="Critic learning rate") # Added critic LR
    gamma: float = Field(0.99, description="Discount factor")
    tau: float = Field(0.005, description="Target network update rate")
    alpha: float = Field(0.2, description="Temperature parameter (Initial value if auto-tuning)")
    auto_tune_alpha: bool = Field(True, description="Whether to auto-tune the alpha parameter")
    use_rnn: bool = Field(False, description="Whether to use RNN layers in Actor/Critic") # DEFAULT FALSE
    rnn_type: Literal['lstm', 'gru'] = Field('lstm', description="Type of RNN cell (Only used if use_rnn is True)")
    rnn_hidden_size: int = Field(128, description="Hidden size of RNN layers (Only used if use_rnn is True)")
    rnn_num_layers: int = Field(1, description="Number of RNN layers (Only used if use_rnn is True)")
    use_state_normalization: bool = Field(False, description="Enable/disable state normalization using RunningMeanStd (operates on potentially already normalized coords)")
    use_reward_normalization: bool = Field(True, description="Enable/disable reward normalization by batch std dev")

class TSACConfig(SACConfig):
    """Configuration for the Transformer-SAC agent, inheriting from SACConfig"""
    use_rnn: bool = Field(False, description="Ensure RNN is disabled for T-SAC's Transformer Critic")
    embedding_dim: int = Field(128, description="Embedding dimension for states and actions in Transformer Critic")
    transformer_n_layers: int = Field(2, description="Number of Transformer encoder layers in Critic")
    transformer_n_heads: int = Field(4, description="Number of attention heads in Transformer Critic")
    transformer_hidden_dim: int = Field(512, description="Hidden dimension within Transformer layers (FeedForward network)")
    use_layer_norm_actor: bool = Field(True, description="Apply Layer Normalization in Actor MLP layers")
    alpha: float = Field(0.1, description="Temperature parameter (Initial value if auto-tuning)")
    # lr: float = Field(1.5e-4, description="Learning rate for T-SAC") # Removed single LR
    actor_lr: float = Field(1.5e-4, description="Actor learning rate for T-SAC") # Added separate LRs
    critic_lr: float = Field(1.5e-4, description="Critic learning rate for T-SAC") # Added separate LRs
    use_state_normalization: bool = Field(True, description="Enable/disable state normalization using RunningMeanStd (operates on potentially already normalized coords)")
    use_reward_normalization: bool = Field(True, description="Enable/disable reward normalization by batch std dev")

class PPOConfig(BaseModel):
    """Configuration for the PPO agent"""
    state_dim: int = Field(CORE_STATE_DIM, description="Dimension of the basic state tuple (sensors + norm_coords)")
    action_dim: int = Field(CORE_ACTION_DIM, description="Action dimension (yaw_change)")
    hidden_dim: int = Field(256, description="Hidden layer dimension")
    log_std_min: int = Field(-20, description="Minimum log std for action distribution")
    log_std_max: int = Field(2, description="Maximum log std for action distribution")
    actor_lr: float = Field(5e-4, description="Actor learning rate") # Adjusted
    critic_lr: float = Field(1e-3, description="Critic learning rate") # Adjusted
    gamma: float = Field(0.99, description="Discount factor")
    gae_lambda: float = Field(0.95, description="GAE lambda parameter")
    policy_clip: float = Field(0.2, description="PPO clipping parameter")
    n_epochs: int = Field(10, description="Number of optimization epochs per update")
    entropy_coef: float = Field(0.01, description="Entropy coefficient for exploration")
    value_coef: float = Field(0.5, description="Value loss coefficient")
    batch_size: int = Field(64, description="Batch size for training")
    steps_per_update: int = Field(2048, description="Environment steps between PPO updates")
    use_state_normalization: bool = Field(True, description="Enable/disable state normalization using RunningMeanStd (operates on potentially already normalized coords)")
    use_reward_normalization: bool = Field(True, description="Enable/disable reward normalization by batch std dev")

# --- Replay Buffer Config (Unchanged) ---
class ReplayBufferConfig(BaseModel):
    """Configuration for the replay buffer (SAC/TSAC)"""
    capacity: int = Field(1500000, description="Maximum capacity of replay buffer (stores full trajectories)")
    gamma: float = Field(0.99, description="Discount factor for returns")

# --- Mapper Config (Simplified) ---
class MapperConfig(BaseModel):
    """Configuration for the Oil Spill Mapper"""
    min_oil_points_for_estimate: int = Field(5, description="Minimum number of oil-detecting sensor locations needed to attempt a Convex Hull estimate")

# --- Training Config ---
class TrainingConfig(BaseModel):
    """Configuration for training"""
    num_episodes: int = Field(30000, description="Number of episodes to train")
    max_steps: int = Field(500, description="Maximum steps per episode (increased)")
    batch_size: int = Field(512, description="Batch size for training (SAC/TSAC: trajectories, PPO: transitions)")
    save_interval: int = Field(200, description="Interval (in episodes) for saving models")
    log_frequency: int = Field(10, description="Frequency (in episodes) for logging to TensorBoard")
    models_dir: str = Field("models/default_mapping/", description="Directory for saving models")
    learning_starts: int = Field(8000, description="Number of steps to collect before starting SAC/TSAC training updates")
    train_freq: int = Field(4, description="Update the policy every n environment steps (SAC/TSAC)")
    gradient_steps: int = Field(1, description="How many gradient steps to perform when training frequency is met (SAC/TSAC)")
    training_seeds: List[int] = Field([], description="List of seeds for environment generation during training. Empty list uses random seeds.")

# --- Evaluation Config ---
class EvaluationConfig(BaseModel):
    """Configuration for evaluation"""
    num_episodes: int = Field(10, description="Number of episodes for evaluation")
    max_steps: int = Field(500, description="Maximum steps per evaluation episode")
    render: bool = Field(True, description="Whether to render the evaluation")
    evaluation_seeds: List[int] = Field([101, 102, 103, 104, 105, 106, 107, 108, 109, 110], description="List of seeds for environment generation during evaluation. Should match num_episodes.")


# --- Pos/Vel/Randomization (Unchanged structure, but values relate to unnormalized world) ---
class Position(BaseModel):
    x: float = 0.0
    y: float = 0.0

class Velocity(BaseModel):
    x: float = 0.0
    y: float = 0.0

class RandomizationRange(BaseModel):
    x_range: Tuple[float, float] = Field((10.0, 90.0), description="Min/Max X range for randomization (unnormalized)")
    y_range: Tuple[float, float] = Field((10.0, 90.0), description="Min/Max Y range for randomization (unnormalized)")

# --- Visualization Config ---
class VisualizationConfig(BaseModel):
    """Configuration for visualization"""
    save_dir: str = Field("mapping_snapshots", description="Directory for saving visualizations")
    figure_size: tuple = Field((10, 10), description="Figure size for visualizations")
    max_trajectory_points: int = Field(200, description="Max trajectory points to display")
    gif_frame_duration: float = Field(0.1, description="Duration of each frame in generated GIFs")
    delete_frames_after_gif: bool = Field(True, description="Delete individual PNG frames after creating GIF")
    sensor_marker_size: int = Field(10, description="Marker size for sensors")
    sensor_color_oil: str = Field("red", description="Color for sensors detecting oil")
    sensor_color_water: str = Field("blue", description="Color for sensors detecting water")
    plot_oil_points: bool = Field(True, description="Whether to plot true oil points (can be slow)")
    plot_water_points: bool = Field(False, description="Whether to plot true water points (can be slow)")
    point_marker_size: int = Field(2, description="Marker size for oil/water points")

# --- World Config ---
class WorldConfig(BaseModel):
    """Configuration for the world"""
    dt: float = Field(1.0, description="Time step")
    world_size: Tuple[float, float] = Field((100.0, 100.0), description="Dimensions (X, Y) of the world (unnormalized)")
    normalize_coords: bool = Field(True, description="Normalize agent coordinates to [0, 1] in the returned state")

    agent_speed: float = Field(1.5, description="Constant speed of the agent (unnormalized units per dt)")
    yaw_angle_range: Tuple[float, float] = Field((-math.pi / 6, math.pi / 6), description="Range of possible yaw angle changes per step [-max_change, max_change]")
    num_sensors: int = Field(5, description="Number of sensors around the agent")
    sensor_distance: float = Field(2.5, description="Distance of sensors from agent center (unnormalized)")
    sensor_radius: float = Field(4.0, description="Radius around each sensor for detecting points (unnormalized)")

    agent_initial_location: Position = Field(default_factory=lambda: Position(x=50, y=10), description="Initial agent position (unnormalized, used if randomize=False)")
    randomize_agent_initial_location: bool = Field(True, description="Randomize agent initial location?")
    agent_randomization_ranges: RandomizationRange = Field(default_factory=lambda: RandomizationRange(x_range=(10.0, 90.0), y_range=(10.0, 90.0)), description="Ranges for agent location randomization (unnormalized)")

    num_oil_points: int = Field(200, description="Number of points representing the true oil spill")
    num_water_points: int = Field(400, description="Number of points representing non-spill areas")
    oil_cluster_std_dev_range: Tuple[float, float] = Field((5.0, 15.0), description="Range for standard deviation of the initial oil point cluster (unnormalized)")
    randomize_oil_cluster: bool = Field(True, description="Randomize oil cluster center and std dev at reset?")
    oil_center_randomization_range: RandomizationRange = Field(default_factory=lambda: RandomizationRange(x_range=(25.0, 75.0), y_range=(25.0, 75.0)), description="Ranges for oil cluster center randomization (unnormalized)")
    initial_oil_center: Position = Field(default_factory=lambda: Position(x=50, y=50), description="Initial oil cluster center (unnormalized, used if randomize=False)")
    initial_oil_std_dev: float = Field(10.0, description="Initial oil cluster std dev (unnormalized, used if randomize=False)")

    trajectory_length: int = Field(10, description="Number of steps (N) included in the trajectory state")
    trajectory_feature_dim: int = Field(CORE_STATE_DIM + CORE_ACTION_DIM + TRAJECTORY_REWARD_DIM, description="Dimension of features per step in trajectory state (normalized_state + prev_action + prev_reward)")

    success_metric_threshold: float = Field(0.90, description="Point inclusion percentage above which the episode is successful")
    terminate_on_success: bool = Field(True, description="Terminate episode immediately upon reaching success threshold?")
    terminate_out_of_bounds: bool = Field(True, description="Terminate episode if agent goes out of world bounds?")

    metric_improvement_scale: float = Field(2.0, description="Scaling factor for performance metric *improvement* reward")
    step_penalty: float = Field(0.01, description="Penalty subtracted each step to encourage speed")
    new_oil_detection_bonus: float = Field(0.1, description="Bonus for *any* sensor detecting oil *when it previously didn't*")
    out_of_bounds_penalty: float = Field(5.0, description="Penalty applied when agent goes out of bounds.")
    success_bonus: float = Field(10.0, description="Bonus reward for reaching success_metric_threshold")
    uninitialized_mapper_penalty: float = Field(0.005, description="Penalty applied if the mapper hasn't produced a valid estimate yet")

    mapper_config: MapperConfig = Field(default_factory=MapperConfig, description="Configuration for the mapper")
    # Add evaluation seeds here (moved from EvaluationConfig for better world reset logic)
    seeds: List[int] = Field([101, 102, 103, 104, 105, 106, 107, 108, 109, 110], description="List of seeds used for environment generation during evaluation/specific resets.")


class DefaultConfig(BaseModel):
    """Default configuration for the entire oil spill mapping application"""
    sac: SACConfig = Field(default_factory=SACConfig)
    tsac: TSACConfig = Field(default_factory=TSACConfig)
    ppo: PPOConfig = Field(default_factory=PPOConfig)
    replay_buffer: ReplayBufferConfig = Field(default_factory=ReplayBufferConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    world: WorldConfig = Field(default_factory=WorldConfig)
    mapper: MapperConfig = Field(default_factory=MapperConfig)
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)
    cuda_device: str = Field("cuda:0", description="CUDA device to use (e.g., 'cuda:0', 'cuda:1', 'cpu')")
    algorithm: str = Field("sac", description="RL algorithm to use ('sac', 'ppo', or 'tsac')")

    def model_post_init(self, __context):
         self.world.mapper_config = self.mapper


# --- Define Specific Configurations ---
# Reconfigure defaults slightly for the new setup using separate LRs
sac_default_settings = SACConfig(
    actor_lr=5e-5, critic_lr=5e-5, # Use previous LR for both actor/critic
    tau=0.005, use_state_normalization=True, use_reward_normalization=True, use_rnn=False
)
tsac_default_settings = TSACConfig(
    actor_lr=1.5e-4, critic_lr=1.5e-4, # Use previous LR for both actor/critic
    alpha=0.1, use_state_normalization=True, use_reward_normalization=True
)
ppo_default_settings = PPOConfig(actor_lr=5e-4, critic_lr=1e-3, use_state_normalization=True, use_reward_normalization=True)

# --- Default Mapping (SAC MLP) ---
default_mapping_config = DefaultConfig(
    sac=sac_default_settings.model_copy(), # Use a copy
    ppo=ppo_default_settings.model_copy(),
    tsac=tsac_default_settings.model_copy()
)
default_mapping_config.algorithm = "sac"
default_mapping_config.training.models_dir = "models/sac_pointcloud/"


# --- SAC RNN Mapping ---
# Create a distinct SACConfig instance with RNN set to True
sac_rnn_sac_settings = sac_default_settings.model_copy(update={'use_rnn': True})
sac_rnn_config = DefaultConfig(
    sac=sac_rnn_sac_settings, # Assign the RNN-specific settings
    ppo=ppo_default_settings.model_copy(),
    tsac=tsac_default_settings.model_copy()
)
sac_rnn_config.algorithm = "sac" # Still SAC algorithm overall
sac_rnn_config.training.models_dir = "models/sac_rnn_pointcloud/"


# --- PPO Mapping ---
ppo_mapping_config = DefaultConfig(
    sac=sac_default_settings.model_copy(),
    ppo=ppo_default_settings.model_copy(), # Use a copy
    tsac=tsac_default_settings.model_copy()
)
ppo_mapping_config.algorithm = "ppo"
ppo_mapping_config.training.models_dir = "models/ppo_pointcloud/"


# --- TSAC Mapping ---
tsac_mapping_config = DefaultConfig(
    sac=sac_default_settings.model_copy(),
    ppo=ppo_default_settings.model_copy(),
    tsac=tsac_default_settings.model_copy() # Use a copy
)
tsac_mapping_config.algorithm = "tsac"
tsac_mapping_config.training.models_dir = "models/tsac_pointcloud/"


# Dictionary to access configurations by name
CONFIGS: Dict[str, DefaultConfig] = {
    "default_mapping": default_mapping_config,
    "sac_rnn_mapping":  sac_rnn_config,
    "ppo_mapping": ppo_mapping_config,
    "tsac_mapping": tsac_mapping_config,
}