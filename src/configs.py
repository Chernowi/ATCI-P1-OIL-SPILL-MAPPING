from typing import Dict, Literal, Tuple, List, Any
from pydantic import BaseModel, Field
import math

# Core dimensions for Oil Spill Mapping
# CORE_STATE_DIM = 7  # Original: 5 sensor booleans (0/1) + agent_x + agent_y
CORE_STATE_DIM = 13 # New: 5 sensors + agent_x + agent_y + heading + map_exists + rel_cx + rel_cy + est_radius + dist_boundary
CORE_ACTION_DIM = 1 # yaw_change_normalized
TRAJECTORY_REWARD_DIM = 1 # Reward stored per step

# --- State Feature Indices (for clarity) ---
# Based on CORE_STATE_DIM = 13
SENSOR_INDICES = slice(0, 5)
AGENT_X_INDEX = 5
AGENT_Y_INDEX = 6
HEADING_INDEX = 7
MAP_EXISTS_INDEX = 8
REL_CENTER_X_INDEX = 9
REL_CENTER_Y_INDEX = 10
EST_RADIUS_INDEX = 11
DIST_BOUNDARY_INDEX = 12
# Action index in trajectory: 13
# Reward index in trajectory: 14
# --- End Indices ---


class SACConfig(BaseModel):
    """Configuration for the SAC agent"""
    state_dim: int = Field(CORE_STATE_DIM, description="Dimension of the enhanced state tuple") # Use new dim
    action_dim: int = Field(CORE_ACTION_DIM, description="Action dimension (yaw_change)")
    hidden_dims: List[int] = Field([256, 256], description="List of hidden layer dimensions for MLP part")
    log_std_min: int = Field(-20, description="Minimum log std for action distribution")
    log_std_max: int = Field(2, description="Maximum log std for action distribution")
    lr: float = Field(1e-6, description="Learning rate")
    gamma: float = Field(0.99, description="Discount factor")
    tau: float = Field(0.001, description="Target network update rate")
    alpha: float = Field(0.2, description="Temperature parameter (Initial value if auto-tuning)")
    auto_tune_alpha: bool = Field(True, description="Whether to auto-tune the alpha parameter")
    use_rnn: bool = Field(False, description="Whether to use RNN layers in Actor/Critic")
    rnn_type: Literal['lstm', 'gru'] = Field('lstm', description="Type of RNN cell (Only used if use_rnn is True)")
    rnn_hidden_size: int = Field(128, description="Hidden size of RNN layers (Only used if use_rnn is True)")
    rnn_num_layers: int = Field(1, description="Number of RNN layers (Only used if use_rnn is True)")

class TSACConfig(SACConfig):
    """Configuration for the Transformer-SAC agent, inheriting from SACConfig"""
    state_dim: int = Field(CORE_STATE_DIM, description="Dimension of the enhanced state tuple") # Use new dim
    use_rnn: bool = Field(False, description="Ensure RNN is disabled for T-SAC's Transformer Critic")
    embedding_dim: int = Field(128, description="Embedding dimension for states and actions in Transformer Critic")
    transformer_n_layers: int = Field(2, description="Number of Transformer encoder layers in Critic")
    transformer_n_heads: int = Field(4, description="Number of attention heads in Transformer Critic")
    transformer_hidden_dim: int = Field(512, description="Hidden dimension within Transformer layers (FeedForward network)")
    use_layer_norm_actor: bool = Field(True, description="Apply Layer Normalization in Actor MLP layers")
    alpha: float = Field(0.1, description="Temperature parameter (Initial value if auto-tuning)") # Override SAC alpha
    lr: float = Field(1e-4, description="Learning rate for T-SAC")

class PPOConfig(BaseModel):
    """Configuration for the PPO agent"""
    state_dim: int = Field(CORE_STATE_DIM, description="Dimension of the enhanced state tuple") # Use new dim
    action_dim: int = Field(CORE_ACTION_DIM, description="Action dimension (yaw_change)")
    hidden_dim: int = Field(256, description="Hidden layer dimension")
    log_std_min: int = Field(-20, description="Minimum log std for action distribution")
    log_std_max: int = Field(2, description="Maximum log std for action distribution")
    actor_lr: float = Field(3e-4, description="Actor learning rate")
    critic_lr: float = Field(1e-3, description="Critic learning rate")
    gamma: float = Field(0.99, description="Discount factor")
    gae_lambda: float = Field(0.95, description="GAE lambda parameter")
    policy_clip: float = Field(0.2, description="PPO clipping parameter")
    n_epochs: int = Field(10, description="Number of optimization epochs per update")
    entropy_coef: float = Field(0.01, description="Entropy coefficient for exploration")
    value_coef: float = Field(0.5, description="Value loss coefficient")
    batch_size: int = Field(64, description="Batch size for training")
    steps_per_update: int = Field(2048, description="Environment steps between PPO updates")


class ReplayBufferConfig(BaseModel):
    """Configuration for the replay buffer (SAC/TSAC)"""
    capacity: int = Field(100000, description="Maximum capacity of replay buffer (stores full trajectories)")
    gamma: float = Field(0.99, description="Discount factor for returns")

class MapperConfig(BaseModel):
    """Configuration for the Oil Spill Mapper"""
    min_oil_points_for_estimate: int = Field(5, description="Minimum number of oil points needed to attempt an estimate")
    min_water_points_for_refinement: int = Field(3, description="Minimum water points needed for boundary refinement")

class TrainingConfig(BaseModel):
    """Configuration for training"""
    num_episodes: int = Field(30000, description="Number of episodes to train")
    max_steps: int = Field(300, description="Maximum steps per episode")
    batch_size: int = Field(128, description="Batch size for training (SAC/TSAC: trajectories, PPO: transitions)")
    save_interval: int = Field(200, description="Interval (in episodes) for saving models")
    log_frequency: int = Field(10, description="Frequency (in episodes) for logging to TensorBoard")
    models_dir: str = Field("models/default_mapping/", description="Directory for saving models")
    learning_starts: int = Field(8000, description="Number of steps to collect before starting SAC/TSAC training updates")
    train_freq: int = Field(4, description="Update the policy every n environment steps (SAC/TSAC)")
    gradient_steps: int = Field(1, description="How many gradient steps to perform when training frequency is met (SAC/TSAC)")

class EvaluationConfig(BaseModel):
    """Configuration for evaluation"""
    num_episodes: int = Field(5, description="Number of episodes for evaluation")
    max_steps: int = Field(300, description="Maximum steps per evaluation episode")
    render: bool = Field(True, description="Whether to render the evaluation")

class Position(BaseModel):
    """Position configuration (X, Y only for this task)"""
    x: float = 0.0
    y: float = 0.0

class Velocity(BaseModel):
    """Velocity configuration (X, Y only for this task)"""
    x: float = 0.0
    y: float = 0.0

class RandomizationRange(BaseModel):
    """Defines ranges for random initialization of position"""
    x_range: Tuple[float, float] = Field((-50.0, 50.0), description="Min/Max X range for randomization")
    y_range: Tuple[float, float] = Field((-50.0, 50.0), description="Min/Max Y range for randomization")

class OilSpillConfig(BaseModel):
    """Configuration for the true oil spill"""
    initial_center: Position = Field(default_factory=lambda: Position(x=0, y=0), description="Initial spill center (used if randomization false)")
    initial_radius: float = Field(20.0, description="Initial spill radius (used if randomization false)")
    radius_range: Tuple[float, float] = Field((10.0, 30.0), description="Min/Max radius range for randomization")
    center_randomization_range: RandomizationRange = Field(default_factory=RandomizationRange, description="Ranges for spill center randomization")
    randomize_oil_spill: bool = Field(True, description="Randomize spill center and radius at reset?")

class VisualizationConfig(BaseModel):
    """Configuration for visualization"""
    save_dir: str = Field("mapping_snapshots", description="Directory for saving visualizations")
    figure_size: tuple = Field((10, 10), description="Figure size for visualizations")
    max_trajectory_points: int = Field(100, description="Max trajectory points to display")
    gif_frame_duration: float = Field(0.15, description="Duration of each frame in generated GIFs")
    delete_frames_after_gif: bool = Field(True, description="Delete individual PNG frames after creating GIF")
    sensor_marker_size: int = Field(20, description="Marker size for sensors")
    sensor_color_oil: str = Field("black", description="Color for sensors detecting oil")
    sensor_color_water: str = Field("cyan", description="Color for sensors detecting water")

class WorldConfig(BaseModel):
    """Configuration for the world"""
    dt: float = Field(1.0, description="Time step")
    agent_speed: float = Field(2.0, description="Constant speed of the agent")
    yaw_angle_range: Tuple[float, float] = Field((-math.pi / 8, math.pi / 8), description="Range of possible yaw angle changes per step [-max_change, max_change]")
    num_sensors: int = Field(5, description="Number of sensors around the agent")
    sensor_distance: float = Field(3.5, description="Distance of sensors from agent center")

    agent_initial_location: Position = Field(default_factory=lambda: Position(x=-40, y=0), description="Initial agent position (used if randomization false)")
    randomize_agent_initial_location: bool = Field(True, description="Randomize agent initial location?")
    agent_randomization_ranges: RandomizationRange = Field(default_factory=RandomizationRange, description="Ranges for agent location randomization")

    oil_spill: OilSpillConfig = Field(default_factory=OilSpillConfig, description="True oil spill configuration")

    trajectory_length: int = Field(10, description="Number of steps (N) included in the trajectory state (SAC/TSAC)")
    # Updated trajectory feature dimension to match new state dim
    trajectory_feature_dim: int = Field(CORE_STATE_DIM + CORE_ACTION_DIM + TRAJECTORY_REWARD_DIM, description="Dimension of features per step in trajectory state (enhanced_state + prev_action + prev_reward)") # 13+1+1=15

    # --- Termination Conditions ---
    success_iou_threshold: float = Field(0.90, description="IoU between estimated and true spill above which the episode is successful")

    # --- Reward Function Parameters ---
    base_iou_reward_scale: float = Field(1.0, description="Scaling factor for CURRENT IoU-based reward")
    iou_improvement_scale: float = Field(1.5, description="Scaling factor for IoU *improvement* reward")
    step_penalty: float = Field(0.05, description="Penalty subtracted each step")
    success_bonus: float = Field(10.0, description="Bonus reward added upon reaching success_iou_threshold")
    uninitialized_mapper_penalty: float = Field(0.05, description="Penalty applied if the mapper hasn't produced a valid estimate yet")

    mapper_config: MapperConfig = Field(default_factory=MapperConfig, description="Configuration for the mapper")


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

    # --- Pass mapper config to world config ---
    def model_post_init(self, __context):
         self.world.mapper_config = self.mapper


# --- Define Specific Configurations ---

# Default config uses SAC and saves to "models/sac_mapping/" (Now uses enhanced state)
default_mapping_config = DefaultConfig()
default_mapping_config.algorithm = "sac"
default_mapping_config.training.models_dir = "models/sac_mapping_enhanced/" # Updated dir name

# SAC RNN config (Now uses enhanced state)
sac_rnn_config = DefaultConfig()
sac_rnn_config.algorithm = "sac"
sac_rnn_config.sac.use_rnn = True
sac_rnn_config.training.models_dir = "models/sac_rnn_mapping_enhanced/" # Updated dir name

# PPO config (Now uses enhanced state)
ppo_mapping_config = DefaultConfig()
ppo_mapping_config.algorithm = "ppo"
ppo_mapping_config.training.models_dir = "models/ppo_mapping_enhanced/" # Updated dir name

# T-SAC config (Now uses enhanced state)
tsac_mapping_config = DefaultConfig()
tsac_mapping_config.algorithm = "tsac"
tsac_mapping_config.training.models_dir = "models/tsac_mapping_enhanced/" # Updated dir name
# tsac_mapping_config.tsac.lr = 1e-4 # Example override
# tsac_mapping_config.world.trajectory_length = 5 # Example override


# Dictionary to access configurations by name
CONFIGS: Dict[str, DefaultConfig] = {
    "default_mapping": default_mapping_config, # Default is SAC Enhanced
    "sac_rnn_mapping":  sac_rnn_config,        # SAC RNN Enhanced
    "ppo_mapping": ppo_mapping_config,         # PPO Enhanced
    "tsac_mapping": tsac_mapping_config,       # T-SAC Enhanced
}

# Example access: CONFIGS["ppo_mapping"].ppo.n_epochs
# Example access: CONFIGS["sac_mapping"].training.models_dir => "models/sac_mapping_enhanced/"