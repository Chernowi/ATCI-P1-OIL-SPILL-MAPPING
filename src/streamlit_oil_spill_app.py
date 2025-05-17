import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import random
import math
from collections import deque
from typing import Dict, Tuple, Any, List, Optional, Literal
from scipy.spatial import ConvexHull, Delaunay
import warnings
import json # For loading config.json
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
import imageio.v2 as imageio # For GIF
from PIL import Image
import matplotlib.animation as animation # For MP4

# --- Pydantic Models (Minimal for loading config.json) ---
from pydantic import BaseModel, Field

# These will be used to parse the config.json file
class SACConfigPydantic(BaseModel):
    state_dim: int
    action_dim: int
    hidden_dims: List[int]
    log_std_min: int
    log_std_max: int
    actor_lr: float
    critic_lr: float
    gamma: float
    tau: float
    alpha: float
    auto_tune_alpha: bool
    use_rnn: bool
    rnn_type: Literal['lstm', 'gru']
    rnn_hidden_size: int
    rnn_num_layers: int
    use_state_normalization: bool
    use_reward_normalization: bool
    use_per: bool
    per_alpha: float
    per_beta_start: float
    per_beta_frames: int
    per_epsilon: float

class PositionPydantic(BaseModel):
    x: float
    y: float

class RandomizationRangePydantic(BaseModel):
    x_range: Tuple[float, float]
    y_range: Tuple[float, float]

class MapperConfigPydantic(BaseModel):
    min_oil_points_for_estimate: int

class WorldConfigPydantic(BaseModel):
    CORE_STATE_DIM: int = Field(8)
    CORE_ACTION_DIM: int = Field(1)
    TRAJECTORY_REWARD_DIM: int = Field(1)
    dt: float
    world_size: Tuple[float, float]
    normalize_coords: bool
    agent_speed: float
    yaw_angle_range: Tuple[float, float]
    num_sensors: int
    sensor_distance: float
    sensor_radius: float
    agent_initial_location: PositionPydantic
    randomize_agent_initial_location: bool
    agent_randomization_ranges: RandomizationRangePydantic
    num_oil_points: int
    num_water_points: int
    oil_cluster_std_dev_range: Tuple[float, float]
    randomize_oil_cluster: bool
    oil_center_randomization_range: RandomizationRangePydantic
    initial_oil_center: PositionPydantic
    initial_oil_std_dev: float
    min_initial_separation_distance: float
    trajectory_length: int
    trajectory_feature_dim: int
    success_metric_threshold: float
    terminate_on_success: bool
    terminate_out_of_bounds: bool
    metric_improvement_scale: float
    step_penalty: float
    new_oil_detection_bonus: float
    out_of_bounds_penalty: float
    success_bonus: float
    uninitialized_mapper_penalty: float
    mapper_config: MapperConfigPydantic
    seeds: List[int]

class VisualizationConfigPydantic(BaseModel):
    save_dir: str = "streamlit_outputs" # Default for streamlit app
    figure_size: Tuple[int, int]
    max_trajectory_points: int
    output_format: Literal['gif', 'mp4']
    video_fps: int
    delete_png_frames: bool
    sensor_marker_size: int
    sensor_color_oil: str
    sensor_color_water: str
    plot_oil_points: bool
    plot_water_points: bool
    point_marker_size: int

class AppConfigPydantic(BaseModel):
    sac: SACConfigPydantic
    world: WorldConfigPydantic
    visualization: VisualizationConfigPydantic
    # Add other top-level keys from config.json if needed for SAC loading or eval
    # For example, if replay_buffer or training configs have params SAC relies on,
    # even if not directly used in this eval-only app.
    # For this lightweight version, we'll assume SAC only needs its own section.
    evaluation: dict # Keep as dict if not fully parsed
    cuda_device: str = "cpu" # Default to CPU for Streamlit

# --- World Objects ---
class Velocity:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
    def is_moving(self) -> bool: return self.x != 0 or self.y != 0
    def get_heading(self) -> float:
        return math.atan2(self.y, self.x) if self.is_moving() else 0.0

class Location:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
    def update(self, velocity: Velocity, dt: float = 1.0):
        self.x += velocity.x * dt
        self.y += velocity.y * dt
    def distance_to(self, other_loc: 'Location') -> float:
        return math.sqrt((self.x - other_loc.x)**2 + (self.y - other_loc.y)**2)
    def get_normalized(self, world_size: Tuple[float, float]) -> Tuple[float, float]:
        ws_x, ws_y = world_size
        norm_x = max(0.0, min(1.0, self.x / ws_x)) if ws_x > 0 else 0.0
        norm_y = max(0.0, min(1.0, self.y / ws_y)) if ws_y > 0 else 0.0
        return norm_x, norm_y

class WorldObject: # Renamed from Object to avoid conflict
    def __init__(self, location: Location, velocity: Optional[Velocity] = None, name: Optional[str] = None):
        self.name = name if name else "Unnamed Object"
        self.location = location
        self.velocity = velocity if velocity is not None else Velocity(0.0, 0.0)
    def update_position(self, dt: float = 1.0):
        if self.velocity and self.velocity.is_moving():
            self.location.update(self.velocity, dt)
    def get_heading(self) -> float: return self.velocity.get_heading()

# --- Mapper ---
class Mapper:
    def __init__(self, config: MapperConfigPydantic):
        self.config = config
        self.oil_sensor_locations: List[Location] = []
        self.water_sensor_locations: List[Location] = [] # Not strictly needed for eval vis
        self.estimated_hull: Optional[ConvexHull] = None
        self.hull_vertices: Optional[np.ndarray] = None
    def reset(self):
        self.oil_sensor_locations = []
        self.water_sensor_locations = []
        self.estimated_hull = None
        self.hull_vertices = None
    def add_measurement(self, sensor_location: Location, is_oil_detected: bool):
        # Simplified: only add oil points for hull
        if is_oil_detected:
            if not any(abs(p.x - sensor_location.x) < 1e-6 and abs(p.y - sensor_location.y) < 1e-6 for p in self.oil_sensor_locations):
                self.oil_sensor_locations.append(sensor_location)
    def estimate_spill(self):
        self.estimated_hull = None; self.hull_vertices = None
        if len(self.oil_sensor_locations) < self.config.min_oil_points_for_estimate: return
        oil_points_np = np.array([[p.x, p.y] for p in self.oil_sensor_locations])
        unique_oil_points = np.unique(oil_points_np, axis=0)
        if unique_oil_points.shape[0] < 3: return
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                hull = ConvexHull(unique_oil_points, qhull_options='QJ')
            self.estimated_hull = hull
            self.hull_vertices = unique_oil_points[hull.vertices]
        except Exception: pass
    def is_inside_estimate(self, point: Location) -> bool:
        if self.estimated_hull is None or self.hull_vertices is None or len(self.hull_vertices) < 3: return False
        point_np = np.array([point.x, point.y])
        try:
            delaunay_hull = Delaunay(self.hull_vertices, qhull_options='QJ')
            return delaunay_hull.find_simplex(point_np) >= 0
        except Exception: return False

# --- World Environment ---
class World:
    def __init__(self, world_config: WorldConfigPydantic):
        self.world_config = world_config
        self.mapper_config = world_config.mapper_config
        self.dt = world_config.dt
        self.agent_speed = world_config.agent_speed
        self.max_yaw_change = world_config.yaw_angle_range[1]
        self.num_sensors = world_config.num_sensors
        self.sensor_distance = world_config.sensor_distance
        self.sensor_radius = world_config.sensor_radius
        self.trajectory_length = world_config.trajectory_length
        self.CORE_STATE_DIM = world_config.CORE_STATE_DIM
        self.CORE_ACTION_DIM = world_config.CORE_ACTION_DIM
        self.TRAJECTORY_REWARD_DIM = world_config.TRAJECTORY_REWARD_DIM
        self.feature_dim = self.CORE_STATE_DIM + self.CORE_ACTION_DIM + self.TRAJECTORY_REWARD_DIM
        self.world_size = world_config.world_size
        self.normalize_coords = world_config.normalize_coords
        self.agent: Optional[WorldObject] = None
        self.true_oil_points: List[Location] = []
        self.mapper: Mapper = Mapper(self.mapper_config)
        self.current_seed = None
        self.reward: float = 0.0 # Simplified for eval
        self.performance_metric: float = 0.0
        self.done: bool = False
        self.current_step: int = 0
        self._trajectory_history = deque(maxlen=self.trajectory_length)

    def _seed_environment(self, seed: Optional[int] = None):
        if seed is None: seed = random.randint(0, 2**32 - 1)
        self.current_seed = seed; random.seed(seed); np.random.seed(seed)

    def reset(self, seed: Optional[int] = None,
              custom_agent_loc: Optional[Tuple[float, float]] = None,
              custom_oil_center: Optional[Tuple[float, float]] = None):
        self.current_step = 0; self.done = False; self.reward = 0.0
        self.performance_metric = 0.0
        self._seed_environment(seed)
        self.true_oil_points = []
        world_w, world_h = self.world_size

        if custom_oil_center:
            oil_center = Location(x=custom_oil_center[0], y=custom_oil_center[1])
            oil_std_dev = np.mean(self.world_config.oil_cluster_std_dev_range) # Use mean for custom
        else: # Fallback to config if not custom
            oil_center = Location(x=self.world_config.initial_oil_center.x, y=self.world_config.initial_oil_center.y)
            oil_std_dev = self.world_config.initial_oil_std_dev
        
        for _ in range(self.world_config.num_oil_points):
            px = np.random.normal(oil_center.x, oil_std_dev)
            py = np.random.normal(oil_center.y, oil_std_dev)
            self.true_oil_points.append(Location(max(0.0, min(world_w, px)), max(0.0, min(world_h, py))))

        if custom_agent_loc:
            agent_location = Location(x=custom_agent_loc[0], y=custom_agent_loc[1])
        else: # Fallback to config
             agent_location = Location(x=self.world_config.agent_initial_location.x, y=self.world_config.agent_initial_location.y)

        initial_heading = random.uniform(-math.pi, math.pi)
        agent_velocity = Velocity(x=self.agent_speed * math.cos(initial_heading), y=self.agent_speed * math.sin(initial_heading))
        self.agent = WorldObject(location=agent_location, velocity=agent_velocity, name="agent")
        self.mapper.reset()
        self._initialize_trajectory_history()
        # Perform initial sensor reading and mapper update for first state
        sensor_locs_t0, sensor_reads_t0 = self._get_sensor_readings()
        for loc, read in zip(sensor_locs_t0, sensor_reads_t0):
             self.mapper.add_measurement(loc, read)
        self.mapper.estimate_spill()
        self._calculate_performance_metric()
        return self.encode_state()

    def _get_sensor_locations(self) -> List[Location]:
        sensor_locations = []
        if not self.agent: return []
        agent_loc = self.agent.location; agent_heading = self.agent.get_heading()
        angle_offsets = np.linspace(-math.pi / 2, math.pi / 2, self.num_sensors) if self.num_sensors > 1 else [0.0]
        for angle_offset in angle_offsets:
            sensor_angle = agent_heading + angle_offset
            sx = agent_loc.x + self.sensor_distance * math.cos(sensor_angle)
            sy = agent_loc.y + self.sensor_distance * math.sin(sensor_angle)
            sensor_locations.append(Location(max(0.0, min(self.world_size[0], sx)), max(0.0, min(self.world_size[1], sy))))
        return sensor_locations

    def _get_sensor_readings(self) -> Tuple[List[Location], List[bool]]:
        sensor_locations = self._get_sensor_locations()
        sensor_readings = [False] * self.num_sensors
        if not self.true_oil_points: return sensor_locations, sensor_readings
        for i, sensor_loc in enumerate(sensor_locations):
            for oil_point in self.true_oil_points:
                if sensor_loc.distance_to(oil_point) <= self.sensor_radius:
                    sensor_readings[i] = True; break
        return sensor_locations, sensor_readings

    def _calculate_performance_metric(self):
        if self.mapper.estimated_hull is None or not self.true_oil_points:
            self.performance_metric = 0.0; return
        points_inside = sum(1 for oil_point in self.true_oil_points if self.mapper.is_inside_estimate(oil_point))
        self.performance_metric = points_inside / len(self.true_oil_points)

    def _get_basic_state_tuple_normalized(self) -> Tuple:
        _, sensor_reads_bool = self._get_sensor_readings()
        sensor_reads_float = [1.0 if r else 0.0 for r in sensor_reads_bool]
        if not self.agent: # Should not happen if reset properly
            return tuple([0.0] * self.CORE_STATE_DIM)
            
        agent_loc_norm = self.agent.location.get_normalized(self.world_size)
        agent_heading_norm = self.agent.get_heading() / math.pi
        state_list = sensor_reads_float + list(agent_loc_norm) + [agent_heading_norm]
        return tuple(state_list)

    def _initialize_trajectory_history(self):
        if not self.agent: # Ensure agent exists
            # Create a dummy agent if needed for initialization path, though reset should handle it
            self.agent = WorldObject(Location(0,0), Velocity(0,0))

        initial_basic_state_normalized = self._get_basic_state_tuple_normalized()
        initial_feature = np.concatenate([
            np.array(initial_basic_state_normalized, dtype=np.float32),
            np.zeros(self.CORE_ACTION_DIM, dtype=np.float32), # Zero action
            np.zeros(self.TRAJECTORY_REWARD_DIM, dtype=np.float32) # Zero reward
        ])
        self._trajectory_history.clear()
        for _ in range(self.trajectory_length): self._trajectory_history.append(initial_feature)

    def step(self, yaw_change_normalized: float):
        if self.done or not self.agent: return self.encode_state()
        
        prev_basic_state_norm = self._get_basic_state_tuple_normalized()
        prev_action = yaw_change_normalized
        reward_from_prev_step = self.reward # r_t

        sensor_locs_t, sensor_reads_t = self._get_sensor_readings()
        
        yaw_change = yaw_change_normalized * self.max_yaw_change
        current_heading = self.agent.get_heading()
        new_heading = (current_heading + yaw_change + math.pi) % (2 * math.pi) - math.pi
        self.agent.velocity = Velocity(self.agent_speed * math.cos(new_heading), self.agent_speed * math.sin(new_heading))
        self.agent.update_position(self.dt)

        terminated_by_bounds = False
        if not (0.0 <= self.agent.location.x <= self.world_size[0] and \
                  0.0 <= self.agent.location.y <= self.world_size[1]):
            terminated_by_bounds = True
        
        if terminated_by_bounds:
            self.done = True
            self.reward = -self.world_config.out_of_bounds_penalty # Simplified for eval
        else:
            for loc, read in zip(sensor_locs_t, sensor_reads_t):
                self.mapper.add_measurement(loc, read)
            self.mapper.estimate_spill()
            self._calculate_performance_metric()
            # Simplified reward for eval visualization
            self.reward = self.performance_metric - self.world_config.step_penalty 
            if self.performance_metric >= self.world_config.success_metric_threshold:
                self.reward += self.world_config.success_bonus
                if self.world_config.terminate_on_success: self.done = True
        
        self.current_step += 1
        if self.current_step >= self.world_config.max_steps: # Using world_config.max_steps here
            self.done = True

        current_feature_vector = np.concatenate([
            np.array(prev_basic_state_norm, dtype=np.float32),
            np.array([prev_action], dtype=np.float32),
            np.array([reward_from_prev_step], dtype=np.float32)
        ])
        self._trajectory_history.append(current_feature_vector)
        return self.encode_state()

    def encode_state(self) -> Dict[str, Any]:
        basic_state_now = self._get_basic_state_tuple_normalized()
        full_traj = np.array(self._trajectory_history, dtype=np.float32)
        if full_traj.shape != (self.trajectory_length, self.feature_dim): # Fallback if history is weird
            self._initialize_trajectory_history() # This will use basic_state_now if agent is set
            full_traj = np.array(self._trajectory_history, dtype=np.float32)

        return {"basic_state": basic_state_now, "full_trajectory": full_traj}

# --- SAC Agent (Lightweight - RNN variant) ---
class RunningMeanStd: # Copied from utils_light.py for self-containment
    def __init__(self, epsilon=1e-4, shape=(), device=None):
        self.device = device or torch.device("cpu")
        self.mean = torch.zeros(shape, dtype=torch.float32, device=self.device)
        self.var = torch.ones(shape, dtype=torch.float32, device=self.device)
        self.count = torch.tensor(epsilon, dtype=torch.float32, device=self.device)
        self.epsilon = epsilon; self._is_eval = False
    def update(self, x: torch.Tensor):
        if self._is_eval: return
        x=x.to(self.device); batch_mean=torch.mean(x,dim=0); batch_var=torch.var(x,dim=0,unbiased=False); batch_count=torch.tensor(x.shape[0],dtype=torch.float32,device=self.device)
        if x.dim()==1: x=x.unsqueeze(0)
        if x.shape[0]==0: return
        delta=batch_mean-self.mean; tot_count=self.count+batch_count; new_mean=self.mean+delta*batch_count/tot_count
        m_a=self.var*self.count; m_b=batch_var*batch_count; M2=m_a+m_b+torch.square(delta)*self.count*batch_count/tot_count
        new_var=M2/tot_count; self.mean=new_mean; self.var=torch.clamp(new_var,min=0.0); self.count=tot_count
    def normalize(self,x:torch.Tensor)->torch.Tensor: x=x.to(self.device); return torch.clamp((x-self.mean)/torch.sqrt(self.var+self.epsilon),-10.0,10.0)
    def state_dict(self): return {'mean':self.mean.cpu(),'var':self.var.cpu(),'count':self.count.cpu()}
    def load_state_dict(self,sd): self.mean=sd['mean'].to(self.device); self.var=sd['var'].to(self.device); self.count=sd['count'].to(self.device)
    def eval(self): self._is_eval=True
    def train(self): self._is_eval=False

class Actor(nn.Module):
    def __init__(self, config: SACConfigPydantic, world_config: WorldConfigPydantic):
        super().__init__()
        self.config = config; self.world_config = world_config
        self.use_rnn = config.use_rnn
        self.state_dim = world_config.CORE_STATE_DIM
        self.action_dim = world_config.CORE_ACTION_DIM
        if self.use_rnn:
            self.rnn_hidden_size = config.rnn_hidden_size; self.rnn_num_layers = config.rnn_num_layers
            rnn_input_dim = self.state_dim
            self.rnn = nn.GRU(rnn_input_dim, config.rnn_hidden_size, config.rnn_num_layers, batch_first=True) if config.rnn_type == 'gru' \
                else nn.LSTM(rnn_input_dim, config.rnn_hidden_size, config.rnn_num_layers, batch_first=True)
            mlp_input_dim = config.rnn_hidden_size
        else: mlp_input_dim = self.state_dim; self.rnn = None
        self.layers = nn.ModuleList()
        current_dim = mlp_input_dim
        for hidden_dim in config.hidden_dims:
            self.layers.append(nn.Linear(current_dim, hidden_dim)); self.layers.append(nn.ReLU()); current_dim = hidden_dim
        self.mean = nn.Linear(current_dim, self.action_dim); self.log_std = nn.Linear(current_dim, self.action_dim)
        self.log_std_min = config.log_std_min; self.log_std_max = config.log_std_max

    def forward(self, net_in: torch.Tensor, hid_st: Optional[Tuple]=None) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple]]:
        next_hid_st = None
        if self.use_rnn and self.rnn: net_out, next_hid_st = self.rnn(net_in, hid_st); mlp_in = net_out[:, -1, :]
        else: mlp_in = net_in
        x = mlp_in
        for layer in self.layers: x = layer(x)
        mean = self.mean(x); log_std = torch.clamp(self.log_std(x), self.log_std_min, self.log_std_max)
        return mean, log_std, next_hid_st
    
    def sample(self, net_in: torch.Tensor, hid_st: Optional[Tuple]=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[Tuple]]:
        mean, log_std, next_hid_st = self.forward(net_in, hid_st)
        std = log_std.exp(); normal = Normal(mean, std); x_t = normal.rsample()
        act_norm = torch.tanh(x_t); log_prob_unb = normal.log_prob(x_t)
        clamp_tanh = act_norm.clamp(-1+1e-6,1-1e-6); log_det_jac = torch.log(1-clamp_tanh.pow(2)+1e-7)
        log_prob = (log_prob_unb-log_det_jac).sum(1,keepdim=True)
        return act_norm, log_prob, torch.tanh(mean), next_hid_st

    def get_initial_hidden_state(self, batch_size: int, device: torch.device) -> Optional[Tuple]:
        if not self.use_rnn: return None
        h0 = torch.zeros(self.rnn_num_layers, batch_size, self.rnn_hidden_size, device=device)
        return (h0, torch.zeros_like(h0)) if self.config.rnn_type == 'lstm' else h0

class SACAgent:
    def __init__(self, config: SACConfigPydantic, world_config: WorldConfigPydantic, device_str: str = "cpu"):
        self.config = config; self.world_config = world_config
        self.device = torch.device(device_str)
        self.actor = Actor(config, world_config).to(self.device)
        # For eval, we only need the actor and normalizer
        self.state_normalizer = None
        if config.use_state_normalization:
            self.state_normalizer = RunningMeanStd(shape=(world_config.CORE_STATE_DIM,), device=self.device)

    def select_action(self, state: dict, actor_hidden_state: Optional[Tuple] = None, evaluate: bool = True) -> Tuple[float, Optional[Tuple]]:
        state_traj = state['full_trajectory']
        state_tensor_full = torch.FloatTensor(state_traj).to(self.device).unsqueeze(0)
        
        with torch.no_grad():
            raw_input_states = state_tensor_full[:, :, :self.world_config.CORE_STATE_DIM]
            actor_input = raw_input_states if self.config.use_rnn else raw_input_states[:, -1, :]
            
            if self.config.use_state_normalization and self.state_normalizer:
                self.state_normalizer.eval()
                norm_actor_input = self.state_normalizer.normalize(actor_input)
            else:
                norm_actor_input = actor_input
            
            self.actor.eval()
            if evaluate: # Always deterministic for this app's eval
                _, _, action_mean_sq, next_hidden = self.actor.sample(norm_actor_input, actor_hidden_state)
                action_norm = action_mean_sq
            else: # Stochastic if needed (not for this app's core use case)
                 action_norm, _, _, next_hidden = self.actor.sample(norm_actor_input, actor_hidden_state)
        
        return action_norm.detach().cpu().numpy()[0,0], next_hidden

    def load_model(self, path: str):
        if not os.path.exists(path):
            st.error(f"Model file not found: {path}"); return
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            if self.config.use_state_normalization and self.state_normalizer and 'state_normalizer_state_dict' in checkpoint:
                self.state_normalizer.load_state_dict(checkpoint['state_normalizer_state_dict'])
            st.sidebar.success(f"Model loaded from {os.path.basename(path)}")
        except Exception as e:
            st.error(f"Error loading model: {e}")

# --- Matplotlib visualization function (adapted from Project 1) ---
_agent_trajectory_viz: List[tuple[float, float]] = []
def visualize_world_mpl(world: World, vis_config: VisualizationConfigPydantic, fig, ax):
    global _agent_trajectory_viz
    if world.agent:
        _agent_trajectory_viz.append((world.agent.location.x, world.agent.location.y))
        if len(_agent_trajectory_viz) > vis_config.max_trajectory_points:
            _agent_trajectory_viz = _agent_trajectory_viz[-vis_config.max_trajectory_points:]
    ax.clear()
    if len(_agent_trajectory_viz) > 1:
        traj_x, traj_y = zip(*_agent_trajectory_viz)
        ax.plot(traj_x, traj_y, 'g-', lw=1, alpha=0.6, label='Agent Traj.')
    if vis_config.plot_oil_points and world.true_oil_points:
        oil_x, oil_y = zip(*[(p.x, p.y) for p in world.true_oil_points])
        ax.scatter(oil_x, oil_y, c='k', marker='.', s=vis_config.point_marker_size, alpha=0.7, label='Oil')
    if world.mapper and world.mapper.hull_vertices is not None:
        hull_poly = Polygon(world.mapper.hull_vertices, ec='r', fc='r', alpha=0.2, lw=1.5, ls='--', label=f'Est. Hull ({world.performance_metric:.2%})')
        ax.add_patch(hull_poly)
    if world.agent:
        ax.scatter(world.agent.location.x, world.agent.location.y, c='b', marker='o', s=60, zorder=5, label='Agent')
        h = world.agent.get_heading(); arrow_len = 3.0
        ax.arrow(world.agent.location.x, world.agent.location.y, arrow_len*math.cos(h), arrow_len*math.sin(h),
                 head_width=1.0, head_length=1.5, fc='b', ec='b', alpha=0.7, zorder=5)
        sensor_locs, sensor_reads = world._get_sensor_readings()
        for i, loc in enumerate(sensor_locs):
            color = vis_config.sensor_color_oil if sensor_reads[i] else vis_config.sensor_color_water
            ax.scatter(loc.x, loc.y, c=color, marker='s', s=vis_config.sensor_marker_size, ec='k', lw=0.5, zorder=4, label='Sensors' if i==0 else "")
            # Sensor circle can be intensive, maybe make optional for streamlit
            # sensor_circle = Circle((loc.x, loc.y), world.sensor_radius, ec=color, fc='none', lw=0.5, ls=':', alpha=0.4, zorder=3)
            # ax.add_patch(sensor_circle)

    world_w, world_h = world.world_size
    ax.set_xlabel('X'); ax.set_ylabel('Y')
    title1 = f"Step: {world.current_step}, Metric: {world.performance_metric:.3f}"
    if world.current_seed is not None: title1 += f", Seed: {world.current_seed}"
    ax.set_title(f'Oil Spill Mapping\n{title1}')
    padding = 5.0
    ax.set_xlim(-padding, world_w + padding); ax.set_ylim(-padding, world_h + padding)
    ax.set_aspect('equal', adjustable='box')
    ax.legend(fontsize='small', loc='upper left', bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout(rect=[0,0,0.85,1])

def reset_viz_trajectories():
    global _agent_trajectory_viz
    _agent_trajectory_viz = []

# --- Streamlit Application ---
st.set_page_config(layout="wide", page_title="RL Agent Oil Spill Mapping Demo")
st.title("🛢️ RL Agent Demo: Oil Spill Mapping")
st.markdown("""
This application demonstrates a pre-trained SAC-RNN agent performing an oil spill mapping task.
1.  **Drag and Drop**: Click on the canvas to set the initial center of the oil spill and the agent's starting position.
2.  **Configuration**: Adjust simulation parameters in the sidebar.
3.  **Run**: Click "Run Simulation" to see the agent in action. The resulting animation will be displayed below.
""")

# --- File Paths (Update these to where your files are) ---
MODEL_DIR = "assets"
CONFIG_FILE_PATH = os.path.join(MODEL_DIR, "config.json")
# IMPORTANT: Rename your model file to 'sac_rnn_oil_spill_model.pt' and place it in 'assets/'
MODEL_FILE_PATH = os.path.join(MODEL_DIR, "sac_rnn_oil_spill_model.pt")

@st.cache_data
def load_app_config(config_path):
    if not os.path.exists(config_path):
        st.error(f"Configuration file not found: {config_path}")
        return None
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    try:
        # Only parse necessary parts for this app
        app_conf = AppConfigPydantic(
            sac=config_dict['sac'],
            world=config_dict['world'],
            visualization=config_dict['visualization'],
            evaluation=config_dict.get('evaluation', {}), # Use dict for flexibility
            cuda_device=config_dict.get('cuda_device', 'cpu') # Default to CPU
        )
        return app_conf
    except Exception as e:
        st.error(f"Error parsing configuration: {e}")
        return None

@st.cache_resource # Cache the loaded agent
def load_sac_agent(_app_config: AppConfigPydantic, model_path: str):
    if not _app_config: return None
    agent = SACAgent(config=_app_config.sac, world_config=_app_config.world, device_str=_app_config.cuda_device)
    agent.load_model(model_path)
    return agent

app_config = load_app_config(CONFIG_FILE_PATH)

if app_config:
    agent = load_sac_agent(app_config, MODEL_FILE_PATH)
else:
    st.stop() # Stop if config fails to load

# --- Sidebar for Configuration ---
st.sidebar.header("Simulation Parameters")
max_eval_steps = st.sidebar.slider("Max Evaluation Steps", 10, app_config.world.max_steps * 2, app_config.evaluation.get('max_steps', 300))
num_oil_points_sidebar = st.sidebar.slider("Number of Oil Points", 50, 500, app_config.world.num_oil_points)
oil_std_dev_sidebar = st.sidebar.slider("Oil Cluster Std Dev", 1.0, 20.0, float(np.mean(app_config.world.oil_cluster_std_dev_range) if app_config.world.oil_cluster_std_dev_range else app_config.world.initial_oil_std_dev))
agent_speed_sidebar = st.sidebar.slider("Agent Speed", 1.0, 10.0, app_config.world.agent_speed)
output_format_sidebar = st.sidebar.selectbox("Output Format", ["mp4", "gif"], index=["mp4", "gif"].index(app_config.visualization.output_format))
stochastic_policy = st.sidebar.checkbox("Use Stochastic Policy for Eval", value=app_config.evaluation.get('use_stochastic_policy_eval', False))


# --- Interactive Canvas for Initial Conditions ---
st.subheader("Set Initial Conditions")
canvas_size = 600 # pixels
world_w, world_h = app_config.world.world_size

# Use session state to store clicks
if 'oil_center_canvas' not in st.session_state:
    st.session_state.oil_center_canvas = None # (canvas_x, canvas_y)
if 'agent_start_canvas' not in st.session_state:
    st.session_state.agent_start_canvas = None # (canvas_x, canvas_y)
if 'last_click_type' not in st.session_state:
    st.session_state.last_click_type = 'oil' # 'oil' or 'agent'

col1, col2 = st.columns(2)
with col1:
    st.write("Click on the canvas to place items:")
    if st.button("Next click places: Oil Spill Center"):
        st.session_state.last_click_type = 'oil'
    if st.button("Next click places: Agent Start"):
        st.session_state.last_click_type = 'agent'
    st.write(f"Current mode: Placing **{st.session_state.last_click_type.replace('_', ' ').title()}**")

# Simple canvas using matplotlib for interaction (Streamlit doesn't have native draggable canvas)
fig_canvas, ax_canvas = plt.subplots(figsize=(canvas_size/100, canvas_size/100)) # Approx square
ax_canvas.set_xlim(0, world_w)
ax_canvas.set_ylim(0, world_h) # Matplotlib's origin is bottom-left
ax_canvas.set_title("Click to Set Positions (Origin: Bottom-Left)")
ax_canvas.set_aspect('equal', adjustable='box')
ax_canvas.grid(True)

# Plot current selections
if st.session_state.oil_center_canvas:
    ax_canvas.plot(st.session_state.oil_center_canvas[0], st.session_state.oil_center_canvas[1], 'ko', markersize=10, label='Oil Center')
if st.session_state.agent_start_canvas:
    ax_canvas.plot(st.session_state.agent_start_canvas[0], st.session_state.agent_start_canvas[1], 'bo', markersize=10, label='Agent Start')
if st.session_state.oil_center_canvas or st.session_state.agent_start_canvas:
    ax_canvas.legend()

# This is a workaround for click events. Streamlit reruns on interaction.
# We can't directly get click coords from `st.pyplot`.
# A more robust solution would use a custom component or a library like `streamlit-drawable-canvas`.
# For simplicity, we'll use text inputs that the user fills after "visualizing" where to click.
st.markdown("""
**Note on Placement:** Due to Streamlit limitations, direct canvas clicking isn't perfectly integrated.
Observe the grid and enter approximate coordinates below.
""")

default_oil_x = st.session_state.oil_center_canvas[0] if st.session_state.oil_center_canvas else app_config.world.initial_oil_center.x
default_oil_y = st.session_state.oil_center_canvas[1] if st.session_state.oil_center_canvas else app_config.world.initial_oil_center.y
default_agent_x = st.session_state.agent_start_canvas[0] if st.session_state.agent_start_canvas else app_config.world.agent_initial_location.x
default_agent_y = st.session_state.agent_start_canvas[1] if st.session_state.agent_start_canvas else app_config.world.agent_initial_location.y

oil_x_input = st.number_input("Oil Center X", min_value=0.0, max_value=float(world_w), value=float(default_oil_x), step=1.0)
oil_y_input = st.number_input("Oil Center Y", min_value=0.0, max_value=float(world_h), value=float(default_oil_y), step=1.0)
agent_x_input = st.number_input("Agent Start X", min_value=0.0, max_value=float(world_w), value=float(default_agent_x), step=1.0)
agent_y_input = st.number_input("Agent Start Y", min_value=0.0, max_value=float(world_h), value=float(default_agent_y), step=1.0)

# Update canvas preview based on number inputs
st.session_state.oil_center_canvas = (oil_x_input, oil_y_input)
st.session_state.agent_start_canvas = (agent_x_input, agent_y_input)

# Re-plot with number input values
ax_canvas.clear()
ax_canvas.set_xlim(0, world_w); ax_canvas.set_ylim(0, world_h)
ax_canvas.set_title("Click to Set Positions (Origin: Bottom-Left) - Preview from Inputs")
ax_canvas.set_aspect('equal', adjustable='box'); ax_canvas.grid(True)
if st.session_state.oil_center_canvas:
    ax_canvas.plot(st.session_state.oil_center_canvas[0], st.session_state.oil_center_canvas[1], 'ko', markersize=10, label='Oil Center')
if st.session_state.agent_start_canvas:
    ax_canvas.plot(st.session_state.agent_start_canvas[0], st.session_state.agent_start_canvas[1], 'bo', markersize=10, label='Agent Start')
if st.session_state.oil_center_canvas or st.session_state.agent_start_canvas:
    ax_canvas.legend()

with col2:
    st.pyplot(fig_canvas)
plt.close(fig_canvas) # Close the figure to free memory

# --- Run Simulation Button ---
if st.button("🚀 Run Simulation"):
    if not agent:
        st.error("Agent could not be loaded. Please check model file and configuration.")
    elif st.session_state.oil_center_canvas is None or st.session_state.agent_start_canvas is None:
        st.warning("Please set both oil spill center and agent start positions using the number inputs.")
    else:
        custom_oil_center_world = st.session_state.oil_center_canvas
        custom_agent_loc_world = st.session_state.agent_start_canvas

        # Create a temporary config for this run
        current_run_world_config = app_config.world.model_copy(deep=True)
        current_run_world_config.randomize_agent_initial_location = False
        current_run_world_config.randomize_oil_cluster = False
        current_run_world_config.agent_initial_location = PositionPydantic(x=custom_agent_loc_world[0], y=custom_agent_loc_world[1])
        current_run_world_config.initial_oil_center = PositionPydantic(x=custom_oil_center_world[0], y=custom_oil_center_world[1])
        
        # Update from sidebar
        current_run_world_config.max_steps = max_eval_steps # This should be eval_config.max_steps
        current_run_world_config.num_oil_points = num_oil_points_sidebar
        current_run_world_config.initial_oil_std_dev = oil_std_dev_sidebar # This is a single value now
        current_run_world_config.oil_cluster_std_dev_range = (oil_std_dev_sidebar, oil_std_dev_sidebar) # Make it a range of one value
        current_run_world_config.agent_speed = agent_speed_sidebar

        current_run_vis_config = app_config.visualization.model_copy(deep=True)
        current_run_vis_config.output_format = output_format_sidebar
        
        # Ensure output directory exists
        os.makedirs(current_run_vis_config.save_dir, exist_ok=True)

        world_instance = World(world_config=current_run_world_config)
        
        st.info(f"Running simulation with Oil Center: {custom_oil_center_world}, Agent Start: {custom_agent_loc_world}")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        state = world_instance.reset(
            custom_agent_loc=custom_agent_loc_world,
            custom_oil_center=custom_oil_center_world
        )
        
        actor_hidden_state = None
        if agent.config.use_rnn: # Agent's SAC config
            actor_hidden_state = agent.actor.get_initial_hidden_state(batch_size=1, device=agent.device)

        frames = []
        reset_viz_trajectories() # Reset for visualization

        output_filename_base = f"oil_spill_sim_{int(time.time())}"
        temp_frame_dir = os.path.join(current_run_vis_config.save_dir, f"{output_filename_base}_frames")
        os.makedirs(temp_frame_dir, exist_ok=True)

        fig_sim, ax_sim = plt.subplots(figsize=current_run_vis_config.figure_size)

        for step in range(max_eval_steps):
            status_text.text(f"Simulating step {step+1}/{max_eval_steps}...")
            action, actor_hidden_state = agent.select_action(
                state, 
                actor_hidden_state=actor_hidden_state, 
                evaluate=not stochastic_policy
            )
            state = world_instance.step(action)
            
            # Generate frame
            visualize_world_mpl(world_instance, current_run_vis_config, fig_sim, ax_sim)
            frame_path = os.path.join(temp_frame_dir, f"frame_{step:04d}.png")
            fig_sim.savefig(frame_path)
            frames.append(frame_path)
            
            progress_bar.progress((step + 1) / max_eval_steps)
            if world_instance.done:
                status_text.text(f"Simulation ended at step {step+1} (Done). Metric: {world_instance.performance_metric:.3f}")
                break
        if not world_instance.done:
             status_text.text(f"Simulation ended at step {max_eval_steps} (Max steps reached). Metric: {world_instance.performance_metric:.3f}")
        
        plt.close(fig_sim) # Close the simulation figure

        if frames:
            media_placeholder = st.empty()
            media_placeholder.info(f"Generating {current_run_vis_config.output_format.upper()}...")
            
            output_media_filename = f"{output_filename_base}.{current_run_vis_config.output_format}"
            output_media_path = os.path.join(current_run_vis_config.save_dir, output_media_filename)

            images_for_media = [imageio.imread(f) for f in frames]

            if current_run_vis_config.output_format == 'gif':
                imageio.mimsave(output_media_path, images_for_media, fps=current_run_vis_config.video_fps)
                media_placeholder.image(output_media_path)
            elif current_run_vis_config.output_format == 'mp4':
                # Use matplotlib.animation for MP4
                fig_anim, ax_anim = plt.subplots(figsize=current_run_vis_config.figure_size)
                ax_anim.set_axis_off() # Turn off axis for cleaner video
                
                # Read the first image to set consistent plot limits for animation
                first_img_pil = Image.open(frames[0])
                # We can't directly use the saved MPL figure limits, so we'll just display images.
                # For a true MPL animation, we'd need to redraw each frame.
                # Simpler: show images in sequence.
                
                # Clean solution: ImageSequenceClip with moviepy (if installed)
                # Fallback: matplotlib.animation.ArtistAnimation
                # Simplest for now: just offer GIF or link to MP4 if saved directly by imageio (might need specific ffmpeg setup)
                
                try: # Try imageio for mp4 first (needs ffmpeg backend)
                    imageio.mimsave(output_media_path, images_for_media, fps=current_run_vis_config.video_fps, format='FFMPEG', output_params=['-vcodec', 'libx264'])
                    media_placeholder.video(output_media_path)
                except Exception as e_mp4:
                    st.warning(f"MP4 creation with imageio failed: {e_mp4}. Try installing ffmpeg or using GIF format.")
                    # Fallback: create a GIF if MP4 fails and user selected MP4
                    gif_fallback_path = output_media_path.replace(".mp4", ".gif")
                    imageio.mimsave(gif_fallback_path, images_for_media, fps=current_run_vis_config.video_fps)
                    media_placeholder.image(gif_fallback_path)
                    st.info("Fell back to GIF format for video.")

            if current_run_vis_config.delete_png_frames:
                for frame_file in frames:
                    try: os.remove(frame_file)
                    except Exception: pass # Ignore errors during cleanup
                try: os.rmdir(temp_frame_dir) # Remove empty frame directory
                except Exception: pass
            st.success(f"Simulation complete. Final Performance Metric: {world_instance.performance_metric:.3f}")
        else:
            st.error("No frames generated for the video.")

else:
    st.error("Failed to load application configuration. Please check `assets/config.json`.")

st.sidebar.markdown("---")
st.sidebar.markdown("Created for visualizing RL agent performance.")