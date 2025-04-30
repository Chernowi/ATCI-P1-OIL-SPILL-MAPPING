from world_objects import Object, Location, Velocity, OilSpillCircle
# Use new state dim constant from configs
from configs import WorldConfig, MapperConfig, CORE_STATE_DIM, CORE_ACTION_DIM, TRAJECTORY_REWARD_DIM
from mapper import Mapper
from utils import calculate_iou_circles # Import IoU calculation
import numpy as np
import random
import time
import math
from collections import deque
from typing import Dict, Tuple, Any, List

class World():
    """
    Represents the oil spill mapping environment.
    Agent uses sensors to detect oil and a Mapper estimates the spill shape.
    Action is yaw_change (normalized float).
    State uses an ENHANCED feature set including map estimate info.
    Trajectory state (for SAC/TSAC) includes history of enhanced state, action, reward.
    """
    def __init__(self, world_config: WorldConfig):
        self.world_config = world_config
        self.mapper_config = world_config.mapper_config
        self.dt = world_config.dt
        self.agent_speed = world_config.agent_speed
        self.max_yaw_change = world_config.yaw_angle_range[1]
        self.num_sensors = world_config.num_sensors
        self.sensor_distance = world_config.sensor_distance
        self.trajectory_length = world_config.trajectory_length
        # Use updated feature_dim from config
        self.feature_dim = world_config.trajectory_feature_dim

        self.agent: Object = None
        self.true_spill: OilSpillCircle = None
        self.mapper: Mapper = Mapper(self.mapper_config)

        # World state variables
        self.reward: float = 0.0
        self.iou: float = 0.0
        self.previous_iou: float = 0.0
        self.done: bool = False
        self.current_step: int = 0

        # Initialize trajectory history (deque stores feature vectors)
        # Feature vector: [enhanced_state (13), prev_action (1), prev_reward (1)]
        self._trajectory_history = deque(maxlen=self.trajectory_length)

        self.reset()

    def reset(self):
        """Resets the environment to a new initial state."""
        self.current_step = 0
        self.done = False
        self.reward = 0.0
        self.iou = 0.0
        self.previous_iou = 0.0

        # --- Initialize True Oil Spill ---
        spill_cfg = self.world_config.oil_spill
        if spill_cfg.randomize_oil_spill:
            center_ranges = spill_cfg.center_randomization_range
            spill_center = Location(
                x=random.uniform(*center_ranges.x_range),
                y=random.uniform(*center_ranges.y_range)
            )
            spill_radius = random.uniform(*spill_cfg.radius_range)
        else:
            spill_center = Location(x=spill_cfg.initial_center.x, y=spill_cfg.initial_center.y)
            spill_radius = spill_cfg.initial_radius
        self.true_spill = OilSpillCircle(center=spill_center, radius=spill_radius)

        # --- Initialize Agent ---
        if self.world_config.randomize_agent_initial_location:
            ranges = self.world_config.agent_randomization_ranges
            agent_location = Location(
                x=random.uniform(*ranges.x_range),
                y=random.uniform(*ranges.y_range)
            )
        else:
            loc_cfg = self.world_config.agent_initial_location
            agent_location = Location(x=loc_cfg.x, y=loc_cfg.y)

        initial_heading = random.uniform(-math.pi, math.pi)
        agent_velocity = Velocity(
            x=self.agent_speed * math.cos(initial_heading),
            y=self.agent_speed * math.sin(initial_heading)
        )
        self.agent = Object(location=agent_location, velocity=agent_velocity, name="agent")

        # --- Reset Mapper ---
        self.mapper.reset()

        # --- Initialize Trajectory History ---
        # Do initial sensor reading/mapper update *before* initializing history
        # This ensures the first element in the history has potentially valid map info
        sensor_locs, sensor_reads = self._get_sensor_readings()
        for loc, read in zip(sensor_locs, sensor_reads):
             self.mapper.add_measurement(loc, read)
        self.mapper.estimate_spill()
        self._calculate_iou()
        self.previous_iou = self.iou

        # Now initialize history with the state *after* the first mapper update
        self._initialize_trajectory_history()

        return self.encode_state()


    def _get_sensor_locations(self) -> List[Location]:
        """Calculates the world coordinates of the agent's sensors."""
        sensor_locations = []
        agent_loc = self.agent.location
        agent_heading = self.agent.get_heading()
        angle_step = 2 * math.pi / self.num_sensors
        for i in range(self.num_sensors):
            relative_angle = (i * angle_step) - (math.pi / 2)
            sensor_angle = agent_heading + relative_angle
            sensor_x = agent_loc.x + self.sensor_distance * math.cos(sensor_angle)
            sensor_y = agent_loc.y + self.sensor_distance * math.sin(sensor_angle)
            sensor_locations.append(Location(x=sensor_x, y=sensor_y))
        return sensor_locations

    def _get_sensor_readings(self) -> Tuple[List[Location], List[bool]]:
        """Gets the locations and oil detection status (True/False) for each sensor."""
        sensor_locations = self._get_sensor_locations()
        sensor_readings = [self.true_spill.is_inside(loc) for loc in sensor_locations]
        return sensor_locations, sensor_readings

    def _calculate_iou(self):
        """Calculates IoU between true spill and estimated spill."""
        if self.mapper.estimated_spill is None:
            self.iou = 0.0
        else:
            self.iou = calculate_iou_circles(
                self.true_spill.center, self.true_spill.radius,
                self.mapper.estimated_spill.center, self.mapper.estimated_spill.radius
            )

    def _get_basic_state_tuple(self) -> Tuple:
        """ Encodes the instantaneous ENHANCED state observation. """
        sensor_locs, sensor_readings_bool = self._get_sensor_readings()
        sensor_readings_float = [1.0 if read else 0.0 for read in sensor_readings_bool]
        agent_loc = self.agent.location
        agent_heading = self.agent.get_heading() # Agent's current heading

        # Map related features - with defaults if no estimate exists
        map_exists_flag = 0.0
        rel_center_x = 0.0
        rel_center_y = 0.0
        est_radius = 0.0
        dist_boundary = 0.0 # Default (could also use a large value)

        if self.mapper.estimated_spill is not None:
            map_exists_flag = 1.0
            est_center = self.mapper.estimated_spill.center
            est_radius = self.mapper.estimated_spill.radius
            rel_center_x = est_center.x - agent_loc.x
            rel_center_y = est_center.y - agent_loc.y
            dist_to_center = math.sqrt(rel_center_x**2 + rel_center_y**2)
            # Distance to boundary: positive if outside, negative if inside
            dist_boundary = dist_to_center - est_radius

        state_list = (
            sensor_readings_float +                         # 5 features
            [agent_loc.x, agent_loc.y] +                    # 2 features
            [agent_heading] +                               # 1 feature
            [map_exists_flag] +                             # 1 feature
            [rel_center_x, rel_center_y] +                  # 2 features
            [est_radius] +                                  # 1 feature
            [dist_boundary]                                 # 1 feature
        )

        # Ensure the length matches the configured dimension
        if len(state_list) != CORE_STATE_DIM:
             raise ValueError(f"Enhanced state dimension mismatch. Expected {CORE_STATE_DIM}, got {len(state_list)}")

        # Check for NaNs before returning
        if any(math.isnan(x) for x in state_list):
            print(f"Warning: NaN detected in basic state tuple generation. State: {state_list}")
            # Replace NaNs with 0 for stability, though this indicates an issue
            state_list = [0.0 if math.isnan(x) else x for x in state_list]

        return tuple(state_list)

    def _initialize_trajectory_history(self):
        """Fills the initial trajectory history with the current enhanced state."""
        if self.agent is None or self.true_spill is None:
            raise ValueError("Agent and Spill must be initialized before trajectory history.")

        # Get the enhanced state tuple based on the *current* world state
        # (which includes mapper results from reset)
        initial_basic_state = self._get_basic_state_tuple()
        initial_action = 0.0
        initial_reward = 0.0 # Reward is 0 at the very start

        initial_feature = np.concatenate([
            np.array(initial_basic_state, dtype=np.float32),       # Enhanced state (13)
            np.array([initial_action], dtype=np.float32),          # Action (1)
            np.array([initial_reward], dtype=np.float32)           # Reward (1)
        ])

        if len(initial_feature) != self.feature_dim:
            raise ValueError(f"Trajectory feature dimension mismatch. Expected {self.feature_dim}, got {len(initial_feature)}")

        self._trajectory_history.clear()
        for _ in range(self.trajectory_length):
            # Check for NaNs in the feature vector before adding
            if np.isnan(initial_feature).any():
                 print("Error: NaN detected in initial feature vector during history initialization!")
                 # Handle error appropriately, maybe default to zeros?
                 initial_feature = np.zeros(self.feature_dim, dtype=np.float32)

            self._trajectory_history.append(initial_feature)


    def step(self, yaw_change_normalized: float, training: bool = True, terminal_step: bool = False):
        """
        Advance the world state by one time step using the ENHANCED state.
        Args:
            yaw_change_normalized (float): The normalized yaw change action [-1, 1].
            training (bool): Flag indicating if rewards should be calculated.
            terminal_step (bool): Flag indicating if this is the forced last step.
        """
        if self.done:
            print("Warning: step() called after environment is done.")
            return self.encode_state()

        # --- Store info needed *before* state changes ---
        # Get state s_t (enhanced tuple) before taking action
        prev_basic_state = self._get_basic_state_tuple()
        prev_action = yaw_change_normalized
        # Get reward r_t calculated at the end of the previous step
        reward_from_previous_step = self.reward

        # 1. Get sensor readings BEFORE moving (at time t)
        sensor_locs, sensor_reads = self._get_sensor_readings()

        # 2. Apply action (a_t): Update agent velocity and position
        yaw_change = yaw_change_normalized * self.max_yaw_change
        current_vx, current_vy = self.agent.velocity.x, self.agent.velocity.y
        current_heading = math.atan2(current_vy, current_vx)
        new_heading = (current_heading + yaw_change)
        new_heading = (new_heading + math.pi) % (2 * math.pi) - math.pi
        new_vx = self.agent_speed * math.cos(new_heading)
        new_vy = self.agent_speed * math.sin(new_heading)
        self.agent.velocity = Velocity(new_vx, new_vy)
        self.agent.update_position(self.dt)
        # Agent is now at position for time t+1

        # 3. Update Mapper with measurements taken BEFORE moving (at time t)
        for loc, read in zip(sensor_locs, sensor_reads):
            self.mapper.add_measurement(loc, read)
        self.mapper.estimate_spill() # Estimate potentially updated based on readings at t

        # --- State at t+1 is now determined ---

        # 4. Calculate IoU based on NEW estimate (IoU at t+1)
        self._calculate_iou() # Updates self.iou

        # 5. Calculate reward r_{t+1} based on the state at t+1 and change from t
        if training:
            self._calculate_reward() # Uses self.iou (current) and self.previous_iou
        else:
            self.reward = 0.0 # No reward calculation if not training

        # 6. Check termination conditions based on state at t+1
        self.current_step += 1
        success = self.iou >= self.world_config.success_iou_threshold
        # Check if max steps reached OR success condition met this step
        self.done = success or terminal_step


        # Apply success bonus if applicable (on top of reward calculated in _calculate_reward)
        # Only add bonus if success was the reason for termination *this* step
        if success and training and self.done and not terminal_step:
            self.reward += self.world_config.success_bonus

        # 7. Update trajectory history with [s_t, a_t, r_t]
        # Use the state (s_t), action (a_t), and reward (r_t) from *before* the step
        current_feature_vector = np.concatenate([
            np.array(prev_basic_state, dtype=np.float32),          # Enhanced state s_t
            np.array([prev_action], dtype=np.float32),             # Action a_t
            np.array([reward_from_previous_step], dtype=np.float32) # Reward r_t
        ])
        if len(current_feature_vector) != self.feature_dim:
             raise ValueError(f"Feature vector dim mismatch. Expected {self.feature_dim}, got {len(current_feature_vector)}")

        # Check for NaNs before appending
        if np.isnan(current_feature_vector).any():
            print(f"Warning: NaN detected in feature vector at step {self.current_step}. Vector: {current_feature_vector}")
            # Replace NaNs with 0 to prevent issues, but indicates a problem
            current_feature_vector = np.nan_to_num(current_feature_vector, nan=0.0)

        self._trajectory_history.append(current_feature_vector)

        # 8. Update previous_iou for the *next* step's reward calculation
        self.previous_iou = self.iou # self.iou holds IoU(t+1)

        # 9. Return state dict representing s_{t+1}
        return self.encode_state() # Includes the newly computed basic_state(t+1) and updated history

    def _calculate_reward(self):
        """
        Calculate reward r_{t+1} based on state at t+1 and change from t.
        Uses self.iou (current IoU at t+1) and self.previous_iou (IoU at t).
        (Function remains the same, but uses the updated self.iou)
        """
        current_iou_value = self.iou
        self.reward = 0.0

        if self.mapper.estimated_spill is None:
            self.reward -= self.world_config.uninitialized_mapper_penalty
        else:
            self.reward += self.world_config.base_iou_reward_scale * current_iou_value
            iou_delta = current_iou_value - self.previous_iou
            self.reward += self.world_config.iou_improvement_scale * max(0, iou_delta)
            self.reward -= self.world_config.step_penalty
        # Success bonus added externally in step()

    def encode_state(self) -> Dict[str, Any]:
        """
        Encodes the full state representation using the ENHANCED basic state.
        """
        # Get the enhanced basic state tuple for the *current* time step
        basic_state = self._get_basic_state_tuple()
        # Get the history buffer (which contains features from t-N+1 to t)
        full_trajectory = np.array(self._trajectory_history, dtype=np.float32)

        if full_trajectory.shape != (self.trajectory_length, self.feature_dim):
             print(f"Warning: Encoded trajectory shape mismatch. Got {full_trajectory.shape}, expected {(self.trajectory_length, self.feature_dim)}. Reinitializing history.")
             self._initialize_trajectory_history()
             full_trajectory = np.array(self._trajectory_history, dtype=np.float32)
             if full_trajectory.shape != (self.trajectory_length, self.feature_dim):
                  raise ValueError(f"Failed to recover trajectory history shape after mismatch. Expected {(self.trajectory_length, self.feature_dim)}")

        # Check for NaNs in the final encoded state
        if np.isnan(full_trajectory).any() or any(math.isnan(x) for x in basic_state):
             print(f"Warning: NaN detected in final encoded state.")
             # Apply nan_to_num for stability before returning
             basic_state = tuple(0.0 if math.isnan(x) else x for x in basic_state)
             full_trajectory = np.nan_to_num(full_trajectory, nan=0.0)


        return {
            "basic_state": basic_state,           # Enhanced state tuple (13 features)
            "full_trajectory": full_trajectory    # History array (N, 15 features)
        }