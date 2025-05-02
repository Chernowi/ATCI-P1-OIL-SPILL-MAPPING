from world_objects import Object, Location, Velocity, OilSpillCircle
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
    State is a tuple of sensor readings (booleans) and agent coordinates (x, y).
    Trajectory state (for SAC/TSAC) includes history of basic state, action, reward.
    Goal: Encourage fast and accurate mapping, preventing excessive wandering.
    """
    def __init__(self, world_config: WorldConfig):
        self.world_config = world_config
        self.mapper_config = world_config.mapper_config # Get mapper config from world config
        self.dt = world_config.dt
        self.agent_speed = world_config.agent_speed
        self.max_yaw_change = world_config.yaw_angle_range[1]
        self.num_sensors = world_config.num_sensors
        self.sensor_distance = world_config.sensor_distance
        self.trajectory_length = world_config.trajectory_length
        self.feature_dim = world_config.trajectory_feature_dim # basic_state + action + reward

        self.agent: Object = None
        self.true_spill: OilSpillCircle = None
        self.mapper: Mapper = Mapper(self.mapper_config)
        self.agent_initial_location: Location = None # Store initial agent location

        # World state variables
        self.reward: float = 0.0
        self.iou: float = 0.0 # Intersection over Union metric
        self.previous_iou: float = 0.0 # Store IoU from previous step
        self.done: bool = False
        self.current_step: int = 0

        # --- ADDED: Reward component tracking ---
        self.reward_components = {
            "iou_base": 0.0,
            "iou_improvement": 0.0,
            "proximity": 0.0,
            "oil_detection": 0.0,
            "step_penalty": 0.0,
            "distance_penalty": 0.0,
            "uninitialized_penalty": 0.0,
            "far_distance_penalty": 0.0, # For termination penalty
            "success_bonus": 0.0,       # For success bonus
            "total": 0.0 # This will store the final calculated reward for the step
        }
        # --- End Added ---

        # Initialize trajectory history (deque stores feature vectors)
        self._trajectory_history = deque(maxlen=self.trajectory_length)

        self.reset() # Initialize agent, spill, mapper, history

    def reset(self):
        """Resets the environment to a new initial state."""
        self.current_step = 0
        self.done = False
        self.reward = 0.0
        self.iou = 0.0
        self.previous_iou = 0.0

        # --- Reset reward components ---
        self.reward_components = {k: 0.0 for k in self.reward_components}

        # Initialize True Oil Spill
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

        # Initialize Agent
        if self.world_config.randomize_agent_initial_location:
            ranges = self.world_config.agent_randomization_ranges
            agent_location = Location(
                x=random.uniform(*ranges.x_range),
                y=random.uniform(*ranges.y_range)
            )
        else:
            loc_cfg = self.world_config.agent_initial_location
            agent_location = Location(x=loc_cfg.x, y=loc_cfg.y)

        # --- Store initial location ---
        self.agent_initial_location = Location(x=agent_location.x, y=agent_location.y)

        initial_heading = random.uniform(-math.pi, math.pi)
        agent_velocity = Velocity(
            x=self.agent_speed * math.cos(initial_heading),
            y=self.agent_speed * math.sin(initial_heading)
        )
        self.agent = Object(location=agent_location, velocity=agent_velocity, name="agent")

        # Reset Mapper
        self.mapper.reset()

        # Perform initial sensor reading and mapper update for the very first state
        sensor_locs, sensor_reads = self._get_sensor_readings()
        for loc, read in zip(sensor_locs, sensor_reads):
             self.mapper.add_measurement(loc, read)
        self.mapper.estimate_spill()
        self._calculate_iou()
        self.previous_iou = self.iou

        # --- Calculate initial reward (updates components) ---
        self._calculate_reward(sensor_reads)
        # Set the main reward attribute from the calculated components' total
        self.reward = self.reward_components["total"]
        # --- End Modification ---

        # Initialize Trajectory History AFTER initial state/reward established
        self._initialize_trajectory_history()

        return self.encode_state()


    def _get_sensor_locations(self) -> List[Location]:
        """Calculates the world coordinates of the agent's sensors."""
        sensor_locations = []
        agent_loc = self.agent.location
        agent_heading = self.agent.get_heading()
        angle_step = 2 * math.pi / self.num_sensors
        # Distribute sensors evenly around the front 180 degrees
        if self.num_sensors == 1:
             angle_offsets = [0.0] # Sensor directly in front
        else:
             angle_offsets = np.linspace(-math.pi / 2, math.pi / 2, self.num_sensors)

        for angle_offset in angle_offsets:
            sensor_angle = agent_heading + angle_offset
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
        """ Encodes the instantaneous basic state observation. """
        sensor_locs, sensor_readings_bool = self._get_sensor_readings()
        sensor_readings_float = [1.0 if read else 0.0 for read in sensor_readings_bool]
        agent_loc = self.agent.location

        state_list = sensor_readings_float + [agent_loc.x, agent_loc.y]
        if len(state_list) != CORE_STATE_DIM:
             raise ValueError(f"Basic state dimension mismatch. Expected {CORE_STATE_DIM}, got {len(state_list)}")

        return tuple(state_list)

    def _initialize_trajectory_history(self):
        """Fills the initial trajectory history with the initial state and REWARD."""
        if self.agent is None or self.true_spill is None:
            raise ValueError("Agent and Spill must be initialized before trajectory history.")

        initial_basic_state = self._get_basic_state_tuple()
        initial_action = 0.0
        # Use the reward calculated *for* the initial state (s0)
        # This reward (r1) is associated with taking a hypothetical initial action a0 in s0
        initial_reward = self.reward

        initial_feature = np.concatenate([
            np.array(initial_basic_state, dtype=np.float32),
            np.array([initial_action], dtype=np.float32),
            np.array([initial_reward], dtype=np.float32) # Reward r1 associated with state s0
        ])

        if len(initial_feature) != self.feature_dim:
            raise ValueError(f"Feature dimension mismatch. Expected {self.feature_dim}, got {len(initial_feature)}")

        self._trajectory_history.clear()
        for _ in range(self.trajectory_length):
            self._trajectory_history.append(initial_feature)


    def step(self, yaw_change_normalized: float, training: bool = True, terminal_step: bool = False):
        """
        Advance the world state by one time step.
        """
        if self.done:
            return self.encode_state()

        # Store info needed *before* state changes (state s_t)
        prev_basic_state = self._get_basic_state_tuple()
        prev_action = yaw_change_normalized # action a_t

        # 1. Get sensor readings BEFORE moving (at time t)
        sensor_locs, sensor_reads_t = self._get_sensor_readings()

        # 2. Apply action (a_t): Update agent velocity and position
        yaw_change = yaw_change_normalized * self.max_yaw_change
        current_vx, current_vy = self.agent.velocity.x, self.agent.velocity.y
        current_heading = math.atan2(current_vy, current_vx)
        new_heading = (current_heading + yaw_change)
        new_heading = (new_heading + math.pi) % (2 * math.pi) - math.pi # Normalize heading
        new_vx = self.agent_speed * math.cos(new_heading)
        new_vy = self.agent_speed * math.sin(new_heading)
        self.agent.velocity = Velocity(new_vx, new_vy)
        self.agent.update_position(self.dt)
        # Agent is now at position for time t+1 (state s_{t+1})

        # 3. Update Mapper with measurements taken BEFORE moving (at time t)
        for loc, read in zip(sensor_locs, sensor_reads_t):
            self.mapper.add_measurement(loc, read)
        self.mapper.estimate_spill() # Estimate potentially updated based on readings at t

        # 4. Calculate IoU based on NEW estimate (IoU at t+1)
        self._calculate_iou()

        # 5. Calculate reward components r_{t+1} based on state at t+1 and change from t
        if training:
            self._calculate_reward(sensor_reads_t) # This updates self.reward_components
        else:
            # Zero out components and total reward for evaluation runs
            self.reward_components = {k: 0.0 for k in self.reward_components}
            self.reward = 0.0

        # 6. Check termination conditions based on state at t+1
        self.current_step += 1
        success = self.iou >= self.world_config.success_iou_threshold

        distance_from_start = 0.0
        terminated_by_distance = False
        if self.agent_initial_location:
            distance_from_start = self.agent.location.distance_to(self.agent_initial_location)
            if distance_from_start > self.world_config.max_distance_from_start:
                terminated_by_distance = True

        # Update done flag
        self.done = success or terminal_step or terminated_by_distance

        # --- Apply terminal bonuses/penalties DIRECTLY to components ---
        # Reset these specific components before potentially applying them
        self.reward_components["success_bonus"] = 0.0
        self.reward_components["far_distance_penalty"] = 0.0

        if success and not terminated_by_distance and training:
             # Only apply bonus if success was achieved *this step*
             if not self.previous_iou >= self.world_config.success_iou_threshold:
                 self.reward_components["success_bonus"] = self.world_config.success_bonus

        if terminated_by_distance and training:
            self.reward_components["far_distance_penalty"] = -self.world_config.far_distance_penalty # Store as negative
        # --- End Modification ---

        # --- Calculate final total reward for this step (r_{t+1}) ---
        # Sum all components *after* potential terminal adjustments
        self.reward = sum(self.reward_components.values())
        self.reward_components["total"] = self.reward # Store the final total
        # ---

        # 7. Update trajectory history with [s_t, a_t, r_{t+1}]
        # The feature vector added corresponds to the transition FROM prev_basic_state
        current_feature_vector = np.concatenate([
            np.array(prev_basic_state, dtype=np.float32),    # s_t
            np.array([prev_action], dtype=np.float32),       # a_t
            np.array([self.reward], dtype=np.float32)        # r_{t+1}
        ])
        if len(current_feature_vector) != self.feature_dim:
             raise ValueError(f"Feature vector dim mismatch. Expected {self.feature_dim}, got {len(current_feature_vector)}")
        self._trajectory_history.append(current_feature_vector)

        # 8. Update previous_iou for the *next* step's calculation
        self.previous_iou = self.iou

        # Return the state dict representing s_{t+1}
        return self.encode_state()

    def _calculate_reward(self, current_sensor_readings: List[bool]):
        """
        Calculate reward components for step r_{t+1} based on state s_{t+1} and transition.
        Updates the self.reward_components dictionary (excluding terminal bonuses/penalties).
        """
        cfg = self.world_config
        current_iou_value = self.iou
        # Reset components dictionary for this calculation step
        components = {k: 0.0 for k in self.reward_components}

        # --- Penalties ---
        components["step_penalty"] = -cfg.step_penalty

        if cfg.distance_penalty_scale > 0 and self.agent_initial_location:
            distance_from_start = self.agent.location.distance_to(self.agent_initial_location)
            penalty = 0.0
            if cfg.distance_penalty_type == 'linear':
                penalty = cfg.distance_penalty_scale * distance_from_start
            elif cfg.distance_penalty_type == 'log':
                penalty = cfg.distance_penalty_scale * math.log(1 + distance_from_start)
            elif cfg.distance_penalty_type == 'quadratic':
                 penalty = cfg.distance_penalty_scale * (distance_from_start ** 2)
            components["distance_penalty"] = -penalty

        if self.mapper.estimated_spill is None:
            components["uninitialized_penalty"] = -cfg.uninitialized_mapper_penalty
        else:
            # --- Positive Rewards (only if estimate exists) ---
            est_spill = self.mapper.estimated_spill
            components["iou_base"] = cfg.base_iou_reward_scale * current_iou_value
            iou_delta = current_iou_value - self.previous_iou
            components["iou_improvement"] = cfg.iou_improvement_scale * max(0, iou_delta)

            distance_to_est_center = self.agent.location.distance_to(est_spill.center)
            distance_error_from_boundary = abs(distance_to_est_center - est_spill.radius)
            proximity_reward = cfg.proximity_to_spill_scale / (1.0 + distance_error_from_boundary + 1e-6)
            components["proximity"] = proximity_reward

            if any(current_sensor_readings):
                components["oil_detection"] = cfg.new_oil_detection_bonus

        # Update the class attribute. Terminal bonuses/penalties are handled in step()
        # Only update the non-terminal components here
        for key in components:
            if key not in ["success_bonus", "far_distance_penalty", "total"]:
                self.reward_components[key] = components[key]

        # Calculate the preliminary total (before terminal adjustments in step())
        preliminary_total = sum(v for k, v in self.reward_components.items() if k not in ["success_bonus", "far_distance_penalty", "total"])
        self.reward_components["total"] = preliminary_total # Store preliminary total temporarily


    def encode_state(self) -> Dict[str, Any]:
        """
        Encodes the full state representation needed by the RL agent.
        Returns a dictionary containing:
            - 'basic_state': Tuple (sensor1..N, agent_x, agent_y) at current time t+1.
            - 'full_trajectory': Numpy array (traj_length, feature_dim) representing
                                 history [ (s_{t-N+1}, a_{t-N+1}, r_{t-N+1}), ..., (s_t, a_t, r_t) ].
                                 Note: r_t here is the reward received after (s_{t-1}, a_{t-1}).
                                 The last element is (s_t, a_t, r_{t+1}).
        """
        basic_state_t_plus_1 = self._get_basic_state_tuple()
        full_trajectory = np.array(self._trajectory_history, dtype=np.float32)

        if full_trajectory.shape != (self.trajectory_length, self.feature_dim):
             print(f"Warning: Encoded trajectory shape mismatch. Got {full_trajectory.shape}, expected {(self.trajectory_length, self.feature_dim)}. Reinitializing history.")
             self._initialize_trajectory_history()
             full_trajectory = np.array(self._trajectory_history, dtype=np.float32)
             if full_trajectory.shape != (self.trajectory_length, self.feature_dim):
                  raise ValueError(f"Failed to recover trajectory history shape after mismatch. Expected {(self.trajectory_length, self.feature_dim)}")

        if np.isnan(basic_state_t_plus_1).any() or np.isnan(full_trajectory).any():
             print(f"Warning: NaN detected in encode_state. Step: {self.current_step}")

        return {
            "basic_state": basic_state_t_plus_1,
            "full_trajectory": full_trajectory
        }