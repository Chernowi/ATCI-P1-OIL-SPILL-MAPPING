from world_objects import Object, Location, Velocity
from configs import WorldConfig, MapperConfig, CORE_STATE_DIM, CORE_ACTION_DIM, TRAJECTORY_REWARD_DIM
from mapper import Mapper
# Removed calculate_iou_circles import
import numpy as np
import random
import time
import math
from collections import deque
from typing import Dict, Tuple, Any, List, Optional

class World():
    """
    Represents the oil spill mapping environment with point clouds and fixed boundaries.
    Agent uses sensors with radii to detect oil points.
    Mapper estimates spill using Convex Hull on oil-detecting sensor locations.
    State coordinates are normalized to [0, 1].
    Goal: Maximize percentage of true oil points captured by estimate, efficiently.
    """
    def __init__(self, world_config: WorldConfig):
        self.world_config = world_config
        self.mapper_config = world_config.mapper_config
        self.dt = world_config.dt
        self.agent_speed = world_config.agent_speed
        self.max_yaw_change = world_config.yaw_angle_range[1]
        self.num_sensors = world_config.num_sensors
        self.sensor_distance = world_config.sensor_distance
        self.sensor_radius = world_config.sensor_radius
        self.trajectory_length = world_config.trajectory_length
        self.feature_dim = world_config.trajectory_feature_dim
        self.world_size = world_config.world_size # Added
        self.normalize_coords = world_config.normalize_coords # Added

        self.agent: Object = None
        # --- Spill representation ---
        self.true_oil_points: List[Location] = [] # Changed from true_spill
        self.true_water_points: List[Location] = [] # Added for density/background
        # ---
        self.mapper: Mapper = Mapper(self.mapper_config)
        self.agent_initial_location: Location = None

        # --- Seeding ---
        self.seeds = world_config.seeds
        self.seed_index = 0
        self.current_seed = None
        # ---

        # World state variables
        self.reward: float = 0.0
        self.performance_metric: float = 0.0 # % oil points inside estimate
        self.previous_performance_metric: float = 0.0 # Store previous metric
        self.done: bool = False
        self.current_step: int = 0
        self.last_sensor_reads: List[bool] = [False] * self.num_sensors # Track previous reads for bonus

        self.reward_components = {
            # Removed iou components
            "metric_improvement": 0.0,
            "new_oil_detection": 0.0,
            "step_penalty": 0.0,
            "uninitialized_penalty": 0.0,
            "out_of_bounds_penalty": 0.0,
            "success_bonus": 0.0,
            "total": 0.0
        }

        self._trajectory_history = deque(maxlen=self.trajectory_length)

        # Call reset during init to set up the first state
        # self.reset() # No, reset is called externally before training starts

    def _seed_environment(self, seed: Optional[int] = None):
        """Sets the random seed for Python and NumPy."""
        if seed is None:
            # Generate a random seed if none is provided
            seed = random.randint(0, 2**32 - 1)
        self.current_seed = seed
        random.seed(self.current_seed)
        np.random.seed(self.current_seed)
        # print(f"World seeded with: {self.current_seed}") # Optional: for debugging

    def reset(self, seed: Optional[int] = None):
        """
        Resets the environment to a new initial state, potentially using a seed.
        If `seed` is provided, it's used. Otherwise, if `seeds` is configured,
        the next seed from the list is used. If `seeds` is empty, a random
        seed is generated.
        """
        self.current_step = 0
        self.done = False
        self.reward = 0.0
        self.performance_metric = 0.0
        self.previous_performance_metric = 0.0
        self.reward_components = {k: 0.0 for k in self.reward_components}

        # --- Seeding Logic ---
        reset_seed = seed # Use provided seed if available
        if reset_seed is None:
            if self.seeds:
                if self.seed_index >= len(self.seeds):
                    self.seed_index = 0 # Cycle through seeds
                reset_seed = self.seeds[self.seed_index]
                self.seed_index += 1
            # else: reset_seed remains None, _seed_environment will generate one
        self._seed_environment(reset_seed)
        # --- End Seeding ---

        # --- Generate Point Clouds ---
        self.true_oil_points = []
        self.true_water_points = []
        world_w, world_h = self.world_size

        # Oil points clustered
        if self.world_config.randomize_oil_cluster:
            center_ranges = self.world_config.oil_center_randomization_range
            oil_center = Location(
                x=random.uniform(*center_ranges.x_range),
                y=random.uniform(*center_ranges.y_range)
            )
            oil_std_dev = random.uniform(*self.world_config.oil_cluster_std_dev_range)
        else:
            oil_center = self.world_config.initial_oil_center
            oil_std_dev = self.world_config.initial_oil_std_dev

        for _ in range(self.world_config.num_oil_points):
            px = np.random.normal(oil_center.x, oil_std_dev)
            py = np.random.normal(oil_center.y, oil_std_dev)
            # Clamp points to be within world boundaries
            px = max(0.0, min(world_w, px))
            py = max(0.0, min(world_h, py))
            self.true_oil_points.append(Location(px, py))

        # Water points spread out (uniform random)
        for _ in range(self.world_config.num_water_points):
            px = random.uniform(0, world_w)
            py = random.uniform(0, world_h)
            # Ensure water point doesn't accidentally land exactly on an oil point (optional)
            is_on_oil = any(abs(p.x - px) < 1e-6 and abs(p.y - py) < 1e-6 for p in self.true_oil_points)
            if not is_on_oil:
                 self.true_water_points.append(Location(px, py))

        # --- Initialize Agent (Unnormalized Coords) ---
        if self.world_config.randomize_agent_initial_location:
            ranges = self.world_config.agent_randomization_ranges
            agent_location = Location(
                x=random.uniform(*ranges.x_range),
                y=random.uniform(*ranges.y_range)
            )
            # Ensure agent starts within bounds
            agent_location.x = max(0.0, min(world_w, agent_location.x))
            agent_location.y = max(0.0, min(world_h, agent_location.y))

        else:
            loc_cfg = self.world_config.agent_initial_location
            # Clamp configured start location just in case
            agent_location = Location(
                 x=max(0.0, min(world_w, loc_cfg.x)),
                 y=max(0.0, min(world_h, loc_cfg.y))
            )

        self.agent_initial_location = Location(x=agent_location.x, y=agent_location.y) # Store unnormalized
        initial_heading = random.uniform(-math.pi, math.pi)
        agent_velocity = Velocity(
            x=self.agent_speed * math.cos(initial_heading),
            y=self.agent_speed * math.sin(initial_heading)
        )
        self.agent = Object(location=agent_location, velocity=agent_velocity, name="agent")

        # --- Reset Mapper ---
        self.mapper.reset()

        # --- Perform initial sensor reading and mapper update for the very first state ---
        sensor_locs_t0, sensor_reads_t0 = self._get_sensor_readings()
        for loc, read in zip(sensor_locs_t0, sensor_reads_t0):
             self.mapper.add_measurement(loc, read) # Add based on initial detection
        self.mapper.estimate_spill()
        self._calculate_performance_metric() # Calculate initial metric
        self.previous_performance_metric = self.performance_metric
        self.last_sensor_reads = sensor_reads_t0 # Store initial reads

        # --- Calculate initial reward (updates components) ---
        self._calculate_reward(sensor_reads_t0) # Pass initial readings
        self.reward = self.reward_components["total"] # Update main reward

        # --- Initialize Trajectory History (uses normalized state from encode_state) ---
        self._initialize_trajectory_history()

        return self.encode_state() # Return normalized state


    def _get_sensor_locations(self) -> List[Location]:
        """Calculates the world coordinates (unnormalized) of the agent's sensors."""
        sensor_locations = []
        agent_loc = self.agent.location
        agent_heading = self.agent.get_heading()
        # Evenly distribute sensors across the front 180 degrees
        if self.num_sensors == 1:
             angle_offsets = [0.0] # Sensor directly in front
        else:
             # Adjust linspace to be -pi/2 to pi/2 relative to heading
             angle_offsets = np.linspace(-math.pi / 2, math.pi / 2, self.num_sensors)

        for angle_offset in angle_offsets:
            # Sensor angle relative to world frame
            sensor_angle = agent_heading + angle_offset
            sensor_x = agent_loc.x + self.sensor_distance * math.cos(sensor_angle)
            sensor_y = agent_loc.y + self.sensor_distance * math.sin(sensor_angle)
            # Clamp sensor location to world bounds (important for simulation stability)
            sensor_x = max(0.0, min(self.world_size[0], sensor_x))
            sensor_y = max(0.0, min(self.world_size[1], sensor_y))
            sensor_locations.append(Location(x=sensor_x, y=sensor_y))
        return sensor_locations

    def _get_sensor_readings(self) -> Tuple[List[Location], List[bool]]:
        """
        Gets the locations (unnormalized) and oil detection status (True/False)
        for each sensor based on radius overlap with true oil points.
        """
        sensor_locations = self._get_sensor_locations()
        sensor_readings = [False] * self.num_sensors

        if not self.true_oil_points: # Handle case with no oil points
             return sensor_locations, sensor_readings

        for i, sensor_loc in enumerate(sensor_locations):
            for oil_point in self.true_oil_points:
                if sensor_loc.distance_to(oil_point) <= self.sensor_radius:
                    sensor_readings[i] = True
                    break # One point is enough to activate the sensor
        return sensor_locations, sensor_readings

    def _calculate_performance_metric(self):
        """Calculates the percentage of true oil points inside the estimated hull."""
        if self.mapper.estimated_hull is None or not self.true_oil_points:
            self.performance_metric = 0.0
            return

        points_inside = 0
        for oil_point in self.true_oil_points:
            if self.mapper.is_inside_estimate(oil_point):
                points_inside += 1

        self.performance_metric = points_inside / len(self.true_oil_points)


    def _get_basic_state_tuple_normalized(self) -> Tuple:
        """ Encodes the instantaneous basic state observation with normalized coords. """
        _, sensor_readings_bool = self._get_sensor_readings()
        sensor_readings_float = [1.0 if read else 0.0 for read in sensor_readings_bool]
        # Get agent's *normalized* coordinates
        agent_loc_normalized = self.agent.location.get_normalized(self.world_size)

        state_list = sensor_readings_float + list(agent_loc_normalized)
        if len(state_list) != CORE_STATE_DIM:
             raise ValueError(f"Normalized state dimension mismatch. Expected {CORE_STATE_DIM}, got {len(state_list)}")
        return tuple(state_list)

    def _initialize_trajectory_history(self):
        """Fills the initial trajectory history with the normalized initial state and reward."""
        if self.agent is None:
            raise ValueError("Agent must be initialized before trajectory history.")

        initial_basic_state_normalized = self._get_basic_state_tuple_normalized()
        initial_action = 0.0
        # Reward r1 is associated with the transition from s0
        initial_reward = self.reward

        initial_feature = np.concatenate([
            np.array(initial_basic_state_normalized, dtype=np.float32),
            np.array([initial_action], dtype=np.float32),
            np.array([initial_reward], dtype=np.float32) # Reward r1
        ])

        if len(initial_feature) != self.feature_dim:
            raise ValueError(f"Feature dimension mismatch. Expected {self.feature_dim}, got {len(initial_feature)}")

        self._trajectory_history.clear()
        for _ in range(self.trajectory_length):
            self._trajectory_history.append(initial_feature)


    def step(self, yaw_change_normalized: float, training: bool = True, terminal_step: bool = False):
        """ Advance the world state by one time step. """
        if self.done:
            # If already done, just return the current state without changes
            return self.encode_state()

        # Store info needed *before* state changes (state s_t)
        # Get normalized state s_t for the trajectory history
        prev_basic_state_normalized = self._get_basic_state_tuple_normalized()
        prev_action = yaw_change_normalized # action a_t

        # 1. Get sensor readings BEFORE moving (at time t)
        sensor_locs_t, sensor_reads_t = self._get_sensor_readings()

        # 2. Apply action (a_t): Update agent velocity and position (unnormalized)
        yaw_change = yaw_change_normalized * self.max_yaw_change
        current_vx, current_vy = self.agent.velocity.x, self.agent.velocity.y
        current_heading = math.atan2(current_vy, current_vx)
        new_heading = (current_heading + yaw_change)
        new_heading = (new_heading + math.pi) % (2 * math.pi) - math.pi # Normalize heading
        new_vx = self.agent_speed * math.cos(new_heading)
        new_vy = self.agent_speed * math.sin(new_heading)
        self.agent.velocity = Velocity(new_vx, new_vy)
        self.agent.update_position(self.dt)
        # Agent is now at unnormalized position for time t+1

        # 3. Check for Out Of Bounds termination
        agent_x, agent_y = self.agent.location.x, self.agent.location.y
        world_w, world_h = self.world_size
        terminated_by_bounds = False
        if self.world_config.terminate_out_of_bounds:
            if not (0.0 <= agent_x <= world_w and 0.0 <= agent_y <= world_h):
                terminated_by_bounds = True
                # Apply penalty immediately if training
                if training:
                    self.reward_components["out_of_bounds_penalty"] = -self.world_config.out_of_bounds_penalty

        # If terminated by bounds, skip further updates and set done flag
        if terminated_by_bounds:
             self.done = True
             # Calculate final reward including the penalty
             self.reward = sum(self.reward_components.values())
             self.reward_components["total"] = self.reward
             # Update history with the state s_t that *led* to termination
             current_feature_vector = np.concatenate([
                 np.array(prev_basic_state_normalized, dtype=np.float32), # s_t (normalized)
                 np.array([prev_action], dtype=np.float32),              # a_t
                 np.array([self.reward], dtype=np.float32)               # r_{t+1} (includes penalty)
             ])
             self._trajectory_history.append(current_feature_vector)
             return self.encode_state() # Return state s_{t+1} (which is OOB)

        # --- Continue if not terminated by bounds ---

        # 4. Update Mapper with measurements taken at time t
        for loc, read in zip(sensor_locs_t, sensor_reads_t):
            self.mapper.add_measurement(loc, read)
        self.mapper.estimate_spill() # Estimate potentially updated

        # 5. Calculate Performance Metric based on NEW estimate (metric at t+1)
        self._calculate_performance_metric()

        # 6. Calculate reward components r_{t+1} based on state at t+1 and change from t
        if training:
            self._calculate_reward(sensor_reads_t) # Pass sensor reads from time t
        else:
            self.reward_components = {k: 0.0 for k in self.reward_components}
            self.reward = 0.0

        # 7. Check other termination conditions based on state at t+1
        self.current_step += 1
        success = self.performance_metric >= self.world_config.success_metric_threshold
        terminated_by_success = success and self.world_config.terminate_on_success
        terminated_by_steps = terminal_step # Reached max steps for episode

        # Update done flag
        self.done = terminated_by_success or terminated_by_steps # Already handled bounds termination

        # Apply success bonus if applicable (and training)
        self.reward_components["success_bonus"] = 0.0
        if terminated_by_success and training:
             if not self.previous_performance_metric >= self.world_config.success_metric_threshold:
                 self.reward_components["success_bonus"] = self.world_config.success_bonus

        # Calculate final total reward for this step (r_{t+1})
        self.reward = sum(self.reward_components.values())
        self.reward_components["total"] = self.reward

        # 8. Update trajectory history with [s_t (normalized), a_t, r_{t+1}]
        current_feature_vector = np.concatenate([
            np.array(prev_basic_state_normalized, dtype=np.float32), # s_t (normalized)
            np.array([prev_action], dtype=np.float32),              # a_t
            np.array([self.reward], dtype=np.float32)               # r_{t+1}
        ])
        if len(current_feature_vector) != self.feature_dim:
             raise ValueError(f"Feature vector dim mismatch. Expected {self.feature_dim}, got {len(current_feature_vector)}")
        self._trajectory_history.append(current_feature_vector)

        # 9. Update previous_metric and last sensor reads for the *next* step's calculation
        self.previous_performance_metric = self.performance_metric
        self.last_sensor_reads = sensor_reads_t

        # Return the state dict representing s_{t+1} (normalized)
        return self.encode_state()

    def _calculate_reward(self, current_sensor_readings: List[bool]):
        """
        Calculate reward components for step r_{t+1} based on state s_{t+1} and transition.
        Updates self.reward_components (excluding terminal adjustments).
        Uses performance metric (point inclusion %).
        """
        cfg = self.world_config
        current_metric_value = self.performance_metric
        # Reset components dictionary for this calculation step (excluding terminal ones)
        components_to_reset = [k for k in self.reward_components if k not in ["out_of_bounds_penalty", "success_bonus", "total"]]
        for key in components_to_reset:
             self.reward_components[key] = 0.0

        # --- Penalties ---
        self.reward_components["step_penalty"] = -cfg.step_penalty

        # Penalty if mapper hasn't estimated yet
        if self.mapper.estimated_hull is None:
            self.reward_components["uninitialized_penalty"] = -cfg.uninitialized_mapper_penalty
        else:
            # --- Positive Rewards (only if estimate exists) ---
            # Reward for improvement in performance metric
            metric_delta = current_metric_value - self.previous_performance_metric
            self.reward_components["metric_improvement"] = cfg.metric_improvement_scale * max(0, metric_delta)

            # Bonus for new oil detections
            new_detections = 0
            for i in range(self.num_sensors):
                if current_sensor_readings[i] and not self.last_sensor_reads[i]:
                    new_detections += 1
            if new_detections > 0:
                 self.reward_components["new_oil_detection"] = cfg.new_oil_detection_bonus # Flat bonus if *any* new detection

        # Calculate the preliminary total (before terminal adjustments in step())
        preliminary_total = sum(v for k, v in self.reward_components.items() if k not in ["success_bonus", "out_of_bounds_penalty", "total"])
        self.reward_components["total"] = preliminary_total


    def encode_state(self) -> Dict[str, Any]:
        """
        Encodes the full state representation needed by the RL agent.
        Returns a dictionary containing:
            - 'basic_state': Tuple (sensor1..N, agent_x_norm, agent_y_norm) at current time t+1.
            - 'full_trajectory': Numpy array (traj_length, feature_dim) where the
                                 state component within each feature vector contains
                                 *normalized* coordinates.
        """
        # 1. Get the basic state tuple with normalized coordinates for the *current* time t+1
        basic_state_t_plus_1_normalized = self._get_basic_state_tuple_normalized()

        # 2. Get the trajectory history (which contains normalized states already)
        full_trajectory_normalized = np.array(self._trajectory_history, dtype=np.float32)

        # Validation checks
        if full_trajectory_normalized.shape != (self.trajectory_length, self.feature_dim):
             # This indicates a problem during history update, attempt recovery
             print(f"CRITICAL WARNING: Encoded trajectory shape mismatch. Got {full_trajectory_normalized.shape}, expected {(self.trajectory_length, self.feature_dim)}. Reinitializing history.")
             self._initialize_trajectory_history() # Reinitialize with the CURRENT state
             full_trajectory_normalized = np.array(self._trajectory_history, dtype=np.float32)
             if full_trajectory_normalized.shape != (self.trajectory_length, self.feature_dim):
                  raise ValueError(f"Failed to recover trajectory history shape after mismatch.")

        if np.isnan(basic_state_t_plus_1_normalized).any() or np.isnan(full_trajectory_normalized).any():
             print(f"Warning: NaN detected in encode_state. Step: {self.current_step}")
             # Consider how to handle NaNs if they persist - maybe return previous state?
             # For now, just print warning. Algorithms might handle NaNs via skipping updates.
             basic_state_t_plus_1_normalized = tuple(np.nan_to_num(list(basic_state_t_plus_1_normalized), nan=0.0))
             full_trajectory_normalized = np.nan_to_num(full_trajectory_normalized, nan=0.0)


        return {
            "basic_state": basic_state_t_plus_1_normalized,
            "full_trajectory": full_trajectory_normalized
        }