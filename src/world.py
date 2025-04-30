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

        # Initialize trajectory history (deque stores feature vectors)
        # Feature vector: [sensor1..5, agent_x, agent_y, prev_action, prev_reward]
        self._trajectory_history = deque(maxlen=self.trajectory_length)

        self.reset() # Initialize agent, spill, mapper, history

    def reset(self):
        """Resets the environment to a new initial state."""
        self.current_step = 0
        self.done = False
        self.reward = 0.0
        self.iou = 0.0
        self.previous_iou = 0.0 # Reset previous IoU

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

        # --- STORE INITIAL LOCATION ---
        self.agent_initial_location = Location(x=agent_location.x, y=agent_location.y) # Store a copy

        initial_heading = random.uniform(-math.pi, math.pi)
        agent_velocity = Velocity(
            x=self.agent_speed * math.cos(initial_heading),
            y=self.agent_speed * math.sin(initial_heading)
        )
        self.agent = Object(location=agent_location, velocity=agent_velocity, name="agent")

        # --- Reset Mapper ---
        self.mapper.reset()

        # --- Initialize Trajectory History ---
        self._initialize_trajectory_history()

        # Perform initial sensor reading and mapper update for the very first state
        sensor_locs, sensor_reads = self._get_sensor_readings()
        for loc, read in zip(sensor_locs, sensor_reads):
             self.mapper.add_measurement(loc, read)
        self.mapper.estimate_spill()
        self._calculate_iou() # Calculate initial IoU
        self.previous_iou = self.iou # Set previous IoU for the first step calculation

        # Calculate reward for the initial state (though it won't be used directly by agent)
        self._calculate_reward(sensor_reads) # Pass initial sensor readings

        return self.encode_state()


    def _get_sensor_locations(self) -> List[Location]:
        """Calculates the world coordinates of the agent's sensors."""
        sensor_locations = []
        agent_loc = self.agent.location
        agent_heading = self.agent.get_heading()
        # Evenly space sensors around the agent
        angle_step = 2 * math.pi / self.num_sensors
        for i in range(self.num_sensors):
            # Angle relative to agent's heading (0 degrees is straight ahead)
            relative_angle = (i * angle_step) - (math.pi / 2) # Start with sensor 0 at agent's left
            # Absolute angle in world frame
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
        """ Encodes the instantaneous basic state observation. """
        sensor_locs, sensor_readings_bool = self._get_sensor_readings()
        sensor_readings_float = [1.0 if read else 0.0 for read in sensor_readings_bool]
        agent_loc = self.agent.location

        state_list = sensor_readings_float + [agent_loc.x, agent_loc.y]
        if len(state_list) != CORE_STATE_DIM:
             raise ValueError(f"Basic state dimension mismatch. Expected {CORE_STATE_DIM}, got {len(state_list)}")

        return tuple(state_list)

    def _initialize_trajectory_history(self):
        """Fills the initial trajectory history with the initial state."""
        if self.agent is None or self.true_spill is None:
            raise ValueError("Agent and Spill must be initialized before trajectory history.")

        initial_basic_state = self._get_basic_state_tuple()
        initial_action = 0.0
        initial_reward = 0.0

        initial_feature = np.concatenate([
            np.array(initial_basic_state, dtype=np.float32),
            np.array([initial_action], dtype=np.float32),
            np.array([initial_reward], dtype=np.float32)
        ])

        if len(initial_feature) != self.feature_dim:
            raise ValueError(f"Feature dimension mismatch. Expected {self.feature_dim}, got {len(initial_feature)}")

        self._trajectory_history.clear()
        for _ in range(self.trajectory_length):
            self._trajectory_history.append(initial_feature)


    def step(self, yaw_change_normalized: float, training: bool = True, terminal_step: bool = False):
        """
        Advance the world state by one time step.
        Args:
            yaw_change_normalized (float): The normalized yaw change action [-1, 1].
            training (bool): Flag indicating if rewards should be calculated.
            terminal_step (bool): Flag indicating if this is the forced last step
                                  due to max_steps being reached in the caller loop.
        """
        if self.done:
            # print("Warning: step() called after environment is done.")
            # Return current state without advancing time if already done
            return self.encode_state()

        # --- Store info needed *before* state changes ---
        prev_basic_state = self._get_basic_state_tuple()
        prev_action = yaw_change_normalized
        # Grab the reward that was calculated at the END of the previous step (r_t)
        reward_from_previous_step = self.reward

        # 1. Get sensor readings BEFORE moving (at time t)
        sensor_locs, sensor_reads_t = self._get_sensor_readings() # Sensor readings at time t

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
        # Agent is now at position for time t+1

        # 3. Update Mapper with measurements taken BEFORE moving (at time t)
        for loc, read in zip(sensor_locs, sensor_reads_t):
            self.mapper.add_measurement(loc, read)
        self.mapper.estimate_spill() # Estimate potentially updated based on readings at t

        # --- State at t+1 is now determined (new agent pos, potentially new estimate) ---

        # 4. Calculate IoU based on NEW estimate (IoU at t+1)
        self._calculate_iou() # Updates self.iou based on state at t+1

        # 5. Calculate reward r_{t+1} based on the state at t+1 and the change from t
        if training:
            # Pass the sensor readings taken at time t (which led to state t+1)
            self._calculate_reward(sensor_reads_t)
        else:
            self.reward = 0.0 # Reset reward calculation for next step if not training

        # 6. Check termination conditions based on state at t+1
        self.current_step += 1
        success = self.iou >= self.world_config.success_iou_threshold
        self.done = success or terminal_step

        # Apply success bonus if applicable (using reward calculated in _calculate_reward)
        if success and training:
             # Add bonus only if the success threshold was crossed in *this* step
             if not self.previous_iou >= self.world_config.success_iou_threshold:
                 self.reward += self.world_config.success_bonus

        # 7. Update trajectory history with [s_t, a_t, r_t]
        # Use the reward that was calculated *at the end of the previous step*
        current_feature_vector = np.concatenate([
            np.array(prev_basic_state, dtype=np.float32),
            np.array([prev_action], dtype=np.float32),
            np.array([reward_from_previous_step], dtype=np.float32) # Reward corresponding to s_t, a_t
        ])
        if len(current_feature_vector) != self.feature_dim:
             raise ValueError(f"Feature vector dim mismatch. Expected {self.feature_dim}, got {len(current_feature_vector)}")
        self._trajectory_history.append(current_feature_vector)

        # 8. Update previous_iou for the *next* step's calculation
        # self.iou holds the value for time t+1, which becomes previous_iou for next step
        self.previous_iou = self.iou

        return self.encode_state() # Return the state dict representing s_{t+1}

    # --- MODIFIED REWARD CALCULATION ---
    def _calculate_reward(self, current_sensor_readings: List[bool]):
        """
        Calculate reward r_{t+1} based on state at t+1 and change from t.
        Uses self.iou (current IoU at t+1), self.previous_iou (IoU at t),
        and the sensor readings taken at time t.
        """
        # Store current IoU before potentially resetting reward
        current_iou_value = self.iou # IoU calculated based on state at t+1

        self.reward = 0.0 # Start fresh for reward r_{t+1}

        # --- Standard Components ---
        # Penalty for not having an estimate
        if self.mapper.estimated_spill is None:
            self.reward -= self.world_config.uninitialized_mapper_penalty
        else:
            # Reward for current IoU level (linear scaling)
            self.reward += self.world_config.base_iou_reward_scale * current_iou_value

            # Reward for IoU improvement (IoU at t+1 vs IoU at t)
            iou_delta = current_iou_value - self.previous_iou
            self.reward += self.world_config.iou_improvement_scale * max(0, iou_delta)

            # Apply step penalty (always applies unless uninitialized penalty was given)
            self.reward -= self.world_config.step_penalty

            # --- NEW: Reward for proximity to estimated spill boundary ---
            est_spill = self.mapper.estimated_spill
            distance_to_est_center = self.agent.location.distance_to(est_spill.center)
            # Ideal distance is the estimated radius
            distance_error_from_boundary = abs(distance_to_est_center - est_spill.radius)
            # Reward decreases as the agent deviates from the boundary
            # Adding a small epsilon to avoid division by zero if error is exactly 0
            proximity_reward = self.world_config.proximity_to_spill_scale / (1.0 + distance_error_from_boundary + 1e-6)
            self.reward += proximity_reward

            # --- NEW: Bonus for detecting oil *when an estimate exists* ---
            # Encourages finding new points inside/near the estimate
            if any(current_sensor_readings):
                self.reward += self.world_config.new_oil_detection_bonus

        # --- NEW: Penalty for distance from start ---
        if self.agent_initial_location and self.world_config.distance_from_start_penalty_scale > 0:
            distance_from_start = self.agent.location.distance_to(self.agent_initial_location)
            # Quadratic penalty increases more sharply with distance
            distance_penalty = self.world_config.distance_from_start_penalty_scale * (distance_from_start ** 2)
            self.reward -= distance_penalty


        # Note: Success bonus is added externally in the step function based on the final reward value

    def encode_state(self) -> Dict[str, Any]:
        """
        Encodes the full state representation needed by the RL agent.
        Includes the basic state tuple and the full trajectory history.
        """
        basic_state = self._get_basic_state_tuple()
        full_trajectory = np.array(self._trajectory_history, dtype=np.float32)

        if full_trajectory.shape != (self.trajectory_length, self.feature_dim):
             # Attempt to recover if deque length is wrong (e.g., after load)
             print(f"Warning: Encoded trajectory shape mismatch. Got {full_trajectory.shape}, expected {(self.trajectory_length, self.feature_dim)}. Reinitializing history.")
             self._initialize_trajectory_history() # This fills with initial state
             # We might lose recent history here, but it prevents crashes.
             full_trajectory = np.array(self._trajectory_history, dtype=np.float32)
             if full_trajectory.shape != (self.trajectory_length, self.feature_dim):
                  raise ValueError(f"Failed to recover trajectory history shape. Expected {(self.trajectory_length, self.feature_dim)}")


        return {
            "basic_state": basic_state,
            "full_trajectory": full_trajectory
        }