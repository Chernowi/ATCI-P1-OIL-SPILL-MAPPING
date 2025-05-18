# Guide to Tuning Configuration Parameters for Oil Spill RL Agent

This guide explains each parameter in `configs.py`, how it can be tuned, and its expected effect on the training and performance of the Reinforcement Learning agent for oil spill mapping.

## General Tuning Philosophy

1.  **Start with Defaults**: The provided default configurations (e.g., `default_sac_mlp`, `default_ppo_mlp`) are often good starting points.
2.  **Tune One Thing at a Time**: Change parameters systematically to understand their individual impact.
3.  **Monitor Learning Curves**: Use TensorBoard (or similar tools) to observe rewards, loss functions, entropy, etc. This is crucial for diagnosing issues.
4.  **Consider Your Specific Problem**: The "best" parameters depend on the complexity of the oil spill scenarios, agent capabilities, and desired performance metrics.
5.  **Computational Budget**: More complex models (larger networks, RNNs, larger buffers/batches) require more computation.
6.  **Exploration vs. Exploitation**: Many parameters influence this balance. Too much exploration can lead to slow learning; too little can lead to suboptimal policies.

---

## Core Dimensions (Module-level constants)

These are generally defined by the environment's observation and action space and are **not typically tuned** unless the environment itself is fundamentally changed.

*   `CORE_STATE_DIM` (Default: 8)
    *   **Purpose**: Dimension of the basic state tuple (sensors + normalized coordinates + normalized heading).
    *   **Tuning**: Fixed by environment design.
*   `CORE_ACTION_DIM` (Default: 1)
    *   **Purpose**: Action dimension (yaw_change).
    *   **Tuning**: Fixed by environment design.
*   `TRAJECTORY_REWARD_DIM` (Default: 1)
    *   **Purpose**: Dimension of the reward part in the trajectory feature vector.
    *   **Tuning**: Fixed by environment design.
*   `DEFAULT_TRAJECTORY_FEATURE_DIM`
    *   **Purpose**: Calculated as `CORE_STATE_DIM + CORE_ACTION_DIM + TRAJECTORY_REWARD_DIM`. Used as the default for `WorldConfig.trajectory_feature_dim`.
    *   **Tuning**: Derived, not directly tuned.

---

## SACConfig (`sac`)

Configuration for the Soft Actor-Critic (SAC) agent.

*   `state_dim`, `action_dim`:
    *   **Tuning**: Inherited from core dimensions. Not tuned directly here.
*   `hidden_dims` (Default: `[128, 128]`)
    *   **Purpose**: List of hidden layer dimensions for the MLP parts of actor and critic.
    *   **Tuning**:
        *   **Increasing neurons/layers** (e.g., `[256, 256]`, `[256, 256, 128]`): Increases model capacity. Can learn more complex functions but risks overfitting, trains slower, and requires more data.
        *   **Decreasing neurons/layers** (e.g., `[64, 64]`): Reduces model capacity. Trains faster, less prone to overfitting on simple tasks, but may underfit complex problems.
    *   **Effect**: Affects the agent's ability to approximate optimal policies and value functions.
*   `log_std_min`, `log_std_max` (Defaults: -20, 1)
    *   **Purpose**: Bounds for the logarithm of the standard deviation of the action distribution. This constrains the exploration noise.
    *   **Tuning**:
        *   Default values are usually robust.
        *   **Widening the range**: Allows for potentially much larger or smaller exploration noise.
        *   **Narrowing `log_std_max` (e.g., to 0 or -1)**: Limits maximum exploration noise, can be useful if actions are too erratic.
        *   **Increasing `log_std_min` (e.g., to -10)**: Prevents exploration noise from becoming too small too quickly.
    *   **Effect**: Controls the stochasticity of the policy.
*   `actor_lr`, `critic_lr` (Defaults: 5e-5)
    *   **Purpose**: Learning rates for the actor and critic networks.
    *   **Tuning**:
        *   **Typical Range**: `1e-5` to `1e-3`.
        *   **Increasing LR**: Faster initial learning, but can become unstable or overshoot optima.
        *   **Decreasing LR**: Slower learning, but more stable and can find better optima if given enough time.
        *   Often, `critic_lr` can be slightly higher than or equal to `actor_lr`.
    *   **Effect**: Determines step size during gradient descent. Critical for convergence.
*   `gamma` (Default: 0.99)
    *   **Purpose**: Discount factor for future rewards.
    *   **Tuning**:
        *   **Range**: `0.8` to `0.999`.
        *   **Increasing `gamma` (closer to 1)**: Agent values future rewards more highly, becomes more farsighted. Suitable for tasks with delayed rewards.
        *   **Decreasing `gamma`**: Agent prioritizes immediate rewards, becomes more shortsighted. Can be useful if future rewards are noisy or irrelevant.
    *   **Effect**: Balances importance of immediate vs. future rewards.
*   `tau` (Default: 0.005)
    *   **Purpose**: Soft update rate for target networks (polyak averaging). `target_weights = tau * local_weights + (1 - tau) * target_weights`.
    *   **Tuning**:
        *   **Range**: `0.001` to `0.1`.
        *   **Increasing `tau`**: Target networks update faster, making learning more responsive but potentially less stable.
        *   **Decreasing `tau`**: Target networks update slower, leading to more stable but potentially slower learning.
    *   **Effect**: Stability of target Q-values.
*   `alpha` (Default: 0.2)
    *   **Purpose**: Initial temperature parameter for entropy regularization. Balances reward maximization and policy entropy maximization (exploration).
    *   **Tuning**:
        *   Only relevant if `auto_tune_alpha` is `False`.
        *   **Increasing `alpha`**: Encourages more exploration (more random actions).
        *   **Decreasing `alpha`**: Encourages less exploration (more deterministic actions).
    *   **Effect**: Exploration-exploitation trade-off.
*   `auto_tune_alpha` (Default: `True`)
    *   **Purpose**: Whether to automatically tune the `alpha` parameter.
    *   **Tuning**:
        *   `True`: Recommended. The agent learns the optimal `alpha`.
        *   `False`: `alpha` is fixed to the value above. Requires manual tuning of `alpha`.
    *   **Effect**: If `True`, adapts exploration level automatically.
*   `use_rnn` (Default: `False`)
    *   **Purpose**: Whether to use RNN (LSTM/GRU) layers in the actor and critic.
    *   **Tuning**:
        *   `True`: Agent uses past trajectory information. Good for POMDPs or when history matters. Increases model complexity and computational cost.
        *   `False`: Agent is memoryless (uses only current state/last step of trajectory). Simpler, faster.
    *   **Effect**: Agent's ability to use historical context.
*   `rnn_type` (Default: 'lstm')
    *   **Purpose**: Type of RNN cell ('lstm' or 'gru').
    *   **Tuning**:
        *   'lstm': Generally more powerful, can capture longer dependencies, but more parameters.
        *   'gru': Simpler, fewer parameters, often performs comparably to LSTM, trains faster.
    *   **Effect**: Type of recurrent processing.
*   `rnn_hidden_size` (Default: 68)
    *   **Purpose**: Hidden size of RNN layers.
    *   **Tuning**: Similar to `hidden_dims`.
        *   **Increasing**: More capacity for RNN to store/process history.
        *   **Decreasing**: Less capacity, faster.
    *   **Effect**: RNN's memory capacity.
*   `rnn_num_layers` (Default: 1)
    *   **Purpose**: Number of RNN layers.
    *   **Tuning**:
        *   **Increasing**: Deeper recurrent processing, more capacity.
        *   **Decreasing (e.g., 1)**: Simpler RNN.
    *   **Effect**: Depth of RNN.
*   `use_state_normalization` (Default: `False`)
    *   **Purpose**: Enable/disable state normalization using `RunningMeanStd`.
    *   **Tuning**:
        *   `True`: Recommended if state features have different scales or are not zero-centered. Can significantly stabilize and speed up learning.
        *   `False`: Use raw state values.
    *   **Effect**: Standardizes input to networks.
*   `use_reward_normalization` (Default: `True`)
    *   **Purpose**: Enable/disable reward normalization (by batch standard deviation). Note: SAC typically normalizes returns, not raw rewards directly in the update, but this flag could control if rewards are scaled before being used in target calculation. The current `SAC.py` normalizes rewards if this is true.
    *   **Tuning**:
        *   `True`: Can help stabilize learning if rewards have high variance or large magnitudes.
        *   `False`: Use raw rewards.
    *   **Effect**: Scales reward signals.
*   `use_per` (Default: `False`)
    *   **Purpose**: Enable Prioritized Experience Replay.
    *   **Tuning**:
        *   `True`: Samples transitions with high TD-error more frequently. Can speed up learning, especially when important transitions are rare. Adds computational overhead.
        *   `False`: Uniform sampling from replay buffer.
    *   **Effect**: Sampling strategy from replay buffer.
*   `per_alpha` (Default: 0.6)
    *   **Purpose**: PER alpha parameter. Controls how much prioritization is used (0: uniform, 1: full prioritization).
    *   **Tuning**:
        *   **Range**: `0.0` to `1.0`.
        *   **Increasing**: More aggressive prioritization.
        *   **Decreasing**: Closer to uniform sampling.
    *   **Effect**: Degree of prioritization.
*   `per_beta_start` (Default: 0.4)
    *   **Purpose**: PER beta initial value. Importance-sampling exponent, anneals towards 1.0.
    *   **Tuning**:
        *   **Range**: `0.0` to `1.0`.
        *   Controls bias correction. Annealing is common.
    *   **Effect**: Initial strength of importance sampling correction.
*   `per_beta_frames` (Default: 100000)
    *   **Purpose**: Number of frames (timesteps or updates) over which `per_beta` anneals from `per_beta_start` to 1.0.
    *   **Tuning**:
        *   **Increasing**: Slower annealing.
        *   **Decreasing**: Faster annealing.
        *   Should generally span a significant portion of early-to-mid training.
    *   **Effect**: Annealing schedule for PER importance sampling.
*   `per_epsilon` (Default: 1e-5)
    *   **Purpose**: Small value added to priorities to ensure non-zero probability for all transitions.
    *   **Tuning**: Usually kept small. `1e-5` to `1e-6` is common.
    *   **Effect**: Prevents transitions from having zero sampling probability.

---

## PPOConfig (`ppo`)

Configuration for the Proximal Policy Optimization (PPO) agent.

*   `state_dim`, `action_dim`:
    *   **Tuning**: Inherited from core dimensions.
*   `hidden_dim` (Default: 256)
    *   **Purpose**: Hidden layer dimension for MLP parts (PPO often uses same size for actor/critic MLPs).
    *   **Tuning**: Similar to `SACConfig.hidden_dims`, but a single value.
        *   **Increasing**: More capacity.
        *   **Decreasing**: Less capacity.
    *   **Effect**: MLP capacity.
*   `log_std_min`, `log_std_max`:
    *   **Purpose/Tuning**: Same as in `SACConfig`.
*   `actor_lr`, `critic_lr` (Defaults: 5e-5)
    *   **Purpose/Tuning**: Same as in `SACConfig`.
*   `gamma` (Default: 0.99)
    *   **Purpose/Tuning**: Same as in `SACConfig`.
*   `gae_lambda` (Default: 0.95)
    *   **Purpose**: Lambda parameter for Generalized Advantage Estimation (GAE).
    *   **Tuning**:
        *   **Range**: `0.9` to `1.0`.
        *   **`gae_lambda` = 1**: High variance (Monte Carlo returns for advantages).
        *   **`gae_lambda` = 0**: High bias (TD(0) error for advantages).
        *   Values like `0.95` to `0.98` often provide a good bias-variance trade-off.
    *   **Effect**: Bias-variance trade-off in advantage estimation.
*   `policy_clip` (Default: 0.2)
    *   **Purpose**: PPO clipping parameter (`epsilon`). Limits the change in policy at each update.
    *   **Tuning**:
        *   **Range**: `0.1` to `0.3`.
        *   **Increasing**: Allows larger policy updates, potentially faster learning but more risk of instability.
        *   **Decreasing**: More conservative updates, more stable but potentially slower learning.
    *   **Effect**: Stability of policy updates.
*   `n_epochs` (Default: 10)
    *   **Purpose**: Number of optimization epochs over the collected rollout data.
    *   **Tuning**:
        *   **Range**: `3` to `20`.
        *   **Increasing**: More gradient updates per rollout, better sample efficiency but can lead to overfitting on the current batch of data.
        *   **Decreasing**: Fewer updates, less risk of overfitting to current batch, but might need more rollouts overall.
    *   **Effect**: How thoroughly each batch of experience is used.
*   `entropy_coef` (Default: 0.25)
    *   **Purpose**: Coefficient for the entropy bonus in the PPO loss. Encourages exploration.
    *   **Tuning**:
        *   **Range**: `0.0` to `0.5` (can be higher or lower depending on action space).
        *   **Increasing**: More exploration (policy becomes more stochastic).
        *   **Decreasing**: Less exploration. If too low, policy might become deterministic too quickly and get stuck in local optima.
        *   Can be annealed (decreased over time).
    *   **Effect**: Exploration level.
*   `value_coef` (Default: 0.5)
    *   **Purpose**: Coefficient for the value loss (critic loss) in the total PPO loss.
    *   **Tuning**: Usually `0.5` or `1.0`.
        *   Adjusting this can balance the importance of fitting the value function vs. improving the policy.
    *   **Effect**: Weight of the critic's loss in the combined loss function.
*   `batch_size` (Default: 64)
    *   **Purpose**: For RNN PPO, this is the number of *rollouts* (sequences) processed in one training batch during the `n_epochs` of updates. For MLP PPO, this could be the number of transitions if data is flattened and shuffled. Given the current `RecurrentPPOMemory`, it's rollouts.
    *   **Tuning**:
        *   **Increasing**: More stable gradient estimates, but larger memory footprint and slower processing per batch.
        *   **Decreasing**: Noisier gradients, but faster processing per batch.
        *   Interacts with `steps_per_update`. Total samples = `batch_size * steps_per_update` if `steps_per_update` is rollout length and `batch_size` is number of parallel environments (not the case here) or number of rollouts grouped together. In this code, it's number of rollouts to make a training minibatch.
    *   **Effect**: Stability and speed of updates.
*   `steps_per_update` (Default: 256)
    *   **Purpose**: Number of environment steps to collect in each rollout before performing a PPO update. This is the length of sequences stored in `RecurrentPPOMemory`.
    *   **Tuning**:
        *   **Increasing**: Longer rollouts, potentially better GAE estimates, but less frequent updates and policy might become stale. More memory per rollout.
        *   **Decreasing**: Shorter rollouts, more frequent updates, but GAE estimates might be noisier.
    *   **Effect**: Frequency of policy updates and length of experience sequences.
*   `use_state_normalization`, `use_reward_normalization`:
    *   **Purpose/Tuning**: Same as in `SACConfig`. PPO often benefits greatly from state normalization. Reward normalization is also common.
*   `use_rnn`, `rnn_type`, `rnn_hidden_size`, `rnn_num_layers`:
    *   **Purpose/Tuning**: Same as in `SACConfig`. PPO can also leverage RNNs for history dependence.

---

## ReplayBufferConfig (`replay_buffer`)

Relevant for SAC and other off-policy algorithms.

*   `capacity` (Default: 3,000,000)
    *   **Purpose**: Maximum number of transitions stored in the replay buffer.
    *   **Tuning**:
        *   **Increasing**: Stores more diverse and older experiences. Can improve stability and prevent catastrophic forgetting, but requires more memory and might slow down learning if very old, irrelevant data is frequently sampled.
        *   **Decreasing**: Uses more recent data, more responsive to changes in policy/environment, but less diverse and might overfit to recent experiences.
    *   **Effect**: Size and diversity of stored experiences.
*   `gamma` (Default: 0.99)
    *   **Purpose**: Discount factor. This is somewhat redundant as agent configs also have `gamma`. Ensure consistency if used by buffer logic (e.g., for N-step returns, though not used here).
    *   **Tuning**: Same as agent's `gamma`.

---

## MapperConfig (`mapper`, also in `WorldConfig.mapper_config`)

Configuration for the oil spill estimation mapper.

*   `min_oil_points_for_estimate` (Default: 3)
    *   **Purpose**: Minimum number of unique oil-detecting sensor locations required to attempt a Convex Hull estimation.
    *   **Tuning**:
        *   **Increasing (e.g., 5, 10)**: Mapper waits for more evidence before forming an estimate. Estimates might be more robust but appear later. Could lead to an "uninitialized_mapper_penalty" for longer if the penalty is active.
        *   **Decreasing (e.g., 2, but 3 is practical min for ConvexHull)**: Mapper forms estimates sooner with less data. Estimates might be noisy or small initially.
    *   **Effect**: How quickly and with how much data the spill shape is estimated.

---

## TrainingConfig (`training`)

General parameters for the training loop.

*   `num_episodes` (Default: 30,000)
    *   **Purpose**: Total number of episodes to train the agent.
    *   **Tuning**: Increase for more training, decrease for less. Depends on task complexity and convergence speed.
    *   **Effect**: Total training duration.
*   `max_steps` (Default: 350)
    *   **Purpose**: Maximum number of steps allowed per episode.
    *   **Tuning**:
        *   **Increasing**: Allows agent more time to solve the task or explore within an episode.
        *   **Decreasing**: Forces episodes to end sooner. Can be useful if tasks should be solved quickly or to prevent agent from getting stuck.
    *   **Effect**: Episode length.
*   `batch_size` (Default: 512) - **This is for SAC updates.**
    *   **Purpose**: Number of transitions sampled from the replay buffer for each SAC gradient update.
    *   **Tuning**:
        *   **Increasing**: More stable gradient estimates, but each update step takes longer.
        *   **Decreasing**: Noisier gradients, faster updates, but might require lower learning rates.
    *   **Effect**: Stability and speed of SAC updates.
*   `save_interval` (Default: 200 episodes)
    *   **Purpose**: Interval (in episodes) for saving model checkpoints.
    *   **Tuning**: Adjust based on how frequently you want backups.
    *   **Effect**: Checkpoint frequency.
*   `log_frequency` (Default: 10 episodes)
    *   **Purpose**: Frequency (in episodes) for logging metrics to TensorBoard.
    *   **Tuning**: Adjust for desired logging granularity.
    *   **Effect**: Logging detail.
*   `learning_starts` (Default: 8000 steps) - **For SAC.**
    *   **Purpose**: Number of environment steps to take (populating the replay buffer with random actions or an initial policy) before starting SAC training updates.
    *   **Tuning**:
        *   **Increasing**: Ensures buffer has more diverse experiences before training, can improve initial stability.
        *   **Decreasing**: Starts learning sooner, but initial updates might be on less representative data.
        *   Should be at least `batch_size`.
    *   **Effect**: When SAC learning begins.
*   `train_freq` (Default: 4 steps) - **For SAC.**
    *   **Purpose**: Perform a training update every `train_freq` environment steps.
    *   **Tuning**:
        *   **Increasing**: Fewer updates relative to environment interaction, potentially more sample efficient but policy updates less frequently.
        *   **Decreasing (e.g., 1)**: More updates, policy changes more rapidly.
        *   Interacts with `gradient_steps`.
    *   **Effect**: Frequency of SAC updates.
*   `gradient_steps` (Default: 1) - **For SAC.**
    *   **Purpose**: Number of gradient updates to perform when `train_freq` is met.
    *   **Tuning**:
        *   **`gradient_steps` > 1 with `train_freq` > 1**: Collect `train_freq` steps, then do `gradient_steps` updates on that data or sampled batches.
        *   If `gradient_steps` = `train_freq` (and `train_freq` > 1), it's like doing one update per env step on average but batched.
        *   Often kept at 1. Some algorithms (like DDPG variants) might use `gradient_steps` = `episode_length`.
    *   **Effect**: Number of learning updates per data collection cycle.
*   `enable_early_stopping` (Default: `False`)
    *   **Purpose**: Whether to enable early stopping based on average reward.
    *   **Tuning**: Set to `True` if you want training to stop once a performance threshold is met.
    *   **Effect**: Conditional training termination.
*   `early_stopping_threshold` (Default: 50)
    *   **Purpose**: Average reward threshold for early stopping.
    *   **Tuning**: Set to a desired performance level.
    *   **Effect**: Target reward for stopping.
*   `early_stopping_window` (Default: 50 episodes)
    *   **Purpose**: Window size (in episodes) for averaging reward for early stopping.
    *   **Tuning**:
        *   **Increasing**: Smoother average, less sensitive to noisy episodes, but slower to react to true improvement.
        *   **Decreasing**: More responsive, but more prone to premature stopping due to lucky streaks.
    *   **Effect**: Sensitivity of early stopping.

---

## EvaluationConfig (`evaluation`)

Parameters for the evaluation phase.

*   `num_episodes` (Default: 6)
    *   **Purpose**: Number of episodes to run for evaluation.
    *   **Tuning**: Increase for more statistically significant evaluation results.
    *   **Effect**: Robustness of evaluation.
*   `max_steps` (Default: 200)
    *   **Purpose**: Maximum steps per evaluation episode.
    *   **Tuning**: Should be sufficient for the agent to demonstrate its learned behavior.
    *   **Effect**: Length of evaluation episodes.
*   `render` (Default: `True`)
    *   **Purpose**: Whether to render evaluation episodes (generate GIFs/videos).
    *   **Tuning**: `True` to see agent behavior, `False` for faster evaluation if visualization isn't needed.
    *   **Effect**: Visual output during evaluation.
*   `use_stochastic_policy_eval` (Default: `False`)
    *   **Purpose**: Whether to use a stochastic (sampled actions) or deterministic (mean actions) policy during evaluation.
    *   **Tuning**:
        *   `False` (deterministic): Standard for evaluating learned policy's direct performance.
        *   `True` (stochastic): Evaluates performance with exploration noise; can reveal robustness or how much exploration impacts outcomes.
    *   **Effect**: Action selection mode during evaluation.

---

## Position, Velocity, RandomizationRange

These are data structures, not directly tuned as hyperparameters, but their instances within `WorldConfig` define environment properties.

---

## VisualizationConfig (`visualization`)

Parameters for controlling visualizations. These do **not** affect agent learning, only the output visuals.

*   `save_dir` (Default: "mapping_snapshots")
    *   **Purpose**: Directory for saving visualizations.
*   `figure_size` (Default: (10, 10))
    *   **Purpose**: Figure size for Matplotlib plots.
*   `max_trajectory_points` (Default: 5)
    *   **Purpose**: Max recent trajectory points of the agent to display.
    *   **Tuning**: Increase for longer trails, decrease for shorter.
*   `output_format` (Default: 'gif')
    *   **Purpose**: 'gif' or 'mp4' for rendered episodes.
*   `video_fps` (Default: 15)
    *   **Purpose**: Frames per second for video/GIF.
*   `delete_png_frames` (Default: `True`)
    *   **Purpose**: Whether to delete individual PNG frames after GIF/video creation.
*   `sensor_marker_size`, `sensor_color_oil`, `sensor_color_water`, `plot_oil_points`, `plot_water_points`, `point_marker_size`:
    *   **Purpose**: Aesthetic parameters for the plot.

---

## WorldConfig (`world`)

Crucial for defining the environment dynamics, observation space, and reward structure. Many of these parameters define the task difficulty.

*   `CORE_STATE_DIM`, `CORE_ACTION_DIM`, `TRAJECTORY_REWARD_DIM`:
    *   **Purpose**: Defaulted from global constants, can be overridden here if a specific world instance needs different core feature dimensions.
    *   **Tuning**: Generally not tuned unless making a custom world variant.
*   `dt` (Default: 1.0)
    *   **Purpose**: Simulation time step.
    *   **Tuning**:
        *   **Increasing**: Agent covers more distance per step (if speed is constant). Can make control harder if too large.
        *   **Decreasing**: Finer control, smoother movement, but more steps needed to cover distance.
    *   **Effect**: Granularity of simulation.
*   `world_size` (Default: (125.0, 125.0))
    *   **Purpose**: Dimensions (X, Y) of the world.
    *   **Tuning**: Defines the scale of the problem. Larger worlds might require more exploration or longer episodes.
    *   **Effect**: Environment scale.
*   `normalize_coords` (Default: `True`)
    *   **Purpose**: Whether to normalize agent coordinates in the state representation to `[0, 1]`.
    *   **Tuning**: `True` is highly recommended as it makes the learning problem easier for the NN by providing consistent input ranges.
    *   **Effect**: Agent's coordinate representation.
*   `agent_speed` (Default: 3 units/dt)
    *   **Purpose**: Agent's movement speed.
    *   **Tuning**: Critical.
        *   **Increasing**: Agent explores faster but may overshoot targets or have difficulty with fine maneuvers.
        *   **Decreasing**: More precise control but slower exploration.
    *   **Effect**: Agent's movement capability.
*   `yaw_angle_range` (Default: `(-math.pi / 6, math.pi / 6)`)
    *   **Purpose**: Min/max change in yaw angle per step. Action output is normalized to `[-1, 1]` and then scaled by `yaw_angle_range[1]`.
    *   **Tuning**:
        *   **Widening range**: Agent can turn more sharply.
        *   **Narrowing range**: Finer turning control, but slower to make large turns.
    *   **Effect**: Agent's turning agility.
*   `num_sensors` (Default: 5)
    *   **Purpose**: Number of sensors on the agent.
    *   **Tuning**:
        *   **Increasing**: More detailed perception of the immediate surroundings. Increases state dimension.
        *   **Decreasing**: Coarser perception.
    *   **Effect**: Agent's perceptual acuity.
*   `sensor_distance` (Default: 2.5 units)
    *   **Purpose**: Distance of sensors from the agent's center.
    *   **Tuning**:
        *   **Increasing**: Sensors "see" further ahead/around.
        *   **Decreasing**: Sensors detect things closer to the agent.
    *   **Effect**: Sensor placement.
*   `sensor_radius` (Default: 4.0 units)
    *   **Purpose**: Detection radius of each sensor.
    *   **Tuning**:
        *   **Increasing**: Sensors detect oil from further away, larger field of view per sensor.
        *   **Decreasing**: Sensors require closer proximity to detect oil.
    *   **Effect**: Sensor sensitivity/range.
*   `agent_initial_location` (Default: `Position(x=50, y=10)`)
    *   **Purpose**: Default initial location if `randomize_agent_initial_location` is `False`.
    *   **Tuning**: Set a fixed start for debugging or specific scenarios.
*   `randomize_agent_initial_location` (Default: `True`)
    *   **Purpose**: Whether to randomize the agent's starting position within `agent_randomization_ranges`.
    *   **Tuning**: `True` for more diverse training starting conditions, `False` for fixed starts.
*   `agent_randomization_ranges` (Default: `x_range=(25.0, 100.0), y_range=(25.0, 100.0)`)
    *   **Purpose**: Min/Max X, Y ranges for randomizing agent start if enabled.
    *   **Tuning**: Define the area where the agent can start.
*   `num_oil_points` (Default: 200)
    *   **Purpose**: Number of true oil spill points.
    *   **Tuning**:
        *   **Increasing**: Denser, potentially larger/more complex spill to map.
        *   **Decreasing**: Sparser, potentially simpler spill.
    *   **Effect**: Spill complexity.
*   `num_water_points` (Default: 400)
    *   **Purpose**: Number of non-spill area points (currently not directly used in reward or state, but could be for visualization or future features).
    *   **Tuning**: Affects visual density if plotted.
*   `oil_cluster_std_dev_range` (Default: `(8.0, 10.0)`)
    *   **Purpose**: Range for standard deviation of the Gaussian cluster used to generate oil points if `randomize_oil_cluster` is `True`.
    *   **Tuning**:
        *   **Increasing std dev**: More spread-out, diffuse oil spills.
        *   **Decreasing std dev**: More compact, concentrated oil spills.
    *   **Effect**: Shape/spread of the oil spill.
*   `randomize_oil_cluster` (Default: `True`)
    *   **Purpose**: Whether to randomize oil center and std dev.
    *   **Tuning**: `True` for diverse spill scenarios, `False` for fixed spills.
*   `oil_center_randomization_range` (Default: `x_range=(25.0, 100.0), y_range=(25.0, 100.0)`)
    *   **Purpose**: Min/Max X, Y ranges for randomizing oil cluster center.
    *   **Tuning**: Defines where spills can appear.
*   `initial_oil_center` (Default: `Position(x=50, y=50)`)
    *   **Purpose**: Default oil center if not randomizing.
*   `initial_oil_std_dev` (Default: 10.0)
    *   **Purpose**: Default oil std dev if not randomizing.
*   `min_initial_separation_distance` (Default: 40.0)
    *   **Purpose**: Minimum distance between agent's starting location and the oil cluster's center.
    *   **Tuning**:
        *   **Increasing**: Ensures agent starts further away, potentially making the initial search harder.
        *   **Decreasing**: Agent can start closer, potentially easier initial detection.
    *   **Effect**: Initial difficulty of finding the spill.
*   `trajectory_length` (Default: 10) - **For RNNs/Trajectory States**
    *   **Purpose**: Number of past steps (N) to include in the trajectory state.
    *   **Tuning**:
        *   **Increasing**: Agent sees more history. Useful if long-term memory is beneficial. Increases input size to RNN.
        *   **Decreasing**: Agent sees less history.
    *   **Effect**: How much historical context the agent uses.
*   `trajectory_feature_dim` (Default: `CORE_STATE_DIM + CORE_ACTION_DIM + TRAJECTORY_REWARD_DIM`)
    *   **Purpose**: Dimension of features per step in the trajectory state. (Normalized state incl. heading + previous action + previous reward).
    *   **Tuning**: Derived from core dimensions, not directly tuned unless modifying what's in a trajectory step.
*   `max_steps` (Default: 350)
    *   **Purpose**: Maximum steps per episode from the world's perspective (can differ from `TrainingConfig.max_steps` if desired, but usually aligned).
    *   **Tuning**: Same as `TrainingConfig.max_steps`.
*   `success_metric_threshold` (Default: 0.95)
    *   **Purpose**: Performance metric (e.g., % points in hull) threshold for an episode to be considered a "success."
    *   **Tuning**: Set based on desired task completion criteria. Higher is harder.
    *   **Effect**: Definition of success.
*   `terminate_on_success` (Default: `True`)
    *   **Purpose**: Whether to end the episode if `success_metric_threshold` is met.
    *   **Tuning**: `True` if task is considered solved upon success, `False` to let agent continue (e.g., for refinement or if reward structure encourages further action).
*   `terminate_out_of_bounds` (Default: `True`)
    *   **Purpose**: Whether to end the episode if the agent goes out of bounds.
    *   **Tuning**: Almost always `True`.
*   **Reward Shaping Parameters**: These are highly influential and often require careful tuning.
    *   `metric_improvement_scale` (Default: 50.0)
        *   **Purpose**: Scaling factor for reward based on positive change in the performance metric.
        *   **Tuning**: Higher values give stronger positive reinforcement for improving the map.
        *   **Effect**: Encourages improving the spill estimate.
    *   `step_penalty` (Default: 0)
        *   **Purpose**: Penalty applied at each step.
        *   **Tuning**:
            *   Small positive value (e.g., `0.01`, `0.1`): Encourages efficiency (solving in fewer steps).
            *   Zero: No penalty for taking time.
        *   **Effect**: Encourages shorter solutions.
    *   `new_oil_detection_bonus` (Default: 0.0)
        *   **Purpose**: Bonus for new oil detections by sensors (sensors that were off and are now on).
        *   **Tuning**: Positive value rewards exploration leading to new information.
        *   **Effect**: Encourages finding new parts of the spill.
    *   `out_of_bounds_penalty` (Default: 20.0)
        *   **Purpose**: Penalty for going out of world bounds.
        *   **Tuning**: Should be significantly negative to discourage this behavior. Magnitude relative to other rewards matters.
        *   **Effect**: Discourages leaving the designated area.
    *   `success_bonus` (Default: 55.0)
        *   **Purpose**: Bonus awarded upon reaching the `success_metric_threshold`.
        *   **Tuning**: Large positive value to strongly reinforce successful task completion.
        *   **Effect**: Reinforces achieving the main goal.
    *   `uninitialized_mapper_penalty` (Default: 0)
        *   **Purpose**: Penalty if the mapper has not yet formed an estimate (due to insufficient points).
        *   **Tuning**: Small negative value can encourage the agent to quickly gather enough points to form an initial estimate.
        *   **Effect**: Encourages initial data gathering for the mapper.
*   `mapper_config`: Nested `MapperConfig` instance.
*   `seeds` (Default: `[]`)
    *   **Purpose**: List of seeds for environment generation during evaluation or specific resets. Used to ensure reproducible evaluation scenarios.
    *   **Tuning**: Populate with specific integers for consistent testing.

---

## DefaultConfig (Top-Level)

Container for all other configurations.

*   `sac`, `ppo`, `replay_buffer`, `training`, `evaluation`, `world`, `mapper`, `visualization`: Instances of the above configurations.
*   `cuda_device` (Default: "cuda:0")
    *   **Purpose**: CUDA device to use (e.g., "cuda:0", "cuda:1", "cpu").
    *   **Tuning**: Set based on available hardware. "cpu" for CPU-only training.
    *   **Effect**: Hardware used for training.
*   `algorithm` (Default: "sac")
    *   **Purpose**: RL algorithm to use ("sac" or "ppo").
    *   **Tuning**: Choose the algorithm you want to run.
    *   **Effect**: Selects the learning algorithm.

