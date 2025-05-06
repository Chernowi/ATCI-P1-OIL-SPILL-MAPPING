# Hyperparameter Tuning Guide for Oil Spill Mapping RL Agents

## Introduction

Hyperparameter tuning is a crucial step in getting the best performance out of Reinforcement Learning (RL) agents. Unlike model parameters learned during training (like neural network weights), hyperparameters are set *before* training starts. They define the agent's architecture, learning process, and interaction with the environment.

Finding the optimal set of hyperparameters is often an empirical process involving experimentation, careful observation, and iteration. There's rarely a single "best" set that works for all problems or even all variations of the same problem.

This guide provides insights into each hyperparameter in `configs.py`, explaining its role and offering strategies for tuning.

**Key Principles:**

1.  **Iterative Process:** Tune one or a small group of related parameters at a time.
2.  **Metrics are Key:** Monitor relevant metrics (e.g., episode reward, success rate, point inclusion metric, loss curves, entropy) using tools like TensorBoard.
3.  **Understand the Trade-offs:** Many parameters involve trade-offs (e.g., learning speed vs. stability, exploration vs. exploitation).
4.  **Start Simple:** Begin with standard values or simpler configurations before adding complexity (like PER or RNNs).
5.  **Computational Cost:** Be mindful of the computational resources required for extensive tuning (consider tools like Optuna or Ray Tune for automated sweeps if necessary).
6.  **Context Matters:** Optimal values depend heavily on the specific environment dynamics, reward structure, and chosen algorithm.

## General Tuning Strategy

1.  **Establish a Baseline:** Run training with default or standard hyperparameter values to get a baseline performance level.
2.  **Identify Key Parameters:** Focus initial tuning efforts on parameters known to have a significant impact (e.g., learning rates, discount factor, entropy coefficient, PPO clip range, buffer size).
3.  **Tune Systematically:**
    *   Change one parameter (or a closely related pair, like actor/critic LRs) at a time.
    *   Try values across a reasonable range (e.g., logarithmic scale for learning rates: 1e-5, 3e-5, 1e-4, 3e-4).
    *   Run multiple seeds for each setting to account for randomness.
4.  **Log Everything:** Use TensorBoard or similar tools to log metrics throughout training. Compare learning curves, final performance, and stability across different runs.
5.  **Analyze Results:** Determine which changes improved performance based on your chosen metrics (e.g., higher average reward, faster convergence, better final performance metric, higher success rate).
6.  **Iterate:** Based on the analysis, refine the ranges for promising parameters or move on to tuning others.

---

## Configuration Class Details

### `SACConfig` / `TSACConfig` (Soft Actor-Critic / Transformer SAC)

These parameters control the SAC/TSAC agent's architecture and learning process.

*   **`state_dim`, `action_dim`**:
    *   **Description:** Dimensions of the environment's state and action spaces.
    *   **Tuning:** Usually determined by the environment (`CORE_STATE_DIM`, `CORE_ACTION_DIM`) and **not tuned**. Ensure they match the `World`'s output.

*   **`hidden_dims` (List[int])**:
    *   **Description:** Defines the size and number of hidden layers in the MLP parts of the Actor and Critic networks (for SAC) or just the Actor (for TSAC).
    *   **Tuning:** Controls network capacity.
        *   *Start:* Common defaults like `[256, 256]` are often sufficient.
        *   *Increase:* If the agent struggles to learn complex relationships (consistently low performance, flat losses), try increasing layer sizes (e.g., `[512, 512]`) or adding a layer (e.g., `[256, 256, 256]`). Be cautious of overfitting and increased computation.
        *   *Decrease:* If training is slow or overfitting is suspected, try smaller layers (e.g., `[128, 128]`).
        *   *Monitor:* Actor/Critic loss curves, overall performance.

*   **`log_std_min`, `log_std_max` (int)**:
    *   **Description:** Bounds for the logarithm of the standard deviation of the action distribution. Prevents policy from becoming too deterministic (log_std -> -inf) or too random (log_std -> +inf).
    *   **Tuning:** Standard values (`-20`, `1` or `2`) usually work well. Significant tuning is rarely needed unless exploration is severely problematic. Adjusting might slightly influence the *range* of exploration noise.

*   **`actor_lr`, `critic_lr` (float)**:
    *   **Description:** Learning rates for the Adam optimizers of the actor and critic networks.
    *   **Tuning:** **Crucial**. Controls how quickly network weights are updated.
        *   *Range:* Typically `1e-5` to `1e-3`. SAC/TSAC often use lower LRs like `3e-5` to `3e-4`.
        *   *Too High:* Can lead to instability (spiky loss curves, diverging performance).
        *   *Too Low:* Can lead to very slow learning.
        *   *Strategy:* Tune logarithmically (e.g., 1e-5, 3e-5, 1e-4, 3e-4). Often tuned together, keeping them equal or critic slightly higher. Monitor loss curves closely. If losses explode, decrease LR. If losses decrease extremely slowly, increase LR cautiously.

*   **`gamma` (float)**:
    *   **Description:** Discount factor for future rewards. Determines the importance of future rewards relative to immediate ones.
    *   **Tuning:**
        *   *Range:* `0.9` to `0.999`. Standard is `0.99`.
        *   *Effect:* Values closer to 1 encourage far-sighted behavior, considering long-term consequences. Values closer to 0 make the agent short-sighted, focusing on immediate rewards.
        *   *Consider:* The effective horizon of the task (`1 / (1 - gamma)`). For tasks requiring long sequences of actions (like efficient mapping), higher gamma (`0.99`, `0.995`) is usually better.

*   **`tau` (float)**:
    *   **Description:** Interpolation factor for the soft target network updates (`target_weights = tau * main_weights + (1 - tau) * target_weights`).
    *   **Tuning:**
        *   *Range:* Typically `0.001` to `0.01`. Standard is `0.005`.
        *   *Effect:* Controls how quickly the target networks track the main networks. Smaller `tau` -> slower updates, more stable targets, potentially slower learning. Larger `tau` -> faster updates, less stable targets, potentially faster learning but risks instability. Usually kept at standard values unless significant stability issues arise.

*   **`alpha` (float)**:
    *   **Description:** (Initial) Temperature parameter for SAC/TSAC entropy regularization. Balances maximizing reward vs. maximizing policy entropy (encouraging exploration).
    *   **Tuning:** Only relevant if `auto_tune_alpha` is `False`.
        *   *Effect:* Higher `alpha` -> more exploration (policy closer to uniform random). Lower `alpha` -> less exploration (policy more deterministic).
        *   *Strategy:* If `auto_tune_alpha=False`, tune based on observed exploration. If the agent explores too little (gets stuck), increase `alpha`. If it explores too much (acts randomly, low reward), decrease `alpha`. Monitor policy entropy if logged. **Using `auto_tune_alpha` is generally recommended.**

*   **`auto_tune_alpha` (bool)**:
    *   **Description:** Whether to automatically tune the `alpha` parameter to target a specific entropy level (`target_entropy`, usually based on action dimension).
    *   **Tuning:** Usually set to `True`. It dynamically adjusts exploration based on policy entropy, often leading to better performance and removing the need to manually tune `alpha`. If set to `True`, `alpha` serves as the *initial* value.

*   **`use_rnn` (bool) (SAC only)**:
    *   **Description:** Whether to use RNN (LSTM/GRU) layers in the actor and critic networks. Useful for Partially Observable Markov Decision Processes (POMDPs) where history matters.
    *   **Tuning:** Set to `True` if you believe temporal context beyond the current `trajectory_length` is important *or* if you want the network to explicitly model sequences *within* the trajectory. Set to `False` (default) for standard MLP architecture processing only the last state (or final RNN hidden state). TSAC uses a Transformer, so `use_rnn` should be `False`.

*   **`rnn_type` ('lstm' or 'gru') (SAC only)**:
    *   **Description:** Type of RNN cell if `use_rnn` is `True`.
    *   **Tuning:** Often an empirical choice. LSTM has memory cells, potentially better for longer dependencies. GRU is simpler, potentially faster. Try both if using RNNs.

*   **`rnn_hidden_size` (int) (SAC only)**:
    *   **Description:** Number of features in the RNN hidden state.
    *   **Tuning:** Similar to `hidden_dims`. Controls RNN capacity. Start reasonably (e.g., `128`, `256`) and adjust based on performance and computational cost.

*   **`rnn_num_layers` (int) (SAC only)**:
    *   **Description:** Number of stacked RNN layers.
    *   **Tuning:** More layers increase capacity but also complexity and computational cost. `1` or `2` layers are common.

*   **`use_state_normalization` (bool)**:
    *   **Description:** Whether to use a `RunningMeanStd` normalizer *within the agent* on the state features received from the environment (which might already be partially normalized, like coordinates). Normalizes features to have zero mean and unit variance based on experience.
    *   **Tuning:** Generally recommended (`True`), especially if state features have different scales (e.g., sensor readings [0,1] vs. potentially large unnormalized coordinates). It stabilizes learning. Turn `False` only if you are certain all input features are already well-scaled and stable. Monitor normalizer stats (mean, std, count) if logged.

*   **`use_reward_normalization` (bool) (SAC only)**:
    *   **Description:** Whether to normalize rewards within the batch before calculating target Q-values. Can help stabilize learning, especially if rewards have high variance or large magnitudes.
    *   **Tuning:** Often beneficial (`True`). Can sometimes hinder learning if rewards are very sparse or already well-scaled. Try both `True` and `False`. PPO handles reward normalization differently (often within GAE).

*   **`use_per` (bool)**:
    *   **Description:** Enables Prioritized Experience Replay. Samples transitions based on their TD error (importance) rather than uniformly.
    *   **Tuning:** Can significantly speed up learning, especially when important transitions are rare. Set to `True` to enable. Requires tuning associated `per_*` parameters. Increases computational overhead slightly.

*   **`per_alpha` (float)**:
    *   **Description:** PER exponent `alpha`. Controls how much prioritization is used (0: uniform sampling, 1: full prioritization based on TD error rank).
    *   **Tuning:**
        *   *Range:* `0.5` to `0.8`. Common start: `0.6`.
        *   *Effect:* Higher `alpha` focuses more strongly on high-error samples. Too high might lead to overfitting on outliers. Tune based on stability and learning speed.

*   **`per_beta_start` (float)**:
    *   **Description:** PER initial value for importance sampling exponent `beta`. Anneals towards 1.0 over `per_beta_frames`. Corrects bias introduced by non-uniform sampling.
    *   **Tuning:**
        *   *Range:* `0.4` to `0.6`. Common start: `0.4`.
        *   *Effect:* Controls the amount of importance sampling correction early in training. Should anneal to 1.0 for unbiased estimates eventually. Standard values usually work well.

*   **`per_beta_frames` (int)**:
    *   **Description:** Number of *training updates* over which `beta` anneals from `per_beta_start` to 1.0.
    *   **Tuning:** Set based on the expected length of training (in terms of updates). Should be long enough for `beta` to reach 1.0 before training converges. E.g., `100,000` to `1,000,000`. Monitor the `beta` value in logs.

*   **`per_epsilon` (float)**:
    *   **Description:** Small positive value added to priorities to ensure even transitions with zero TD error have a non-zero chance of being sampled.
    *   **Tuning:** Usually kept at a small value like `1e-5` or `1e-6`. Not typically tuned unless numerical issues arise.

*   **`embedding_dim` (int) (TSAC only)**:
    *   **Description:** Dimensionality of the embeddings used for states and actions within the Transformer Critic.
    *   **Tuning:** Controls capacity of the embedding layers. Related to `transformer_hidden_dim`. Start reasonably (e.g., `128`, `256`). Tune based on performance and computational cost.

*   **`transformer_n_layers`, `transformer_n_heads`, `transformer_hidden_dim` (int) (TSAC only)**:
    *   **Description:** Core parameters of the Transformer Encoder layers in the Critic. Control its capacity and complexity. `n_heads` must be a divisor of `embedding_dim`.
    *   **Tuning:** Sensitive parameters.
        *   *Start:* Begin with smaller values (e.g., `n_layers=2`, `n_heads=4`, `hidden_dim=256` or `512`).
        *   *Increase:* If critic loss remains high or performance plateaus, cautiously increase complexity (more layers, heads, or hidden dim). Monitor VRAM usage and training time.
        *   *Monitor:* Critic loss curves. Ensure they are decreasing.

*   **`use_layer_norm_actor` (bool) (TSAC only)**:
    *   **Description:** Whether to apply Layer Normalization within the Actor's MLP layers.
    *   **Tuning:** Often helps stabilize training (`True`). Try `False` only if experiencing issues potentially related to normalization.

### `PPOConfig` (Proximal Policy Optimization)

*   **`state_dim`, `action_dim`**:
    *   **Description:** Environment dimensions.
    *   **Tuning:** Fixed by the environment.

*   **`hidden_dim` (int)**:
    *   **Description:** Size of hidden layers in Actor and Critic MLPs.
    *   **Tuning:** Similar to SAC `hidden_dims`. `256` is a common starting point. Adjust based on problem complexity and performance.

*   **`log_std_min`, `log_std_max` (int)**:
    *   **Description:** Bounds for the learned log standard deviation parameter in the policy.
    *   **Tuning:** Similar to SAC. Standard values (`-20`, `2`) usually work.

*   **`actor_lr`, `critic_lr` (float)**:
    *   **Description:** Learning rates for Actor and Critic.
    *   **Tuning:** **Crucial**. PPO often tolerates slightly higher LRs than SAC.
        *   *Range:* `1e-5` to `5e-4`. Common start: `1e-4` or `3e-4`.
        *   *Strategy:* Tune logarithmically. Monitor actor/critic losses and policy entropy. Ensure stability.

*   **`gamma` (float)**:
    *   **Description:** Discount factor.
    *   **Tuning:** Same as SAC. Usually `0.99`.

*   **`gae_lambda` (float)**:
    *   **Description:** Lambda parameter for Generalized Advantage Estimation (GAE). Trades off bias and variance in advantage estimates.
    *   **Tuning:**
        *   *Range:* `0.9` to `1.0`. Common default: `0.95`.
        *   *Effect:* `lambda=1` is high variance (Monte Carlo returns). `lambda=0` is high bias (TD(0) error). `0.95`-`0.98` often provides a good balance.

*   **`policy_clip` (float)**:
    *   **Description:** PPO's clipping parameter (`epsilon`). Restricts the change in the policy during updates to prevent large, destabilizing steps.
    *   **Tuning:** **Crucial PPO parameter.**
        *   *Range:* `0.1` to `0.3`. Common default: `0.2`.
        *   *Effect:* Smaller values -> smaller policy updates, potentially more stable but slower learning. Larger values -> larger policy updates, potentially faster learning but risks instability.
        *   *Strategy:* Start with `0.2`. If updates seem unstable (losses spike, performance collapses), decrease to `0.15` or `0.1`. If learning is stable but slow, cautiously increase to `0.25` or `0.3`. Monitor approximate KL divergence between old and new policies (if logged) - it should stay relatively small.

*   **`n_epochs` (int)**:
    *   **Description:** Number of optimization epochs to run on the collected batch of data (`steps_per_update`) during each PPO update cycle.
    *   **Tuning:**
        *   *Range:* `3` to `20`. Common: `10`.
        *   *Effect:* More epochs utilize the collected data more thoroughly but increase computational cost per update cycle and risk overfitting to the current batch. Fewer epochs are faster per update but might be less sample efficient.
        *   *Trade-off:* Balance with `steps_per_update`. Often requires joint tuning.

*   **`entropy_coef` (float)**:
    *   **Description:** Coefficient for the entropy bonus added to the PPO objective. Encourages exploration by penalizing overly deterministic policies.
    *   **Tuning:**
        *   *Range:* `0.0` to `0.1`. Common start: `0.01`.
        *   *Effect:* Higher values promote more randomness (exploration). Lower values allow the policy to become more deterministic (exploitation).
        *   *Strategy:* Monitor policy entropy (if logged). If entropy collapses too quickly and the agent gets stuck, increase the coefficient. If the agent explores too much and fails to converge, decrease it.

*   **`value_coef` (float)**:
    *   **Description:** Coefficient for the value function loss (critic loss) in the total PPO loss.
    *   **Tuning:** Standard value is `0.5`. Usually kept fixed unless the value loss dominates or becomes negligible compared to the policy loss. Adjust if critic learning seems problematic.

*   **`batch_size` (int)**:
    *   **Description:** **Mini-batch size** used within each PPO optimization epoch. *Not* the total data collected per update. Must be less than or equal to `steps_per_update`.
    *   **Tuning:**
        *   *Range:* `32` to `512`. Common: `64`, `128`.
        *   *Effect:* Smaller mini-batches lead to noisier gradient estimates but can sometimes escape local optima. Larger mini-batches provide more stable gradients. Ensure `steps_per_update` is divisible by `batch_size` for efficient processing.

*   **`steps_per_update` (int)**:
    *   **Description:** Number of environment steps (transitions) collected between each PPO policy update. Defines the size of the rollout buffer.
    *   **Tuning:**
        *   *Range:* `512` to `8192` or more. Common: `2048`, `4096`.
        *   *Effect:* Larger values provide more data for each update, leading to more stable estimates of advantages and value targets, but updates are less frequent. Smaller values lead to more frequent updates but potentially higher variance.
        *   *Trade-off:* Balance stability, sample efficiency, and wall-clock time. Must be >= `batch_size`.

*   **`use_state_normalization`, `use_reward_normalization` (bool)**:
    *   **Description:** Use `RunningMeanStd` for states and normalize rewards (typically by batch std dev within GAE calculation).
    *   **Tuning:** Similar to SAC. Generally recommended (`True`) for PPO to stabilize learning. Monitor normalizer stats and ensure rewards aren't scaled down excessively if they are already small.

### `ReplayBufferConfig`

*   **`capacity` (int)**:
    *   **Description:** Maximum number of transitions (SAC/TSAC) or trajectories stored in the replay buffer.
    *   **Tuning:**
        *   *Range:* `1e5` to `2e6`. Common: `1e6`.
        *   *Effect:* Larger capacity stores more diverse, older data, reducing correlation but potentially slowing learning from recent experience and increasing memory usage. Smaller capacity focuses on recent data but might lead to catastrophic forgetting or instability due to correlated samples.
        *   *Strategy:* Choose based on memory constraints and observed stability. Ensure `learning_starts` allows the buffer to fill adequately before training starts.

*   **`gamma` (float)**:
    *   **Description:** Discount factor (should match agent's gamma).
    *   **Tuning:** Not tuned independently here; set by the agent's config. (Note: Duplication exists in the provided `configs.py`).

### `MapperConfig`

*   **`min_oil_points_for_estimate` (int)**:
    *   **Description:** Minimum number of unique sensor locations detecting oil required before the `Mapper` attempts to compute a Convex Hull estimate.
    *   **Tuning:** Domain-specific.
        *   *Effect:* Lower values (e.g., 3) allow estimates sooner but might be unstable/inaccurate initially. Higher values (e.g., 5, 10) delay the estimate until more data is gathered, potentially leading to a more stable first estimate but incurring an `uninitialized_mapper_penalty` for longer.
        *   *Strategy:* Tune based on how critical an early estimate is versus initial stability. Visual inspection during evaluation can help. Depends on sensor density and spill characteristics.

### `TrainingConfig`

*   **`num_episodes` (int)**:
    *   **Description:** Total number of episodes to train for.
    *   **Tuning:** Not a typical hyperparameter. Set based on observing convergence on the learning curve (e.g., when performance plateaus) and computational budget.

*   **`max_steps` (int)**:
    *   **Description:** Maximum number of steps allowed per episode.
    *   **Tuning:** Affects episode length.
        *   *Effect:* Longer episodes allow more time for exploration and achieving goals but can slow down learning (fewer episode resets). Shorter episodes provide faster learning signals (rewards/termination) but might prevent the agent from reaching distant goals.
        *   *Strategy:* Ensure it's long enough for the agent to potentially solve the task. Consider the impact on credit assignment (related to `gamma`).

*   **`batch_size` (int)**:
    *   **Description:** Batch size for SAC/TSAC updates (sampling from replay buffer). **Note:** PPO uses `PPOConfig.batch_size` for its mini-batch size during optimization epochs.
    *   **Tuning (SAC/TSAC):**
        *   *Range:* `128` to `1024`. Common: `256`, `512`.
        *   *Effect:* Larger batches provide more stable gradient estimates but require more computation per update. Smaller batches are faster per update but have noisier gradients. Often tuned in conjunction with learning rates.

*   **`save_interval`, `log_frequency` (int)**:
    *   **Description:** Frequency (in episodes) for saving model checkpoints and logging to TensorBoard.
    *   **Tuning:** Convenience parameters. Set based on desired monitoring frequency and storage space. Do not directly impact performance.

*   **`models_dir` (str)**:
    *   **Description:** Directory path for saving models.
    *   **Tuning:** Convenience parameter.

*   **`learning_starts` (int)**:
    *   **Description:** Number of environment steps to collect randomly before starting SAC/TSAC training updates. Allows the replay buffer to populate.
    *   **Tuning (SAC/TSAC):**
        *   *Range:* `batch_size` up to `50,000` or more. Common: `1000` to `10000`.
        *   *Effect:* Ensures initial updates use diverse data. Too small -> updates on highly correlated data initially, potentially unstable. Too large -> delays learning unnecessarily.

*   **`train_freq` (int)**:
    *   **Description:** Frequency (in environment steps) at which SAC/TSAC training updates are performed.
    *   **Tuning (SAC/TSAC):**
        *   *Range:* `1` to `10` or higher.
        *   *Effect:* `train_freq=1` means update after every environment step (most computationally intensive, potentially most sample efficient). Higher values reduce computation but update the policy less frequently.

*   **`gradient_steps` (int)**:
    *   **Description:** Number of gradient updates performed each time the `train_freq` condition is met (for SAC/TSAC).
    *   **Tuning (SAC/TSAC):**
        *   *Effect:* `gradient_steps=1` is standard. Values > 1 mean performing multiple updates using the same batch(es) sampled when the `train_freq` occurred. Can increase computational load per interaction but might improve learning speed if `train_freq > 1`. Often kept at `1`.

### `EvaluationConfig`

*   **`num_episodes`, `max_steps` (int)**:
    *   **Description:** Parameters controlling the evaluation process.
    *   **Tuning:** Set based on desired statistical significance and evaluation time constraints. Higher `num_episodes` gives more reliable performance estimates. `max_steps` should generally match or exceed the training `max_steps`.

*   **`render` (bool)**:
    *   **Description:** Whether to render the environment during evaluation (requires visualization libraries).
    *   **Tuning:** Convenience/Debugging. Set `True` to watch the agent's behavior.

### `VisualizationConfig`

*   **All parameters (`save_dir`, `figure_size`, etc.)**:
    *   **Description:** Control the output of visualizations.
    *   **Tuning:** Convenience/Aesthetics. Adjust for desired output format and clarity. Do not directly impact agent performance.

### `WorldConfig` (Environment Parameters - Tuning these changes the *problem*)

It's crucial to distinguish between agent hyperparameters and environment parameters. Tuning `WorldConfig` parameters changes the task the agent is trying to solve. If you change these, you will likely need to retune the *agent's* hyperparameters.

*   **`dt`, `agent_speed`, `yaw_angle_range`, `num_sensors`, `sensor_distance`, `sensor_radius`**: Define the physics and agent capabilities. Changing these alters task difficulty and dynamics.
*   **`normalize_coords`**: Should generally be `True` for stable NN inputs.
*   **`agent_initial_location`, `randomize_agent_initial_location`, `agent_randomization_ranges`**: Control starting conditions and generalization requirements. Randomization (`True`) is vital for training robust agents. Ranges define the state space explored.
*   **`num_oil_points`, `num_water_points`, `oil_cluster_std_dev_range`, `randomize_oil_cluster`, `oil_center_randomization_range`, `initial_oil_center`, `initial_oil_std_dev`**: Define the oil spill characteristics (size, shape, location variability). Directly impacts task difficulty and the observations the agent receives. Randomization (`True`) is vital.
*   **`trajectory_length`**: Defines the length of the state history provided to SAC/TSAC. Needs to be long enough to capture relevant temporal dependencies if using RNNs or Transformers. Tune based on perceived task memory requirements vs. computational cost.
*   **`success_metric_threshold`**: Defines what constitutes "success" for the task. Set based on application requirements.
*   **`terminate_on_success`, `terminate_out_of_bounds`**: Affect episode termination logic, influencing effective episode length and reward signals.
*   **Reward Function Parameters (`metric_improvement_scale`, `step_penalty`, `new_oil_detection_bonus`, `out_of_bounds_penalty`, `success_bonus`, `uninitialized_mapper_penalty`)**: **Reward Engineering**. These are **extremely influential**.
    *   **Description:** Shape the agent's behavior by defining what is "good".
    *   **Tuning:** Requires careful iteration and observation.
        *   *Start Simple:* Begin with only essential components (e.g., metric improvement, step penalty).
        *   *Observe Behavior:* Does the agent explore? Does it stay within bounds? Does it try to improve the map?
        *   *Adjust Weights:* If the agent is too slow, increase `step_penalty`. If it doesn't care about mapping, increase `metric_improvement_scale`. If it never finds oil, consider `new_oil_detection_bonus`. Use penalties (`out_of_bounds`, `uninitialized`) to discourage undesirable states. Use `success_bonus` for goal achievement.
        *   *Balance:* Ensure components don't counteract each other excessively or create perverse incentives. Monitor the contribution of each component to the total reward in logs.
*   **`seeds`**: Used for reproducible evaluation, not typically tuned during training setup.

---

## Conclusion

Hyperparameter tuning is an art and a science. Start with sensible defaults, change parameters methodically, rely on logging and metrics, and understand the trade-offs involved. Remember that environment parameters (`WorldConfig`) change the task itself, while agent parameters (`SACConfig`, `PPOConfig`, etc.) tune the agent's learning strategy for that task. Good luck!