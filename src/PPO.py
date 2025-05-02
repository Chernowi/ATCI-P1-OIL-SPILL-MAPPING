import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import time
from collections import deque
from tqdm import tqdm

# Local imports
from world import World # Mapping world
from configs import DefaultConfig, PPOConfig, TrainingConfig, CORE_STATE_DIM
from torch.utils.tensorboard import SummaryWriter
import math
from utils import RunningMeanStd # Normalization utility

class PPOMemory:
    # ... (PPOMemory code remains unchanged) ...
    """Memory buffer for PPO algorithm. Stores individual basic states (sensor readings + coords)."""
    def __init__(self, batch_size=64):
        # Stores individual transitions, not trajectories
        self.states = [] # List of basic_state tuples (raw, unnormalized, e.g., 7 floats)
        self.actions = [] # List of float actions (normalized yaw change)
        self.probs = []   # List of float log probs
        self.vals = []    # List of float values (from critic)
        self.rewards = [] # List of float rewards (received after taking action)
        self.dones = []   # List of bool dones (observed after taking action)
        self.batch_size = batch_size

    def store(self, basic_state, action, probs, vals, reward, done):
        """Store a transition corresponding to taking 'action' in 'basic_state'."""
        if not isinstance(basic_state, tuple) or len(basic_state) != CORE_STATE_DIM:
            print(f"Warning: PPO Memory storing state with unexpected format/length: {type(basic_state)}, len={len(basic_state) if isinstance(basic_state, tuple) else 'N/A'}. Expected tuple of length {CORE_STATE_DIM}.")
            return
        # Check for NaNs in the state tuple itself
        if any(np.isnan(x) for x in basic_state):
            return # Avoid storing NaN states

        self.states.append(basic_state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward) # Reward r_{t+1}
        self.dones.append(done)     # Done d_{t+1}

    def clear(self):
        self.states, self.actions, self.probs, self.vals, self.rewards, self.dones = [], [], [], [], [], []

    def generate_batches(self):
        """Generates batches of indices for sampling from stored transitions."""
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        try:
            states_arr = np.array(self.states, dtype=np.float32)
            if np.isnan(states_arr).any():
                 print("Error: NaN found in PPO memory states_arr during batch generation!")
                 return None, None, None, None, None, None, None

            actions_arr = np.array(self.actions, dtype=np.float32).reshape(-1, 1)
            probs_arr = np.array(self.probs, dtype=np.float32).reshape(-1, 1)
            vals_arr = np.array(self.vals, dtype=np.float32).reshape(-1, 1)
            rewards_arr = np.array(self.rewards, dtype=np.float32)
            dones_arr = np.array(self.dones, dtype=np.float32)

        except Exception as e:
             print(f"Error converting PPO memory to arrays: {e}")
             return None, None, None, None, None, None, None

        return states_arr, actions_arr, probs_arr, vals_arr, rewards_arr, dones_arr, batches

    def __len__(self):
        return len(self.states)

class PolicyNetwork(nn.Module):
    # ... (PolicyNetwork code remains unchanged) ...
    """Actor network for PPO. Takes normalized basic_state (sensors + coords) as input."""
    def __init__(self, config: PPOConfig):
        super(PolicyNetwork, self).__init__()
        self.state_dim = config.state_dim
        self.action_dim = config.action_dim
        self.hidden_dim = config.hidden_dim
        self.log_std_min = config.log_std_min
        self.log_std_max = config.log_std_max

        self.fc1 = nn.Linear(self.state_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.mean = nn.Linear(self.hidden_dim, self.action_dim)
        self.log_std = nn.Parameter(torch.zeros(1, self.action_dim))

    def forward(self, network_input): # Takes potentially normalized basic_state
        x = F.relu(self.fc1(network_input))
        x = F.relu(self.fc2(x))
        action_mean = self.mean(x)
        action_log_std = torch.clamp(self.log_std, self.log_std_min, self.log_std_max)
        action_std = torch.exp(action_log_std)
        return action_mean, action_std

    def sample(self, network_input):
        """Sample action (normalized) and calculate log_prob."""
        mean, std = self.forward(network_input)
        distribution = Normal(mean, std)
        x_t = distribution.sample()
        action_normalized = torch.tanh(x_t)

        log_prob_unbounded = distribution.log_prob(x_t)
        clamped_tanh = action_normalized.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        log_det_jacobian = torch.log(1.0 - clamped_tanh.pow(2) + 1e-7)
        log_prob = log_prob_unbounded - log_det_jacobian
        log_prob = log_prob.sum(1, keepdim=True)

        return action_normalized, log_prob

    def evaluate(self, network_input, action_normalized):
        """Evaluate log_prob and entropy for a given state and action."""
        mean, std = self.forward(network_input)
        distribution = Normal(mean, std)

        action_tanh = torch.clamp(action_normalized, -1.0 + 1e-6, 1.0 - 1e-6)
        action_original_space = torch.atanh(action_tanh)

        log_prob_unbounded = distribution.log_prob(action_original_space)
        log_det_jacobian = torch.log(1.0 - action_tanh.pow(2) + 1e-7)
        log_prob = log_prob_unbounded - log_det_jacobian
        log_prob = log_prob.sum(1, keepdim=True)

        entropy = distribution.entropy().sum(1, keepdim=True)
        return log_prob, entropy

class ValueNetwork(nn.Module):
    # ... (ValueNetwork code remains unchanged) ...
    """Critic network for PPO. Takes normalized basic_state (sensors + coords) as input."""
    def __init__(self, config: PPOConfig):
        super(ValueNetwork, self).__init__()
        self.state_dim = config.state_dim
        self.hidden_dim = config.hidden_dim
        self.fc1 = nn.Linear(self.state_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.value = nn.Linear(self.hidden_dim, 1)

    def forward(self, network_input): # Takes potentially normalized basic_state
        x = F.relu(self.fc1(network_input))
        x = F.relu(self.fc2(x))
        value = self.value(x)
        return value

class PPO:
    """Proximal Policy Optimization algorithm implementation with optional state normalization."""
    def __init__(self, config: PPOConfig, device: torch.device = None):
        self.config = config
        self.gamma = config.gamma
        self.gae_lambda = config.gae_lambda
        self.policy_clip = config.policy_clip
        self.n_epochs = config.n_epochs
        self.entropy_coef = config.entropy_coef
        self.value_coef = config.value_coef
        self.state_dim = config.state_dim
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"PPO Agent using device: {self.device}")

        # --- Conditional Normalization ---
        self.use_state_normalization = config.use_state_normalization
        self.state_normalizer = None
        if self.use_state_normalization:
            self.state_normalizer = RunningMeanStd(shape=(self.state_dim,), device=self.device)
            print(f"PPO Agent state normalization ENABLED for dim: {self.state_dim}")
        else:
            print(f"PPO Agent state normalization DISABLED.")
        # --- End Conditional Normalization ---

        self.actor = PolicyNetwork(config).to(self.device)
        self.critic = ValueNetwork(config).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.critic_lr)
        self.memory = PPOMemory(batch_size=config.batch_size)

    def select_action(self, state: dict, evaluate=False):
        """Select action based on the potentially normalized *basic state* tuple."""
        basic_state_tuple = state['basic_state'] # Tuple (sensor1..5, agent_x, agent_y)
        state_tensor_raw = torch.FloatTensor(basic_state_tuple).to(self.device).unsqueeze(0) # (1, state_dim)

        # --- Conditionally Normalize State ---
        if self.use_state_normalization and self.state_normalizer:
            # Use normalizer in eval mode for selection
            self.state_normalizer.eval()
            network_input = self.state_normalizer.normalize(state_tensor_raw)
            self.state_normalizer.train() # Switch back to train mode
        else:
            network_input = state_tensor_raw
        # --- End Conditional Normalize ---

        self.actor.eval()
        self.critic.eval()

        with torch.no_grad():
            if evaluate:
                action_mean, _ = self.actor.forward(network_input)
                action_normalized = torch.tanh(action_mean)
            else:
                action_normalized, log_prob = self.actor.sample(network_input)
                value = self.critic(network_input) # Get value from same (potentially normalized) input

                # Store raw state tuple
                self.memory.store(
                    basic_state_tuple,
                    action_normalized.cpu().numpy()[0, 0],
                    log_prob.cpu().numpy()[0, 0],
                    value.cpu().numpy()[0, 0],
                    0, False # Placeholder reward/done
                )

        self.actor.train()
        self.critic.train()

        return action_normalized.detach().cpu().numpy()[0, 0]

    def store_reward_done(self, reward, done):
        # ... (store_reward_done logic remains unchanged) ...
        """Store reward and done flag for the *last* transition added to memory."""
        if len(self.memory.rewards) < len(self.memory.states):
             idx_to_update = len(self.memory.rewards)
             self.memory.rewards.insert(idx_to_update, reward)
             self.memory.dones.insert(idx_to_update, done)
        elif len(self.memory.rewards) == len(self.memory.states) and len(self.memory.states) > 0:
             if self.memory.rewards[-1] == 0 and self.memory.dones[-1] is False:
                  self.memory.rewards[-1] = reward
                  self.memory.dones[-1] = done
             else:
                  pass # Entry already populated


    def update_parameters(self):
        """Update policy and value networks using PPO algorithm and collected transitions."""
        if len(self.memory) < self.config.batch_size:
            return None, 0.0

        states_arr_raw, actions_arr, old_log_probs_arr, values_arr, rewards_arr, dones_arr, batches = self.memory.generate_batches()

        if states_arr_raw is None:
            print("Error: Failed to generate batches from PPO memory. Skipping PPO update.")
            self.memory.clear()
            return None, 0.0

        # --- Conditionally Update State Normalizer Stats ---
        if self.use_state_normalization and self.state_normalizer:
            self.state_normalizer.update(torch.from_numpy(states_arr_raw).to(self.device))
        # --- End Update ---

        advantages, returns = self._compute_advantages_returns(rewards_arr, dones_arr, values_arr.squeeze())
        if advantages is None:
            print("Error: Failed to compute returns/advantages. Skipping PPO update.")
            self.memory.clear()
            return None, 0.0

        advantages_tensor = torch.tensor(advantages, dtype=torch.float32).unsqueeze(1).to(self.device)
        returns_tensor = torch.tensor(returns, dtype=torch.float32).unsqueeze(1).to(self.device)
        if len(advantages_tensor) > 1: advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)

        states_arr_raw_tensor = torch.from_numpy(states_arr_raw).to(self.device)
        actions_arr_tensor = torch.from_numpy(actions_arr).to(self.device)
        old_log_probs_arr_tensor = torch.from_numpy(old_log_probs_arr).to(self.device)

        actor_losses, critic_losses, entropies = [], [], []

        for _ in range(self.n_epochs):
            for batch_indices in batches:
                batch_states_raw = states_arr_raw_tensor[batch_indices]

                # --- Conditionally Normalize states for this batch ---
                if self.use_state_normalization and self.state_normalizer:
                    with torch.no_grad():
                        batch_network_input = self.state_normalizer.normalize(batch_states_raw)
                else:
                    batch_network_input = batch_states_raw
                # --- End Normalize ---

                batch_actions = actions_arr_tensor[batch_indices]
                batch_old_log_probs = old_log_probs_arr_tensor[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]

                # Evaluate current policy on potentially NORMALIZED states
                new_log_probs, entropy = self.actor.evaluate(batch_network_input, batch_actions)
                # Get value predictions from current critic on potentially NORMALIZED states
                new_values = self.critic(batch_network_input)

                # PPO Loss Calculation
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.policy_clip, 1.0 + self.policy_clip) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(new_values, batch_returns)
                entropy_loss = -entropy.mean()
                total_loss = actor_loss + self.value_coef * critic_loss + self.entropy_coef * entropy_loss

                # Optimization Step
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.actor_optimizer.step()
                self.critic_optimizer.step()

                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
                entropies.append(entropy.mean().item())

        avg_actor_loss = np.mean(actor_losses) if actor_losses else 0.0
        avg_critic_loss = np.mean(critic_losses) if critic_losses else 0.0
        avg_entropy = np.mean(entropies) if entropies else 0.0

        self.memory.clear()

        return {'actor_loss': avg_actor_loss, 'critic_loss': avg_critic_loss, 'entropy': avg_entropy}


    def _compute_advantages_returns(self, rewards, dones, values):
        """Compute advantages and returns using GAE, handling conditional normalization for V(s_N)."""
        n_steps = len(rewards)
        advantages = np.zeros(n_steps, dtype=np.float32)
        returns = np.zeros(n_steps, dtype=np.float32)
        last_gae_lam = 0.0
        last_value = 0.0

        if not dones[-1]:
            if self.memory.states:
                last_basic_state_tuple = self.memory.states[-1]
                last_state_tensor_raw = torch.FloatTensor(last_basic_state_tuple).to(self.device).unsqueeze(0)
                with torch.no_grad():
                    # --- Conditionally Normalize last state for V(s_N) ---
                    if self.use_state_normalization and self.state_normalizer:
                         # Use normalizer in eval mode here
                         self.state_normalizer.eval()
                         network_input_last = self.state_normalizer.normalize(last_state_tensor_raw)
                         self.state_normalizer.train() # Switch back
                    else:
                         network_input_last = last_state_tensor_raw
                    # --- End Conditional ---
                    last_value = self.critic(network_input_last).cpu().numpy()[0, 0]
            else:
                print("Warning: GAE calculation attempted with empty memory states.")
                return None, None

        for t in reversed(range(n_steps)):
            if t == n_steps - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value = last_value
            else:
                next_non_terminal = 1.0 - dones[t]
                next_value = values[t+1]

            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            advantages[t] = last_gae_lam

        returns = advantages + values
        return advantages, returns

    def save_model(self, path: str):
        print(f"Saving PPO model to {path}...")
        save_dict = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'use_state_normalization': self.use_state_normalization, # Save the flag
            'device_type': self.device.type
        }
        # --- Conditionally save normalizer ---
        if self.use_state_normalization and self.state_normalizer:
            save_dict['state_normalizer_state_dict'] = self.state_normalizer.state_dict()
        # --- End Conditional ---
        torch.save(save_dict, path)

    def load_model(self, path: str):
        if not os.path.exists(path):
            print(f"Warning: PPO model file not found: {path}. Skipping loading.")
            return
        print(f"Loading PPO model from {path}...")
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])

        # --- Load Normalizer State Conditionally ---
        loaded_use_norm = checkpoint.get('use_state_normalization', True)
        if loaded_use_norm != self.use_state_normalization:
            print(f"Warning: Loaded model normalization setting ({loaded_use_norm}) differs from current config ({self.use_state_normalization}). Using current config setting.")

        if self.use_state_normalization and self.state_normalizer:
            if 'state_normalizer_state_dict' in checkpoint:
                try:
                    self.state_normalizer.load_state_dict(checkpoint['state_normalizer_state_dict'])
                    print("Loaded PPO state normalizer statistics.")
                except Exception as e:
                     print(f"Warning: Failed to load PPO state normalizer stats: {e}. Using initial values.")
            else:
                print("Warning: PPO state normalizer statistics not found in checkpoint, but normalization is enabled. Using initial values.")
        elif not self.use_state_normalization and 'state_normalizer_state_dict' in checkpoint:
             print("Warning: Checkpoint contains PPO normalizer stats, but normalization is disabled in current config. Ignoring saved stats.")
        # --- End Load Normalizer ---

        self.actor.train()
        self.critic.train()
        # --- Conditionally set normalizer mode ---
        if self.use_state_normalization and self.state_normalizer:
            self.state_normalizer.train()
        # --- End Conditional ---

        print(f"PPO model loaded successfully from {path}")


# --- Training Loop (train_ppo) ---
def train_ppo(config: DefaultConfig, use_multi_gpu: bool = False, run_evaluation: bool = True):
    # ... (Setup remains the same) ...
    ppo_config = config.ppo
    train_config = config.training
    world_config = config.world
    cuda_device = config.cuda_device

    log_dir = os.path.join("runs", f"ppo_mapping_{int(time.time())}")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logs: {log_dir}")

    # Device Setup
    # ... (remains same) ...

    # Agent initialization now uses the config flag
    agent = PPO(config=ppo_config, device=device)
    os.makedirs(train_config.models_dir, exist_ok=True)

    # Checkpoint loading
    # ... (remains same, agent.load_model handles conditional norm loading) ...

    # Training Loop
    episode_rewards = []
    timing_metrics = {'env_step_time': deque(maxlen=100), 'parameter_update_time': deque(maxlen=100)}
    update_frequency = ppo_config.steps_per_update
    pbar = tqdm(range(start_episode, train_config.num_episodes + 1),
                desc="Training PPO Mapping", unit="episode", initial=start_episode-1, total=train_config.num_episodes)

    world = World(world_config=world_config)
    learn_steps = 0

    for episode in pbar:
        state = world.reset()
        episode_reward = 0
        episode_steps = 0
        episode_ious = []

        for step_in_episode in range(train_config.max_steps):
            # select_action uses normalizer conditionally
            action_normalized = agent.select_action(state, evaluate=False)

            step_start_time = time.time()
            next_state = world.step(action_normalized, training=True, terminal_step=(step_in_episode == train_config.max_steps - 1))
            step_time = time.time() - step_start_time
            timing_metrics['env_step_time'].append(step_time)

            reward = world.reward
            done = world.done
            current_iou = world.iou

            agent.store_reward_done(reward, done)

            state = next_state
            episode_reward += reward
            episode_steps += 1
            total_steps += 1
            learn_steps += 1
            episode_ious.append(current_iou)

            # update_parameters uses normalizer conditionally
            if learn_steps >= update_frequency:
                update_start_time = time.time()
                losses = agent.update_parameters()
                update_time = time.time() - update_start_time
                if losses:
                    timing_metrics['parameter_update_time'].append(update_time)
                    writer.add_scalar('Loss/Actor', losses['actor_loss'], total_steps)
                    writer.add_scalar('Loss/Critic', losses['critic_loss'], total_steps)
                    writer.add_scalar('Policy/Entropy', losses['entropy'], total_steps)
                learn_steps = 0

            if done:
                break

        # --- Logging (End of Episode) ---
        episode_rewards.append(episode_reward)
        avg_iou = np.mean(episode_ious) if episode_ious else 0.0

        if episode % train_config.log_frequency == 0:
            # ... (Log times, rewards, steps, IoU, buffer size - unchanged) ...
            if timing_metrics['env_step_time']: writer.add_scalar('Time/Environment_Step_ms_Avg100', np.mean(timing_metrics['env_step_time']) * 1000, total_steps)
            if timing_metrics['parameter_update_time']: writer.add_scalar('Time/Parameter_Update_ms_Avg100', np.mean(timing_metrics['parameter_update_time']) * 1000, total_steps)
            elif learn_steps > 0 and total_steps > ppo_config.steps_per_update: writer.add_scalar('Time/Parameter_Update_ms_Avg100', 0, total_steps)

            writer.add_scalar('Reward/Episode', episode_reward, total_steps)
            writer.add_scalar('Steps/Episode', episode_steps, total_steps)
            writer.add_scalar('Progress/Total_Steps', total_steps, episode)
            writer.add_scalar('Performance/IoU_AvgEp', avg_iou, total_steps)
            writer.add_scalar('Performance/IoU_EndEp', world.iou, total_steps)
            writer.add_scalar('Buffer/PPO_Memory_Size', len(agent.memory), total_steps)

            # --- Conditionally log normalizer stats ---
            if agent.use_state_normalization and agent.state_normalizer and agent.state_normalizer.count > agent.state_normalizer.epsilon:
                writer.add_scalar('Stats/PPO_Normalizer_Count', agent.state_normalizer.count.item(), total_steps)
                writer.add_scalar('Stats/PPO_Normalizer_Mean_Sensor0', agent.state_normalizer.mean[0].item(), total_steps)
                writer.add_scalar('Stats/PPO_Normalizer_Std_Sensor0', torch.sqrt(agent.state_normalizer.var[0].clamp(min=1e-8)).item(), total_steps)
                writer.add_scalar('Stats/PPO_Normalizer_Mean_AgentX', agent.state_normalizer.mean[5].item(), total_steps)
                writer.add_scalar('Stats/PPO_Normalizer_Std_AgentX', torch.sqrt(agent.state_normalizer.var[5].clamp(min=1e-8)).item(), total_steps)
            # --- End Conditional Log ---

            lookback = min(100, len(episode_rewards))
            avg_reward_100 = np.mean(episode_rewards[-lookback:]) if episode_rewards else 0
            writer.add_scalar('Reward/Average_100', avg_reward_100, total_steps)

        if episode % 10 == 0:
             # ... (Update pbar postfix - unchanged) ...
            lookback = min(10, len(episode_rewards))
            avg_reward_10 = np.mean(episode_rewards[-lookback:]) if episode_rewards else 0
            pbar.set_postfix({'avg_rew_10': f'{avg_reward_10:.2f}', 'steps': total_steps, 'IoU': f"{world.iou:.3f}"})


        if episode % train_config.save_interval == 0:
            save_path = os.path.join(train_config.models_dir, f"ppo_ep{episode}_step{total_steps}.pt")
            agent.save_model(save_path)

    # ... (Final cleanup and evaluation call - unchanged) ...
    pbar.close()
    writer.close()
    print(f"PPO Training finished. Total steps: {total_steps}")

    final_save_path = os.path.join(train_config.models_dir, f"ppo_final_ep{train_config.num_episodes}_step{total_steps}.pt")
    agent.save_model(final_save_path)

    if run_evaluation:
         print("\nStarting evaluation after training...")
         evaluate_ppo(agent=agent, config=config) # Pass full config

    return agent, episode_rewards


# --- Evaluation Loop (evaluate_ppo) ---
def evaluate_ppo(agent: PPO, config: DefaultConfig):
    eval_config = config.evaluation
    world_config = config.world
    vis_config = config.visualization

    # --- Conditional Visualization Import ---
    # ... (remains same) ...
    vis_available = False
    visualize_world, reset_trajectories, save_gif = None, None, None
    if eval_config.render:
        try:
            from visualization import visualize_world, reset_trajectories, save_gif
            import imageio.v2 as imageio
            vis_available = True; print("Visualization enabled.")
        except ImportError:
            print("Vis libs not found. Rendering disabled."); vis_available = False; eval_config.render = False
    else: vis_available = False; print("Rendering disabled by config.")


    eval_rewards = []
    eval_ious = []
    success_count = 0
    all_episode_gif_paths = []

    # --- Set Agent and Normalizer (if exists) to Evaluation Mode ---
    agent.actor.eval()
    agent.critic.eval()
    if agent.use_state_normalization and agent.state_normalizer:
        agent.state_normalizer.eval()
    # --- End Set Eval Mode ---

    print(f"\nRunning PPO Evaluation for {eval_config.num_episodes} episodes...")
    world = World(world_config=world_config)

    for episode in range(eval_config.num_episodes):
        state = world.reset()
        episode_reward = 0
        episode_frames = []
        episode_ious = []

        if eval_config.render and vis_available:
            # ... (Initial frame visualization remains same) ...
            os.makedirs(vis_config.save_dir, exist_ok=True)
            if reset_trajectories: reset_trajectories()
            try:
                fname = f"ppo_eval_ep{episode+1}_frame_000_initial.png"
                initial_frame_file = visualize_world(world, vis_config, filename=fname)
                if initial_frame_file and os.path.exists(initial_frame_file): episode_frames.append(initial_frame_file)
            except Exception as e: print(f"Warn: Vis failed init state ep {episode+1}. E: {e}")


        for step in range(eval_config.max_steps):
            # select_action uses normalizer conditionally and in eval mode
            action_normalized = agent.select_action(state, evaluate=True)
            next_state = world.step(action_normalized, training=False, terminal_step=(step == eval_config.max_steps - 1))
            reward = world.reward
            done = world.done
            current_iou = world.iou

            if eval_config.render and vis_available:
                 # ... (Step frame visualization remains same) ...
                try:
                    fname = f"ppo_eval_ep{episode+1}_frame_{step+1:03d}.png"
                    frame_file = visualize_world(world, vis_config, filename=fname)
                    if frame_file and os.path.exists(frame_file): episode_frames.append(frame_file)
                except Exception as e: print(f"Warn: Vis failed step {step+1} ep {episode+1}. E: {e}")


            state = next_state
            episode_reward += reward
            episode_ious.append(current_iou)
            if done:
                break

        # --- Episode End ---
        # ... (Episode end logic remains same) ...
        final_iou = world.iou
        eval_rewards.append(episode_reward)
        eval_ious.append(final_iou)

        success = final_iou >= world_config.success_iou_threshold
        if success: success_count += 1
        status = "Success!" if success else "Failure."

        if world.done and not world.current_step >= config.training.max_steps: # Terminated early
            print(f"  Episode {episode+1}: Terminated Step {world.current_step}. Final IoU: {final_iou:.3f}. {status}")
        else: # Max steps reached
             print(f"  Episode {episode+1}: Finished (Step {world.current_step}). Final IoU: {final_iou:.3f}. {status}")

        if eval_config.render and vis_available and episode_frames and save_gif:
             # ... (GIF saving remains same) ...
            gif_filename = f"ppo_mapping_eval_episode_{episode+1}.gif"
            try:
                print(f"  Saving GIF for PPO episode {episode+1} with {len(episode_frames)} frames...")
                gif_path = save_gif(output_filename=gif_filename, vis_config=vis_config, frame_paths=episode_frames, delete_frames=vis_config.delete_frames_after_gif)
                if gif_path: all_episode_gif_paths.append(gif_path)
            except Exception as e: print(f"  Warn: Failed GIF save ep {episode+1}. E: {e}")


    # --- Set Agent and Normalizer (if exists) back to Training Mode ---
    agent.actor.train()
    agent.critic.train()
    if agent.use_state_normalization and agent.state_normalizer:
        agent.state_normalizer.train()
    # --- End Set Train Mode ---

    # ... (Evaluation summary printing remains same) ...
    avg_eval_reward = np.mean(eval_rewards) if eval_rewards else 0.0
    std_eval_reward = np.std(eval_rewards) if eval_rewards else 0.0
    avg_eval_iou = np.mean(eval_ious) if eval_ious else 0.0
    std_eval_iou = np.std(eval_ious) if eval_ious else 0.0
    success_rate = success_count / eval_config.num_episodes if eval_config.num_episodes > 0 else 0.0

    print("\n--- PPO Evaluation Summary ---")
    print(f"Episodes: {eval_config.num_episodes}")
    print(f"Average Final IoU: {avg_eval_iou:.3f} +/- {std_eval_iou:.3f}")
    print(f"Success Rate (IoU >= {world_config.success_iou_threshold:.2f}): {success_rate:.2%} ({success_count}/{eval_config.num_episodes})")
    if eval_config.render and vis_available and all_episode_gif_paths: print(f"GIFs saved to: '{os.path.abspath(vis_config.save_dir)}'")
    elif eval_config.render and not vis_available: print("Rendering enabled but libs not found.")
    elif not eval_config.render: print("Rendering disabled.")
    print("--- End PPO Evaluation ---\n")


    return eval_rewards, success_rate, avg_eval_iou
