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
            # print("Warning: PPO Memory storing NaN state.")
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
            # Convert lists to numpy arrays for easier batching
            # States are stored raw, convert to numpy here
            states_arr = np.array(self.states, dtype=np.float32) # (N, state_dim)
            if np.isnan(states_arr).any():
                 print("Error: NaN found in PPO memory states_arr during batch generation!")
                 return None, None, None, None, None, None, None # Indicate failure

            actions_arr = np.array(self.actions, dtype=np.float32).reshape(-1, 1) # (N, 1)
            probs_arr = np.array(self.probs, dtype=np.float32).reshape(-1, 1) # (N, 1)
            vals_arr = np.array(self.vals, dtype=np.float32).reshape(-1, 1) # (N, 1)
            rewards_arr = np.array(self.rewards, dtype=np.float32) # (N,) - r_{t+1}
            dones_arr = np.array(self.dones, dtype=np.float32) # (N,) - d_{t+1}

        except Exception as e:
             print(f"Error converting PPO memory to arrays: {e}")
             return None, None, None, None, None, None, None

        return states_arr, actions_arr, probs_arr, vals_arr, rewards_arr, dones_arr, batches

    def __len__(self):
        return len(self.states)


class PolicyNetwork(nn.Module):
    """Actor network for PPO. Takes normalized basic_state (sensors + coords) as input."""
    def __init__(self, config: PPOConfig):
        super(PolicyNetwork, self).__init__()
        # *** Use state_dim from config (should be CORE_STATE_DIM, e.g., 7) ***
        self.state_dim = config.state_dim
        self.action_dim = config.action_dim
        self.hidden_dim = config.hidden_dim
        self.log_std_min = config.log_std_min
        self.log_std_max = config.log_std_max

        self.fc1 = nn.Linear(self.state_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.mean = nn.Linear(self.hidden_dim, self.action_dim)
        # Learnable log_std, initialized near zero
        self.log_std = nn.Parameter(torch.zeros(1, self.action_dim))

    def forward(self, normalized_basic_state): # Takes normalized basic_state now
        x = F.relu(self.fc1(normalized_basic_state))
        x = F.relu(self.fc2(x))
        action_mean = self.mean(x)
        # Clamp the learned log_std
        action_log_std = torch.clamp(self.log_std, self.log_std_min, self.log_std_max)
        action_std = torch.exp(action_log_std)
        return action_mean, action_std

    def sample(self, normalized_basic_state):
        """Sample action (normalized) and calculate log_prob."""
        mean, std = self.forward(normalized_basic_state)
        distribution = Normal(mean, std)
        x_t = distribution.sample() # Sample in unbounded space
        action_normalized = torch.tanh(x_t) # Squash to [-1, 1]

        # Log prob with tanh correction (safer)
        log_prob_unbounded = distribution.log_prob(x_t)
        # Clamp before log_det_jacobian calculation
        clamped_tanh = action_normalized.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        log_det_jacobian = torch.log(1.0 - clamped_tanh.pow(2) + 1e-7) # Add epsilon
        log_prob = log_prob_unbounded - log_det_jacobian
        log_prob = log_prob.sum(1, keepdim=True) # Sum across action dim if multi-dim

        return action_normalized, log_prob

    def evaluate(self, normalized_basic_state, action_normalized):
        """Evaluate log_prob and entropy for a given state and action."""
        mean, std = self.forward(normalized_basic_state)
        distribution = Normal(mean, std)

        # Inverse tanh transformation + clamping for stability
        action_tanh = torch.clamp(action_normalized, -1.0 + 1e-6, 1.0 - 1e-6) # Clamp before atanh
        action_original_space = torch.atanh(action_tanh) # Unsquash action

        # Log prob with tanh correction (safer)
        log_prob_unbounded = distribution.log_prob(action_original_space)
        log_det_jacobian = torch.log(1.0 - action_tanh.pow(2) + 1e-7) # Use clamped tanh here too
        log_prob = log_prob_unbounded - log_det_jacobian
        log_prob = log_prob.sum(1, keepdim=True) # Sum across action dim if multi-dim

        entropy = distribution.entropy().sum(1, keepdim=True) # Sum entropy across action dim
        return log_prob, entropy


class ValueNetwork(nn.Module):
    """Critic network for PPO. Takes normalized basic_state (sensors + coords) as input."""
    def __init__(self, config: PPOConfig):
        super(ValueNetwork, self).__init__()
        # *** Use state_dim from config (should be CORE_STATE_DIM, e.g., 7) ***
        self.state_dim = config.state_dim
        self.hidden_dim = config.hidden_dim
        self.fc1 = nn.Linear(self.state_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.value = nn.Linear(self.hidden_dim, 1)

    def forward(self, normalized_basic_state): # Takes normalized basic_state now
        x = F.relu(self.fc1(normalized_basic_state))
        x = F.relu(self.fc2(x))
        value = self.value(x)
        return value

class PPO:
    """Proximal Policy Optimization algorithm implementation with state normalization."""
    def __init__(self, config: PPOConfig, device: torch.device = None):
        self.config = config
        self.gamma = config.gamma
        self.gae_lambda = config.gae_lambda
        self.policy_clip = config.policy_clip
        self.n_epochs = config.n_epochs
        self.entropy_coef = config.entropy_coef
        self.value_coef = config.value_coef
        # *** Use state_dim from config ***
        self.state_dim = config.state_dim
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"PPO Agent using device: {self.device}")

        # --- Normalization ---
        # *** Initialize with the correct basic state dimension ***
        self.state_normalizer = RunningMeanStd(shape=(self.state_dim,), device=self.device)
        print(f"PPO Agent state normalization enabled for dim: {self.state_dim}")

        self.actor = PolicyNetwork(config).to(self.device)
        self.critic = ValueNetwork(config).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.critic_lr)
        self.memory = PPOMemory(batch_size=config.batch_size)

    def select_action(self, state: dict, evaluate=False):
        """Select action based on the normalized *basic state* tuple."""
        # state is dict {'basic_state': tuple, 'full_trajectory': array}
        basic_state_tuple = state['basic_state'] # Tuple (sensor1..5, agent_x, agent_y)

        # Convert to tensor for normalization and network input
        state_tensor_raw = torch.FloatTensor(basic_state_tuple).to(self.device).unsqueeze(0) # (1, state_dim)

        # --- Normalize State ---
        # Use current stats (do not update normalizer here)
        normalized_state_tensor = self.state_normalizer.normalize(state_tensor_raw)
        # --- End Normalize ---

        self.actor.eval() # Set to eval mode for selection/evaluation
        self.critic.eval()

        with torch.no_grad():
            if evaluate:
                # Get deterministic action (mean of distribution)
                action_mean, _ = self.actor.forward(normalized_state_tensor)
                action_normalized = torch.tanh(action_mean)
            else:
                # Sample action and get log_prob from normalized state
                action_normalized, log_prob = self.actor.sample(normalized_state_tensor)
                # Get value from normalized state
                value = self.critic(normalized_state_tensor)

                # Store the *raw (unnormalized)* basic state in memory, along with action, log_prob, value
                # Reward and done will be added later via store_reward_done
                self.memory.store(
                    basic_state_tuple, # Store raw state tuple
                    action_normalized.cpu().numpy()[0, 0], # Store action taken
                    log_prob.cpu().numpy()[0, 0], # Store log_prob of action taken
                    value.cpu().numpy()[0, 0], # Store value V(s_t)
                    0, False # Placeholder reward r_{t+1}, done d_{t+1}
                )

        self.actor.train() # Revert to train mode if needed
        self.critic.train()

        return action_normalized.detach().cpu().numpy()[0, 0] # Return normalized action


    def store_reward_done(self, reward, done):
        """Store reward and done flag for the *last* transition added to memory."""
        if len(self.memory.rewards) < len(self.memory.states):
            # Add reward and done to the most recently stored transition
             idx_to_update = len(self.memory.rewards)
             self.memory.rewards.insert(idx_to_update, reward) # r_{t+1}
             self.memory.dones.insert(idx_to_update, done) # d_{t+1}
        elif len(self.memory.rewards) == len(self.memory.states) and len(self.memory.states) > 0:
             # Update the last entry if it's still the placeholder
             if self.memory.rewards[-1] == 0 and self.memory.dones[-1] is False:
                  self.memory.rewards[-1] = reward
                  self.memory.dones[-1] = done
             else:
                  # This case might happen if store_reward_done is called unexpectedly
                  # print(f"Warning: PPO store_reward_done called but last entry seems populated. Len states: {len(self.memory.states)}")
                  pass # Or decide how to handle - maybe append if lengths match?
        # else: memory is empty or lengths mismatch significantly - handled in store


    def update_parameters(self):
        """Update policy and value networks using PPO algorithm and collected transitions."""
        if len(self.memory) < self.config.batch_size:
            # print(f"Skipping PPO update: Memory size {len(self.memory)} < Batch size {self.config.batch_size}")
            return None, 0.0 # Not enough samples

        # 1. Generate Batches using raw states stored in memory
        # states_arr_raw: (N, state_dim), actions_arr: (N, 1), old_log_probs_arr: (N, 1)
        # values_arr: (N, 1) [V(s_t)], rewards_arr: (N,) [r_{t+1}], dones_arr: (N,) [d_{t+1}]
        states_arr_raw, actions_arr, old_log_probs_arr, values_arr, rewards_arr, dones_arr, batches = self.memory.generate_batches()

        if states_arr_raw is None: # Handle batch generation failure
            print("Error: Failed to generate batches from PPO memory. Skipping PPO update.")
            self.memory.clear()
            return None, 0.0

        # --- Update State Normalizer Stats ---
        # Use the raw states collected over the update interval
        self.state_normalizer.update(torch.from_numpy(states_arr_raw).to(self.device))
        # --- End Update ---

        # 2. Compute Advantages and Returns (GAE)
        advantages, returns = self._compute_advantages_returns(rewards_arr, dones_arr, values_arr.squeeze())
        if advantages is None: # Handle GAE failure
            print("Error: Failed to compute returns/advantages. Skipping PPO update.")
            self.memory.clear()
            return None, 0.0

        # Convert to tensors and normalize advantages
        advantages_tensor = torch.tensor(advantages, dtype=torch.float32).unsqueeze(1).to(self.device) # (N, 1)
        returns_tensor = torch.tensor(returns, dtype=torch.float32).unsqueeze(1).to(self.device) # (N, 1)
        if len(advantages_tensor) > 1: advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)

        # Convert raw state array to tensor for normalization within the epoch loop
        states_arr_raw_tensor = torch.from_numpy(states_arr_raw).to(self.device)
        actions_arr_tensor = torch.from_numpy(actions_arr).to(self.device)
        old_log_probs_arr_tensor = torch.from_numpy(old_log_probs_arr).to(self.device)

        actor_losses, critic_losses, entropies = [], [], []

        # 3. Optimize policy and value networks over N epochs
        for _ in range(self.n_epochs):
            for batch_indices in batches:
                # Get raw states for this batch
                batch_states_raw = states_arr_raw_tensor[batch_indices] # (batch, s_dim)

                # --- Normalize states for this batch ---
                # Normalize using the updated normalizer stats
                with torch.no_grad(): # Normalization doesn't need gradients here
                    batch_states_normalized = self.state_normalizer.normalize(batch_states_raw)
                # --- End Normalize ---

                # Get corresponding actions, old log probs, advantages, returns for the batch
                batch_actions = actions_arr_tensor[batch_indices]
                batch_old_log_probs = old_log_probs_arr_tensor[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]

                # Evaluate current policy on NORMALIZED states and actions
                # new_log_probs: Log prob of batch_actions under current policy
                # entropy: Entropy of current policy distribution at batch_states
                new_log_probs, entropy = self.actor.evaluate(batch_states_normalized, batch_actions)

                # Get value predictions from current critic on NORMALIZED states
                new_values = self.critic(batch_states_normalized)

                # --- PPO Loss Calculation ---
                # Ratio of new policy probability to old policy probability
                ratio = torch.exp(new_log_probs - batch_old_log_probs)

                # Clipped surrogate objective
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.policy_clip, 1.0 + self.policy_clip) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean() # Maximize objective -> Minimize negative objective

                # Value loss (MSE between predicted values and actual returns)
                critic_loss = F.mse_loss(new_values, batch_returns)

                # Entropy bonus (encourage exploration)
                entropy_loss = -entropy.mean() # Maximize entropy -> Minimize negative entropy

                # Total loss
                total_loss = actor_loss + self.value_coef * critic_loss + self.entropy_coef * entropy_loss

                # --- Optimization Step ---
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()
                # Optional: Gradient clipping
                nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.actor_optimizer.step()
                self.critic_optimizer.step()

                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
                entropies.append(entropy.mean().item())

        # 4. Clear memory after update
        avg_actor_loss = np.mean(actor_losses) if actor_losses else 0.0
        avg_critic_loss = np.mean(critic_losses) if critic_losses else 0.0
        avg_entropy = np.mean(entropies) if entropies else 0.0

        self.memory.clear() # Clear memory for next batch of transitions

        return {'actor_loss': avg_actor_loss, 'critic_loss': avg_critic_loss, 'entropy': avg_entropy}


    def _compute_advantages_returns(self, rewards, dones, values):
        """
        Compute advantages and returns using Generalized Advantage Estimation (GAE).
        Args:
            rewards (np.ndarray): Array of rewards r_{t+1}. Shape (N,).
            dones (np.ndarray): Array of done flags d_{t+1}. Shape (N,).
            values (np.ndarray): Array of value estimates V(s_t). Shape (N,).
        Returns:
            Tuple[np.ndarray, np.ndarray]: advantages, returns. Shape (N,).
        """
        n_steps = len(rewards)
        advantages = np.zeros(n_steps, dtype=np.float32)
        returns = np.zeros(n_steps, dtype=np.float32)
        last_gae_lam = 0.0
        last_value = 0 # V(s_{N}) - bootstrap value

        # Check if the last state was terminal. If not, need to bootstrap V(s_N)
        if not dones[-1]:
            # Get last *raw* basic state tuple and compute its value
            if self.memory.states: # Ensure memory is not empty
                last_basic_state_tuple = self.memory.states[-1]
                last_state_tensor_raw = torch.FloatTensor(last_basic_state_tuple).to(self.device).unsqueeze(0)
                with torch.no_grad():
                    # Normalize the raw state before getting value
                    normalized_last_state = self.state_normalizer.normalize(last_state_tensor_raw)
                    last_value = self.critic(normalized_last_state).cpu().numpy()[0, 0]
            else:
                # Should not happen if update_parameters checks len(memory) > 0
                print("Warning: GAE calculation attempted with empty memory states.")
                return None, None


        # Iterate backwards from T-1 to 0
        for t in reversed(range(n_steps)):
            if t == n_steps - 1:
                next_non_terminal = 1.0 - dones[t] # Is s_{t+1} (i.e., s_N) non-terminal?
                next_value = last_value           # V(s_{t+1}) = V(s_N)
            else:
                next_non_terminal = 1.0 - dones[t] # Is s_{t+1} non-terminal?
                next_value = values[t+1]           # V(s_{t+1})

            # Calculate delta_t = r_{t+1} + gamma * V(s_{t+1}) * (1 - d_{t+1}) - V(s_t)
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]

            # Calculate advantage A_t = delta_t + gamma * lambda * A_{t+1} * (1 - d_{t+1})
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            advantages[t] = last_gae_lam

        # Calculate returns R_t = A_t + V(s_t)
        returns = advantages + values

        return advantages, returns

    def save_model(self, path: str):
        print(f"Saving PPO model to {path}...")
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'state_normalizer_state_dict': self.state_normalizer.state_dict(), # Save normalizer
            'device_type': self.device.type
        }, path)

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

        # Load normalizer
        if 'state_normalizer_state_dict' in checkpoint:
            self.state_normalizer.load_state_dict(checkpoint['state_normalizer_state_dict'])
            print("Loaded PPO state normalizer statistics.")
        else:
            print("Warning: PPO state normalizer statistics not found in checkpoint.")

        self.actor.train()
        self.critic.train()
        self.state_normalizer.train() # Ensure normalizer is in train mode

        print(f"PPO model loaded successfully from {path}")


# --- Training Loop (train_ppo) ---
# (Mostly unchanged, uses updated config defaults and logs IoU)
def train_ppo(config: DefaultConfig, use_multi_gpu: bool = False, run_evaluation: bool = True):
    ppo_config = config.ppo
    train_config = config.training # Uses updated dir based on config.algorithm
    world_config = config.world
    cuda_device = config.cuda_device

    log_dir = os.path.join("runs", f"ppo_mapping_{int(time.time())}")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logs: {log_dir}")

    # Device Setup
    if torch.cuda.is_available():
        if use_multi_gpu: print(f"Warn: Multi-GPU not standard for PPO. Using: {cuda_device}")
        device = torch.device(cuda_device)
        print(f"Using device: {device}")
        if 'cuda' in cuda_device:
            try: torch.cuda.set_device(device); print(f"GPU: {torch.cuda.get_device_name(device)}")
            except Exception as e: print(f"Warn: Could not set CUDA device {cuda_device}. E: {e}"); device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu"); print("GPU not available, using CPU.")

    agent = PPO(config=ppo_config, device=device)
    os.makedirs(train_config.models_dir, exist_ok=True)

    # Checkpoint loading
    model_files = [f for f in os.listdir(train_config.models_dir) if f.startswith("ppo_") and f.endswith(".pt")]
    latest_model_path = None
    if model_files:
        try: latest_model_path = max([os.path.join(train_config.models_dir, f) for f in model_files], key=os.path.getmtime)
        except Exception as e: print(f"Could not find latest model: {e}")

    total_steps = 0
    start_episode = 1
    if latest_model_path and os.path.exists(latest_model_path):
        print(f"\nResuming PPO training from: {latest_model_path}")
        agent.load_model(latest_model_path)
        try: # Best effort parsing
            parts = os.path.basename(latest_model_path).split('_')
            ep_part = next((p for p in parts if p.startswith('ep')), None)
            step_part = next((p for p in parts if p.startswith('step')), None)
            if ep_part: start_episode = int(ep_part.replace('ep', '')) + 1
            if step_part: total_steps = int(step_part.replace('step', '').split('.')[0])
            print(f"Resuming at step {total_steps}, episode {start_episode}")
        except Exception as e: print(f"Warn: File parse failed: {e}. Start fresh."); total_steps=0; start_episode=1
    else:
        print("\nStarting PPO training from scratch.")

    # --- Training Loop ---
    episode_rewards = []
    timing_metrics = {'env_step_time': deque(maxlen=100), 'parameter_update_time': deque(maxlen=100)}
    update_frequency = ppo_config.steps_per_update
    pbar = tqdm(range(start_episode, train_config.num_episodes + 1),
                desc="Training PPO Mapping", unit="episode", initial=start_episode-1, total=train_config.num_episodes)

    world = World(world_config=world_config) # Mapping world
    learn_steps = 0 # Track steps collected since last update

    for episode in pbar:
        state = world.reset() # state is dict with 'basic_state' tuple
        episode_reward = 0
        episode_steps = 0
        episode_ious = []

        for step_in_episode in range(train_config.max_steps):
            # Select action (agent normalizes state internally)
            # Stores raw basic_state (s_t) and action (a_t) in memory
            action_normalized = agent.select_action(state, evaluate=False)

            step_start_time = time.time()
            # Step world with action a_t
            next_state = world.step(action_normalized, training=True, terminal_step=(step_in_episode == train_config.max_steps - 1))
            step_time = time.time() - step_start_time
            timing_metrics['env_step_time'].append(step_time)

            # Get reward r_{t+1} and done d_{t+1} resulting from action a_t
            reward = world.reward
            done = world.done
            current_iou = world.iou

            # Store reward and done for the *last* transition (s_t, a_t) stored in memory
            agent.store_reward_done(reward, done)

            state = next_state
            episode_reward += reward # Accumulate r_{t+1}
            episode_steps += 1
            total_steps += 1
            learn_steps += 1
            episode_ious.append(current_iou)

            # PPO Update condition
            if learn_steps >= update_frequency:
                update_start_time = time.time()
                losses = agent.update_parameters() # Agent handles normalization internally
                update_time = time.time() - update_start_time
                if losses:
                    timing_metrics['parameter_update_time'].append(update_time)
                    writer.add_scalar('Loss/Actor', losses['actor_loss'], total_steps)
                    writer.add_scalar('Loss/Critic', losses['critic_loss'], total_steps)
                    writer.add_scalar('Policy/Entropy', losses['entropy'], total_steps)
                learn_steps = 0 # Reset counter

            if done:
                break

        # --- Logging (End of Episode) ---
        episode_rewards.append(episode_reward)
        avg_iou = np.mean(episode_ious) if episode_ious else 0.0

        if episode % train_config.log_frequency == 0:
            if timing_metrics['env_step_time']: writer.add_scalar('Time/Environment_Step_ms_Avg100', np.mean(timing_metrics['env_step_time']) * 1000, total_steps)
            if timing_metrics['parameter_update_time']: writer.add_scalar('Time/Parameter_Update_ms_Avg100', np.mean(timing_metrics['parameter_update_time']) * 1000, total_steps)
            elif learn_steps > 0 and total_steps > ppo_config.steps_per_update: writer.add_scalar('Time/Parameter_Update_ms_Avg100', 0, total_steps)

            writer.add_scalar('Reward/Episode', episode_reward, total_steps)
            writer.add_scalar('Steps/Episode', episode_steps, total_steps)
            writer.add_scalar('Progress/Total_Steps', total_steps, episode)
            writer.add_scalar('Performance/IoU_AvgEp', avg_iou, total_steps)
            writer.add_scalar('Performance/IoU_EndEp', world.iou, total_steps)
            writer.add_scalar('Buffer/PPO_Memory_Size', len(agent.memory), total_steps) # Log transitions stored before update

            # Log normalizer stats
            if agent.state_normalizer.count > agent.state_normalizer.epsilon:
                writer.add_scalar('Stats/PPO_Normalizer_Count', agent.state_normalizer.count.item(), total_steps)
                writer.add_scalar('Stats/PPO_Normalizer_Mean_Sensor0', agent.state_normalizer.mean[0].item(), total_steps)
                writer.add_scalar('Stats/PPO_Normalizer_Std_Sensor0', torch.sqrt(agent.state_normalizer.var[0].clamp(min=1e-8)).item(), total_steps)
                writer.add_scalar('Stats/PPO_Normalizer_Mean_AgentX', agent.state_normalizer.mean[5].item(), total_steps)
                writer.add_scalar('Stats/PPO_Normalizer_Std_AgentX', torch.sqrt(agent.state_normalizer.var[5].clamp(min=1e-8)).item(), total_steps)

            lookback = min(100, len(episode_rewards))
            avg_reward_100 = np.mean(episode_rewards[-lookback:]) if episode_rewards else 0
            writer.add_scalar('Reward/Average_100', avg_reward_100, total_steps)

        if episode % 10 == 0:
            lookback = min(10, len(episode_rewards))
            avg_reward_10 = np.mean(episode_rewards[-lookback:]) if episode_rewards else 0
            pbar.set_postfix({'avg_rew_10': f'{avg_reward_10:.2f}', 'steps': total_steps, 'IoU': f"{world.iou:.3f}"})

        if episode % train_config.save_interval == 0:
            save_path = os.path.join(train_config.models_dir, f"ppo_ep{episode}_step{total_steps}.pt")
            agent.save_model(save_path)

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
# (Requires setting normalizer to eval mode and logging IoU)
def evaluate_ppo(agent: PPO, config: DefaultConfig):
    eval_config = config.evaluation
    world_config = config.world
    vis_config = config.visualization

    # --- Conditional Visualization Import ---
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
    # --- End Conditional Import ---

    eval_rewards = []
    eval_ious = []
    success_count = 0
    all_episode_gif_paths = []

    # --- Set Agent and Normalizer to Evaluation Mode ---
    agent.actor.eval()
    agent.critic.eval()
    agent.state_normalizer.eval()
    # --- End Set Eval Mode ---

    print(f"\nRunning PPO Evaluation for {eval_config.num_episodes} episodes...")
    world = World(world_config=world_config) # Mapping world

    for episode in range(eval_config.num_episodes):
        state = world.reset()
        episode_reward = 0
        episode_frames = []
        episode_ious = []

        # Visualize initial state
        if eval_config.render and vis_available:
            os.makedirs(vis_config.save_dir, exist_ok=True)
            if reset_trajectories: reset_trajectories()
            try:
                fname = f"ppo_eval_ep{episode+1}_frame_000_initial.png"
                initial_frame_file = visualize_world(world, vis_config, filename=fname)
                if initial_frame_file and os.path.exists(initial_frame_file): episode_frames.append(initial_frame_file)
            except Exception as e: print(f"Warn: Vis failed init state ep {episode+1}. E: {e}")

        for step in range(eval_config.max_steps):
            # Select deterministic action using normalized state
            action_normalized = agent.select_action(state, evaluate=True)
            # Step world (training=False)
            next_state = world.step(action_normalized, training=False, terminal_step=(step == eval_config.max_steps - 1))
            reward = world.reward # Should be 0
            done = world.done
            current_iou = world.iou

            # Visualize current state
            if eval_config.render and vis_available:
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

        # Save GIF
        if eval_config.render and vis_available and episode_frames and save_gif:
            gif_filename = f"ppo_mapping_eval_episode_{episode+1}.gif"
            try:
                print(f"  Saving GIF for PPO episode {episode+1} with {len(episode_frames)} frames...")
                gif_path = save_gif(output_filename=gif_filename, vis_config=vis_config, frame_paths=episode_frames, delete_frames=vis_config.delete_frames_after_gif)
                if gif_path: all_episode_gif_paths.append(gif_path)
            except Exception as e: print(f"  Warn: Failed GIF save ep {episode+1}. E: {e}")

    # --- Set Agent and Normalizer back to Training Mode ---
    agent.actor.train()
    agent.critic.train()
    agent.state_normalizer.train()
    # --- End Set Train Mode ---

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
