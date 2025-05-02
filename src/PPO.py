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
import random # Added for evaluation seeding

# Local imports
from world import World # Mapping world
from configs import DefaultConfig, PPOConfig, TrainingConfig, CORE_STATE_DIM
from torch.utils.tensorboard import SummaryWriter
import math
from utils import RunningMeanStd # Normalization utility

class PPOMemory:
    """Memory buffer for PPO algorithm. Stores individual normalized basic states."""
    def __init__(self, batch_size=64):
        # Stores individual transitions, not trajectories
        self.states = [] # List of normalized basic_state tuples (e.g., 7 floats)
        self.actions = [] # List of float actions (normalized yaw change)
        self.probs = []   # List of float log probs
        self.vals = []    # List of float values (from critic)
        self.rewards = [] # List of float rewards (received after taking action)
        self.dones = []   # List of bool dones (observed after taking action)
        self.batch_size = batch_size

    def store(self, normalized_basic_state, action, probs, vals, reward, done):
        """Store a transition corresponding to taking 'action' in 'normalized_basic_state'."""
        if not isinstance(normalized_basic_state, tuple) or len(normalized_basic_state) != CORE_STATE_DIM:
            # print(f"Warning: PPO Memory storing state with unexpected format/length: {type(normalized_basic_state)}, len={len(normalized_basic_state) if isinstance(normalized_basic_state, tuple) else 'N/A'}. Expected tuple of length {CORE_STATE_DIM}.")
            return # Reduced verbosity
        # Check for NaNs in the state tuple itself
        if any(np.isnan(x) for x in normalized_basic_state):
            # print(f"Warning: NaN detected in PPO state to be stored: {normalized_basic_state}. Skipping store.")
            return # Avoid storing NaN states

        self.states.append(normalized_basic_state)
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
        if n_states == 0: return None, None, None, None, None, None, None # Handle empty memory
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        try:
            # States are already normalized tuples, convert to array
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
    """Actor network for PPO. Takes normalized basic_state (sensors + norm_coords) as input."""
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
        self.log_std = nn.Parameter(torch.zeros(1, self.action_dim)) # Learnable log_std

    def forward(self, network_input): # Takes already normalized basic_state
        x = F.relu(self.fc1(network_input))
        x = F.relu(self.fc2(x))
        action_mean = self.mean(x)
        # Clamp the LEARNED log_std parameter for stability
        action_log_std = torch.clamp(self.log_std, self.log_std_min, self.log_std_max)
        # Expand the learned log_std to match the batch size of the mean
        action_std = action_log_std.exp().expand_as(action_mean)
        return action_mean, action_std

    def sample(self, network_input):
        """Sample action (normalized) and calculate log_prob."""
        mean, std = self.forward(network_input)
        distribution = Normal(mean, std)
        x_t = distribution.sample() # Sample in unbounded space
        action_normalized = torch.tanh(x_t) # Squash to [-1, 1]

        # Calculate log prob using the change of variables formula (log_prob(action) = log_prob(x_t) - log(abs(dy/dx)))
        log_prob_unbounded = distribution.log_prob(x_t)
        # log(abs(dy/dx)) = log(abs(d(tanh(x))/dx)) = log(1 - tanh^2(x))
        # Add epsilon for numerical stability, clamp tanh output before pow(2)
        clamped_tanh = action_normalized.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        log_det_jacobian = torch.log(1.0 - clamped_tanh.pow(2) + 1e-7)
        log_prob = log_prob_unbounded - log_det_jacobian
        log_prob = log_prob.sum(1, keepdim=True) # Sum across action dimension (if > 1)

        return action_normalized, log_prob

    def evaluate(self, network_input, action_normalized):
        """Evaluate log_prob and entropy for a given state and action."""
        mean, std = self.forward(network_input)
        distribution = Normal(mean, std)

        # Inverse tanh to get the action in the unbounded space (pre-squashing)
        # Clamp input to avoid +/- infinity from atanh
        action_tanh = torch.clamp(action_normalized, -1.0 + 1e-6, 1.0 - 1e-6)
        action_original_space = torch.atanh(action_tanh)

        log_prob_unbounded = distribution.log_prob(action_original_space)
        log_det_jacobian = torch.log(1.0 - action_tanh.pow(2) + 1e-7)
        log_prob = log_prob_unbounded - log_det_jacobian
        log_prob = log_prob.sum(1, keepdim=True)

        entropy = distribution.entropy().sum(1, keepdim=True)
        return log_prob, entropy

class ValueNetwork(nn.Module):
    """Critic network for PPO. Takes normalized basic_state (sensors + norm_coords) as input."""
    def __init__(self, config: PPOConfig):
        super(ValueNetwork, self).__init__()
        self.state_dim = config.state_dim
        self.hidden_dim = config.hidden_dim
        self.fc1 = nn.Linear(self.state_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.value = nn.Linear(self.hidden_dim, 1)

    def forward(self, network_input): # Takes already normalized basic_state
        x = F.relu(self.fc1(network_input))
        x = F.relu(self.fc2(x))
        value = self.value(x)
        return value

class PPO:
    """Proximal Policy Optimization algorithm implementation with optional state/reward normalization."""
    def __init__(self, config: PPOConfig, device: torch.device = None):
        self.config = config
        self.gamma = config.gamma
        self.gae_lambda = config.gae_lambda
        self.policy_clip = config.policy_clip
        self.n_epochs = config.n_epochs
        self.entropy_coef = config.entropy_coef
        self.value_coef = config.value_coef
        self.state_dim = config.state_dim # Dimension of normalized state
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"PPO Agent using device: {self.device}")

        # --- Agent-Level State Normalization ---
        self.use_state_normalization = config.use_state_normalization
        self.state_normalizer = None
        if self.use_state_normalization:
            self.state_normalizer = RunningMeanStd(shape=(self.state_dim,), device=self.device)
            print(f"PPO Agent state normalization ENABLED for dim: {self.state_dim} (operates on world's output state)")
        else:
            print(f"PPO Agent state normalization DISABLED.")
        # --- End Agent-Level State Normalization ---

        # --- Agent-Level Reward Normalization --- # MODIFIED
        self.use_reward_normalization = config.use_reward_normalization
        if self.use_reward_normalization:
             print("PPO Agent reward normalization ENABLED (within GAE calculation).")
        else:
             print("PPO Agent reward normalization DISABLED.")
        # --- End Agent-Level Reward Normalization --- # END MODIFIED

        self.actor = PolicyNetwork(config).to(self.device)
        self.critic = ValueNetwork(config).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.critic_lr)
        self.memory = PPOMemory(batch_size=config.batch_size)

    def select_action(self, state: dict, evaluate=False):
        """Select action based on the normalized *basic state* tuple provided by world."""
        # state['basic_state'] is already normalized by world.encode_state()
        basic_state_normalized_tuple = state['basic_state']
        state_tensor_normalized = torch.FloatTensor(basic_state_normalized_tuple).to(self.device).unsqueeze(0) # (1, state_dim)

        # --- Conditionally Apply Agent-Level Normalization ---
        if self.use_state_normalization and self.state_normalizer:
            self.state_normalizer.eval() # Use normalizer in eval mode for selection
            network_input = self.state_normalizer.normalize(state_tensor_normalized)
            self.state_normalizer.train() # Switch back to train mode
        else:
            network_input = state_tensor_normalized # Use world's normalized state directly
        # --- End Conditional Normalize ---

        self.actor.eval()
        self.critic.eval()

        with torch.no_grad():
            if evaluate:
                # Get deterministic action (mean) for evaluation
                action_mean, _ = self.actor.forward(network_input)
                action_normalized = torch.tanh(action_mean)
            else:
                # Sample action for training
                action_normalized, log_prob = self.actor.sample(network_input)
                value = self.critic(network_input) # Get value from same (potentially normalized) input

                # Store the normalized state tuple provided by the world
                self.memory.store(
                    basic_state_normalized_tuple, # Store the normalized state
                    action_normalized.cpu().numpy()[0, 0],
                    log_prob.cpu().numpy()[0, 0],
                    value.cpu().numpy()[0, 0],
                    0, False # Placeholder reward/done (updated later)
                )

        self.actor.train()
        self.critic.train()

        return action_normalized.detach().cpu().numpy()[0, 0] # Return single float action

    def store_reward_done(self, reward, done):
        """Store reward and done flag for the *last* transition added to memory."""
        # Check if the last added state has corresponding reward/done entries yet
        # Ensure index exists before trying to access/modify
        if len(self.memory.states) > len(self.memory.rewards):
             idx_to_update = len(self.memory.rewards) # Index for the missing reward/done
             self.memory.rewards.insert(idx_to_update, reward)
             self.memory.dones.insert(idx_to_update, done)
        elif len(self.memory.states) == len(self.memory.rewards) and len(self.memory.states) > 0:
             # This case handles overwriting the placeholder if it was added
             # Only update if the placeholder values (0, False) are still there
             if self.memory.rewards[-1] == 0 and self.memory.dones[-1] is False:
                  self.memory.rewards[-1] = reward
                  self.memory.dones[-1] = done
             else:
                  # This state already has reward/done, likely from a previous step's update.
                  pass # Reduced verbosity
        # else: memory is empty or lengths mismatch unexpectedly


    def update_parameters(self):
        """Update policy and value networks using PPO algorithm and collected transitions."""
        if len(self.memory) < self.config.batch_size:
            return None # Not enough samples yet

        # Returns arrays of normalized states, actions, etc.
        states_arr_normalized, actions_arr, old_log_probs_arr, values_arr, rewards_arr, dones_arr, batches = self.memory.generate_batches()

        if states_arr_normalized is None:
            print("Error: Failed to generate batches from PPO memory. Skipping PPO update.")
            self.memory.clear()
            return None

        # --- Conditionally Update Agent-Level State Normalizer Stats ---
        if self.use_state_normalization and self.state_normalizer:
            # Update normalizer with the batch of normalized states from memory
            self.state_normalizer.update(torch.from_numpy(states_arr_normalized).to(self.device))
        # --- End Update ---

        # Compute Advantages and Returns using GAE
        # Reward normalization happens INSIDE this function if enabled
        advantages, returns = self._compute_advantages_returns(rewards_arr, dones_arr, values_arr.squeeze())
        if advantages is None:
            print("Error: Failed to compute returns/advantages. Skipping PPO update.")
            self.memory.clear()
            return None

        # Normalize advantages for stability
        advantages_tensor = torch.tensor(advantages, dtype=torch.float32).unsqueeze(1).to(self.device)
        returns_tensor = torch.tensor(returns, dtype=torch.float32).unsqueeze(1).to(self.device)
        if len(advantages_tensor) > 1:
            advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)

        # Convert other arrays to tensors
        states_arr_normalized_tensor = torch.from_numpy(states_arr_normalized).to(self.device)
        actions_arr_tensor = torch.from_numpy(actions_arr).to(self.device)
        old_log_probs_arr_tensor = torch.from_numpy(old_log_probs_arr).to(self.device)

        actor_losses, critic_losses, entropies = [], [], []

        # PPO Update Epochs
        for _ in range(self.n_epochs):
            for batch_indices in batches:
                batch_states_normalized = states_arr_normalized_tensor[batch_indices]

                # --- Conditionally Normalize states for this batch ---
                if self.use_state_normalization and self.state_normalizer:
                    with torch.no_grad(): # Don't track gradients for normalization itself
                        # Normalize the already-normalized state again using agent's RMS
                        batch_network_input = self.state_normalizer.normalize(batch_states_normalized)
                else:
                    batch_network_input = batch_states_normalized # Use world's normalized state
                # --- End Normalize ---

                batch_actions = actions_arr_tensor[batch_indices]
                batch_old_log_probs = old_log_probs_arr_tensor[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices] # These returns are based on potentially normalized rewards

                # Evaluate current policy on potentially re-normalized states
                new_log_probs, entropy = self.actor.evaluate(batch_network_input, batch_actions)
                # Get value predictions from current critic on potentially re-normalized states
                new_values = self.critic(batch_network_input)

                # PPO Loss Calculation
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.policy_clip, 1.0 + self.policy_clip) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                # Critic loss compares current value prediction with GAE returns (based on potentially normalized rewards)
                critic_loss = F.mse_loss(new_values, batch_returns)
                entropy_loss = -entropy.mean()
                total_loss = actor_loss + self.value_coef * critic_loss + self.entropy_coef * entropy_loss

                # Optimization Step
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5) # Gradient clipping
                nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.actor_optimizer.step()
                self.critic_optimizer.step()

                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
                entropies.append(entropy.mean().item())

        avg_actor_loss = np.mean(actor_losses) if actor_losses else 0.0
        avg_critic_loss = np.mean(critic_losses) if critic_losses else 0.0
        avg_entropy = np.mean(entropies) if entropies else 0.0

        self.memory.clear() # Clear memory after update

        return {'actor_loss': avg_actor_loss, 'critic_loss': avg_critic_loss, 'entropy': avg_entropy}


    # MODIFIED Function Signature and Implementation
    def _compute_advantages_returns(self, rewards: np.ndarray, dones: np.ndarray, values: np.ndarray):
        """Compute advantages and returns using GAE, optionally normalizing rewards."""
        n_steps = len(rewards)
        advantages = np.zeros(n_steps, dtype=np.float32)
        returns = np.zeros(n_steps, dtype=np.float32)
        last_gae_lam = 0.0
        last_value = 0.0 # V(s_N)

        # --- Conditionally Normalize Rewards ---
        rewards_to_use = rewards # Default to raw rewards
        if self.use_reward_normalization:
            if n_steps > 1: # Need at least 2 rewards to compute std dev
                 reward_mean = np.mean(rewards)
                 reward_std = np.std(rewards)
                 # Avoid division by zero if std is very small
                 if reward_std > 1e-8:
                      rewards_to_use = (rewards - reward_mean) / reward_std
                 else:
                      # If std is near zero, just center the rewards
                      rewards_to_use = rewards - reward_mean
                      # print("Info: PPO Reward std near zero, only centering rewards.")
            # else: Cannot normalize single reward, use raw value

        # --- End Conditional Reward Normalization ---


        # Estimate value of the last state V(s_N) if the episode didn't end
        if not dones[-1]: # If the last transition was not terminal
            if self.memory.states: # Ensure memory is not empty
                last_normalized_state_tuple = self.memory.states[-1]
                last_state_tensor_normalized = torch.FloatTensor(last_normalized_state_tuple).to(self.device).unsqueeze(0)
                with torch.no_grad():
                    # --- Conditionally Normalize last state for V(s_N) ---
                    if self.use_state_normalization and self.state_normalizer:
                         self.state_normalizer.eval() # Use normalizer in eval mode here
                         network_input_last = self.state_normalizer.normalize(last_state_tensor_normalized)
                         self.state_normalizer.train() # Switch back
                    else:
                         network_input_last = last_state_tensor_normalized
                    # --- End Conditional ---
                    last_value = self.critic(network_input_last).cpu().numpy()[0, 0]
            else:
                print("Warning: GAE calculation attempted with empty memory states after non-terminal step.")
                return None, None # Cannot compute if memory is empty

        # Calculate GAE starting from the last step
        for t in reversed(range(n_steps)):
            if t == n_steps - 1:
                next_non_terminal = 1.0 - dones[t] # 0 if done, 1 if not
                next_value = last_value # Use V(s_N) calculated above
            else:
                next_non_terminal = 1.0 - dones[t]
                next_value = values[t+1] # Use critic's prediction for V(s_{t+1})

            # Calculate TD error (delta) using potentially normalized rewards
            delta = rewards_to_use[t] + self.gamma * next_value * next_non_terminal - values[t]
            # Calculate GAE for this step
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            advantages[t] = last_gae_lam

        # Calculate returns by adding advantages to values
        # Note: Values are predictions based on potentially normalized states,
        # and advantages are derived from potentially normalized rewards.
        # The relationship Return = Advantage + Value holds regardless of reward normalization.
        returns = advantages + values
        return advantages, returns
    # --- END MODIFIED Function ---

    def save_model(self, path: str):
        print(f"Saving PPO model to {path}...")
        save_dict = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'use_state_normalization': self.use_state_normalization, # Save the flag
            'use_reward_normalization': self.use_reward_normalization, # Save reward norm flag
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

        # --- Load State Normalizer State Conditionally ---
        loaded_use_state_norm = checkpoint.get('use_state_normalization', True) # Default True if missing
        if 'use_state_normalization' in checkpoint and loaded_use_state_norm != self.use_state_normalization:
            print(f"Warning: Loaded model STATE normalization setting ({loaded_use_state_norm}) differs from current config ({self.use_state_normalization}). Using current config setting.")

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
             print("Warning: Checkpoint contains PPO state normalizer stats, but normalization is disabled in current config. Ignoring saved stats.")
        # --- End Load State Normalizer ---

        # --- Load Reward Normalization Flag --- # MODIFIED
        loaded_use_reward_norm = checkpoint.get('use_reward_normalization', True) # Default True if missing
        if 'use_reward_normalization' in checkpoint and loaded_use_reward_norm != self.use_reward_normalization:
             print(f"Warning: Loaded model REWARD normalization setting ({loaded_use_reward_norm}) differs from current config ({self.use_reward_normalization}). Using current config setting.")
        # --- End Load Reward Normalization Flag --- # END MODIFIED

        self.actor.train()
        self.critic.train()
        # --- Conditionally set normalizer mode ---
        if self.use_state_normalization and self.state_normalizer:
            self.state_normalizer.train()
        # --- End Conditional ---

        print(f"PPO model loaded successfully from {path}")


# --- Training Loop (train_ppo) ---
def train_ppo(config: DefaultConfig, use_multi_gpu: bool = False, run_evaluation: bool = True):
    ppo_config = config.ppo
    train_config = config.training
    world_config = config.world
    cuda_device = config.cuda_device

    log_dir = os.path.join("runs", f"ppo_pointcloud_{int(time.time())}") # Changed log dir name
    os.makedirs(log_dir, exist_ok=True)

    # --- Save Configuration ---
    config_save_path = os.path.join(log_dir, "config.json")
    try:
        with open(config_save_path, "w") as f:
            f.write(config.model_dump_json(indent=2))
        print(f"Configuration saved to: {config_save_path}")
    except Exception as e:
        print(f"Error saving configuration to {config_save_path}: {e}")
    # --- End Save Configuration ---

    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logs: {log_dir}")

    # --- Device Setup ---
    device = torch.device(cuda_device if torch.cuda.is_available() and cuda_device != 'cpu' else "cpu")
    if device.type == 'cuda' and cuda_device != 'cpu':
        try:
            if ':' in cuda_device:
                device_index = int(cuda_device.split(':')[-1])
                if device_index < torch.cuda.device_count():
                    torch.cuda.set_device(device)
                else:
                    print(f"Warning: CUDA device index {device_index} out of range ({torch.cuda.device_count()} available). Falling back to cuda:0.")
                    device = torch.device("cuda:0")
                    torch.cuda.set_device(device)
            print(f"PPO Training using CUDA device: {torch.cuda.current_device()} ({device})")
        except Exception as e:
                print(f"Warning: Error setting CUDA device {cuda_device}: {e}. Falling back to CPU.")
                device = torch.device("cpu")
                print("PPO Training using CPU.")
    else:
        device = torch.device("cpu") # Ensure device is 'cpu' if condition fails
        print("PPO Training using CPU.")
    # --- End Device Setup ---

    # Agent initialization now uses the config flag (handles norm printouts internally)
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
        agent.load_model(latest_model_path) # Handles conditional norm loading
        try:
            parts = os.path.basename(latest_model_path).split('_')
            ep_part = next((p for p in parts if p.startswith('ep')), None)
            step_part = next((p for p in parts if p.startswith('step')), None)
            if ep_part: start_episode = int(ep_part.replace('ep', '')) + 1
            if step_part: total_steps = int(step_part.replace('step', '').split('.')[0])
            print(f"Resuming at step {total_steps}, episode {start_episode}")
        except Exception as e: print(f"Warn: Could not parse file: {e}. Starting fresh."); total_steps = 0; start_episode = 1
    else:
        print("\nStarting PPO training from scratch.")


    # Training Loop
    episode_rewards = []
    timing_metrics = {'env_step_time': deque(maxlen=100), 'parameter_update_time': deque(maxlen=100)}
    update_frequency = ppo_config.steps_per_update
    pbar = tqdm(range(start_episode, train_config.num_episodes + 1),
                desc="Training PPO Mapping", unit="episode", initial=start_episode-1, total=train_config.num_episodes)

    world = World(world_config=world_config) # Initialize world outside loop
    learn_steps = 0

    for episode in pbar:
        # Reset world (uses internal seeding)
        state = world.reset() # state contains normalized coords
        episode_reward = 0
        episode_steps = 0
        episode_metrics = [] # Changed from episode_ious

        for step_in_episode in range(train_config.max_steps):
            # select_action takes state dict (with normalized state)
            action_normalized = agent.select_action(state, evaluate=False)

            step_start_time = time.time()
            next_state = world.step(action_normalized, training=True, terminal_step=(step_in_episode == train_config.max_steps - 1))
            step_time = time.time() - step_start_time
            timing_metrics['env_step_time'].append(step_time)

            reward = world.reward
            done = world.done
            current_metric = world.performance_metric # Changed from current_iou

            # Store reward/done for the transition that just happened
            agent.store_reward_done(reward, done)

            state = next_state # next_state contains normalized coords
            episode_reward += reward
            episode_steps += 1
            total_steps += 1
            learn_steps += 1
            episode_metrics.append(current_metric) # Changed from episode_ious

            # Update PPO parameters periodically
            if learn_steps >= update_frequency:
                update_start_time = time.time()
                losses = agent.update_parameters() # Uses agent's internal memory
                update_time = time.time() - update_start_time
                if losses:
                    timing_metrics['parameter_update_time'].append(update_time)
                    writer.add_scalar('Loss/Actor', losses['actor_loss'], total_steps)
                    writer.add_scalar('Loss/Critic', losses['critic_loss'], total_steps)
                    writer.add_scalar('Policy/Entropy', losses['entropy'], total_steps)
                learn_steps = 0 # Reset learn step counter

            if done:
                break

        # --- Logging (End of Episode) ---
        episode_rewards.append(episode_reward)
        avg_metric = np.mean(episode_metrics) if episode_metrics else 0.0 # Changed from avg_iou

        if episode % train_config.log_frequency == 0:
            if timing_metrics['env_step_time']: writer.add_scalar('Time/Environment_Step_ms_Avg100', np.mean(timing_metrics['env_step_time']) * 1000, total_steps)
            if timing_metrics['parameter_update_time']: writer.add_scalar('Time/Parameter_Update_ms_Avg100', np.mean(timing_metrics['parameter_update_time']) * 1000, total_steps)
            elif learn_steps > 0 and total_steps > ppo_config.steps_per_update: writer.add_scalar('Time/Parameter_Update_ms_Avg100', 0, total_steps) # Log 0 if update pending

            writer.add_scalar('Reward/Episode', episode_reward, total_steps)
            writer.add_scalar('Steps/Episode', episode_steps, total_steps)
            writer.add_scalar('Progress/Total_Steps', total_steps, episode)
            writer.add_scalar('Performance/Metric_AvgEp', avg_metric, total_steps) # Changed tag
            writer.add_scalar('Performance/Metric_EndEp', world.performance_metric, total_steps) # Changed tag
            writer.add_scalar('Buffer/PPO_Memory_Size', len(agent.memory), total_steps)
            if world.current_seed is not None: writer.add_scalar('Environment/Current_Seed', world.current_seed, episode)

            # --- Conditionally log normalizer stats ---
            if agent.use_state_normalization and agent.state_normalizer and agent.state_normalizer.count > agent.state_normalizer.epsilon:
                writer.add_scalar('Stats/PPO_Normalizer_Count', agent.state_normalizer.count.item(), total_steps)
                # Log mean/std for a few representative dimensions
                writer.add_scalar('Stats/PPO_Normalizer_Mean_Sensor0', agent.state_normalizer.mean[0].item(), total_steps)
                writer.add_scalar('Stats/PPO_Normalizer_Std_Sensor0', torch.sqrt(agent.state_normalizer.var[0].clamp(min=1e-8)).item(), total_steps)
                if agent.state_dim > 5:
                     writer.add_scalar('Stats/PPO_Normalizer_Mean_AgentXNorm', agent.state_normalizer.mean[5].item(), total_steps)
                     writer.add_scalar('Stats/PPO_Normalizer_Std_AgentXNorm', torch.sqrt(agent.state_normalizer.var[5].clamp(min=1e-8)).item(), total_steps)
            # --- End Conditional Log ---

            lookback = min(100, len(episode_rewards))
            avg_reward_100 = np.mean(episode_rewards[-lookback:]) if episode_rewards else 0
            writer.add_scalar('Reward/Average_100', avg_reward_100, total_steps)

        if episode % 10 == 0:
            lookback = min(10, len(episode_rewards))
            avg_reward_10 = np.mean(episode_rewards[-lookback:]) if episode_rewards else 0
            pbar.set_postfix({
                'avg_rew_10': f'{avg_reward_10:.2f}',
                'steps': total_steps,
                'Metric': f"{world.performance_metric:.3f}" # Changed label
            })


        if episode % train_config.save_interval == 0 and episode > 0: # Avoid saving at ep 0
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
            import imageio.v2 as imageio # Keep import here
            vis_available = True; print("Visualization enabled.")
        except ImportError:
            print("Vis libs not found. Rendering disabled."); vis_available = False
    else: vis_available = False; print("Rendering disabled by config.")


    eval_rewards = []
    eval_metrics = [] # Changed from eval_ious
    success_count = 0
    all_episode_gif_paths = []

    # --- Set Agent and Normalizer (if exists) to Evaluation Mode ---
    agent.actor.eval()
    agent.critic.eval()
    if agent.use_state_normalization and agent.state_normalizer:
        agent.state_normalizer.eval()
    # --- End Set Eval Mode ---

    print(f"\nRunning PPO Evaluation for {eval_config.num_episodes} episodes...")
    # Use evaluation seeds if provided
    eval_seeds = world_config.seeds
    if len(eval_seeds) != eval_config.num_episodes:
         print(f"Warning: Number of evaluation seeds ({len(eval_seeds)}) doesn't match num_episodes ({eval_config.num_episodes}). Using first {eval_config.num_episodes} seeds or generating if needed.")
         if len(eval_seeds) < eval_config.num_episodes:
              extra_seeds_needed = eval_config.num_episodes - len(eval_seeds)
              eval_seeds.extend([random.randint(0, 2**32 - 1) for _ in range(extra_seeds_needed)])
         eval_seeds = eval_seeds[:eval_config.num_episodes]

    world = World(world_config=world_config) # Create world instance

    for episode in range(eval_config.num_episodes):
        seed_to_use = eval_seeds[episode] if eval_seeds else None
        state = world.reset(seed=seed_to_use) # state contains normalized coords
        episode_reward = 0
        episode_frames = []
        episode_metrics_current = [] # Changed from episode_ious

        if eval_config.render and vis_available:
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
            # Set training=False for evaluation
            next_state = world.step(action_normalized, training=False, terminal_step=(step == eval_config.max_steps - 1))
            reward = world.reward # Will be 0 if training=False
            done = world.done
            current_metric = world.performance_metric # Changed from current_iou

            if eval_config.render and vis_available:
                try:
                    fname = f"ppo_eval_ep{episode+1}_frame_{step+1:03d}.png"
                    frame_file = visualize_world(world, vis_config, filename=fname)
                    if frame_file and os.path.exists(frame_file): episode_frames.append(frame_file)
                except Exception as e: print(f"Warn: Vis failed step {step+1} ep {episode+1}. E: {e}")


            state = next_state # next_state contains normalized coords
            episode_reward += reward
            episode_metrics_current.append(current_metric) # Changed from episode_ious
            if done:
                break

        # --- Episode End ---
        final_metric = world.performance_metric # Changed from final_iou
        eval_rewards.append(episode_reward)
        eval_metrics.append(final_metric) # Changed from eval_ious

        success = final_metric >= world_config.success_metric_threshold
        if success: success_count += 1
        status = "Success!" if success else "Failure."

        print(f"  Ep {episode+1}/{eval_config.num_episodes} (Seed:{world.current_seed}): Steps={world.current_step}, Terminated={world.done}, Final Metric: {final_metric:.3f}. {status}")


        if eval_config.render and vis_available and episode_frames and save_gif:
            gif_filename = f"ppo_mapping_eval_episode_{episode+1}_seed{world.current_seed}.gif"
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

    avg_eval_reward = np.mean(eval_rewards) if eval_rewards else 0.0
    std_eval_reward = np.std(eval_rewards) if eval_rewards else 0.0
    avg_eval_metric = np.mean(eval_metrics) if eval_metrics else 0.0 # Changed from avg_eval_iou
    std_eval_metric = np.std(eval_metrics) if eval_metrics else 0.0 # Changed from std_eval_iou
    success_rate = success_count / eval_config.num_episodes if eval_config.num_episodes > 0 else 0.0

    print("\n--- PPO Evaluation Summary ---")
    print(f"Episodes: {eval_config.num_episodes}")
    print(f"Average Final Metric (Point Inclusion): {avg_eval_metric:.3f} +/- {std_eval_metric:.3f}") # Changed label
    print(f"Success Rate (Metric >= {world_config.success_metric_threshold:.2f}): {success_rate:.2%} ({success_count}/{eval_config.num_episodes})")
    print(f"Average Episode Reward: {avg_eval_reward:.3f} +/- {std_eval_reward:.3f}") # Keep reward report
    if eval_config.render and vis_available and all_episode_gif_paths: print(f"GIFs saved to: '{os.path.abspath(vis_config.save_dir)}'")
    elif eval_config.render and not vis_available: print("Rendering enabled but libs not found.")
    elif not eval_config.render: print("Rendering disabled.")
    print("--- End PPO Evaluation ---\n")


    return eval_rewards, success_rate, avg_eval_metric # Return metric instead of iou