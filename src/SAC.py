import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import random
import time
from collections import deque
from tqdm import tqdm

# Local imports
from world import World # Now the mapping world
from configs import DefaultConfig, ReplayBufferConfig, SACConfig, WorldConfig, CORE_STATE_DIM
from torch.utils.tensorboard import SummaryWriter
from typing import Tuple, Optional
from utils import RunningMeanStd # Normalization utility

class ReplayBuffer:
    """Experience replay buffer storing full state trajectories."""
    def __init__(self, config: ReplayBufferConfig, world_config: WorldConfig):
        self.buffer = deque(maxlen=config.capacity)
        self.config = config
        self.trajectory_length = world_config.trajectory_length
        self.feature_dim = world_config.trajectory_feature_dim # Includes state+action+reward

    def push(self, state, action, reward, next_state, done):
        """ Add a new experience to memory. """
        trajectory = state['full_trajectory'] # Shape (N, feat_dim) - state is normalized
        next_trajectory = next_state['full_trajectory'] # Shape (N, feat_dim) - state is normalized

        if isinstance(action, (np.ndarray, list)): action = action[0]
        if isinstance(reward, (np.ndarray)): reward = reward.item()

        # Basic validation
        if not isinstance(trajectory, np.ndarray) or trajectory.shape != (self.trajectory_length, self.feature_dim):
            # print(f"Warning: Skipping push due to trajectory shape mismatch: {trajectory.shape}")
            return
        if not isinstance(next_trajectory, np.ndarray) or next_trajectory.shape != (self.trajectory_length, self.feature_dim):
            # print(f"Warning: Skipping push due to next_trajectory shape mismatch: {next_trajectory.shape}")
            return
        if np.isnan(trajectory).any() or np.isnan(next_trajectory).any():
             # print("Warning: Skipping push due to NaN in trajectory.")
             return # Skip pushing if NaNs are present

        self.buffer.append((trajectory, float(action), float(reward), next_trajectory, done))

    def sample(self, batch_size: int) -> Optional[Tuple]:
        """Sample a batch of experiences from memory."""
        if len(self.buffer) < batch_size:
            return None
        try:
            batch = random.sample(self.buffer, batch_size)
        except ValueError:
            print(f"Warning: ReplayBuffer sampling failed. len(buffer)={len(self.buffer)}, batch_size={batch_size}")
            return None

        # state/next_state are full trajectories (batch, N, feat_dim) - state part is normalized
        trajectory, action, reward, next_trajectory, done = zip(*batch)

        try:
            trajectory_arr = np.array(trajectory, dtype=np.float32)
            action_arr = np.array(action, dtype=np.float32).reshape(-1, 1) # Shape (batch, 1)
            reward_arr = np.array(reward, dtype=np.float32) # Shape (batch,)
            next_trajectory_arr = np.array(next_trajectory, dtype=np.float32)
            done_arr = np.array(done, dtype=np.float32) # Shape (batch,)

            # Check for NaNs after conversion (optional but good practice)
            if np.isnan(trajectory_arr).any() or np.isnan(next_trajectory_arr).any():
                print("Warning: Sampled batch contains NaN values. Returning None.")
                return None

        except Exception as e:
            print(f"Error converting sampled batch to numpy arrays: {e}")
            return None

        return (trajectory_arr, action_arr, reward_arr, next_trajectory_arr, done_arr)

    def __len__(self):
        return len(self.buffer)


class Actor(nn.Module):
    """Policy network (Actor) for SAC, optionally with RNN."""
    def __init__(self, config: SACConfig, world_config: WorldConfig):
        super(Actor, self).__init__()
        self.config = config
        self.world_config = world_config
        self.use_rnn = config.use_rnn
        # Input state dim (already potentially normalized)
        self.state_dim = config.state_dim
        self.action_dim = config.action_dim
        self.trajectory_length = world_config.trajectory_length

        if self.use_rnn:
            self.rnn_hidden_size = config.rnn_hidden_size
            self.rnn_num_layers = config.rnn_num_layers
            # RNN processes the CORE_STATE_DIM features (normalized)
            rnn_input_dim = self.state_dim
            if config.rnn_type == 'lstm':
                self.rnn = nn.LSTM(input_size=rnn_input_dim, hidden_size=config.rnn_hidden_size,
                                   num_layers=config.rnn_num_layers, batch_first=True)
            elif config.rnn_type == 'gru':
                self.rnn = nn.GRU(input_size=rnn_input_dim, hidden_size=config.rnn_hidden_size,
                                  num_layers=config.rnn_num_layers, batch_first=True)
            else:
                raise ValueError(f"Unsupported RNN type: {config.rnn_type}")
            mlp_input_dim = config.rnn_hidden_size
        else:
            # MLP uses only the last normalized basic state
            mlp_input_dim = self.state_dim
            self.rnn = None

        self.layers = nn.ModuleList()
        current_dim = mlp_input_dim
        hidden_dims = config.hidden_dims if config.hidden_dims else [256, 256] # Ensure default
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(current_dim, hidden_dim))
            self.layers.append(nn.ReLU())
            current_dim = hidden_dim

        self.mean = nn.Linear(current_dim, self.action_dim)
        self.log_std = nn.Linear(current_dim, self.action_dim)

        self.log_std_min = config.log_std_min
        self.log_std_max = config.log_std_max

    def forward(self, network_input: torch.Tensor, hidden_state: Optional[Tuple] = None) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple]]:
        """Forward pass through the network.
           Expects pre-normalized basic state(s).
           network_input shape:
             - RNN: (batch, seq_len, state_dim) - normalized basic state sequence
             - MLP: (batch, state_dim) - normalized last basic state
        """
        next_hidden_state = None
        if self.use_rnn:
            # RNN processing (input is already normalized basic state sequence)
            rnn_output, next_hidden_state = self.rnn(network_input, hidden_state)
            # Use the output corresponding to the last time step
            mlp_input = rnn_output[:, -1, :] # Shape: (batch, rnn_hidden_size)
        else:
            # MLP input is the normalized last basic state
            mlp_input = network_input # Shape: (batch, state_dim)

        x = mlp_input
        for layer in self.layers:
            x = layer(x)

        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std, next_hidden_state

    def sample(self, network_input: torch.Tensor, hidden_state: Optional[Tuple] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[Tuple]]:
        """Sample action (normalized) from the policy distribution."""
        mean, log_std, next_hidden_state = self.forward(network_input, hidden_state)
        std = log_std.exp()
        normal = Normal(mean, std)

        x_t = normal.rsample()
        action_normalized = torch.tanh(x_t)

        # Log prob with tanh correction (safer calculation)
        log_prob_unbounded = normal.log_prob(x_t)
        clamped_tanh = action_normalized.clamp(-0.999999, 0.999999) # Avoid log(0)
        log_det_jacobian = torch.log(1.0 - clamped_tanh.pow(2) + 1e-7) # Add epsilon
        log_prob = log_prob_unbounded - log_det_jacobian
        log_prob = log_prob.sum(1, keepdim=True)

        return action_normalized, log_prob, torch.tanh(mean), next_hidden_state

    def get_initial_hidden_state(self, batch_size: int, device: torch.device) -> Optional[Tuple]:
        """Return initial hidden state for RNN."""
        if not self.use_rnn: return None
        h_zeros = torch.zeros(self.rnn_num_layers, batch_size, self.rnn_hidden_size).to(device)
        if self.config.rnn_type == 'lstm':
            c_zeros = torch.zeros(self.rnn_num_layers, batch_size, self.rnn_hidden_size).to(device)
            return (h_zeros, c_zeros)
        elif self.config.rnn_type == 'gru':
            return h_zeros
        return None

class Critic(nn.Module):
    """Q-function network (Critic) for SAC, optionally with RNN."""
    def __init__(self, config: SACConfig, world_config: WorldConfig):
        super(Critic, self).__init__()
        self.config = config
        self.world_config = world_config
        self.use_rnn = config.use_rnn
        # Input state dim (already potentially normalized)
        self.state_dim = config.state_dim
        self.action_dim = config.action_dim
        self.trajectory_length = world_config.trajectory_length

        if self.use_rnn:
            self.rnn_hidden_size = config.rnn_hidden_size
            self.rnn_num_layers = config.rnn_num_layers
            # RNN processes the CORE_STATE_DIM features (normalized)
            rnn_input_dim = self.state_dim
            rnn_cell = nn.LSTM if config.rnn_type == 'lstm' else nn.GRU

            self.rnn1 = rnn_cell(input_size=rnn_input_dim, hidden_size=config.rnn_hidden_size,
                                  num_layers=config.rnn_num_layers, batch_first=True)
            self.rnn2 = rnn_cell(input_size=rnn_input_dim, hidden_size=config.rnn_hidden_size,
                                  num_layers=config.rnn_num_layers, batch_first=True)
            mlp_input_dim = config.rnn_hidden_size + self.action_dim # MLP takes final RNN state + action
        else:
            # MLP uses only the normalized last basic state + action
            mlp_input_dim = self.state_dim + self.action_dim
            self.rnn1, self.rnn2 = None, None

        hidden_dims = config.hidden_dims if config.hidden_dims else [256, 256] # Ensure default

        # Q1 architecture
        self.q1_layers = nn.ModuleList()
        q1_mlp_input = mlp_input_dim
        for hidden_dim in hidden_dims:
            self.q1_layers.append(nn.Linear(q1_mlp_input, hidden_dim))
            self.q1_layers.append(nn.ReLU())
            q1_mlp_input = hidden_dim
        self.q1_out = nn.Linear(q1_mlp_input, 1)

        # Q2 architecture
        self.q2_layers = nn.ModuleList()
        q2_mlp_input = mlp_input_dim
        for hidden_dim in hidden_dims:
            self.q2_layers.append(nn.Linear(q2_mlp_input, hidden_dim))
            self.q2_layers.append(nn.ReLU())
            q2_mlp_input = hidden_dim
        self.q2_out = nn.Linear(q2_mlp_input, 1)

    def forward(self, network_input: torch.Tensor, action: torch.Tensor, hidden_state: Optional[Tuple] = None) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple]]:
        """Forward pass returning both Q-values.
           Expects pre-normalized basic state(s).
           network_input shape:
             - RNN: (batch, seq_len, state_dim) - normalized basic state sequence
             - MLP: (batch, state_dim) - normalized last basic state
           action shape: (batch, action_dim)
        """
        next_hidden_state = None
        if self.use_rnn:
            h1_in, h2_in = None, None
            if isinstance(hidden_state, tuple) and len(hidden_state) == 2:
                 h1_in, h2_in = hidden_state

            # RNN processing (input is already normalized basic state sequence)
            rnn_out1, next_h1 = self.rnn1(network_input, h1_in)
            rnn_out2, next_h2 = self.rnn2(network_input, h2_in)
            next_hidden_state = (next_h1, next_h2)

            # Use final RNN hidden state and concatenate with action
            mlp_input1 = torch.cat([rnn_out1[:, -1, :], action], dim=1)
            mlp_input2 = torch.cat([rnn_out2[:, -1, :], action], dim=1)
        else:
            # Use normalized last basic state and concatenate with action
            # network_input is shape (batch, state_dim) here
            mlp_input1 = torch.cat([network_input, action], dim=1)
            mlp_input2 = torch.cat([network_input, action], dim=1)

        # Q1 calculation
        x1 = mlp_input1
        for layer in self.q1_layers:
            x1 = layer(x1)
        q1 = self.q1_out(x1)

        # Q2 calculation
        x2 = mlp_input2
        for layer in self.q2_layers:
            x2 = layer(x2)
        q2 = self.q2_out(x2)

        return q1, q2, next_hidden_state

    def q1_forward(self, network_input: torch.Tensor, action: torch.Tensor, hidden_state: Optional[Tuple] = None) -> Tuple[torch.Tensor, Optional[Tuple]]:
        """ Forward pass for Q1 only. """
        next_hidden_state = None
        if self.use_rnn:
            h1_in = hidden_state[0] if isinstance(hidden_state, tuple) else hidden_state
            rnn_out1, next_h1 = self.rnn1(network_input, h1_in)
            next_hidden_state = (next_h1, None) # Return only Q1's hidden state progression
            mlp_input1 = torch.cat([rnn_out1[:, -1, :], action], dim=1)
        else:
            mlp_input1 = torch.cat([network_input, action], dim=1)

        x1 = mlp_input1
        for layer in self.q1_layers:
            x1 = layer(x1)
        q1 = self.q1_out(x1)
        return q1, next_hidden_state

    def get_initial_hidden_state(self, batch_size: int, device: torch.device) -> Optional[Tuple]:
        """Return initial hidden state tuple (h1, h2) for RNNs."""
        if not self.use_rnn: return None
        h_zeros = lambda: torch.zeros(self.rnn_num_layers, batch_size, self.rnn_hidden_size).to(device)
        if self.config.rnn_type == 'lstm':
            c_zeros = lambda: torch.zeros(self.rnn_num_layers, batch_size, self.rnn_hidden_size).to(device)
            h1 = (h_zeros(), c_zeros())
            h2 = (h_zeros(), c_zeros())
            return (h1, h2)
        elif self.config.rnn_type == 'gru':
            h1 = h_zeros()
            h2 = h_zeros()
            return (h1, h2)
        return None


class SAC:
    """Soft Actor-Critic algorithm implementation with trajectory states and normalization."""
    def __init__(self, config: SACConfig, world_config: WorldConfig, device: torch.device = None):
        self.config = config
        self.world_config = world_config
        self.gamma = config.gamma
        self.tau = config.tau
        self.alpha = config.alpha
        self.auto_tune_alpha = config.auto_tune_alpha
        self.use_rnn = config.use_rnn
        self.trajectory_length = world_config.trajectory_length
        self.state_dim = config.state_dim # Dimension of the (potentially normalized) state input
        self.action_dim = config.action_dim
        # --- Store normalization flags ---
        self.use_state_normalization = config.use_state_normalization
        self.use_reward_normalization = config.use_reward_normalization

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"SAC Agent using device: {self.device}")
        if self.use_rnn: print(f"SAC Agent using RNN: Type={config.rnn_type}, SeqLen={self.trajectory_length}")
        else: print(f"SAC Agent using MLP (Processing last state of trajectory)")

        # --- Conditional State Normalization ---
        self.state_normalizer = None
        if self.use_state_normalization:
            # Initialize normalizer with the expected input shape
            self.state_normalizer = RunningMeanStd(shape=(self.state_dim,), device=self.device)
            print(f"SAC Agent state normalization ENABLED for dim: {self.state_dim}")
        else:
            print(f"SAC Agent state normalization DISABLED.")
        # --- End Conditional State Normalization ---

        # --- Print Reward Norm Status ---
        if self.use_reward_normalization:
            print(f"SAC Agent reward normalization ENABLED.")
        else:
            print(f"SAC Agent reward normalization DISABLED.")
        # --- End Status Print ---

        self.actor = Actor(config, world_config).to(self.device)
        self.critic = Critic(config, world_config).to(self.device)
        self.critic_target = Critic(config, world_config).to(self.device)

        self.critic_target.load_state_dict(self.critic.state_dict())
        for target_param in self.critic_target.parameters():
            target_param.requires_grad = False

        # Use separate learning rates from config
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.critic_lr)

        if self.auto_tune_alpha:
            self.target_entropy = -torch.prod(torch.Tensor([self.action_dim]).to(self.device)).item()
            self.log_alpha = torch.tensor(np.log(self.alpha), requires_grad=True, device=self.device)
            # Use actor learning rate for alpha optimizer
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=config.actor_lr)
        else:
            self.log_alpha = torch.tensor(np.log(self.alpha)).to(self.device)

    def select_action(self, state: dict, actor_hidden_state: Optional[Tuple] = None, evaluate: bool = False) -> Tuple[float, Optional[Tuple]]:
        """Select action (normalized yaw change [-1, 1]) based on normalized state (if enabled)."""
        # state['full_trajectory'] already contains normalized states if world.normalize_coords is True
        state_trajectory = state['full_trajectory'] # Get the full trajectory array (N, feat_dim)
        state_tensor_full = torch.FloatTensor(state_trajectory).to(self.device).unsqueeze(0) # (1, N, feat_dim)

        with torch.no_grad():
            # Extract the state part (already normalized by world.encode_state)
            state_part_normalized = state_tensor_full[:, :, :self.state_dim] # (1, N, state_dim)

            # Determine the input for the network based on RNN/MLP
            if self.use_rnn:
                raw_network_input = state_part_normalized # (1, N, state_dim)
            else:
                raw_network_input = state_part_normalized[:, -1, :] # (1, state_dim) - Last normalized state

            # --- Conditionally Apply Agent-Level Normalization ---
            if self.use_state_normalization and self.state_normalizer:
                self.state_normalizer.eval()
                # Normalize the already-normalized state (can help stabilize if ranges vary)
                network_input_processed = self.state_normalizer.normalize(raw_network_input)
                self.state_normalizer.train()
            else:
                # Use the state as provided by the world (which might be normalized)
                network_input_processed = raw_network_input
            # --- End Conditional Normalize ---

            # Set actor to eval mode for selection
            self.actor.eval()
            if evaluate:
                _, _, action_mean_squashed, next_actor_hidden_state = self.actor.sample(network_input_processed, actor_hidden_state)
                action_normalized = action_mean_squashed
            else:
                action_normalized, _, _, next_actor_hidden_state = self.actor.sample(network_input_processed, actor_hidden_state)
            # Set actor back to train mode
            self.actor.train()

        action_normalized_float = action_normalized.detach().cpu().numpy()[0, 0]
        return action_normalized_float, next_actor_hidden_state

    def update_parameters(self, memory: ReplayBuffer, batch_size: int):
        """Perform a single SAC update step using a batch of trajectories with conditional normalization."""
        sampled_batch = memory.sample(batch_size)
        if sampled_batch is None: return None

        # Shapes: state/next_state (b, N, feat_dim), action (b, 1), reward/done (b,)
        # State parts within state_batch/next_state_batch are already normalized if world.normalize_coords is True
        state_batch, action_batch_normalized, reward_batch, next_state_batch, done_batch = sampled_batch

        state_batch_tensor = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch_tensor = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch_tensor = torch.FloatTensor(action_batch_normalized).to(self.device)
        reward_batch_tensor_raw = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        done_batch_tensor = torch.FloatTensor(done_batch).to(self.device).unsqueeze(1)

        # --- Conditionally Normalize Rewards ---
        if self.use_reward_normalization:
            reward_mean = reward_batch_tensor_raw.mean()
            reward_std = reward_batch_tensor_raw.std()
            reward_batch_tensor = (reward_batch_tensor_raw - reward_mean) / (reward_std + 1e-8)
        else:
            reward_batch_tensor = reward_batch_tensor_raw # Use raw rewards
        # --- End Conditional Reward Normalization ---

        # Extract state parts (already normalized by world)
        state_part_normalized = state_batch_tensor[:, :, :self.state_dim]
        next_state_part_normalized = next_state_batch_tensor[:, :, :self.state_dim]

        # Conditionally Update Running State Statistics (using the potentially normalized states from world)
        if self.use_state_normalization and self.state_normalizer:
             # Update normalizer with the normalized states from the buffer
            self.state_normalizer.update(state_part_normalized.reshape(-1, self.state_dim))

        # Conditionally Prepare Network Inputs (Apply agent-level normalization if enabled)
        if self.use_rnn:
            # Input is the state sequence (potentially normalized by world)
            raw_current_input = state_part_normalized
            raw_next_input = next_state_part_normalized
            if self.use_state_normalization and self.state_normalizer:
                 batch, seq_len, _ = raw_current_input.shape
                 current_network_input = self.state_normalizer.normalize(
                     raw_current_input.reshape(-1, self.state_dim)
                 ).reshape(batch, seq_len, self.state_dim)
                 batch_next, seq_len_next, _ = raw_next_input.shape
                 next_network_input = self.state_normalizer.normalize(
                      raw_next_input.reshape(-1, self.state_dim)
                 ).reshape(batch_next, seq_len_next, self.state_dim)
            else:
                 current_network_input = raw_current_input
                 next_network_input = raw_next_input
        else:
            # Input is the last state (potentially normalized by world)
            raw_current_input = state_part_normalized[:, -1, :]
            raw_next_input = next_state_part_normalized[:, -1, :]
            if self.use_state_normalization and self.state_normalizer:
                 current_network_input = self.state_normalizer.normalize(raw_current_input)
                 next_network_input = self.state_normalizer.normalize(raw_next_input)
            else:
                 current_network_input = raw_current_input
                 next_network_input = raw_next_input

        initial_actor_hidden = self.actor.get_initial_hidden_state(batch_size, self.device) if self.use_rnn else None
        initial_critic_hidden = self.critic.get_initial_hidden_state(batch_size, self.device) if self.use_rnn else None
        initial_critic_target_hidden = self.critic_target.get_initial_hidden_state(batch_size, self.device) if self.use_rnn else None

        # --- Critic Update ---
        with torch.no_grad():
            # Pass potentially normalized input to actor
            next_action, next_log_prob, _, _ = self.actor.sample(next_network_input, initial_actor_hidden)
            # Pass potentially normalized input to critic target
            target_q1, target_q2, _ = self.critic_target(next_network_input, next_action, initial_critic_target_hidden)
            target_q_min = torch.min(target_q1, target_q2)
            current_alpha = self.log_alpha.exp().item()
            target_q_entropy = target_q_min - current_alpha * next_log_prob
            # Use the (potentially normalized) reward_batch_tensor here
            y = reward_batch_tensor + (1.0 - done_batch_tensor) * self.gamma * target_q_entropy

        # --- Calculate Current Q values ---
        # Pass potentially normalized input to critic
        current_q1, current_q2, _ = self.critic(current_network_input, action_batch_tensor, initial_critic_hidden)
        critic_loss = F.mse_loss(current_q1, y) + F.mse_loss(current_q2, y)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # --- Actor Update ---
        for param in self.critic.parameters(): param.requires_grad = False
        # Pass potentially normalized input to actor
        action_pi, log_prob_pi, _, _ = self.actor.sample(current_network_input, initial_actor_hidden)
        # Pass potentially normalized input to critic
        q1_pi, q2_pi, _ = self.critic(current_network_input, action_pi, initial_critic_hidden)
        q_pi_min = torch.min(q1_pi, q2_pi)
        current_alpha = self.log_alpha.exp().item()
        actor_loss = (current_alpha * log_prob_pi - q_pi_min).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        for param in self.critic.parameters(): param.requires_grad = True

        # --- Alpha Update ---
        alpha_loss_item = None
        if self.auto_tune_alpha:
            alpha_loss = -(self.log_alpha * (log_prob_pi.detach() + self.target_entropy)).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            nn.utils.clip_grad_norm_([self.log_alpha], max_norm=1.0)
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()
            alpha_loss_item = alpha_loss.item()

        # --- Target Network Update ---
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha': self.alpha,
            'alpha_loss': alpha_loss_item if alpha_loss_item is not None else 0.0
        }

    def save_model(self, path: str):
        print(f"Saving SAC model to {path}...")
        save_dict = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'use_state_normalization': self.use_state_normalization, # Save the flag
            'device_type': self.device.type
        }
        if self.use_state_normalization and self.state_normalizer:
            save_dict['state_normalizer_state_dict'] = self.state_normalizer.state_dict()
        if self.auto_tune_alpha:
            save_dict['log_alpha'] = self.log_alpha
            save_dict['alpha_optimizer_state_dict'] = self.alpha_optimizer.state_dict()
        torch.save(save_dict, path)

    def load_model(self, path: str):
        if not os.path.exists(path):
            print(f"Warning: SAC model file not found at {path}. Skipping loading.")
            return
        print(f"Loading SAC model from {path}...")
        checkpoint = torch.load(path, map_location=self.device)

        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])

        # Load Normalizer State Conditionally
        loaded_use_norm = checkpoint.get('use_state_normalization', True) # Default to True if missing
        # Warn if mismatch, but proceed with the current config setting
        if 'use_state_normalization' in checkpoint and loaded_use_norm != self.use_state_normalization:
            print(f"Warning: Loaded model normalization setting ({loaded_use_norm}) differs from current config ({self.use_state_normalization}). Using current config setting.")

        if self.use_state_normalization and self.state_normalizer:
            if 'state_normalizer_state_dict' in checkpoint:
                try:
                    self.state_normalizer.load_state_dict(checkpoint['state_normalizer_state_dict'])
                    print("Loaded state normalizer statistics.")
                except Exception as e:
                    print(f"Warning: Failed to load state normalizer stats: {e}. Using initial values.")
            else:
                print("Warning: State normalizer statistics not found in checkpoint, but normalization is enabled. Using initial values.")
        elif not self.use_state_normalization and 'state_normalizer_state_dict' in checkpoint:
             print("Warning: Checkpoint contains normalizer stats, but normalization is disabled in current config. Ignoring saved stats.")

        # Load Alpha
        if self.auto_tune_alpha and 'log_alpha' in checkpoint:
            # Ensure log_alpha exists and is a parameter before copying data
            if not hasattr(self, 'log_alpha') or not isinstance(self.log_alpha, torch.Tensor):
                 # Initialize if it doesn't exist (e.g., if loading into a differently configured agent)
                 self.log_alpha = torch.tensor(0.0, requires_grad=True, device=self.device)
            self.log_alpha.data.copy_(checkpoint['log_alpha'].to(self.device))
            if not self.log_alpha.requires_grad: self.log_alpha.requires_grad_(True) # Ensure grad is enabled

            # Safely initialize or reinitialize optimizer (use actor LR from CURRENT config)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.config.actor_lr)
            try:
                if 'alpha_optimizer_state_dict' in checkpoint:
                     self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
            except ValueError as e:
                 print(f"Warning: Could not load SAC alpha optimizer state: {e}. Reinitializing.")
                 self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.config.actor_lr) # Reinitialize fresh
            self.alpha = self.log_alpha.exp().item()

        elif not self.auto_tune_alpha:
             if 'log_alpha' in checkpoint:
                  try:
                       # Ensure log_alpha exists and is a tensor
                       if not hasattr(self, 'log_alpha') or not isinstance(self.log_alpha, torch.Tensor):
                            self.log_alpha = torch.tensor(0.0, device=self.device)
                       self.log_alpha.data.copy_(checkpoint['log_alpha'].to(self.device))
                       self.alpha = self.log_alpha.exp().item()
                  except Exception as e: print(f"Warning: Could not load fixed log_alpha: {e}.")

        self.critic_target.load_state_dict(self.critic.state_dict())
        for target_param in self.critic_target.parameters():
            target_param.requires_grad = False

        self.actor.train()
        self.critic.train()
        self.critic_target.train() # Should be eval, but set target weights above. Does not matter much.
        if self.use_state_normalization and self.state_normalizer:
            self.state_normalizer.train()

        print(f"SAC model loaded successfully from {path}")


# --- Training Loop (train_sac) ---
def train_sac(config: DefaultConfig, use_multi_gpu: bool = False, run_evaluation: bool = True):
    sac_config = config.sac
    train_config = config.training
    buffer_config = config.replay_buffer
    world_config = config.world
    cuda_device = config.cuda_device # String like "cuda:0" or "cpu"

    learning_starts = train_config.learning_starts
    gradient_steps = train_config.gradient_steps
    train_freq = train_config.train_freq
    batch_size = train_config.batch_size
    log_frequency_ep = train_config.log_frequency
    save_interval_ep = train_config.save_interval

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
            print(f"SAC Training using CUDA device: {torch.cuda.current_device()} ({device})")
        except Exception as e:
                print(f"Warning: Error setting CUDA device {cuda_device}: {e}. Falling back to CPU.")
                device = torch.device("cpu")
                print("SAC Training using CPU.")
    else:
        device = torch.device("cpu") # Ensure device is 'cpu' if condition fails
        print("SAC Training using CPU.")
    # --- End Device Setup ---

    name_in_logdir = "sac_pointcloud_rnn_" if config.sac.use_rnn else "sac_pointcloud_"
    log_dir = os.path.join("runs", name_in_logdir + str(int(time.time())))
    os.makedirs(log_dir, exist_ok=True)

    # --- Save Configuration --- # MODIFIED
    config_save_path = os.path.join(log_dir, "config.json")
    try:
        with open(config_save_path, "w") as f:
            # Use model_dump_json for Pydantic v2+ with indent for readability
            f.write(config.model_dump_json(indent=2))
        print(f"Configuration saved to: {config_save_path}")
    except Exception as e:
        print(f"Error saving configuration to {config_save_path}: {e}")
    # --- End Save Configuration --- # END MODIFIED

    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logs: {log_dir}")

    # Now 'device' is defined and passed to the agent
    agent = SAC(config=sac_config, world_config=world_config, device=device) # Agent uses config flags
    memory = ReplayBuffer(config=buffer_config, world_config=world_config)
    os.makedirs(train_config.models_dir, exist_ok=True)

    # Checkpoint loading
    model_files = [f for f in os.listdir(train_config.models_dir) if f.startswith("sac_") and f.endswith(".pt")]
    latest_model_path = None
    if model_files:
        try: latest_model_path = max([os.path.join(train_config.models_dir, f) for f in model_files], key=os.path.getmtime)
        except Exception as e: print(f"Could not find latest model: {e}")

    total_steps = 0
    start_episode = 1
    if latest_model_path and os.path.exists(latest_model_path):
        print(f"\nResuming SAC training from: {latest_model_path}")
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
        print("\nStarting SAC training from scratch.")

    # Training Loop
    episode_rewards = []
    all_losses = {'critic_loss': [], 'actor_loss': [], 'alpha': [], 'alpha_loss': []}
    timing_metrics = { 'env_step_time': deque(maxlen=100), 'parameter_update_time': deque(maxlen=100) }

    # Initialize world outside the loop to manage seeds correctly
    world = World(world_config=world_config)

    # Reward component tracking initialization
    reward_component_accumulator = {k: [] for k in world.reward_components}


    pbar = tqdm(range(start_episode, train_config.num_episodes + 1),
                desc="Training SAC Mapping", unit="episode", initial=start_episode-1, total=train_config.num_episodes)

    for episode in pbar:
        # Reset world (uses seeding mechanism internally)
        state = world.reset()
        episode_reward = 0
        episode_steps = 0
        actor_hidden_state = agent.actor.get_initial_hidden_state(batch_size=1, device=device) if agent.use_rnn else None
        episode_losses_temp = {'critic_loss': [], 'actor_loss': [], 'alpha': [], 'alpha_loss': []}
        updates_made_this_episode = 0
        episode_metrics = [] # Changed from episode_ious
        episode_reward_components = {k: 0.0 for k in reward_component_accumulator}

        for step_in_episode in range(train_config.max_steps):
            # select_action uses normalizer conditionally
            action_normalized, next_actor_hidden_state = agent.select_action(
                state, actor_hidden_state=actor_hidden_state, evaluate=False
            )

            step_start_time = time.time()
            next_state = world.step(action_normalized, training=True, terminal_step=(step_in_episode == train_config.max_steps - 1))
            step_time = time.time() - step_start_time
            timing_metrics['env_step_time'].append(step_time)

            reward = world.reward
            done = world.done
            current_metric = world.performance_metric # Changed from current_iou

            # Accumulate reward components for logging
            for key, value in world.reward_components.items():
                 if key in episode_reward_components:
                    episode_reward_components[key] += value
                 else:
                    print(f"Warning: Unexpected reward component key '{key}' from world.")

            memory.push(state, action_normalized, reward, next_state, done)

            state = next_state
            if agent.use_rnn: actor_hidden_state = next_actor_hidden_state
            episode_reward += reward
            episode_steps += 1
            total_steps += 1
            episode_metrics.append(current_metric) # Changed from episode_ious

            # update_parameters uses normalizer conditionally
            if total_steps >= learning_starts and total_steps % train_freq == 0:
                for _ in range(gradient_steps):
                    if len(memory) >= batch_size:
                        update_start_time = time.time()
                        losses = agent.update_parameters(memory, batch_size)
                        update_time = time.time() - update_start_time
                        if losses:
                            timing_metrics['parameter_update_time'].append(update_time)
                            if not any(np.isnan(v) for v in losses.values() if isinstance(v, (float, np.float64))):
                                for key, val in losses.items():
                                     if isinstance(val, (float, np.float64)):
                                        episode_losses_temp[key].append(val)
                                updates_made_this_episode += 1
                            else:
                                print(f"INFO: Skipping SAC loss logging step {total_steps} due to NaN.")
                        else:
                             break
                    else:
                         break

            if done:
                break

        # --- Logging (End of Episode) ---
        episode_rewards.append(episode_reward)
        avg_losses = {k: np.mean(v) if v else float('nan') for k, v in episode_losses_temp.items()}
        avg_metric = np.mean(episode_metrics) if episode_metrics else 0.0 # Changed from avg_iou

        for key, value in episode_reward_components.items():
            if key in reward_component_accumulator:
                 reward_component_accumulator[key].append(value)
            else:
                 print(f"Warning: Key '{key}' from episode components not found in accumulator.")

        if updates_made_this_episode > 0:
             if not np.isnan(avg_losses['critic_loss']): all_losses['critic_loss'].append(avg_losses['critic_loss'])
             if not np.isnan(avg_losses['actor_loss']): all_losses['actor_loss'].append(avg_losses['actor_loss'])
             if not np.isnan(avg_losses['alpha']): all_losses['alpha'].append(avg_losses['alpha'])
             if agent.auto_tune_alpha and not np.isnan(avg_losses['alpha_loss']): all_losses['alpha_loss'].append(avg_losses['alpha_loss'])

        if episode % log_frequency_ep == 0:
            if timing_metrics['env_step_time']: writer.add_scalar('Time/Environment_Step_ms_Avg100', np.mean(timing_metrics['env_step_time']) * 1000, total_steps)
            if timing_metrics['parameter_update_time']: writer.add_scalar('Time/Parameter_Update_ms_Avg100', np.mean(timing_metrics['parameter_update_time']) * 1000, total_steps)

            writer.add_scalar('Reward/Episode', episode_reward, total_steps)
            writer.add_scalar('Steps/Episode', episode_steps, total_steps)
            writer.add_scalar('Progress/Total_Steps', total_steps, episode)
            writer.add_scalar('Progress/Buffer_Size', len(memory), total_steps)
            writer.add_scalar('Performance/Metric_AvgEp', avg_metric, total_steps) # Changed tag
            writer.add_scalar('Performance/Metric_EndEp', world.performance_metric, total_steps) # Changed tag

            if updates_made_this_episode > 0:
                if not np.isnan(avg_losses['critic_loss']): writer.add_scalar('Loss/Critic_AvgEp', avg_losses['critic_loss'], total_steps)
                if not np.isnan(avg_losses['actor_loss']): writer.add_scalar('Loss/Actor_AvgEp', avg_losses['actor_loss'], total_steps)
                if not np.isnan(avg_losses['alpha']): writer.add_scalar('Alpha/Value', avg_losses['alpha'], total_steps)
                if agent.auto_tune_alpha and not np.isnan(avg_losses['alpha_loss']): writer.add_scalar('Loss/Alpha_AvgEp', avg_losses['alpha_loss'], total_steps)
            else:
                 writer.add_scalar('Alpha/Value', agent.alpha, total_steps)


            # Log average reward components
            # Check if 'total' exists before iterating
            if "total" in reward_component_accumulator:
                for name, component_list in reward_component_accumulator.items():
                    avg_component_value = np.mean(component_list) if component_list else 0.0
                    # Only log if the list wasn't empty (avoid logging 0.0 for unused components)
                    if component_list:
                        writer.add_scalar(f'RewardComponents_AvgEp/{name}', avg_component_value, total_steps)
                # Reset accumulator after logging average
                reward_component_accumulator = {k: [] for k in reward_component_accumulator}

            # Conditionally log normalizer stats
            if agent.use_state_normalization and agent.state_normalizer and agent.state_normalizer.count > agent.state_normalizer.epsilon:
                writer.add_scalar('Stats/Normalizer_Count', agent.state_normalizer.count.item(), total_steps)
                # Log mean/std for a few representative dimensions
                # Assuming state_dim >= 7 (5 sensors + 2 coords)
                writer.add_scalar('Stats/Normalizer_Mean_Sensor0', agent.state_normalizer.mean[0].item(), total_steps)
                writer.add_scalar('Stats/Normalizer_Std_Sensor0', torch.sqrt(agent.state_normalizer.var[0].clamp(min=1e-8)).item(), total_steps)
                if agent.state_dim > 5:
                    writer.add_scalar('Stats/Normalizer_Mean_AgentXNorm', agent.state_normalizer.mean[5].item(), total_steps)
                    writer.add_scalar('Stats/Normalizer_Std_AgentXNorm', torch.sqrt(agent.state_normalizer.var[5].clamp(min=1e-8)).item(), total_steps)


            lookback = min(100, len(episode_rewards))
            avg_reward_100 = np.mean(episode_rewards[-lookback:]) if episode_rewards else 0
            writer.add_scalar('Reward/Average_100', avg_reward_100, total_steps)

        if episode % 10 == 0:
            lookback = min(10, len(episode_rewards))
            avg_reward_10 = np.mean(episode_rewards[-lookback:]) if episode_rewards else 0
            pbar_postfix = {
                'avg_rew_10': f'{avg_reward_10:.2f}',
                'steps': total_steps,
                'Metric': f"{world.performance_metric:.3f}", # Changed label
                'alpha': f"{agent.alpha:.3f}"
            }
            if updates_made_this_episode and not np.isnan(avg_losses['critic_loss']):
                 pbar_postfix['crit_loss'] = f"{avg_losses['critic_loss']:.2f}"
            pbar.set_postfix(pbar_postfix)


        if episode % save_interval_ep == 0 and episode > 0: # Avoid saving at ep 0
            save_path = os.path.join(train_config.models_dir, f"sac_ep{episode}_step{total_steps}.pt")
            agent.save_model(save_path)

    pbar.close()
    writer.close()
    print(f"SAC Training finished. Total steps: {total_steps}")
    final_save_path = os.path.join(train_config.models_dir, f"sac_final_ep{train_config.num_episodes}_step{total_steps}.pt")
    agent.save_model(final_save_path)

    if run_evaluation:
         print("\nStarting evaluation after training...")
         # Pass the config object which might have evaluation seeds
         evaluate_sac(agent=agent, config=config)

    return agent, episode_rewards

# --- Evaluation Loop (evaluate_sac) ---
def evaluate_sac(agent: SAC, config: DefaultConfig):
    eval_config = config.evaluation
    world_config = config.world
    vis_config = config.visualization

    # Conditional Visualization Import
    vis_available = False
    visualize_world, reset_trajectories, save_gif = None, None, None
    if eval_config.render:
        try:
            from visualization import visualize_world, reset_trajectories, save_gif
            import imageio.v2 as imageio # Keep import here
            vis_available = True
            print("Visualization enabled.")
        except ImportError:
            print("Visualization libraries (matplotlib/imageio/PIL) not found. Rendering disabled.")
            # Don't modify config here, just track availability
            vis_available = False
    else:
        print("Rendering disabled by config.")
        vis_available = False

    eval_rewards = [] # Still track rewards, though less critical than metric
    eval_metrics = [] # Changed from eval_ious
    success_count = 0
    all_episode_gif_paths = []

    # --- Set Agent and Normalizer (if exists) to Evaluation Mode ---
    agent.actor.eval()
    agent.critic.eval()
    if agent.use_state_normalization and agent.state_normalizer:
        agent.state_normalizer.eval()
    # --- End Set Eval Mode ---

    print(f"\nRunning SAC Evaluation for {eval_config.num_episodes} episodes...")
    # Use evaluation seeds if provided
    eval_seeds = world_config.seeds
    if len(eval_seeds) != eval_config.num_episodes:
         print(f"Warning: Number of evaluation seeds ({len(eval_seeds)}) doesn't match num_episodes ({eval_config.num_episodes}). Using first {eval_config.num_episodes} seeds or generating if needed.")
         if len(eval_seeds) < eval_config.num_episodes:
              # Generate additional random seeds if needed
              extra_seeds_needed = eval_config.num_episodes - len(eval_seeds)
              eval_seeds.extend([random.randint(0, 2**32 - 1) for _ in range(extra_seeds_needed)])
         eval_seeds = eval_seeds[:eval_config.num_episodes]

    world = World(world_config=world_config) # Create world instance

    for episode in range(eval_config.num_episodes):
        seed_to_use = eval_seeds[episode] if eval_seeds else None
        state = world.reset(seed=seed_to_use) # Reset with specific seed
        episode_reward = 0
        episode_frames = []
        actor_hidden_state = agent.actor.get_initial_hidden_state(batch_size=1, device=agent.device) if agent.use_rnn else None
        episode_metrics_current = [] # Track metric per step

        if eval_config.render and vis_available:
            os.makedirs(vis_config.save_dir, exist_ok=True)
            if reset_trajectories: reset_trajectories()
            try:
                fname = f"sac_eval_ep{episode+1}_frame_000_initial.png"
                initial_frame_file = visualize_world(world, vis_config=vis_config, filename=fname)
                if initial_frame_file and os.path.exists(initial_frame_file): episode_frames.append(initial_frame_file)
            except Exception as e: print(f"Warn: Vis failed init state ep {episode+1}. E: {e}")

        for step in range(eval_config.max_steps):
            # select_action uses normalizer conditionally and in eval mode
            action_normalized, next_actor_hidden_state = agent.select_action(
                state, actor_hidden_state=actor_hidden_state, evaluate=True
            )

            # Set training=False for evaluation steps
            next_state = world.step(action_normalized, training=False, terminal_step=(step == eval_config.max_steps - 1))
            reward = world.reward # Reward will be 0 if training=False
            done = world.done
            current_metric = world.performance_metric # Changed from iou
            episode_metrics_current.append(current_metric) # Track metric at each step

            if eval_config.render and vis_available:
                try:
                    fname = f"sac_eval_ep{episode+1}_frame_{step+1:03d}.png"
                    frame_file = visualize_world(world, vis_config=vis_config, filename=fname)
                    if frame_file and os.path.exists(frame_file): episode_frames.append(frame_file)
                except Exception as e: print(f"Warn: Vis failed step {step+1} ep {episode+1}. E: {e}")

            state = next_state
            if agent.use_rnn: actor_hidden_state = next_actor_hidden_state
            episode_reward += reward # Sum rewards (mostly 0 during eval)
            if done:
                break

        final_metric = world.performance_metric # Changed from final_iou
        eval_rewards.append(episode_reward)
        eval_metrics.append(final_metric) # Store final metric for summary

        success = final_metric >= world_config.success_metric_threshold
        if success: success_count += 1
        status = "Success!" if success else "Failure."

        print(f"  Ep {episode+1}/{eval_config.num_episodes} (Seed:{world.current_seed}): Steps={world.current_step}, Terminated={world.done}, Final Metric: {final_metric:.3f}. {status}")

        if eval_config.render and vis_available and episode_frames and save_gif:
            gif_filename = f"sac_mapping_eval_episode_{episode+1}_seed{world.current_seed}.gif"
            try:
                print(f"  Saving GIF for SAC episode {episode+1} with {len(episode_frames)} frames...")
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

    print("\n--- SAC Evaluation Summary ---")
    print(f"Episodes: {eval_config.num_episodes}")
    print(f"Average Final Metric (Point Inclusion): {avg_eval_metric:.3f} +/- {std_eval_metric:.3f}") # Changed label
    print(f"Success Rate (Metric >= {world_config.success_metric_threshold:.2f}): {success_rate:.2%} ({success_count}/{eval_config.num_episodes})")
    print(f"Average Episode Reward: {avg_eval_reward:.3f} +/- {std_eval_reward:.3f}") # Keep reward reporting
    if eval_config.render and vis_available and all_episode_gif_paths: print(f"GIFs saved to: '{os.path.abspath(vis_config.save_dir)}'")
    elif eval_config.render and not vis_available: print("Rendering was enabled but visualization libraries were not found.")
    elif not eval_config.render: print("Rendering disabled.")
    print("--- End SAC Evaluation ---\n")

    return eval_rewards, success_rate, avg_eval_metric # Return avg metric instead of IoU