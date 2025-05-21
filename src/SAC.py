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
import math 
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Local imports
from world import World 
from configs import DefaultConfig, ReplayBufferConfig, SACConfig, WorldConfig, CORE_STATE_DIM
from torch.utils.tensorboard import SummaryWriter
from typing import Tuple, Optional, Dict, Any, Union, List
from utils import RunningMeanStd 

# --- SumTree Class (for PER) ---
class SumTree:
    """
    Simple SumTree implementation for Prioritized Experience Replay.
    Stores priorities in the tree and pointers to transitions in a separate memory array.
    """
    write = 0 # Write pointer

    def __init__(self, capacity):
        self.capacity = capacity
        # Tree structure: internal nodes + leaves. Size 2*capacity - 1
        self.tree = np.zeros(2 * capacity - 1)
        # Data storage (stores indices to the actual buffer)
        self.data_indices = np.zeros(capacity, dtype=int)
        # Track max priority for new samples
        self.max_priority = 1.0 # Initialize with 1.0

    def _propagate(self, idx, change):
        """Propagate priority change up the tree."""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        """Find sample index and priority for a given sum s."""
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree): # Leaf node
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        """Total priority sum."""
        return self.tree[0]

    def add(self, priority, data_idx):
        """Add priority and associated data index to the tree."""
        # Use max priority for new experiences
        priority = self.max_priority
        idx = self.write + self.capacity - 1 # Tree index for leaf

        self.data_indices[self.write] = data_idx # Store data index
        self.update(idx, priority) # Update tree with priority

        self.write += 1
        if self.write >= self.capacity: # Wrap around pointer
            self.write = 0

    def update(self, idx, priority):
        """Update priority of a node."""
        # Update max priority tracker
        if priority > self.max_priority:
            self.max_priority = priority

        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s):
        """Get leaf index, priority value, and data index for a priority sum s."""
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1 # Convert tree index to data_indices index
        actual_data_idx = self.data_indices[data_idx] # Get the index pointing to buffer memory
        return (idx, self.tree[idx], actual_data_idx)


# --- Prioritized Replay Buffer ---
class PrioritizedReplayBuffer:
    """ Prioritized Experience Replay buffer using a SumTree. """
    def __init__(self, config: SACConfig, buffer_config: ReplayBufferConfig, world_config: WorldConfig):
        self.tree = SumTree(buffer_config.capacity)
        self.capacity = buffer_config.capacity
        self.memory = deque(maxlen=self.capacity) # Stores actual transitions
        self.config = config
        self.buffer_config = buffer_config
        self.trajectory_length = world_config.trajectory_length
        self.feature_dim = world_config.trajectory_feature_dim # Includes heading
        self.alpha = config.per_alpha
        self.beta_start = config.per_beta_start
        self.beta = self.beta_start
        self.beta_frames = config.per_beta_frames
        self.epsilon = config.per_epsilon
        self.frame = 0 # For beta annealing (tracks updates)
        self._next_data_idx = 0 # Index for the memory deque
        self._current_size = 0 # Track current filled size

    def push(self, state, action, reward, next_state, done):
        """ Add a new experience with max priority. """
        trajectory = state['full_trajectory'] # Includes heading
        next_trajectory = next_state['full_trajectory'] # Includes heading
        if isinstance(action, (np.ndarray, list)): action = action[0]
        if isinstance(reward, (np.ndarray)): reward = reward.item()

        if not isinstance(trajectory, np.ndarray) or trajectory.shape != (self.trajectory_length, self.feature_dim): return
        if not isinstance(next_trajectory, np.ndarray) or next_trajectory.shape != (self.trajectory_length, self.feature_dim): return
        if np.isnan(trajectory).any() or np.isnan(next_trajectory).any(): return

        experience = (trajectory, float(action), float(reward), next_trajectory, done)

        # Store the actual experience
        if self._current_size < self.capacity:
             self.memory.append(experience)
             data_idx = self._current_size
             self._current_size += 1
        else:
             # Overwrite oldest experience in deque
             data_idx = self.tree.write # The index being overwritten in SumTree
             self.memory[data_idx] = experience # Replace in deque


        # Add to SumTree with max priority
        # The data index stored in SumTree refers to the position in the conceptual array of size capacity
        self.tree.add(self.tree.max_priority ** self.alpha, data_idx)


    def sample(self, batch_size: int) -> Optional[Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]]:
        """ Sample a batch, returning transitions, indices, and IS weights. """
        if self._current_size < batch_size:
            return None

        batch_data = {'state': [], 'action': [], 'reward': [], 'next_state': [], 'done': []}
        idxs = np.empty((batch_size,), dtype=int) # Tree indices
        priorities = np.empty((batch_size,), dtype=float)
        data_indices = np.empty((batch_size,), dtype=int) # Indices into self.memory

        # Calculate current beta
        self.beta = min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)
        self.frame += 1

        segment = self.tree.total() / batch_size
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, p, data_idx = self.tree.get(s)

            idxs[i] = idx
            priorities[i] = p
            data_indices[i] = data_idx # Store the index in the memory deque

        # Retrieve data using data_indices
        try:
            sampled_experiences = [self.memory[di] for di in data_indices]
            trajectory, action, reward, next_trajectory, done = zip(*sampled_experiences)

            batch_data['state'] = np.array(trajectory, dtype=np.float32) # Includes heading
            batch_data['action'] = np.array(action, dtype=np.float32).reshape(-1, 1)
            batch_data['reward'] = np.array(reward, dtype=np.float32)
            batch_data['next_state'] = np.array(next_trajectory, dtype=np.float32) # Includes heading
            batch_data['done'] = np.array(done, dtype=np.float32)

            # Calculate Importance Sampling weights
            sampling_probabilities = priorities / self.tree.total()
            # Add small epsilon to prevent division by zero if probability is zero
            weights = np.power(self._current_size * sampling_probabilities + 1e-10, -self.beta)
            weights /= weights.max() # Normalize weights

        except IndexError:
            # print(f"IndexError during PER sampling. data_indices: {data_indices}, current_size: {self._current_size}, len(memory): {len(self.memory)}")
            return None # Or handle error appropriately
        except Exception as e:
            # print(f"Error converting PER sampled batch: {e}")
            return None

        return batch_data, idxs, weights

    def update_priorities(self, batch_indices, batch_priorities):
        """ Update priorities of sampled transitions. """
        for idx, priority in zip(batch_indices, batch_priorities):
            priority = abs(priority) + self.epsilon # Ensure positive
            self.tree.update(idx, priority ** self.alpha)
        # After updating, find the new max priority among the leaves actually used.
        # This prevents max_priority from growing indefinitely if old high-priority samples are never resampled.
        active_leaf_indices = np.arange(self.capacity) + self.capacity - 1
        self.tree.max_priority = np.max(self.tree.tree[active_leaf_indices[:self._current_size]]) if self._current_size > 0 else 1.0


    def __len__(self):
        return self._current_size


# --- Standard Replay Buffer (Unchanged) ---
class ReplayBuffer:
    """Experience replay buffer storing full state trajectories."""
    def __init__(self, config: ReplayBufferConfig, world_config: WorldConfig):
        self.buffer = deque(maxlen=config.capacity)
        self.config = config
        self.trajectory_length = world_config.trajectory_length
        self.feature_dim = world_config.trajectory_feature_dim # Includes state+action+reward (state includes heading)

    def push(self, state, action, reward, next_state, done):
        """ Add a new experience to memory. """
        trajectory = state['full_trajectory'] # Shape (N, feat_dim) - state is normalized (incl heading)
        next_trajectory = next_state['full_trajectory'] # Shape (N, feat_dim) - state is normalized (incl heading)

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

    def sample(self, batch_size: int) -> Optional[Dict[str, np.ndarray]]:
        """Sample a batch of experiences from memory."""
        if len(self.buffer) < batch_size:
            return None
        try:
            batch = random.sample(self.buffer, batch_size)
        except ValueError:
            # print(f"Warning: ReplayBuffer sampling failed. len(buffer)={len(self.buffer)}, batch_size={batch_size}")
            return None

        # state/next_state are full trajectories (batch, N, feat_dim) - state part is normalized (incl heading)
        trajectory, action, reward, next_trajectory, done = zip(*batch)
        batch_data = {}
        try:
            batch_data['state'] = np.array(trajectory, dtype=np.float32)
            batch_data['action'] = np.array(action, dtype=np.float32).reshape(-1, 1) # Shape (batch, 1)
            batch_data['reward'] = np.array(reward, dtype=np.float32) # Shape (batch,)
            batch_data['next_state'] = np.array(next_trajectory, dtype=np.float32)
            batch_data['done'] = np.array(done, dtype=np.float32) # Shape (batch,)

            # Check for NaNs after conversion (optional but good practice)
            if np.isnan(batch_data['state']).any() or np.isnan(batch_data['next_state']).any():
                # print("Warning: Sampled batch contains NaN values. Returning None.")
                return None

        except Exception as e:
            # print(f"Error converting sampled batch to numpy arrays: {e}")
            return None

        return batch_data

    def __len__(self):
        return len(self.buffer)

# --- Actor and Critic Classes (Unchanged structure, but input dim handled by config) ---
class Actor(nn.Module):
    """Policy network (Actor) for SAC, optionally with RNN."""
    def __init__(self, config: SACConfig, world_config: WorldConfig):
        super(Actor, self).__init__()
        self.config = config
        self.world_config = world_config
        self.use_rnn = config.use_rnn
        # Input state dim (already potentially normalized, now includes heading)
        self.state_dim = config.state_dim
        self.action_dim = config.action_dim
        self.trajectory_length = world_config.trajectory_length

        if self.use_rnn:
            self.rnn_hidden_size = config.rnn_hidden_size
            self.rnn_num_layers = config.rnn_num_layers
            # RNN processes the CORE_STATE_DIM features (normalized, incl heading)
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
            # MLP uses only the last normalized basic state (incl heading)
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
             - RNN: (batch, seq_len, state_dim) - normalized basic state sequence (incl heading)
             - MLP: (batch, state_dim) - normalized last basic state (incl heading)
        """
        next_hidden_state = None
        if self.use_rnn and self.rnn:
            # RNN processing (input is already normalized basic state sequence incl heading)
            rnn_output, next_hidden_state = self.rnn(network_input, hidden_state)
            # Use the output corresponding to the last time step
            mlp_input = rnn_output[:, -1, :] # Shape: (batch, rnn_hidden_size)
        else:
            # MLP input is the normalized last basic state (incl heading)
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
        # Input state dim (already potentially normalized, includes heading)
        self.state_dim = config.state_dim
        self.action_dim = config.action_dim
        self.trajectory_length = world_config.trajectory_length

        if self.use_rnn:
            self.rnn_hidden_size = config.rnn_hidden_size
            self.rnn_num_layers = config.rnn_num_layers
            # RNN processes the CORE_STATE_DIM features (normalized, incl heading)
            rnn_input_dim = self.state_dim
            rnn_cell = nn.LSTM if config.rnn_type == 'lstm' else nn.GRU

            self.rnn1 = rnn_cell(input_size=rnn_input_dim, hidden_size=config.rnn_hidden_size,
                                  num_layers=config.rnn_num_layers, batch_first=True)
            self.rnn2 = rnn_cell(input_size=rnn_input_dim, hidden_size=config.rnn_hidden_size,
                                  num_layers=config.rnn_num_layers, batch_first=True)
            mlp_input_dim = config.rnn_hidden_size + self.action_dim # MLP takes final RNN state + action
        else:
            # MLP uses only the normalized last basic state (incl heading) + action
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
             - RNN: (batch, seq_len, state_dim) - normalized basic state sequence (incl heading)
             - MLP: (batch, state_dim) - normalized last basic state (incl heading)
           action shape: (batch, action_dim)
        """
        next_hidden_state = None
        if self.use_rnn:
            h1_in, h2_in = None, None
            if isinstance(hidden_state, tuple) and len(hidden_state) == 2:
                 h1_in, h2_in = hidden_state

            # RNN processing (input is already normalized basic state sequence incl heading)
            rnn_out1, next_h1 = self.rnn1(network_input, h1_in)
            rnn_out2, next_h2 = self.rnn2(network_input, h2_in)
            next_hidden_state = (next_h1, next_h2)

            # Use final RNN hidden state and concatenate with action
            mlp_input1 = torch.cat([rnn_out1[:, -1, :], action], dim=1)
            mlp_input2 = torch.cat([rnn_out2[:, -1, :], action], dim=1)
        else:
            # Use normalized last basic state (incl heading) and concatenate with action
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


# --- Updated SAC Agent ---
class SAC:
    """Soft Actor-Critic algorithm implementation with trajectory states, normalization, and optional PER."""
    def __init__(self, config: SACConfig, world_config: WorldConfig, buffer_config: ReplayBufferConfig, device: torch.device = None): # Added buffer_config
        self.config = config
        self.world_config = world_config
        self.buffer_config = buffer_config # Store buffer config
        self.gamma = config.gamma
        self.tau = config.tau
        self.alpha = config.alpha
        self.auto_tune_alpha = config.auto_tune_alpha
        self.use_rnn = config.use_rnn
        self.trajectory_length = world_config.trajectory_length
        self.state_dim = config.state_dim # Dimension of the (potentially normalized) basic state input (now includes heading)
        self.action_dim = config.action_dim
        self.use_state_normalization = config.use_state_normalization
        self.use_reward_normalization = config.use_reward_normalization
        self.use_per = config.use_per # Store PER flag

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"SAC Agent using device: {self.device}")
        if self.use_rnn: print(f"SAC Agent using RNN: Type={config.rnn_type}, SeqLen={self.trajectory_length}")
        else: print(f"SAC Agent using MLP (Processing last state of trajectory)")
        if self.use_per: print(f"SAC Agent using Prioritized Replay Buffer (alpha={config.per_alpha}, beta0={config.per_beta_start})")
        else: print(f"SAC Agent using Standard Replay Buffer")

        # --- Conditional State Normalization ---
        self.state_normalizer = None
        if self.use_state_normalization:
             # Initialize with the new state_dim
            self.state_normalizer = RunningMeanStd(shape=(self.state_dim,), device=self.device)
            print(f"SAC Agent state normalization ENABLED for dim: {self.state_dim} (incl. heading)") # Updated print
        else:
            print(f"SAC Agent state normalization DISABLED.")

        if self.use_reward_normalization: print(f"SAC Agent reward normalization ENABLED.")
        else: print(f"SAC Agent reward normalization DISABLED.")

        # --- Instantiate Networks ---
        self.actor = Actor(config, world_config).to(self.device)
        self.critic = Critic(config, world_config).to(self.device)
        self.critic_target = Critic(config, world_config).to(self.device)

        self.critic_target.load_state_dict(self.critic.state_dict())
        for target_param in self.critic_target.parameters():
            target_param.requires_grad = False

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.critic_lr)

        if self.auto_tune_alpha:
            self.target_entropy = -torch.prod(torch.Tensor([self.action_dim]).to(self.device)).item()
            self.log_alpha = torch.tensor(np.log(self.alpha), requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=config.actor_lr)
        else:
            self.log_alpha = torch.tensor(np.log(self.alpha)).to(self.device)

        # --- Instantiate Replay Buffer ---
        if self.use_per:
            self.memory = PrioritizedReplayBuffer(config=config, buffer_config=buffer_config, world_config=world_config)
        else:
            self.memory = ReplayBuffer(config=buffer_config, world_config=world_config)

        self.total_updates = 0 # Track updates for beta annealing if using PER

    def select_action(self, state: dict, actor_hidden_state: Optional[Tuple] = None, evaluate: bool = False) -> Tuple[float, Optional[Tuple]]:
        """Select action (normalized yaw change [-1, 1]) based on normalized state (if enabled).
           'evaluate=True' means deterministic action (mean), 'evaluate=False' means stochastic action (sample).
        """
        state_trajectory = state['full_trajectory'] # Includes normalized heading
        state_tensor_full = torch.FloatTensor(state_trajectory).to(self.device).unsqueeze(0) # (1, N, feat_dim)

        with torch.no_grad():
            # Extract the basic state part (sensors + coords + heading)
            state_part_normalized = state_tensor_full[:, :, :self.state_dim] # (1, N, state_dim)

            if self.use_rnn:
                raw_network_input = state_part_normalized # Use full sequence
            else:
                raw_network_input = state_part_normalized[:, -1, :] # Use last state (incl heading)

            if self.use_state_normalization and self.state_normalizer:
                self.state_normalizer.eval()
                network_input_processed = self.state_normalizer.normalize(raw_network_input)
                self.state_normalizer.train()
            else:
                network_input_processed = raw_network_input

            self.actor.eval()
            if evaluate:
                # Get deterministic action (tanh of mean)
                _, _, action_mean_squashed, next_actor_hidden_state = self.actor.sample(network_input_processed, actor_hidden_state)
                action_normalized = action_mean_squashed
            else:
                # Sample stochastic action
                action_normalized, _, _, next_actor_hidden_state = self.actor.sample(network_input_processed, actor_hidden_state)
            self.actor.train()

        action_normalized_float = action_normalized.detach().cpu().numpy()[0, 0]
        return action_normalized_float, next_actor_hidden_state

    # Modified update_parameters for PER
    def update_parameters(self, batch_size: int):
        """Perform a single SAC update step using a batch of trajectories with conditional normalization and PER."""
        if self.use_per:
            sampled_batch_info = self.memory.sample(batch_size)
            if sampled_batch_info is None: return None
            # batch_data is dict, idxs is ndarray, weights is ndarray
            batch_data, idxs, weights = sampled_batch_info
            weights_tensor = torch.FloatTensor(weights).unsqueeze(1).to(self.device) # (batch, 1)
        else:
            batch_data = self.memory.sample(batch_size)
            if batch_data is None: return None
            idxs = None # No indices needed for standard buffer
            weights_tensor = torch.ones(batch_size, 1, device=self.device) # Weights are 1

        # Extract data from dictionary
        state_batch = batch_data['state'] # Includes heading
        action_batch_normalized = batch_data['action']
        reward_batch = batch_data['reward']
        next_state_batch = batch_data['next_state'] # Includes heading
        done_batch = batch_data['done']

        state_batch_tensor = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch_tensor = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch_tensor = torch.FloatTensor(action_batch_normalized).to(self.device)
        reward_batch_tensor_raw = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        done_batch_tensor = torch.FloatTensor(done_batch).to(self.device).unsqueeze(1)

        # --- Reward Normalization ---
        if self.use_reward_normalization:
            reward_mean = reward_batch_tensor_raw.mean()
            reward_std = reward_batch_tensor_raw.std()
            reward_batch_tensor = (reward_batch_tensor_raw - reward_mean) / (reward_std + 1e-8)
        else:
            reward_batch_tensor = reward_batch_tensor_raw

        # --- State Processing and Normalization ---
        # Extract basic state part (incl heading)
        state_part_normalized = state_batch_tensor[:, :, :self.state_dim]
        next_state_part_normalized = next_state_batch_tensor[:, :, :self.state_dim]

        if self.use_state_normalization and self.state_normalizer:
            # Update normalizer with all basic states (incl heading) from the batch
            self.state_normalizer.update(state_part_normalized.reshape(-1, self.state_dim))

        if self.use_rnn:
            raw_current_input = state_part_normalized # (batch, N, state_dim)
            raw_next_input = next_state_part_normalized # (batch, N, state_dim)
            if self.use_state_normalization and self.state_normalizer:
                 batch, seq_len, _ = raw_current_input.shape
                 current_network_input = self.state_normalizer.normalize(raw_current_input.reshape(-1, self.state_dim)).reshape(batch, seq_len, self.state_dim)
                 batch_next, seq_len_next, _ = raw_next_input.shape
                 next_network_input = self.state_normalizer.normalize(raw_next_input.reshape(-1, self.state_dim)).reshape(batch_next, seq_len_next, self.state_dim)
            else: current_network_input, next_network_input = raw_current_input, raw_next_input
        else:
            raw_current_input = state_part_normalized[:, -1, :] # (batch, state_dim) - last state incl heading
            raw_next_input = next_state_part_normalized[:, -1, :] # (batch, state_dim) - last state incl heading
            if self.use_state_normalization and self.state_normalizer:
                 current_network_input = self.state_normalizer.normalize(raw_current_input)
                 next_network_input = self.state_normalizer.normalize(raw_next_input)
            else: current_network_input, next_network_input = raw_current_input, raw_next_input

        # --- Initialize Hidden States (if RNN) ---
        initial_actor_hidden = self.actor.get_initial_hidden_state(batch_size, self.device) if self.use_rnn else None
        initial_critic_hidden = self.critic.get_initial_hidden_state(batch_size, self.device) if self.use_rnn else None
        initial_critic_target_hidden = self.critic_target.get_initial_hidden_state(batch_size, self.device) if self.use_rnn else None

        # --- Critic Update ---
        with torch.no_grad():
            next_action, next_log_prob, _, _ = self.actor.sample(next_network_input, initial_actor_hidden)
            target_q1, target_q2, _ = self.critic_target(next_network_input, next_action, initial_critic_target_hidden)
            target_q_min = torch.min(target_q1, target_q2)
            current_alpha_val = self.log_alpha.exp().detach() # Use detach here
            target_q_entropy = target_q_min - current_alpha_val * next_log_prob
            y = reward_batch_tensor + (1.0 - done_batch_tensor) * self.gamma * target_q_entropy # Target Q value

        current_q1, current_q2, _ = self.critic(current_network_input, action_batch_tensor, initial_critic_hidden)

        # --- PER Integration: Weighted Loss and Priority Calculation ---
        td_error1 = (current_q1 - y).abs()
        td_error2 = (current_q2 - y).abs()
        # Use 'none' reduction for element-wise loss before weighting
        critic1_loss = F.mse_loss(current_q1, y, reduction='none')
        critic2_loss = F.mse_loss(current_q2, y, reduction='none')
        # Apply IS weights
        weighted_critic_loss = (weights_tensor * (critic1_loss + critic2_loss)).mean()

        self.critic_optimizer.zero_grad()
        weighted_critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # --- PER: Update Priorities ---
        if self.use_per and idxs is not None:
            # Calculate new priorities based on TD error (e.g., max of the two errors)
            new_priorities = torch.max(td_error1, td_error2).squeeze(1).detach().cpu().numpy()
            self.memory.update_priorities(idxs, new_priorities)

        # --- Actor Update ---
        for param in self.critic.parameters(): param.requires_grad = False
        action_pi, log_prob_pi, _, _ = self.actor.sample(current_network_input, initial_actor_hidden)
        q1_pi, q2_pi, _ = self.critic(current_network_input, action_pi, initial_critic_hidden)
        q_pi_min = torch.min(q1_pi, q2_pi)
        current_alpha_val = self.log_alpha.exp() # Get current alpha (can require grad for alpha tuning)
        # Actor loss doesn't typically use IS weights, but some variations do. Sticking to standard SAC here.
        actor_loss = (current_alpha_val.detach() * log_prob_pi - q_pi_min).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        for param in self.critic.parameters(): param.requires_grad = True

        # --- Alpha Update ---
        alpha_loss_item = 0.0
        if self.auto_tune_alpha:
            # Use the same log_prob_pi as actor loss (already detached there)
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

        self.total_updates += 1 # Increment update counter for beta annealing

        return {
            'critic_loss': weighted_critic_loss.item(), # Log weighted loss
            'actor_loss': actor_loss.item(),
            'alpha': self.alpha,
            'alpha_loss': alpha_loss_item,
            'beta': self.memory.beta if self.use_per else 0.0 # Log beta
        }

    def save_model(self, path: str):
        print(f"Saving SAC model to {path}...")
        os.makedirs(os.path.dirname(path), exist_ok=True) # Ensure directory exists
        save_dict = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'use_state_normalization': self.use_state_normalization,
            'use_per': self.use_per, # Save PER flag
            'per_beta': self.memory.beta if self.use_per else 0.0, # Save current beta
            'per_frame': self.memory.frame if self.use_per else 0, # Save beta frame
            'total_updates': self.total_updates, # Save update count
            'device_type': self.device.type
        }
        if self.use_state_normalization and self.state_normalizer:
            save_dict['state_normalizer_state_dict'] = self.state_normalizer.state_dict()
        if self.auto_tune_alpha:
            save_dict['log_alpha'] = self.log_alpha
            save_dict['alpha_optimizer_state_dict'] = self.alpha_optimizer.state_dict()
        # Save SumTree max_priority (important for PER restart)
        if self.use_per:
            save_dict['per_max_priority'] = self.memory.tree.max_priority

        torch.save(save_dict, path)

    def load_model(self, path: str):
        if not os.path.exists(path):
            print(f"Warning: SAC model file not found at {path}. Skipping loading.")
            return
        print(f"Loading SAC model from {path}...")
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])

        # Load State Normalizer
        loaded_use_state_norm = checkpoint.get('use_state_normalization', False) # Default False if missing
        if 'use_state_normalization' in checkpoint and loaded_use_state_norm != self.use_state_normalization:
            print(f"Warning: Loaded model STATE normalization setting ({loaded_use_state_norm}) differs from current config ({self.use_state_normalization}). Using current config setting.")
        if self.use_state_normalization and self.state_normalizer:
            if 'state_normalizer_state_dict' in checkpoint:
                try:
                    # Check shapes before loading
                    loaded_mean_shape = checkpoint['state_normalizer_state_dict']['mean'].shape
                    if loaded_mean_shape != self.state_normalizer.mean.shape:
                         print(f"Warning: Mismatch in loaded state normalizer shape ({loaded_mean_shape}) and current ({self.state_normalizer.mean.shape}). Reinitializing normalizer.")
                    else:
                         self.state_normalizer.load_state_dict(checkpoint['state_normalizer_state_dict'])
                         print(f"Loaded state normalizer statistics (dim={self.state_dim}).") 
                except Exception as e: print(f"Warning: Failed to load state normalizer stats: {e}. Using initial values.")
            else: print("Warning: State normalizer stats not found in checkpoint, but normalization is enabled.")

        # Load PER State
        loaded_use_per = checkpoint.get('use_per', False) # Default False if missing
        if 'use_per' in checkpoint and loaded_use_per != self.use_per:
            print(f"Warning: Loaded model PER setting ({loaded_use_per}) differs from current config ({self.use_per}). Using current config setting. Replay buffer type mismatch might cause issues.")
        if self.use_per and isinstance(self.memory, PrioritizedReplayBuffer):
             self.memory.beta = checkpoint.get('per_beta', self.config.per_beta_start)
             self.memory.frame = checkpoint.get('per_frame', 0)
             self.memory.tree.max_priority = checkpoint.get('per_max_priority', 1.0) # Load max priority
             print(f"Loaded PER state: beta={self.memory.beta:.4f}, frame={self.memory.frame}, max_p={self.memory.tree.max_priority:.4f}")
        elif not self.use_per and loaded_use_per:
             print("Warning: Checkpoint used PER, but current config doesn't. Ignoring PER state.")


        # Load Alpha
        if self.auto_tune_alpha and 'log_alpha' in checkpoint:
            if not hasattr(self, 'log_alpha') or not isinstance(self.log_alpha, torch.Tensor):
                 self.log_alpha = torch.tensor(0.0, requires_grad=True, device=self.device)
            self.log_alpha.data.copy_(checkpoint['log_alpha'].to(self.device))
            if not self.log_alpha.requires_grad: self.log_alpha.requires_grad_(True)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.config.actor_lr)
            try:
                if 'alpha_optimizer_state_dict' in checkpoint:
                     self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
            except ValueError as e: print(f"Warning: Could not load SAC alpha optimizer state: {e}. Reinitializing.")
            self.alpha = self.log_alpha.exp().item()
        elif not self.auto_tune_alpha and 'log_alpha' in checkpoint:
             if not hasattr(self, 'log_alpha') or not isinstance(self.log_alpha, torch.Tensor):
                  self.log_alpha = torch.tensor(0.0, device=self.device)
             self.log_alpha.data.copy_(checkpoint['log_alpha'].to(self.device))
             self.alpha = self.log_alpha.exp().item()


        self.critic_target.load_state_dict(self.critic.state_dict())
        for target_param in self.critic_target.parameters():
            target_param.requires_grad = False

        self.total_updates = checkpoint.get('total_updates', 0) # Load update count

        self.actor.train()
        self.critic.train()
        self.critic_target.train() 
        if self.use_state_normalization and self.state_normalizer:
            self.state_normalizer.train()

        print(f"SAC model loaded successfully from {path}")


# --- Updated Training Loop (train_sac) ---
def train_sac(original_config_name: str, config: DefaultConfig, use_multi_gpu: bool = False, run_evaluation: bool = True):
    sac_config = config.sac
    train_config = config.training
    buffer_config = config.replay_buffer 
    world_config = config.world
    cuda_device = config.cuda_device

    learning_starts = train_config.learning_starts
    gradient_steps = train_config.gradient_steps
    train_freq = train_config.train_freq
    batch_size = train_config.batch_size
    log_frequency_ep = train_config.log_frequency
    save_interval_ep = train_config.save_interval

    # --- Experiment Directory Setup ---
    timestamp = int(time.time())
    experiment_folder_name = f"{original_config_name}_{config.algorithm}_{timestamp}"
    base_experiments_dir = "experiments"
    current_experiment_path = os.path.join(base_experiments_dir, experiment_folder_name)
    
    actual_tb_log_dir = os.path.join(current_experiment_path, "tensorboard")
    actual_models_save_dir = os.path.join(current_experiment_path, "models")
    
    os.makedirs(current_experiment_path, exist_ok=True)
    os.makedirs(actual_tb_log_dir, exist_ok=True)
    os.makedirs(actual_models_save_dir, exist_ok=True)
    print(f"SAC Experiment data will be saved in: {os.path.abspath(current_experiment_path)}")


    # --- Config Saving ---
    config_save_path = os.path.join(current_experiment_path, "config.json")
    try:
        with open(config_save_path, "w") as f:
            f.write(config.model_dump_json(indent=2))
        print(f"Configuration saved to: {config_save_path}")
    except Exception as e: print(f"Error saving configuration: {e}")

    writer = SummaryWriter(log_dir=actual_tb_log_dir)
    # print(f"TensorBoard logs: {actual_tb_log_dir}") # Covered by experiment path print

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
        device = torch.device("cpu") 
        print("SAC Training using CPU.")

    # --- Agent and Memory Initialization ---
    agent = SAC(config=sac_config, world_config=world_config, buffer_config=buffer_config, device=device)
    memory = agent.memory
    # os.makedirs(train_config.models_dir, exist_ok=True) # Superseded by actual_models_save_dir

    # --- Checkpoint Loading ---
    model_prefix = "sac_" # SAC model filenames are simpler
    latest_model_path = None
    if os.path.exists(actual_models_save_dir):
        model_files = [f for f in os.listdir(actual_models_save_dir) if f.startswith(model_prefix) and f.endswith(".pt")]
        if model_files:
            try: latest_model_path = max([os.path.join(actual_models_save_dir, f) for f in model_files], key=os.path.getmtime)
            except Exception as e: print(f"Could not find latest SAC model in {actual_models_save_dir}: {e}")

    total_steps = 0 # Environment steps, will be re-estimated if loading
    start_episode = 1
    if latest_model_path and os.path.exists(latest_model_path):
        print(f"\nResuming SAC training from: {latest_model_path}")
        agent.load_model(latest_model_path) 
        total_steps = agent.total_updates * train_freq 
        try: 
            parts = os.path.basename(latest_model_path).split('_')
            ep_part = next((p for p in parts if p.startswith('ep')), None)
            if ep_part: start_episode = int(ep_part.replace('ep', '')) + 1
            print(f"Resuming at approx step {total_steps}, episode {start_episode}")
        except Exception as e: print(f"Warn: Could not parse ep from file: {e}. Starting fresh ep count."); start_episode = 1
    else:
        print(f"\nStarting SAC training from scratch in {actual_models_save_dir}.")

    # Training Loop
    episode_rewards = []
    all_losses = {'critic_loss': [], 'actor_loss': [], 'alpha': [], 'alpha_loss': [], 'beta': []} 
    timing_metrics = { 'env_step_time': deque(maxlen=100), 'parameter_update_time': deque(maxlen=100) }
    world = World(world_config=world_config)
    
    _world_reset_for_keys = world.reset() # Get keys for reward_component_accumulator
    reward_component_accumulator = {k: [] for k in world.reward_components}
    del _world_reset_for_keys


    pbar = tqdm(range(start_episode, train_config.num_episodes + 1),
                desc="Training SAC Mapping", unit="episode", initial=start_episode-1, total=train_config.num_episodes)

    total_env_steps = total_steps # Initialize total_env_steps from loaded checkpoint if any

    for episode in pbar:
        state = world.reset() 
        episode_reward = 0
        episode_steps = 0
        actor_hidden_state = agent.actor.get_initial_hidden_state(batch_size=1, device=device) if agent.use_rnn else None
        episode_losses_temp = {'critic_loss': [], 'actor_loss': [], 'alpha': [], 'alpha_loss': [], 'beta': []} 
        updates_made_this_episode = 0
        episode_metrics = []
        episode_reward_components = {k: 0.0 for k in reward_component_accumulator}

        for step_in_episode in range(train_config.max_steps):
            action_normalized, next_actor_hidden_state = agent.select_action(
                state, actor_hidden_state=actor_hidden_state, evaluate=False 
            )

            step_start_time = time.time()
            next_state = world.step(action_normalized, training=True, terminal_step=(step_in_episode == train_config.max_steps - 1))
            timing_metrics['env_step_time'].append(time.time() - step_start_time)

            reward = world.reward
            done = world.done
            current_metric = world.performance_metric

            for key, value in world.reward_components.items():
                 if key in episode_reward_components: episode_reward_components[key] += value

            agent.memory.push(state, action_normalized, reward, next_state, done)

            state = next_state
            if agent.use_rnn: actor_hidden_state = next_actor_hidden_state
            episode_reward += reward
            episode_steps += 1
            total_env_steps += 1 
            episode_metrics.append(current_metric) 

            if total_env_steps >= learning_starts and total_env_steps % train_freq == 0:
                for _ in range(gradient_steps):
                    update_start_time = time.time()
                    losses = agent.update_parameters(batch_size)
                    update_time = time.time() - update_start_time
                    if losses:
                        timing_metrics['parameter_update_time'].append(update_time)
                        if not any(np.isnan(v) for v in losses.values() if isinstance(v, (float, np.float64))):
                            for key, val in losses.items():
                                 if isinstance(val, (float, np.float64)): # Check val type
                                    episode_losses_temp[key].append(val)
                            updates_made_this_episode += 1
                        else: pass # print(f"INFO: Skipping SAC loss logging step {total_env_steps} due to NaN.")
                    else: break 
            if done: break

        # --- Logging (End of Episode) ---
        episode_rewards.append(episode_reward)
        avg_losses = {k: np.mean(v) if v else float('nan') for k, v in episode_losses_temp.items()}
        avg_metric = np.mean(episode_metrics) if episode_metrics else 0.0

        for key, value in episode_reward_components.items():
            if key in reward_component_accumulator: reward_component_accumulator[key].append(value)

        if updates_made_this_episode > 0:
             if not np.isnan(avg_losses['critic_loss']): all_losses['critic_loss'].append(avg_losses['critic_loss'])
             if not np.isnan(avg_losses['actor_loss']): all_losses['actor_loss'].append(avg_losses['actor_loss'])
             if not np.isnan(avg_losses['alpha']): all_losses['alpha'].append(avg_losses['alpha'])
             if agent.auto_tune_alpha and not np.isnan(avg_losses['alpha_loss']): all_losses['alpha_loss'].append(avg_losses['alpha_loss'])
             if agent.use_per and not np.isnan(avg_losses['beta']): all_losses['beta'].append(avg_losses['beta']) 

        if episode % log_frequency_ep == 0:
            log_step = agent.total_updates 
            if timing_metrics['env_step_time']: writer.add_scalar('Time/Environment_Step_ms_Avg100', np.mean(timing_metrics['env_step_time']) * 1000, log_step)
            if timing_metrics['parameter_update_time']: writer.add_scalar('Time/Parameter_Update_ms_Avg100', np.mean(timing_metrics['parameter_update_time']) * 1000, log_step)

            writer.add_scalar('Reward/Episode', episode_reward, log_step)
            writer.add_scalar('Steps/Episode', episode_steps, log_step)
            writer.add_scalar('Progress/Total_Env_Steps', total_env_steps, episode) 
            writer.add_scalar('Progress/Total_Updates', agent.total_updates, log_step) 
            writer.add_scalar('Progress/Buffer_Size', len(agent.memory), log_step)
            writer.add_scalar('Performance/Metric_AvgEp', avg_metric, log_step)
            writer.add_scalar('Performance/Metric_EndEp', world.performance_metric, log_step)
            if world.current_seed is not None: writer.add_scalar('Environment/Current_Seed', world.current_seed, episode)

            if updates_made_this_episode > 0:
                if not np.isnan(avg_losses['critic_loss']): writer.add_scalar('Loss/Critic_AvgEp', avg_losses['critic_loss'], log_step)
                if not np.isnan(avg_losses['actor_loss']): writer.add_scalar('Loss/Actor_AvgEp', avg_losses['actor_loss'], log_step)
                if agent.auto_tune_alpha and not np.isnan(avg_losses['alpha_loss']): writer.add_scalar('Loss/Alpha_AvgEp', avg_losses['alpha_loss'], log_step)
                if not np.isnan(avg_losses['alpha']): writer.add_scalar('Params/Alpha', avg_losses['alpha'], log_step)
                if agent.use_per and not np.isnan(avg_losses['beta']): writer.add_scalar('Params/PER_Beta', avg_losses['beta'], log_step) 
            else:
                 writer.add_scalar('Params/Alpha', agent.alpha, log_step)
                 if agent.use_per: writer.add_scalar('Params/PER_Beta', agent.memory.beta, log_step)

            if reward_component_accumulator:
                avg_components_logged_this_interval = {}
                for name, component_list in reward_component_accumulator.items():
                    if component_list:
                        valid_values = [v for v in component_list if not np.isnan(v)]
                        if valid_values:
                            avg_component_value = np.mean(valid_values)
                            writer.add_scalar(f'RewardComponents_AvgEp/{name}', avg_component_value, log_step)
                reward_component_accumulator = {k: [] for k in reward_component_accumulator}


            if agent.use_state_normalization and agent.state_normalizer and agent.state_normalizer.count > agent.state_normalizer.epsilon:
                writer.add_scalar('Stats/Normalizer_Count', agent.state_normalizer.count.item(), log_step)
                writer.add_scalar('Stats/Normalizer_Mean_Sensor0', agent.state_normalizer.mean[0].item(), log_step)
                writer.add_scalar('Stats/Normalizer_Std_Sensor0', torch.sqrt(agent.state_normalizer.var[0].clamp(min=1e-8)).item(), log_step)
                if agent.state_dim > 5: 
                     writer.add_scalar('Stats/Normalizer_Mean_AgentXNorm', agent.state_normalizer.mean[5].item(), log_step)
                     writer.add_scalar('Stats/Normalizer_Std_AgentXNorm', torch.sqrt(agent.state_normalizer.var[5].clamp(min=1e-8)).item(), log_step)
                if agent.state_dim > 6: 
                     writer.add_scalar('Stats/Normalizer_Mean_AgentYNorm', agent.state_normalizer.mean[6].item(), log_step)
                     writer.add_scalar('Stats/Normalizer_Std_AgentYNorm', torch.sqrt(agent.state_normalizer.var[6].clamp(min=1e-8)).item(), log_step)
                if agent.state_dim > 7: 
                     writer.add_scalar('Stats/Normalizer_Mean_AgentHeadingNorm', agent.state_normalizer.mean[7].item(), log_step)
                     writer.add_scalar('Stats/Normalizer_Std_AgentHeadingNorm', torch.sqrt(agent.state_normalizer.var[7].clamp(min=1e-8)).item(), log_step)


            lookback = min(100, len(episode_rewards))
            avg_reward_100 = np.mean(episode_rewards[-lookback:]) if episode_rewards else 0
            writer.add_scalar('Reward/Average_100', avg_reward_100, log_step)

        if episode % 10 == 0: 
            lookback = min(10, len(episode_rewards))
            avg_reward_10 = np.mean(episode_rewards[-lookback:]) if episode_rewards else 0
            pbar_postfix = {
                'avg_rew_10': f'{avg_reward_10:.2f}',
                'steps': total_env_steps,
                'updates': agent.total_updates,
                'Metric': f"{world.performance_metric:.3f}",
                'alpha': f"{agent.alpha:.3f}"
            }
            if agent.use_per: pbar_postfix['beta'] = f"{agent.memory.beta:.3f}"
            if updates_made_this_episode and not np.isnan(avg_losses['critic_loss']):
                 pbar_postfix['crit_loss'] = f"{avg_losses['critic_loss']:.2f}"
            pbar.set_postfix(pbar_postfix)


        if episode % save_interval_ep == 0 and episode > 0:
            save_path = os.path.join(actual_models_save_dir, f"sac_ep{episode}_updates{agent.total_updates}.pt")
            agent.save_model(save_path)

        if train_config.enable_early_stopping and len(episode_rewards) >= train_config.early_stopping_window:
            avg_reward_window = np.mean(episode_rewards[-train_config.early_stopping_window:])
            if avg_reward_window >= train_config.early_stopping_threshold:
                print(f"\nEarly stopping triggered at episode {episode}!")
                print(f"Average reward over last {train_config.early_stopping_window} episodes ({avg_reward_window:.2f}) >= threshold ({train_config.early_stopping_threshold:.2f}).")
                final_save_path_early = os.path.join(actual_models_save_dir, f"sac_earlystop_ep{episode}_updates{agent.total_updates}.pt")
                agent.save_model(final_save_path_early)
                print(f"Final model saved due to early stopping: {final_save_path_early}")
                pbar.close()
                writer.close()
                return agent, episode_rewards, current_experiment_path 


    pbar.close()
    writer.close()
    final_model_saved_path = None
    if episode < train_config.num_episodes: 
        print(f"SAC Training finished early at episode {episode}. Total env steps: {total_env_steps}, Total updates: {agent.total_updates}")
        if 'final_save_path_early' in locals(): final_model_saved_path = final_save_path_early
    else: 
        print(f"SAC Training finished. Total env steps: {total_env_steps}, Total updates: {agent.total_updates}")
        final_save_path = os.path.join(actual_models_save_dir, f"sac_final_ep{train_config.num_episodes}_updates{agent.total_updates}.pt")
        agent.save_model(final_save_path)
        final_model_saved_path = final_save_path
    
    if final_model_saved_path: print(f"Final model saved to: {final_model_saved_path}")
    else: print("No final model saved (likely due to early exit without meeting early stop criteria).")


    return agent, episode_rewards, current_experiment_path


def evaluate_sac(agent: SAC, config: DefaultConfig):
    eval_config = config.evaluation
    world_config = config.world
    vis_config = config.visualization
    use_stochastic_policy = eval_config.use_stochastic_policy_eval

    vis_available = False
    
    algo_name = "sac" 

    if eval_config.render:
        try:
            from visualization import visualize_world, reset_trajectories, save_gif
            vis_available = True
            print("Visualization enabled.")

            if vis_config.output_format == 'mp4':
                try:
                    animation.FFMpegWriterName = "ffmpeg"
                    if not animation.writers.is_available(animation.FFMpegWriterName):
                        print(f"FFMpeg writer not available. Install ffmpeg and add to PATH. Falling back to GIF.")
                        current_vis_format = 'gif' 
                    else:
                        print("Using FFMpeg for MP4 output.")
                        current_vis_format = 'mp4'
                except Exception as e:
                    print(f"Error checking/setting FFMpeg writer: {e}. Falling back to GIF.")
                    current_vis_format = 'gif'
            else:
                current_vis_format = 'gif'

        except ImportError:
            print("Visualization libraries (matplotlib/imageio/PIL) not found. Rendering disabled.")
            vis_available = False
            current_vis_format = 'none'
    else:
        print("Rendering disabled by config.")
        vis_available = False
        current_vis_format = 'none'


    eval_rewards = []
    eval_metrics = []
    success_count = 0
    all_episode_media_paths = [] 

    agent.actor.eval()
    agent.critic.eval()
    if agent.use_state_normalization and agent.state_normalizer:
        agent.state_normalizer.eval()

    policy_mode = "Stochastic" if use_stochastic_policy else "Deterministic"
    print(f"\nRunning {algo_name.upper()} Evaluation ({policy_mode} Policy) for {eval_config.num_episodes} episodes...")
    
    eval_seeds = world_config.seeds
    num_eval_episodes = eval_config.num_episodes
    if len(eval_seeds) < num_eval_episodes:
         print(f"Warning: Number of evaluation seeds ({len(eval_seeds)}) < num_episodes ({num_eval_episodes}). Generating if needed.")
         eval_seeds.extend([random.randint(0, 2**32 - 1) for _ in range(num_eval_episodes - len(eval_seeds))])
    eval_seeds_to_use = eval_seeds[:num_eval_episodes]


    world = World(world_config=world_config)

    for episode in range(num_eval_episodes):
        seed_to_use = eval_seeds_to_use[episode] if eval_seeds_to_use else None
        state = world.reset(seed=seed_to_use)
        episode_reward = 0
        
        actor_hidden_state = agent.actor.get_initial_hidden_state(batch_size=1, device=agent.device) if agent.use_rnn else None
        
        fig, ax = None, None
        writer_mp4 = None
        episode_png_frames = [] 

        if eval_config.render and vis_available:
            # Note: Evaluation media is saved based on vis_config.save_dir
            os.makedirs(vis_config.save_dir, exist_ok=True)
            if reset_trajectories: reset_trajectories()
            
            fig, ax = plt.subplots(figsize=vis_config.figure_size)
            
            this_episode_format = current_vis_format

            if this_episode_format == 'mp4':
                FFMpegWriter_cls = animation.writers['ffmpeg']
                metadata = dict(title=f'{algo_name.upper()} Eval Ep {episode+1} Seed {seed_to_use}', artist='Matplotlib')
                _writer_instance = FFMpegWriter_cls(fps=vis_config.video_fps, metadata=metadata)
                
                media_filename = f"{algo_name}_eval_{policy_mode.lower()}_ep{episode+1}_seed{seed_to_use}.mp4"
                media_path = os.path.join(vis_config.save_dir, media_filename)
                try:
                    _writer_instance.setup(fig, media_path, dpi=100) 
                    writer_mp4 = _writer_instance 
                except Exception as e:
                    print(f"Error setting up FFMpegWriter for {media_path}: {e}. Falling back to GIF for this episode.")
                    this_episode_format = 'gif' 
                    # plt.close(fig) # Close the problematic figure - No, keep it for GIF
                    # fig, ax = plt.subplots(figsize=vis_config.figure_size) # Recreate for GIF
            
            try:
                visualize_world(world, vis_config, fig, ax) 
                if writer_mp4 and this_episode_format == 'mp4': 
                    writer_mp4.grab_frame()
                elif this_episode_format == 'gif': 
                    png_filename = f"{algo_name}_eval_{policy_mode.lower()}_ep{episode+1}_frame_000_initial.png"
                    png_path = os.path.join(vis_config.save_dir, png_filename)
                    fig.savefig(png_path)
                    if os.path.exists(png_path): episode_png_frames.append(png_path)
            except Exception as e: 
                print(f"Warn: Vis failed for initial state ep {episode+1}. E: {e}")


        for step in range(eval_config.max_steps):
            action_normalized, next_actor_hidden_state = agent.select_action(
                state, actor_hidden_state=actor_hidden_state, evaluate=(not use_stochastic_policy)
            )
            next_state = world.step(action_normalized, training=False, terminal_step=(step == eval_config.max_steps - 1))
            reward = world.reward
            done = world.done
            
            if eval_config.render and vis_available and fig is not None:
                this_episode_format_render_step = current_vis_format
                if writer_mp4 is None and this_episode_format_render_step == 'mp4':
                     this_episode_format_render_step = 'gif'
                try:
                    visualize_world(world, vis_config, fig, ax) 
                    if writer_mp4 and this_episode_format_render_step == 'mp4':
                        writer_mp4.grab_frame()
                    elif this_episode_format_render_step == 'gif':
                        png_filename = f"{algo_name}_eval_{policy_mode.lower()}_ep{episode+1}_frame_{step+1:03d}.png"
                        png_path = os.path.join(vis_config.save_dir, png_filename)
                        fig.savefig(png_path)
                        if os.path.exists(png_path): episode_png_frames.append(png_path)
                except Exception as e: 
                    print(f"Warn: Vis failed step {step+1} ep {episode+1}. E: {e}")

            state = next_state
            if agent.use_rnn: actor_hidden_state = next_actor_hidden_state
            episode_reward += reward
            if done: break

        # --- Episode End ---
        if eval_config.render and vis_available:
            this_episode_format_finish = current_vis_format
            if writer_mp4 is None and this_episode_format_finish == 'mp4':
                 this_episode_format_finish = 'gif'

            if writer_mp4 and this_episode_format_finish == 'mp4':
                try:
                    writer_mp4.finish()
                    print(f"  MP4 video saved: {media_path}")
                    all_episode_media_paths.append(media_path)
                except Exception as e:
                    print(f"  Error finishing MP4 for ep {episode+1}: {e}")
            elif this_episode_format_finish == 'gif' and episode_png_frames:
                gif_filename = f"{algo_name}_eval_{policy_mode.lower()}_ep{episode+1}_seed{seed_to_use}.gif"
                print(f"  Saving GIF for {algo_name.upper()} episode {episode+1} ({policy_mode}) with {len(episode_png_frames)} frames...")
                gif_path = save_gif(output_filename=gif_filename, vis_config=vis_config, frame_paths=episode_png_frames)
                if gif_path: all_episode_media_paths.append(gif_path)
            
            if fig is not None:
                plt.close(fig) 

        final_metric = world.performance_metric
        eval_rewards.append(episode_reward)
        eval_metrics.append(final_metric)

        success = final_metric >= world_config.success_metric_threshold
        if success: success_count += 1
        status = "Success!" if success else "Failure."
        print(f"  Ep {episode+1}/{eval_config.num_episodes} (Seed:{world.current_seed}): Steps={world.current_step}, Terminated={world.done}, Final Metric: {final_metric:.3f}, Accum Reward: {episode_reward:.2f}. {status}")

    agent.actor.train()
    agent.critic.train()
    if agent.use_state_normalization and agent.state_normalizer:
        agent.state_normalizer.train()

    avg_eval_reward = np.mean(eval_rewards) if eval_rewards else 0.0
    std_eval_reward = np.std(eval_rewards) if eval_rewards else 0.0
    avg_eval_metric = np.mean(eval_metrics) if eval_metrics else 0.0
    std_eval_metric = np.std(eval_metrics) if eval_metrics else 0.0
    success_rate = success_count / eval_config.num_episodes if eval_config.num_episodes > 0 else 0.0

    print(f"\n--- {algo_name.upper()} Evaluation Summary ({policy_mode} Policy) ---")
    print(f"Episodes: {eval_config.num_episodes}")
    print(f"Average Final Metric (Point Inclusion): {avg_eval_metric:.3f} +/- {std_eval_metric:.3f}")
    print(f"Success Rate (Metric >= {world_config.success_metric_threshold:.2f}): {success_rate:.2%} ({success_count}/{eval_config.num_episodes})")
    print(f"Average Accumulated Episode Reward: {avg_eval_reward:.3f} +/- {std_eval_reward:.3f}")
    if eval_config.render and vis_available and all_episode_media_paths: print(f"Media saved to: '{os.path.abspath(vis_config.save_dir)}'")
    elif eval_config.render and not vis_available: print("Rendering was enabled but visualization libraries were not found.")
    elif not eval_config.render: print("Rendering disabled.")
    print(f"--- End {algo_name.upper()} Evaluation ({policy_mode} Policy) ---\n")

    return eval_rewards, success_rate, avg_eval_metric