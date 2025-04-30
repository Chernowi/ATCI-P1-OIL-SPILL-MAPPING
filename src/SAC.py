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
    """Experience replay buffer storing full state trajectories. (Unchanged from original)"""
    def __init__(self, config: ReplayBufferConfig, world_config: WorldConfig):
        self.buffer = deque(maxlen=config.capacity)
        self.config = config
        self.trajectory_length = world_config.trajectory_length
        self.feature_dim = world_config.trajectory_feature_dim # Includes state+action+reward

    def push(self, state, action, reward, next_state, done):
        """ Add a new experience to memory. """
        trajectory = state['full_trajectory'] # Shape (N, feat_dim)
        next_trajectory = next_state['full_trajectory'] # Shape (N, feat_dim)

        if isinstance(action, (np.ndarray, list)): action = action[0]
        if isinstance(reward, (np.ndarray)): reward = reward.item()

        # Basic validation
        if not isinstance(trajectory, np.ndarray) or trajectory.shape != (self.trajectory_length, self.feature_dim):
            return
        if not isinstance(next_trajectory, np.ndarray) or next_trajectory.shape != (self.trajectory_length, self.feature_dim):
            return
        if np.isnan(trajectory).any() or np.isnan(next_trajectory).any():
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

        # state/next_state are full trajectories (batch, N, feat_dim)
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
        # *** Basic state dimension is now CORE_STATE_DIM (e.g., 7) ***
        self.state_dim = config.state_dim # Should match CORE_STATE_DIM
        self.action_dim = config.action_dim
        self.trajectory_length = world_config.trajectory_length

        if self.use_rnn:
            self.rnn_hidden_size = config.rnn_hidden_size
            self.rnn_num_layers = config.rnn_num_layers
            rnn_input_dim = self.state_dim # RNN processes normalized basic states
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
            # MLP uses only the normalized last basic state
            mlp_input_dim = self.state_dim
            self.rnn = None

        self.layers = nn.ModuleList()
        current_dim = mlp_input_dim
        for hidden_dim in config.hidden_dims:
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
        # *** Basic state dimension is now CORE_STATE_DIM (e.g., 7) ***
        self.state_dim = config.state_dim # Should match CORE_STATE_DIM
        self.action_dim = config.action_dim
        self.trajectory_length = world_config.trajectory_length

        if self.use_rnn:
            self.rnn_hidden_size = config.rnn_hidden_size
            self.rnn_num_layers = config.rnn_num_layers
            rnn_input_dim = self.state_dim # RNN processes normalized basic states
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

        # Q1 architecture
        self.q1_layers = nn.ModuleList()
        q1_mlp_input = mlp_input_dim
        for hidden_dim in config.hidden_dims:
            self.q1_layers.append(nn.Linear(q1_mlp_input, hidden_dim))
            self.q1_layers.append(nn.ReLU())
            q1_mlp_input = hidden_dim
        self.q1_out = nn.Linear(q1_mlp_input, 1)

        # Q2 architecture
        self.q2_layers = nn.ModuleList()
        q2_mlp_input = mlp_input_dim
        for hidden_dim in config.hidden_dims:
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
        # *** Basic state dimension ***
        self.state_dim = config.state_dim # Should match CORE_STATE_DIM (e.g., 7)
        self.action_dim = config.action_dim

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"SAC Agent using device: {self.device}")
        if self.use_rnn: print(f"SAC Agent using RNN: Type={config.rnn_type}, SeqLen={self.trajectory_length}")
        else: print(f"SAC Agent using MLP (Processing last state of trajectory)")

        # --- Normalization ---
        # *** Initialize with the correct basic state dimension ***
        self.state_normalizer = RunningMeanStd(shape=(self.state_dim,), device=self.device)
        print(f"SAC Agent state normalization enabled for dim: {self.state_dim}")

        self.actor = Actor(config, world_config).to(self.device)
        self.critic = Critic(config, world_config).to(self.device)
        self.critic_target = Critic(config, world_config).to(self.device)

        self.critic_target.load_state_dict(self.critic.state_dict())
        for target_param in self.critic_target.parameters():
            target_param.requires_grad = False

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.lr)

        if self.auto_tune_alpha:
            self.target_entropy = -torch.prod(torch.Tensor([self.action_dim]).to(self.device)).item()
            self.log_alpha = torch.tensor(np.log(self.alpha), requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=config.lr)
        else:
            self.log_alpha = torch.tensor(np.log(self.alpha)).to(self.device)

    def select_action(self, state: dict, actor_hidden_state: Optional[Tuple] = None, evaluate: bool = False) -> Tuple[float, Optional[Tuple]]:
        """Select action (normalized yaw change [-1, 1]) based on normalized state."""
        state_trajectory = state['full_trajectory'] # Get the full trajectory array (N, feat_dim)
        state_tensor_full = torch.FloatTensor(state_trajectory).to(self.device).unsqueeze(0) # (1, N, feat_dim)

        with torch.no_grad():
            # --- Normalize Input State(s) ---
            if self.use_rnn:
                 # Extract basic state sequence: (1, N, state_dim)
                 raw_basic_state_seq = state_tensor_full[:, :, :self.state_dim]
                 # Normalize using current stats (do not update stats here)
                 normalized_basic_state_seq = self.state_normalizer.normalize(raw_basic_state_seq)
                 actor_input = normalized_basic_state_seq
            else:
                 # Extract last basic state: (1, state_dim)
                 raw_last_basic_state = state_tensor_full[:, -1, :self.state_dim]
                 # Normalize using current stats (do not update stats here)
                 normalized_last_basic_state = self.state_normalizer.normalize(raw_last_basic_state)
                 actor_input = normalized_last_basic_state
            # --- End Normalize ---

            # Set actor to eval mode for selection
            self.actor.eval()
            if evaluate:
                # Pass normalized state(s) to actor
                _, _, action_mean_squashed, next_actor_hidden_state = self.actor.sample(actor_input, actor_hidden_state)
                action_normalized = action_mean_squashed
            else:
                # Pass normalized state(s) to actor
                action_normalized, _, _, next_actor_hidden_state = self.actor.sample(actor_input, actor_hidden_state)
            # Set actor back to train mode
            self.actor.train()

        action_normalized_float = action_normalized.detach().cpu().numpy()[0, 0]
        return action_normalized_float, next_actor_hidden_state

    def update_parameters(self, memory: ReplayBuffer, batch_size: int):
        """Perform a single SAC update step using a batch of trajectories with normalization."""
        sampled_batch = memory.sample(batch_size)
        if sampled_batch is None: return None

        # Shapes: state/next_state (b, N, feat_dim), action (b, 1), reward/done (b,)
        state_batch, action_batch_normalized, reward_batch, next_state_batch, done_batch = sampled_batch

        state_batch_tensor = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch_tensor = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch_tensor = torch.FloatTensor(action_batch_normalized).to(self.device)
        reward_batch_tensor = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        done_batch_tensor = torch.FloatTensor(done_batch).to(self.device).unsqueeze(1)

        # --- Update Running State Statistics ---
        # Use all basic states from the current trajectories in the batch
        # Extract basic state part: (batch, N, state_dim)
        raw_basic_states_batch = state_batch_tensor[:, :, :self.state_dim]
        # Update normalizer (expects flat batch)
        self.state_normalizer.update(raw_basic_states_batch.reshape(-1, self.state_dim))
        # --- End Update ---

        # --- Prepare Normalized Network Inputs ---
        # Normalize using the *updated* stats
        if self.use_rnn:
            # Normalize the sequence of basic states (already extracted)
            batch, seq_len, _ = raw_basic_states_batch.shape
            normalized_state_seq = self.state_normalizer.normalize(
                raw_basic_states_batch.reshape(-1, self.state_dim) # Flatten
            ).reshape(batch, seq_len, self.state_dim) # Reshape back

            # Normalize next state sequence
            raw_next_basic_state_seq = next_state_batch_tensor[:, :, :self.state_dim]
            batch_next, seq_len_next, _ = raw_next_basic_state_seq.shape
            normalized_next_state_seq = self.state_normalizer.normalize(
                 raw_next_basic_state_seq.reshape(-1, self.state_dim)
            ).reshape(batch_next, seq_len_next, self.state_dim)

            # Network inputs are the normalized sequences
            current_network_input = normalized_state_seq
            next_network_input = normalized_next_state_seq
        else:
            # Normalize only the last basic state
            raw_last_basic_state = raw_basic_states_batch[:, -1, :] # (batch, state_dim)
            normalized_last_basic_state = self.state_normalizer.normalize(raw_last_basic_state)

            # Normalize last basic state of next trajectory
            raw_next_last_basic_state = next_state_batch_tensor[:, -1, :self.state_dim] # (batch, state_dim)
            normalized_next_last_basic_state = self.state_normalizer.normalize(raw_next_last_basic_state)

            # Network inputs are the normalized last states
            current_network_input = normalized_last_basic_state
            next_network_input = normalized_next_last_basic_state
        # --- End Prepare ---

        # Get initial hidden states for RNNs if used (no change needed)
        initial_actor_hidden = self.actor.get_initial_hidden_state(batch_size, self.device) if self.use_rnn else None
        initial_critic_hidden = self.critic.get_initial_hidden_state(batch_size, self.device) if self.use_rnn else None
        initial_critic_target_hidden = self.critic_target.get_initial_hidden_state(batch_size, self.device) if self.use_rnn else None

        # --- Critic Update ---
        with torch.no_grad():
            # Pass normalized next state(s) to actor
            next_action, next_log_prob, _, _ = self.actor.sample(next_network_input, initial_actor_hidden)

            # Pass normalized next state(s) to target critic
            target_q1, target_q2, _ = self.critic_target(next_network_input, next_action, initial_critic_target_hidden)
            target_q_min = torch.min(target_q1, target_q2)

            current_alpha = self.log_alpha.exp().item()
            target_q_entropy = target_q_min - current_alpha * next_log_prob
            y = reward_batch_tensor + (1.0 - done_batch_tensor) * self.gamma * target_q_entropy

        # --- Calculate Current Q values ---
        # Pass normalized current state(s) and actual action taken to critic
        current_q1, current_q2, _ = self.critic(current_network_input, action_batch_tensor, initial_critic_hidden)

        critic_loss = F.mse_loss(current_q1, y) + F.mse_loss(current_q2, y)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # Optional: Clip critic gradients
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # --- Actor Update ---
        # Freeze critic parameters
        for param in self.critic.parameters(): param.requires_grad = False

        # Pass normalized current state(s) to actor
        action_pi, log_prob_pi, _, _ = self.actor.sample(current_network_input, initial_actor_hidden)
        # Pass normalized current state(s) and policy action to critic
        q1_pi, q2_pi, _ = self.critic(current_network_input, action_pi, initial_critic_hidden)
        q_pi_min = torch.min(q1_pi, q2_pi)

        current_alpha = self.log_alpha.exp().item() # Use current alpha
        actor_loss = (current_alpha * log_prob_pi - q_pi_min).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        # Optional: Clip actor gradients
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        # Unfreeze critic parameters
        for param in self.critic.parameters(): param.requires_grad = True

        # --- Alpha Update ---
        alpha_loss_item = None
        if self.auto_tune_alpha:
            # Use detached log_prob_pi
            alpha_loss = -(self.log_alpha * (log_prob_pi.detach() + self.target_entropy)).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item() # Update Python variable
            alpha_loss_item = alpha_loss.item()

        # --- Target Network Update (Soft Update) ---
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
            'state_normalizer_state_dict': self.state_normalizer.state_dict(), # Save normalizer state
            'device_type': self.device.type
        }
        if self.auto_tune_alpha:
            save_dict['log_alpha'] = self.log_alpha # Save tensor directly
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

        # --- Load Normalizer State ---
        if 'state_normalizer_state_dict' in checkpoint:
            self.state_normalizer.load_state_dict(checkpoint['state_normalizer_state_dict'])
            print("Loaded state normalizer statistics.")
        else:
            print("Warning: State normalizer statistics not found in checkpoint. Using initial values.")
        # --- End Load Normalizer ---

        if self.auto_tune_alpha and 'log_alpha' in checkpoint:
            # Ensure log_alpha exists and is a tensor before loading
            if not hasattr(self, 'log_alpha') or not isinstance(self.log_alpha, torch.Tensor):
                 self.log_alpha = torch.tensor(0.0, requires_grad=True, device=self.device) # Initialize if missing
            self.log_alpha.data.copy_(checkpoint['log_alpha'].to(self.device))
            if not self.log_alpha.requires_grad: self.log_alpha.requires_grad_(True)

            # Re-initialize optimizer after loading log_alpha
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.config.lr)
            try:
                if 'alpha_optimizer_state_dict' in checkpoint:
                     self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
            except ValueError as e:
                 print(f"Warning: Could not load SAC alpha optimizer state: {e}. Reinitializing.")
                 self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.config.lr) # Recreate if needed
            self.alpha = self.log_alpha.exp().item()
        elif not self.auto_tune_alpha:
             if 'log_alpha' in checkpoint:
                  try:
                       self.log_alpha = checkpoint['log_alpha'].to(self.device)
                       self.alpha = self.log_alpha.exp().item()
                  except Exception as e: print(f"Warning: Could not load fixed log_alpha: {e}.")
             # else: keep default alpha

        # Sync target AFTER loading main critic
        self.critic_target.load_state_dict(self.critic.state_dict())
        for target_param in self.critic_target.parameters():
            target_param.requires_grad = False

        self.actor.train()
        self.critic.train()
        self.critic_target.train() # Set target to train mode, though grads are off
        self.state_normalizer.train() # Ensure normalizer is in train mode after loading

        print(f"SAC model loaded successfully from {path}")


# --- Training Loop (train_sac) ---
# (Largely unchanged, but uses updated config defaults and logs IoU)
def train_sac(config: DefaultConfig, use_multi_gpu: bool = False, run_evaluation: bool = True):
    sac_config = config.sac
    train_config = config.training # Now points to correct dir via DefaultConfig logic
    buffer_config = config.replay_buffer
    world_config = config.world
    cuda_device = config.cuda_device

    learning_starts = train_config.learning_starts
    gradient_steps = train_config.gradient_steps
    train_freq = train_config.train_freq
    batch_size = train_config.batch_size
    log_frequency_ep = train_config.log_frequency
    save_interval_ep = train_config.save_interval

    # Device Setup
    if torch.cuda.is_available():
        if use_multi_gpu and torch.cuda.device_count() > 1:
            print(f"Warning: Multi-GPU not standard for SAC. Using: {cuda_device}")
        device = torch.device(cuda_device)
        print(f"Using device: {device}")
        if 'cuda' in cuda_device:
            try: torch.cuda.set_device(device); print(f"GPU: {torch.cuda.get_device_name(device)}")
            except Exception as e: print(f"Warn: Could not set CUDA device {cuda_device}. E: {e}"); device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu"); print("GPU not available, using CPU.")

    # --- Initialization ---
    name_in_logdir = "sac_mapping_rnn_" if config.sac.use_rnn else "sac_mapping_"
    log_dir = os.path.join("runs", name_in_logdir + str(int(time.time())))
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logs: {log_dir}")

    agent = SAC(config=sac_config, world_config=world_config, device=device)
    memory = ReplayBuffer(config=buffer_config, world_config=world_config)
    os.makedirs(train_config.models_dir, exist_ok=True) # Use potentially updated models_dir

    # --- Load Checkpoint ---
    model_files = [f for f in os.listdir(train_config.models_dir) if f.startswith("sac_") and f.endswith(".pt")]
    latest_model_path = None
    if model_files:
        try: latest_model_path = max([os.path.join(train_config.models_dir, f) for f in model_files], key=os.path.getmtime)
        except Exception as e: print(f"Could not find latest model: {e}")

    total_steps = 0
    start_episode = 1
    if latest_model_path and os.path.exists(latest_model_path):
        print(f"\nResuming SAC training from: {latest_model_path}")
        agent.load_model(latest_model_path)
        # Parse steps/episode from filename (best effort)
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

    # --- Training Loop ---
    episode_rewards = []
    all_losses = {'critic_loss': [], 'actor_loss': [], 'alpha': [], 'alpha_loss': []}
    timing_metrics = { 'env_step_time': deque(maxlen=100), 'parameter_update_time': deque(maxlen=100) }

    pbar = tqdm(range(start_episode, train_config.num_episodes + 1),
                desc="Training SAC Mapping", unit="episode", initial=start_episode-1, total=train_config.num_episodes)

    world = World(world_config=world_config) # Initialize the mapping world

    for episode in pbar:
        state = world.reset() # Get initial state dictionary
        episode_reward = 0
        episode_steps = 0
        actor_hidden_state = agent.actor.get_initial_hidden_state(batch_size=1, device=device) if agent.use_rnn else None
        episode_losses_temp = {'critic_loss': [], 'actor_loss': [], 'alpha': [], 'alpha_loss': []}
        updates_made_this_episode = 0
        episode_ious = []

        for step_in_episode in range(train_config.max_steps):
            # Agent selects action based on potentially normalized state
            action_normalized, next_actor_hidden_state = agent.select_action(
                state, actor_hidden_state=actor_hidden_state, evaluate=False
            )

            step_start_time = time.time()
            # Pass terminal_step flag for correct history update on last step
            next_state = world.step(action_normalized, training=True, terminal_step=(step_in_episode == train_config.max_steps - 1))
            step_time = time.time() - step_start_time
            timing_metrics['env_step_time'].append(step_time)

            reward = world.reward # Reward r_{t+1}
            done = world.done

            # Push (s_t, a_t, r_{t+1}, s_{t+1}, d_{t+1}) - buffer expects s, a, r, next_s, done
            # The 'state' dictionary holds the trajectory ending at t, 'next_state' holds trajectory ending at t+1
            memory.push(state, action_normalized, reward, next_state, done)

            state = next_state
            if agent.use_rnn: actor_hidden_state = next_actor_hidden_state
            episode_reward += reward # Accumulate r_{t+1}
            episode_steps += 1
            total_steps += 1
            episode_ious.append(world.iou) # Log IoU at each step

            # Perform Updates
            if total_steps >= learning_starts and total_steps % train_freq == 0:
                for _ in range(gradient_steps):
                    if len(memory) >= batch_size:
                        update_start_time = time.time()
                        losses = agent.update_parameters(memory, batch_size) # Agent handles normalization internally
                        update_time = time.time() - update_start_time
                        if losses:
                            timing_metrics['parameter_update_time'].append(update_time)
                            # Check for NaN losses before storing
                            if not any(np.isnan(v) for v in losses.values() if isinstance(v, (float, np.float64))):
                                for key, val in losses.items():
                                     if isinstance(val, (float, np.float64)):
                                        episode_losses_temp[key].append(val)
                                updates_made_this_episode += 1
                            else:
                                print(f"INFO: Skipping SAC loss logging step {total_steps} due to NaN.")
                        else:
                             break # Stop gradient steps if update fails
                    else:
                         break # Stop gradient steps if buffer too small

            if done:
                break

        # --- Logging (End of Episode) ---
        episode_rewards.append(episode_reward)
        avg_losses = {k: np.mean(v) if v else float('nan') for k, v in episode_losses_temp.items()}
        avg_iou = np.mean(episode_ious) if episode_ious else 0.0

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
            writer.add_scalar('Performance/IoU_AvgEp', avg_iou, total_steps)
            writer.add_scalar('Performance/IoU_EndEp', world.iou, total_steps) # Log final IoU

            if updates_made_this_episode > 0:
                if not np.isnan(avg_losses['critic_loss']): writer.add_scalar('Loss/Critic_AvgEp', avg_losses['critic_loss'], total_steps)
                if not np.isnan(avg_losses['actor_loss']): writer.add_scalar('Loss/Actor_AvgEp', avg_losses['actor_loss'], total_steps)
                if not np.isnan(avg_losses['alpha']): writer.add_scalar('Alpha/Value', avg_losses['alpha'], total_steps)
                if agent.auto_tune_alpha and not np.isnan(avg_losses['alpha_loss']): writer.add_scalar('Loss/Alpha_AvgEp', avg_losses['alpha_loss'], total_steps)
            else:
                 writer.add_scalar('Alpha/Value', agent.alpha, total_steps) # Log current alpha if no updates

            # Log normalizer stats
            if agent.state_normalizer.count > agent.state_normalizer.epsilon:
                writer.add_scalar('Stats/Normalizer_Count', agent.state_normalizer.count.item(), total_steps)
                writer.add_scalar('Stats/Normalizer_Mean_Sensor0', agent.state_normalizer.mean[0].item(), total_steps)
                writer.add_scalar('Stats/Normalizer_Std_Sensor0', torch.sqrt(agent.state_normalizer.var[0].clamp(min=1e-8)).item(), total_steps)
                writer.add_scalar('Stats/Normalizer_Mean_AgentX', agent.state_normalizer.mean[5].item(), total_steps)
                writer.add_scalar('Stats/Normalizer_Std_AgentX', torch.sqrt(agent.state_normalizer.var[5].clamp(min=1e-8)).item(), total_steps)


            lookback = min(100, len(episode_rewards))
            avg_reward_100 = np.mean(episode_rewards[-lookback:]) if episode_rewards else 0
            writer.add_scalar('Reward/Average_100', avg_reward_100, total_steps)

        if episode % 10 == 0:
            lookback = min(10, len(episode_rewards))
            avg_reward_10 = np.mean(episode_rewards[-lookback:]) if episode_rewards else 0
            pbar_postfix = {'avg_rew_10': f'{avg_reward_10:.2f}', 'steps': total_steps, 'IoU': f"{world.iou:.3f}", 'alpha': f"{agent.alpha:.3f}"}
            if updates_made_this_episode and not np.isnan(avg_losses['critic_loss']): pbar_postfix['crit_loss'] = f"{avg_losses['critic_loss']:.2f}"
            pbar.set_postfix(pbar_postfix)

        if episode % save_interval_ep == 0:
            save_path = os.path.join(train_config.models_dir, f"sac_ep{episode}_step{total_steps}.pt")
            agent.save_model(save_path)

    pbar.close()
    writer.close()
    print(f"SAC Training finished. Total steps: {total_steps}")
    final_save_path = os.path.join(train_config.models_dir, f"sac_final_ep{train_config.num_episodes}_step{total_steps}.pt")
    agent.save_model(final_save_path)

    if run_evaluation:
         print("\nStarting evaluation after training...")
         evaluate_sac(agent=agent, config=config) # Pass full config

    return agent, episode_rewards

# --- Evaluation Loop (evaluate_sac) ---
# (Requires setting normalizer to eval mode and logging IoU)
def evaluate_sac(agent: SAC, config: DefaultConfig):
    eval_config = config.evaluation
    world_config = config.world
    vis_config = config.visualization

    # --- Conditional Visualization Import ---
    vis_available = False
    visualize_world, reset_trajectories, save_gif = None, None, None
    if eval_config.render:
        try:
            from visualization import visualize_world, reset_trajectories, save_gif
            # Check if imageio is available too, as it's needed for GIFs
            import imageio.v2 as imageio
            vis_available = True
            print("Visualization enabled.")
        except ImportError:
            print("Visualization libraries (matplotlib/imageio/PIL) not found. Rendering disabled.")
            eval_config.render = False # Force disable rendering
    else:
        print("Rendering disabled by config.")
    # --- End Conditional Import ---

    eval_rewards = []
    eval_ious = [] # Store final IoU for each episode
    success_count = 0
    all_episode_gif_paths = []

    # --- Set Agent and Normalizer to Evaluation Mode ---
    agent.actor.eval()
    agent.critic.eval()
    agent.state_normalizer.eval()
    # --- End Set Eval Mode ---

    print(f"\nRunning SAC Evaluation for {eval_config.num_episodes} episodes...")
    world = World(world_config=world_config) # Create world instance for evaluation

    for episode in range(eval_config.num_episodes):
        state = world.reset()
        episode_reward = 0
        episode_frames = []
        actor_hidden_state = agent.actor.get_initial_hidden_state(batch_size=1, device=agent.device) if agent.use_rnn else None

        # Visualize initial state if rendering
        if eval_config.render and vis_available:
            os.makedirs(vis_config.save_dir, exist_ok=True)
            if reset_trajectories: reset_trajectories()
            try:
                fname = f"sac_eval_ep{episode+1}_frame_000_initial.png"
                initial_frame_file = visualize_world(world, vis_config=vis_config, filename=fname)
                if initial_frame_file and os.path.exists(initial_frame_file): episode_frames.append(initial_frame_file)
            except Exception as e: print(f"Warn: Vis failed init state ep {episode+1}. E: {e}")

        for step in range(eval_config.max_steps):
            # Agent selects action in evaluation mode (uses normalized state)
            action_normalized, next_actor_hidden_state = agent.select_action(
                state, actor_hidden_state=actor_hidden_state, evaluate=True
            )

            # Step world (training=False prevents reward calculation/termination inside step)
            next_state = world.step(action_normalized, training=False, terminal_step=(step == eval_config.max_steps - 1))
            # Get reward, done status, and IoU from world state *after* the step
            reward = world.reward # Should be 0 if training=False, but let's track anyway if logic changes
            done = world.done
            current_iou = world.iou # Get IoU after the step

            # Visualize current state if rendering
            if eval_config.render and vis_available:
                try:
                    fname = f"sac_eval_ep{episode+1}_frame_{step+1:03d}.png"
                    frame_file = visualize_world(world, vis_config=vis_config, filename=fname)
                    if frame_file and os.path.exists(frame_file): episode_frames.append(frame_file)
                except Exception as e: print(f"Warn: Vis failed step {step+1} ep {episode+1}. E: {e}")

            state = next_state
            if agent.use_rnn: actor_hidden_state = next_actor_hidden_state
            episode_reward += reward # Accumulate reward
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
        else: # Max steps reached or terminated on last step
             print(f"  Episode {episode+1}: Finished (Step {world.current_step}). Final IoU: {final_iou:.3f}. {status}")
        # print(f"  Episode {episode+1}: Total Reward: {episode_reward:.2f}") # Reward is less informative here

        # Save GIF if rendering
        if eval_config.render and vis_available and episode_frames and save_gif:
            gif_filename = f"sac_mapping_eval_episode_{episode+1}.gif"
            try:
                print(f"  Saving GIF for SAC episode {episode+1} with {len(episode_frames)} frames...")
                gif_path = save_gif(output_filename=gif_filename, vis_config=vis_config, frame_paths=episode_frames, delete_frames=vis_config.delete_frames_after_gif)
                if gif_path: all_episode_gif_paths.append(gif_path)
            except Exception as e: print(f"  Warn: Failed GIF save ep {episode+1}. E: {e}")
            # Frame deletion happens inside save_gif if requested


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

    print("\n--- SAC Evaluation Summary ---")
    print(f"Episodes: {eval_config.num_episodes}")
    print(f"Average Final IoU: {avg_eval_iou:.3f} +/- {std_eval_iou:.3f}")
    # print(f"Average Reward: {avg_eval_reward:.2f} +/- {std_eval_reward:.2f}") # Reward less useful
    print(f"Success Rate (IoU >= {world_config.success_iou_threshold:.2f}): {success_rate:.2%} ({success_count}/{eval_config.num_episodes})")
    if eval_config.render and vis_available and all_episode_gif_paths: print(f"GIFs saved to: '{os.path.abspath(vis_config.save_dir)}'")
    elif eval_config.render and not vis_available: print("Rendering was enabled but visualization libraries were not found.")
    elif not eval_config.render: print("Rendering disabled.")
    print("--- End SAC Evaluation ---\n")

    return eval_rewards, success_rate, avg_eval_iou # Return IoU as well
