import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import random
import time
import math
from collections import deque
from tqdm import tqdm

# Local imports
from world import World # Mapping world
# Updated config import
from configs import DefaultConfig, ReplayBufferConfig, TSACConfig, WorldConfig, CORE_STATE_DIM, CORE_ACTION_DIM, TRAJECTORY_REWARD_DIM
from torch.utils.tensorboard import SummaryWriter
from typing import Tuple, Optional, List, Dict, Any, Union
from utils import RunningMeanStd # Normalization utility

class SumTree:
    write = 0
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data_indices = np.zeros(capacity, dtype=int)
        self.max_priority = 1.0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, priority, data_idx):
        priority = self.max_priority
        idx = self.write + self.capacity - 1
        self.data_indices[self.write] = data_idx
        self.update(idx, priority)
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

    def update(self, idx, priority):
        if priority > self.max_priority:
            self.max_priority = priority
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        actual_data_idx = self.data_indices[data_idx]
        return (idx, self.tree[idx], actual_data_idx)

# --- Prioritized Replay Buffer (Adapted for TSACConfig) ---
class PrioritizedReplayBuffer:
    def __init__(self, config: TSACConfig, buffer_config: ReplayBufferConfig, world_config: WorldConfig): # Uses TSACConfig
        self.tree = SumTree(buffer_config.capacity)
        self.capacity = buffer_config.capacity
        self.memory = deque(maxlen=self.capacity)
        self.config = config # Store TSACConfig
        self.buffer_config = buffer_config
        self.trajectory_length = world_config.trajectory_length
        self.feature_dim = world_config.trajectory_feature_dim
        self.alpha = config.per_alpha
        self.beta_start = config.per_beta_start
        self.beta = self.beta_start
        self.beta_frames = config.per_beta_frames
        self.epsilon = config.per_epsilon
        self.frame = 0
        self._next_data_idx = 0
        self._current_size = 0

    def push(self, state, action, reward, next_state, done):
        trajectory = state['full_trajectory']
        next_trajectory = next_state['full_trajectory']
        if isinstance(action, (np.ndarray, list)): action = action[0]
        if isinstance(reward, (np.ndarray)): reward = reward.item()

        if not isinstance(trajectory, np.ndarray) or trajectory.shape != (self.trajectory_length, self.feature_dim): return
        if not isinstance(next_trajectory, np.ndarray) or next_trajectory.shape != (self.trajectory_length, self.feature_dim): return
        if np.isnan(trajectory).any() or np.isnan(next_trajectory).any(): return

        experience = (trajectory, float(action), float(reward), next_trajectory, done)

        if self._current_size < self.capacity:
             self.memory.append(experience)
             data_idx = self._current_size
             self._current_size += 1
        else:
             data_idx = self.tree.write
             self.memory[data_idx] = experience

        self.tree.add(self.tree.max_priority ** self.alpha, data_idx)

    def sample(self, batch_size: int) -> Optional[Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]]:
        if self._current_size < batch_size:
            return None

        batch_data = {'state': [], 'action': [], 'reward': [], 'next_state': [], 'done': []}
        idxs = np.empty((batch_size,), dtype=int)
        priorities = np.empty((batch_size,), dtype=float)
        data_indices = np.empty((batch_size,), dtype=int)

        self.beta = min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)
        self.frame += 1

        segment = self.tree.total() / batch_size
        for i in range(batch_size):
            s = random.uniform(segment * i, segment * (i + 1))
            idx, p, data_idx = self.tree.get(s)
            idxs[i] = idx
            priorities[i] = p
            data_indices[i] = data_idx

        try:
            sampled_experiences = [self.memory[di] for di in data_indices]
            trajectory, action, reward, next_trajectory, done = zip(*sampled_experiences)

            batch_data['state'] = np.array(trajectory, dtype=np.float32)
            batch_data['action'] = np.array(action, dtype=np.float32).reshape(-1, 1)
            batch_data['reward'] = np.array(reward, dtype=np.float32)
            batch_data['next_state'] = np.array(next_trajectory, dtype=np.float32)
            batch_data['done'] = np.array(done, dtype=np.float32)

            sampling_probabilities = priorities / self.tree.total()
            weights = np.power(self._current_size * sampling_probabilities + 1e-10, -self.beta)
            weights /= weights.max()
        except IndexError:
            print(f"IndexError during T-SAC PER sampling. data_indices: {data_indices}, current_size: {self._current_size}, len(memory): {len(self.memory)}")
            return None
        except Exception as e:
            print(f"Error converting T-SAC PER sampled batch: {e}")
            return None

        return batch_data, idxs, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, priority in zip(batch_indices, batch_priorities):
            priority = abs(priority) + self.epsilon
            self.tree.update(idx, priority ** self.alpha)
        # Update max priority
        active_leaf_indices = np.arange(self.capacity) + self.capacity - 1
        self.tree.max_priority = np.max(self.tree.tree[active_leaf_indices[:self._current_size]]) if self._current_size > 0 else 1.0


    def __len__(self):
        return self._current_size


# --- Standard Replay Buffer (Unchanged) ---
class ReplayBuffer:
    """Experience replay buffer storing full state trajectories (with normalized states)."""
    def __init__(self, config: ReplayBufferConfig, world_config: WorldConfig):
        self.buffer = deque(maxlen=config.capacity)
        self.config = config
        self.trajectory_length = world_config.trajectory_length
        self.feature_dim = world_config.trajectory_feature_dim # normalized_state + action + reward

    def push(self, state, action, reward, next_state, done):
        # state/next_state dicts contain 'full_trajectory' with normalized coords
        trajectory = state['full_trajectory'] # (N, feat_dim)
        next_trajectory = next_state['full_trajectory'] # (N, feat_dim)
        if isinstance(action, (np.ndarray, list)): action = action[0]
        if isinstance(reward, (np.ndarray)): reward = reward.item()

        if not isinstance(trajectory, np.ndarray) or trajectory.shape != (self.trajectory_length, self.feature_dim): return
        if not isinstance(next_trajectory, np.ndarray) or next_trajectory.shape != (self.trajectory_length, self.feature_dim): return
        if np.isnan(trajectory).any() or np.isnan(next_trajectory).any(): return

        self.buffer.append((trajectory, float(action), float(reward), next_trajectory, done))

    def sample(self, batch_size: int) -> Optional[Dict[str, np.ndarray]]: # Return type change
        if len(self.buffer) < batch_size: return None
        try: batch = random.sample(self.buffer, batch_size)
        except ValueError: return None

        # Trajectories contain normalized states
        trajectory, action, reward, next_trajectory, done = zip(*batch)
        batch_data = {}
        try:
            batch_data['state'] = np.array(trajectory, dtype=np.float32) # (b, N, feat_dim)
            batch_data['action'] = np.array(action, dtype=np.float32).reshape(-1, 1) # (b, 1)
            batch_data['reward'] = np.array(reward, dtype=np.float32) # (b,)
            batch_data['next_state'] = np.array(next_trajectory, dtype=np.float32) # (b, N, feat_dim)
            batch_data['done'] = np.array(done, dtype=np.float32) # (b,)

            if np.isnan(batch_data['state']).any() or np.isnan(batch_data['next_state']).any(): return None
        except Exception as e: print(f"Error converting TSAC sampled batch: {e}"); return None

        return batch_data # Return dict

    def __len__(self):
        return len(self.buffer)

# --- Network Classes (PositionalEncoding, Actor, TransformerCritic) - Unchanged ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        pe_batch_first = self.pe.squeeze(1).unsqueeze(0)
        return x + pe_batch_first[:, :x.size(1), :]

class Actor(nn.Module):
    """Policy network (Actor) for T-SAC. Uses the normalized last basic_state."""
    def __init__(self, config: TSACConfig, world_config: WorldConfig):
        super(Actor, self).__init__()
        self.config = config
        self.world_config = world_config
        self.use_layer_norm = config.use_layer_norm_actor
        self.state_dim = config.state_dim # Dim of normalized state
        self.action_dim = config.action_dim

        self.layers = nn.ModuleList()
        mlp_input_dim = self.state_dim
        current_dim = mlp_input_dim
        hidden_dims = config.hidden_dims if config.hidden_dims else [256, 256]

        for hidden_dim in hidden_dims:
            linear_layer = nn.Linear(current_dim, hidden_dim)
            self.layers.append(linear_layer)
            if self.use_layer_norm:
                self.layers.append(nn.LayerNorm(hidden_dim))
            self.layers.append(nn.ReLU())
            current_dim = hidden_dim

        self.mean = nn.Linear(current_dim, self.action_dim)
        self.log_std = nn.Linear(current_dim, self.action_dim)
        self.log_std_min = config.log_std_min
        self.log_std_max = config.log_std_max

    def forward(self, network_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass using potentially re-normalized last basic state."""
        x = network_input
        for layer in self.layers:
            x = layer(x)
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, network_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action (normalized) from the policy distribution."""
        mean, log_std = self.forward(network_input)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        action_normalized = torch.tanh(x_t)

        log_prob_unbounded = normal.log_prob(x_t)
        clamped_tanh = action_normalized.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        log_det_jacobian = torch.log(1.0 - clamped_tanh.pow(2) + 1e-7)
        log_prob = log_prob_unbounded - log_det_jacobian
        log_prob = log_prob.sum(1, keepdim=True)

        if torch.isnan(log_prob).any() or torch.isinf(log_prob).any():
             log_prob = torch.nan_to_num(log_prob, nan=-1e6, posinf=-1e6, neginf=-1e6)

        return action_normalized, log_prob, torch.tanh(mean)

class TransformerCritic(nn.Module):
    """Q-function network (Critic) for T-SAC using a Transformer.
       Takes potentially re-normalized initial state and UNNORMALIZED action sequence.
    """
    def __init__(self, config: TSACConfig, world_config: WorldConfig):
        super(TransformerCritic, self).__init__()
        self.config = config
        self.world_config = world_config
        self.state_dim = config.state_dim # Dim of normalized state
        self.action_dim = config.action_dim
        self.embedding_dim = config.embedding_dim
        self.sequence_length = world_config.trajectory_length
        # Input sequence to transformer is [s0, a1, a2, ..., aN] -> length N+1
        self.transformer_seq_len = self.sequence_length + 1

        self.state_embed = nn.Linear(self.state_dim, self.embedding_dim)
        self.action_embed = nn.Linear(self.action_dim, self.embedding_dim)
        # Adjust max_len if needed, should be >= transformer_seq_len
        self.pos_encoder = PositionalEncoding(self.embedding_dim, max_len=self.transformer_seq_len + 10)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim, nhead=config.transformer_n_heads,
            dim_feedforward=config.transformer_hidden_dim, batch_first=True,
            activation='gelu', dropout=0.1, norm_first=True) # norm_first recommended
        encoder_norm = nn.LayerNorm(self.embedding_dim)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.transformer_n_layers, norm=encoder_norm)

        # Output head predicts Q-value
        self.q_head = nn.Linear(self.embedding_dim, 1)
        self.mask = None

    def _generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """Generates an upper-triangular matrix of -inf, with zeros on diag."""
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, initial_state_input: torch.Tensor, action_sequence: torch.Tensor) -> torch.Tensor:
        """Forward pass.
           initial_state_input: (batch, state_dim) - potentially re-normalized
           action_sequence: (batch, n, action_dim) where n <= N (raw actions a1..aN)
           Returns Q-predictions corresponding to actions: (batch, n, 1)
        """
        batch_size, current_seq_len, _ = action_sequence.shape
        device = initial_state_input.device
        transformer_input_len = current_seq_len + 1

        state_emb = self.state_embed(initial_state_input).unsqueeze(1)
        action_emb = self.action_embed(action_sequence)

        transformer_input_emb = torch.cat([state_emb, action_emb], dim=1)
        transformer_input_pos = self.pos_encoder(transformer_input_emb)

        if self.mask is None or self.mask.size(0) != transformer_input_len:
             self.mask = self._generate_square_subsequent_mask(transformer_input_len, device)

        transformer_output = self.transformer_encoder(transformer_input_pos, mask=self.mask)
        action_transformer_output = transformer_output[:, 1:, :]
        q_predictions = self.q_head(action_transformer_output)

        if torch.isnan(q_predictions).any():
             # print("WARNING: TransformerCritic forward produced NaN!") # Reduced verbosity
             q_predictions = torch.nan_to_num(q_predictions, nan=0.0)

        return q_predictions


# --- Updated TSAC Agent ---
class TSAC:
    """Transformer-based Soft Actor-Critic with gradient averaging and optional PER."""
    def __init__(self, config: TSACConfig, world_config: WorldConfig, buffer_config: ReplayBufferConfig, device: torch.device = None): # Added buffer_config
        self.config = config
        self.world_config = world_config
        self.buffer_config = buffer_config # Store buffer config
        self.gamma = config.gamma
        self.tau = config.tau
        self.alpha = config.alpha
        self.auto_tune_alpha = config.auto_tune_alpha
        self.sequence_length = world_config.trajectory_length
        self.state_dim = config.state_dim
        self.action_dim = config.action_dim
        self.use_state_normalization = config.use_state_normalization
        self.use_per = config.use_per # Store PER flag

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"T-SAC Agent using device: {self.device} (Grad Avg)")
        print(f"T-SAC Max Sequence Length (N): {self.sequence_length}")
        if self.use_per: print(f"T-SAC Agent using Prioritized Replay Buffer (alpha={config.per_alpha}, beta0={config.per_beta_start})")
        else: print(f"T-SAC Agent using Standard Replay Buffer")

        # --- Conditional Agent-Level Normalization ---
        self.state_normalizer = None
        if self.use_state_normalization:
            self.state_normalizer = RunningMeanStd(shape=(self.state_dim,), device=self.device)
            print(f"T-SAC Agent state normalization ENABLED for dim: {self.state_dim}")
        else: print(f"T-SAC Agent state normalization DISABLED.")

        # --- Network Instantiation ---
        self.actor = Actor(config, world_config).to(self.device)
        self.critic1 = TransformerCritic(config, world_config).to(self.device)
        self.critic2 = TransformerCritic(config, world_config).to(self.device)
        self.critic1_target = TransformerCritic(config, world_config).to(self.device)
        self.critic2_target = TransformerCritic(config, world_config).to(self.device)

        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        for p in self.critic1_target.parameters(): p.requires_grad = False
        for p in self.critic2_target.parameters(): p.requires_grad = False

        # --- Optimizers ---
        actor_lr = config.actor_lr
        critic_lr = config.critic_lr
        if actor_lr != critic_lr: print(f"Using separate T-SAC LRs: Actor={actor_lr}, Critic={critic_lr}")
        else: print(f"Using single T-SAC LR: {actor_lr}")
        self.actor_optimizer = optim.AdamW(self.actor.parameters(), lr=actor_lr, weight_decay=1e-4)
        self.critic1_optimizer = optim.AdamW(self.critic1.parameters(), lr=critic_lr, weight_decay=1e-4)
        self.critic2_optimizer = optim.AdamW(self.critic2.parameters(), lr=critic_lr, weight_decay=1e-4)

        # --- Alpha ---
        if self.auto_tune_alpha:
            self.target_entropy = -torch.prod(torch.Tensor([self.action_dim]).to(self.device)).item()
            self.log_alpha = nn.Parameter(torch.tensor(np.log(self.alpha), device=self.device, dtype=torch.float32))
            self.alpha_optimizer = optim.AdamW([self.log_alpha], lr=actor_lr, weight_decay=1e-4)
        else:
            self.log_alpha = torch.tensor(np.log(self.alpha), device=self.device, dtype=torch.float32)

        # --- Instantiate Replay Buffer ---
        if self.use_per:
            self.memory = PrioritizedReplayBuffer(config=config, buffer_config=buffer_config, world_config=world_config)
        else:
            self.memory = ReplayBuffer(config=buffer_config, world_config=world_config)

        self.gamma_powers = torch.pow(self.gamma, torch.arange(self.sequence_length + 1, device=self.device)).unsqueeze(0)
        self.total_updates = 0 # Track updates for beta annealing

    def select_action(self, state: dict, evaluate: bool = False) -> float:
        """Select action (normalized) based on the normalized last state."""
        state_trajectory = state['full_trajectory']
        if np.isnan(state_trajectory).any(): return 0.0

        last_basic_state_normalized = torch.FloatTensor(state_trajectory[-1, :self.state_dim]).to(self.device).unsqueeze(0)

        if self.use_state_normalization and self.state_normalizer:
             with torch.no_grad():
                self.state_normalizer.eval()
                network_input = self.state_normalizer.normalize(last_basic_state_normalized)
                self.state_normalizer.train()
        else: network_input = last_basic_state_normalized

        self.actor.eval()
        with torch.no_grad():
            if evaluate:
                mean, _ = self.actor.forward(network_input)
                action_normalized = torch.tanh(mean)
            else:
                action_normalized, _, _ = self.actor.sample(network_input)
        self.actor.train()

        if torch.isnan(action_normalized).any(): return 0.0
        return action_normalized.detach().cpu().numpy()[0, 0]

    # Modified update_parameters for PER
    def update_parameters(self, batch_size: int):
        """Perform T-SAC update using gradient averaging and optional PER."""
        if self.use_per:
            sampled_batch_info = self.memory.sample(batch_size)
            if sampled_batch_info is None: return None
            batch_data, idxs, weights = sampled_batch_info
            weights_tensor = torch.FloatTensor(weights).unsqueeze(1).to(self.device) # (batch, 1)
        else:
            batch_data = self.memory.sample(batch_size)
            if batch_data is None: return None
            idxs = None
            weights_tensor = torch.ones(batch_size, 1, device=self.device)

        # Extract data from dictionary
        state_batch = batch_data['state']
        action_batch_current = batch_data['action']
        reward_batch = batch_data['reward']
        next_state_batch = batch_data['next_state']
        done_batch = batch_data['done']

        state_batch_tensor = torch.FloatTensor(state_batch).to(self.device)
        action_batch_current_tensor = torch.FloatTensor(action_batch_current).to(self.device)
        reward_batch_tensor = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        next_state_batch_tensor = torch.FloatTensor(next_state_batch).to(self.device)
        done_batch_tensor = torch.FloatTensor(done_batch).to(self.device).unsqueeze(1)

        # --- State Processing and Normalization ---
        normalized_basic_states_batch = state_batch_tensor[:, :, :self.state_dim]
        sN_normalized = next_state_batch_tensor[:, -1, :self.state_dim]

        if self.use_state_normalization and self.state_normalizer:
            self.state_normalizer.update(normalized_basic_states_batch.reshape(-1, self.state_dim))
            self.state_normalizer.update(sN_normalized)

        initial_state_normalized = normalized_basic_states_batch[:, 0, :]
        target_states_normalized = torch.cat([
            normalized_basic_states_batch[:, 1:, :],
            sN_normalized.unsqueeze(1)
        ], dim=1)

        if self.use_state_normalization and self.state_normalizer:
            initial_state_input = self.state_normalizer.normalize(initial_state_normalized)
            batch_norm, seq_len_norm, state_dim_norm = target_states_normalized.shape
            target_states_input = self.state_normalizer.normalize(
                target_states_normalized.reshape(-1, state_dim_norm)
            ).reshape(batch_norm, seq_len_norm, state_dim_norm)
            actor_initial_state_input = initial_state_input
        else:
            initial_state_input = initial_state_normalized
            target_states_input = target_states_normalized
            actor_initial_state_input = initial_state_normalized

        # --- Action and Reward Sequences ---
        action_sequence_prefix = state_batch_tensor[:, 1:, self.state_dim:(self.state_dim + self.action_dim)]
        action_sequence = torch.cat([action_sequence_prefix, action_batch_current_tensor.unsqueeze(1)], dim=1)
        rewards_sequence = state_batch_tensor[:, :, -1]

        current_alpha = self.log_alpha.exp().detach()

        # --- Calculate N-Step Targets (G_n) ---
        targets_G_n = []
        self.actor.eval()
        self.critic1_target.eval()
        self.critic2_target.eval()
        with torch.no_grad():
            batch_size_b, seq_len_n, state_dim_s = target_states_input.shape
            flat_target_states_input = target_states_input.reshape(-1, state_dim_s)
            next_actions_flat, next_log_probs_flat, _ = self.actor.sample(flat_target_states_input)
            next_actions_target = next_actions_flat.reshape(batch_size_b, seq_len_n, self.action_dim)
            next_log_probs_target = next_log_probs_flat.reshape(batch_size_b, seq_len_n, 1)

            Q1_target_values = []
            Q2_target_values = []
            for n_idx in range(self.sequence_length):
                state_sn_input = target_states_input[:, n_idx, :]
                action_an_prime = next_actions_target[:, n_idx, :]
                # Use target transformer critic
                q1_sn_an_prime = self.critic1_target(state_sn_input, action_an_prime.unsqueeze(1)).squeeze(1)
                q2_sn_an_prime = self.critic2_target(state_sn_input, action_an_prime.unsqueeze(1)).squeeze(1)
                Q1_target_values.append(q1_sn_an_prime)
                Q2_target_values.append(q2_sn_an_prime)

            Q1_target_all = torch.stack(Q1_target_values, dim=1)
            Q2_target_all = torch.stack(Q2_target_values, dim=1)
            Q_target_min_all = torch.min(Q1_target_all, Q2_target_all)
            V_target_all = Q_target_min_all - current_alpha * next_log_probs_target

            for n in range(1, self.sequence_length + 1):
                 reward_sum = torch.sum(rewards_sequence[:, :n] * self.gamma_powers[:, :n], dim=1, keepdim=True)
                 V_target_sn = V_target_all[:, n-1, :]
                 done_mask_n = done_batch_tensor if n == self.sequence_length else torch.zeros_like(done_batch_tensor)
                 target_G = reward_sum + self.gamma_powers[:, n] * V_target_sn * (1.0 - done_mask_n)
                 targets_G_n.append(target_G)

        # --- Critic Update with PER ---
        self.critic1.train(); self.critic2.train()
        self.critic1_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()

        Q1_pred_seq = self.critic1(initial_state_input, action_sequence)
        Q2_pred_seq = self.critic2(initial_state_input, action_sequence)

        total_weighted_loss1 = 0.0
        total_weighted_loss2 = 0.0
        td_errors_n_step = [] # Store TD errors for priority update

        # Average the weighted loss gradient over N steps
        for n in range(1, self.sequence_length + 1):
            q1_pred_n = Q1_pred_seq[:, n-1, :]
            q2_pred_n = Q2_pred_seq[:, n-1, :]
            target_G_n = targets_G_n[n-1].detach()

            # Calculate element-wise loss
            loss1_n_unreduced = F.mse_loss(q1_pred_n, target_G_n, reduction='none')
            loss2_n_unreduced = F.mse_loss(q2_pred_n, target_G_n, reduction='none')

            # Apply IS weights and average over batch dimension
            weighted_loss1_n = (weights_tensor * loss1_n_unreduced).mean()
            weighted_loss2_n = (weights_tensor * loss2_n_unreduced).mean()

            # Accumulate weighted losses (will divide by N later)
            total_weighted_loss1 += weighted_loss1_n
            total_weighted_loss2 += weighted_loss2_n

            # Store TD error for priority update (e.g., using N-step error)
            if n == self.sequence_length: # Or use n==1 for 1-step TD error
                 td_error_n = torch.max((q1_pred_n - target_G_n).abs(), (q2_pred_n - target_G_n).abs())
                 td_errors_n_step = td_error_n.squeeze(1).detach().cpu().numpy()

        # Average total weighted loss over N steps
        avg_weighted_critic1_loss = total_weighted_loss1 / self.sequence_length
        avg_weighted_critic2_loss = total_weighted_loss2 / self.sequence_length
        total_critic_loss = avg_weighted_critic1_loss + avg_weighted_critic2_loss

        # Single backward pass for the averaged weighted loss
        total_critic_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), max_norm=1.0)
        self.critic1_optimizer.step()
        self.critic2_optimizer.step()

        # --- PER: Update Priorities ---
        if self.use_per and idxs is not None and len(td_errors_n_step) > 0:
            # Use TD errors calculated (e.g., from N-step or 1-step)
            self.memory.update_priorities(idxs, td_errors_n_step)
        elif self.use_per and idxs is not None:
             # Fallback or warning if TD errors weren't computed
             print("Warning: TD errors for PER update not available in T-SAC.")
             # Optionally update with a default priority or skip update

        # --- Actor and Alpha Update ---
        for p in self.critic1.parameters(): p.requires_grad = False
        for p in self.critic2.parameters(): p.requires_grad = False
        self.actor.train()
        action_pi_t0, log_prob_pi_t0, _ = self.actor.sample(actor_initial_state_input)
        q1_pi_t0 = self.critic1(initial_state_input, action_pi_t0.unsqueeze(1)).squeeze(1)
        q2_pi_t0 = self.critic2(initial_state_input, action_pi_t0.unsqueeze(1)).squeeze(1)
        q_pi_min_t0 = torch.min(q1_pi_t0, q2_pi_t0)

        current_alpha_val = self.log_alpha.exp()
        actor_loss = (current_alpha_val.detach() * log_prob_pi_t0 - q_pi_min_t0).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        for p in self.critic1.parameters(): p.requires_grad = True
        for p in self.critic2.parameters(): p.requires_grad = True

        # --- Alpha Update ---
        alpha_loss_val = 0.0
        if self.auto_tune_alpha:
            alpha_loss = -(self.log_alpha * (log_prob_pi_t0.detach() + self.target_entropy)).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            torch.nn.utils.clip_grad_norm_([self.log_alpha], max_norm=1.0)
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()
            alpha_loss_val = alpha_loss.item()

        # --- Target Network Update ---
        with torch.no_grad():
            for target_param, param in zip(self.critic1_target.parameters(), self.critic1.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
            for target_param, param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

        self.total_updates += 1 # Increment update counter

        return {
            'critic1_loss': avg_weighted_critic1_loss.item(), # Log weighted loss
            'critic2_loss': avg_weighted_critic2_loss.item(), # Log weighted loss
            'actor_loss': actor_loss.item(),
            'alpha_loss': alpha_loss_val,
            'alpha': self.alpha,
            'beta': self.memory.beta if self.use_per else 0.0 # Log beta
        }

    def save_model(self, path: str):
        """Saves the model state including normalizer and PER state."""
        print(f"Saving T-SAC model to {path}...")
        save_dict = {
            'actor_state_dict': self.actor.state_dict(),
            'critic1_state_dict': self.critic1.state_dict(),
            'critic2_state_dict': self.critic2.state_dict(),
            'critic1_target_state_dict': self.critic1_target.state_dict(),
            'critic2_target_state_dict': self.critic2_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic1_optimizer_state_dict': self.critic1_optimizer.state_dict(),
            'critic2_optimizer_state_dict': self.critic2_optimizer.state_dict(),
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
            save_dict['log_alpha_value'] = self.log_alpha.data
            if hasattr(self, 'alpha_optimizer'):
                save_dict['alpha_optimizer_state_dict'] = self.alpha_optimizer.state_dict()
        if self.use_per:
            save_dict['per_max_priority'] = self.memory.tree.max_priority # Save max priority

        torch.save(save_dict, path)

    def load_model(self, path: str):
        """Loads the model state including normalizer and PER state."""
        if not os.path.exists(path):
            print(f"Warn: T-SAC model not found: {path}. Skipping load.")
            return
        print(f"Loading T-SAC model from {path}...")
        try:
            checkpoint = torch.load(path, map_location=self.device)

            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.critic1.load_state_dict(checkpoint['critic1_state_dict'])
            self.critic2.load_state_dict(checkpoint['critic2_state_dict'])

            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
            self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer_state_dict'])
            self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer_state_dict'])

            # Load State Normalizer
            loaded_use_state_norm = checkpoint.get('use_state_normalization', False) # Default False
            if 'use_state_normalization' in checkpoint and loaded_use_state_norm != self.use_state_normalization:
                 print(f"Warning: Loaded model STATE normalization setting ({loaded_use_state_norm}) differs from current config ({self.use_state_normalization}). Using current config setting.")
            if self.use_state_normalization and self.state_normalizer:
                 if 'state_normalizer_state_dict' in checkpoint:
                     try: self.state_normalizer.load_state_dict(checkpoint['state_normalizer_state_dict']); print("Loaded T-SAC state normalizer statistics.")
                     except Exception as e: print(f"Warning: Failed to load T-SAC state normalizer stats: {e}.")
                 else: print("Warning: T-SAC state normalizer stats not found in checkpoint.")

            # Load PER State
            loaded_use_per = checkpoint.get('use_per', False) # Default False
            if 'use_per' in checkpoint and loaded_use_per != self.use_per:
                print(f"Warning: Loaded model PER setting ({loaded_use_per}) differs from current config ({self.use_per}). Using current config setting. Replay buffer type mismatch.")
            if self.use_per and isinstance(self.memory, PrioritizedReplayBuffer):
                 self.memory.beta = checkpoint.get('per_beta', self.config.per_beta_start)
                 self.memory.frame = checkpoint.get('per_frame', 0)
                 self.memory.tree.max_priority = checkpoint.get('per_max_priority', 1.0) # Load max priority
                 print(f"Loaded PER state: beta={self.memory.beta:.4f}, frame={self.memory.frame}, max_p={self.memory.tree.max_priority:.4f}")
            elif not self.use_per and loaded_use_per: print("Warning: Checkpoint used PER, but current config doesn't.")


            # Load Alpha
            if self.auto_tune_alpha:
                if 'log_alpha_value' in checkpoint:
                    if not hasattr(self, 'log_alpha') or not isinstance(self.log_alpha, nn.Parameter):
                         self.log_alpha = nn.Parameter(torch.zeros_like(checkpoint['log_alpha_value']))
                    self.log_alpha.data.copy_(checkpoint['log_alpha_value'])
                    if not self.log_alpha.requires_grad: self.log_alpha.requires_grad_(True)
                else: print("Warn: log_alpha_value not found.")
                self.alpha_optimizer = optim.AdamW([self.log_alpha], lr=self.config.actor_lr, weight_decay=1e-4)
                if 'alpha_optimizer_state_dict' in checkpoint:
                     try: self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
                     except Exception as e: print(f"Warn: Load alpha optim failed: {e}. Reinitializing.")
                else: print("Warn: Alpha optim state dict not found.")
                self.alpha = self.log_alpha.exp().item()
            elif 'log_alpha_value' in checkpoint:
                 if not hasattr(self, 'log_alpha') or not isinstance(self.log_alpha, torch.Tensor):
                      self.log_alpha = torch.tensor(0.0, device=self.device)
                 self.log_alpha.data.copy_(checkpoint['log_alpha_value'].to(self.device))
                 self.alpha = self.log_alpha.exp().item()

            # Load target networks
            self.critic1_target.load_state_dict(self.critic1.state_dict())
            self.critic2_target.load_state_dict(self.critic2.state_dict())
            for p in self.critic1_target.parameters(): p.requires_grad = False
            for p in self.critic2_target.parameters(): p.requires_grad = False

            self.total_updates = checkpoint.get('total_updates', 0) # Load update count

            print(f"T-SAC model loaded successfully from {path}")
            self.actor.train(); self.critic1.train(); self.critic2.train()
            self.critic1_target.eval(); self.critic2_target.eval()
            if self.use_state_normalization and self.state_normalizer: self.state_normalizer.train()

        except KeyError as e: print(f"Error loading T-SAC model: Missing key {e}.")
        except Exception as e: print(f"Error loading T-SAC model from {path}: {e}")


# --- Updated Training Loop (train_tsac) ---
def train_tsac(config: DefaultConfig, use_multi_gpu: bool = False, run_evaluation: bool = True):
    tsac_config = config.tsac
    train_config = config.training
    buffer_config = config.replay_buffer # Get buffer config
    world_config = config.world
    cuda_device = config.cuda_device

    learning_starts = train_config.learning_starts
    gradient_steps = train_config.gradient_steps
    train_freq = train_config.train_freq
    batch_size = train_config.batch_size
    log_frequency_ep = train_config.log_frequency
    save_interval_ep = train_config.save_interval

    # --- Device Setup ---
    device = torch.device(cuda_device if torch.cuda.is_available() and cuda_device != 'cpu' else "cpu")
    # ... (rest of device setup code is unchanged) ...
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
            print(f"T-SAC Training using CUDA device: {torch.cuda.current_device()} ({device})")
        except Exception as e:
                print(f"Warning: Error setting CUDA device {cuda_device}: {e}. Falling back to CPU.")
                device = torch.device("cpu")
                print("T-SAC Training using CPU.")
    else:
        device = torch.device("cpu") # Ensure device is 'cpu' if condition fails
        print("T-SAC Training using CPU.")

    # --- Logging and Config Saving ---
    name_prefix = "tsac_per_" if config.tsac.use_per else "tsac_"
    log_dir_name = f"{name_prefix}pointcloud_{int(time.time())}"
    log_dir = os.path.join("runs", log_dir_name)
    os.makedirs(log_dir, exist_ok=True)

    config_save_path = os.path.join(log_dir, "config.json")
    try:
        with open(config_save_path, "w") as f: f.write(config.model_dump_json(indent=2))
        print(f"Configuration saved to: {config_save_path}")
    except Exception as e: print(f"Error saving configuration: {e}")

    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logs: {log_dir}")

    # --- Agent and Memory Initialization ---
    # Pass buffer_config to TSAC agent constructor
    agent = TSAC(config=tsac_config, world_config=world_config, buffer_config=buffer_config, device=device)
    memory = agent.memory # Get buffer from agent
    os.makedirs(train_config.models_dir, exist_ok=True)

    # --- Checkpoint Loading ---
    model_files = [f for f in os.listdir(train_config.models_dir) if f.startswith("tsac_") and f.endswith(".pt")]
    latest_model_path = None
    if model_files:
        try: latest_model_path = max([os.path.join(train_config.models_dir, f) for f in model_files], key=os.path.getmtime)
        except Exception as e: print(f"Could not find latest model: {e}")

    total_steps = 0 # Environment steps
    start_episode = 1
    if latest_model_path and os.path.exists(latest_model_path):
        print(f"\nResuming T-SAC training from: {latest_model_path}")
        agent.load_model(latest_model_path) # Agent loads internal state (like total_updates, PER state)
        total_steps = agent.total_updates * train_freq # Estimate env steps
        try: # Parse episode from filename for tqdm start
            parts = os.path.basename(latest_model_path).split('_')
            ep_part = next((p for p in parts if p.startswith('ep')), None)
            if ep_part: start_episode = int(ep_part.replace('ep', '')) + 1
            print(f"Resuming at approx step {total_steps}, episode {start_episode}")
        except Exception as e: print(f"Warn: Could not parse ep from file: {e}."); start_episode = 1
    else:
        print("\nStarting T-SAC training from scratch.")

    # Training Loop
    episode_rewards = []
    all_losses = {'critic1_loss': [], 'critic2_loss': [], 'actor_loss': [], 'alpha': [], 'alpha_loss': [], 'beta': []} # Added beta
    timing_metrics = {'env_step_time': deque(maxlen=100), 'parameter_update_time': deque(maxlen=100)}
    world = World(world_config=world_config)
    total_env_steps = 0 # Actual env step counter

    pbar = tqdm(range(start_episode, train_config.num_episodes + 1), desc="Training T-SAC Mapping", unit="episode", initial=start_episode-1, total=train_config.num_episodes)

    for episode in pbar:
        state = world.reset()
        episode_reward = 0
        episode_steps = 0
        episode_losses_temp = {'critic1_loss': [], 'critic2_loss': [], 'actor_loss': [], 'alpha': [], 'alpha_loss': [], 'beta': []} # Added beta
        updates_made_this_episode = 0
        episode_metrics = []

        for step_in_episode in range(train_config.max_steps):
            action_normalized = agent.select_action(state, evaluate=False)

            step_start_time = time.time()
            next_state = world.step(action_normalized, training=True, terminal_step=(step_in_episode == train_config.max_steps - 1))
            timing_metrics['env_step_time'].append(time.time() - step_start_time)

            reward = world.reward
            done = world.done
            current_metric = world.performance_metric

            agent.memory.push(state, action_normalized, reward, next_state, done)

            state = next_state
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
                                 if isinstance(val, (float, np.float64)):
                                    episode_losses_temp[key].append(val)
                            updates_made_this_episode += 1
                        else: print(f"INFO: Skipping T-SAC loss log step {total_env_steps} due to NaN.")
                    else: break
            if done: break

        # --- Logging ---
        episode_rewards.append(episode_reward)
        avg_losses = {k: np.mean(v) if v else float('nan') for k, v in episode_losses_temp.items()}
        avg_metric = np.mean(episode_metrics) if episode_metrics else 0.0

        if updates_made_this_episode > 0:
             if not np.isnan(avg_losses['critic1_loss']): all_losses['critic1_loss'].append(avg_losses['critic1_loss'])
             if not np.isnan(avg_losses['critic2_loss']): all_losses['critic2_loss'].append(avg_losses['critic2_loss'])
             if not np.isnan(avg_losses['actor_loss']): all_losses['actor_loss'].append(avg_losses['actor_loss'])
             if not np.isnan(avg_losses['alpha']): all_losses['alpha'].append(avg_losses['alpha'])
             if agent.auto_tune_alpha and not np.isnan(avg_losses['alpha_loss']): all_losses['alpha_loss'].append(avg_losses['alpha_loss'])
             if agent.use_per and not np.isnan(avg_losses['beta']): all_losses['beta'].append(avg_losses['beta']) # Log beta

        if episode % log_frequency_ep == 0:
            log_step = agent.total_updates # Log vs updates
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
                if not np.isnan(avg_losses['critic1_loss']): writer.add_scalar('Loss/Critic1_AvgEp', avg_losses['critic1_loss'], log_step)
                if not np.isnan(avg_losses['critic2_loss']): writer.add_scalar('Loss/Critic2_AvgEp', avg_losses['critic2_loss'], log_step)
                if not np.isnan(avg_losses['actor_loss']): writer.add_scalar('Loss/Actor_AvgEp', avg_losses['actor_loss'], log_step)
                if agent.auto_tune_alpha and not np.isnan(avg_losses['alpha_loss']): writer.add_scalar('Loss/Alpha_AvgEp', avg_losses['alpha_loss'], log_step)
                if not np.isnan(avg_losses['alpha']): writer.add_scalar('Params/Alpha', avg_losses['alpha'], log_step)
                if agent.use_per and not np.isnan(avg_losses['beta']): writer.add_scalar('Params/PER_Beta', avg_losses['beta'], log_step) # Log beta
            else:
                 writer.add_scalar('Params/Alpha', agent.alpha, log_step)
                 if agent.use_per: writer.add_scalar('Params/PER_Beta', agent.memory.beta, log_step)

            if agent.use_state_normalization and agent.state_normalizer and agent.state_normalizer.count > agent.state_normalizer.epsilon:
                writer.add_scalar('Stats/TSAC_Normalizer_Count', agent.state_normalizer.count.item(), log_step)
                writer.add_scalar('Stats/TSAC_Normalizer_Mean_Sensor0', agent.state_normalizer.mean[0].item(), log_step)
                writer.add_scalar('Stats/TSAC_Normalizer_Std_Sensor0', torch.sqrt(agent.state_normalizer.var[0].clamp(min=1e-8)).item(), log_step)
                if agent.state_dim > 5:
                     writer.add_scalar('Stats/TSAC_Normalizer_Mean_AgentXNorm', agent.state_normalizer.mean[5].item(), log_step)
                     writer.add_scalar('Stats/TSAC_Normalizer_Std_AgentXNorm', torch.sqrt(agent.state_normalizer.var[5].clamp(min=1e-8)).item(), log_step)

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
            if updates_made_this_episode and not np.isnan(avg_losses['critic1_loss']):
                pbar_postfix['c1_loss'] = f"{avg_losses['critic1_loss']:.2f}"
            pbar.set_postfix(pbar_postfix)

        if episode % save_interval_ep == 0 and episode > 0:
            save_path = os.path.join(train_config.models_dir, f"tsac_ep{episode}_updates{agent.total_updates}.pt")
            agent.save_model(save_path)

        # --- Early Stopping Check --- ADDED BLOCK ---
        if train_config.enable_early_stopping and len(episode_rewards) >= train_config.early_stopping_window:
            avg_reward_window = np.mean(episode_rewards[-train_config.early_stopping_window:])
            if avg_reward_window >= train_config.early_stopping_threshold:
                print(f"\nEarly stopping triggered at episode {episode}!")
                print(f"Average reward over last {train_config.early_stopping_window} episodes ({avg_reward_window:.2f}) >= threshold ({train_config.early_stopping_threshold:.2f}).")
                # Save final model before breaking
                final_save_path_early = os.path.join(train_config.models_dir, f"tsac_earlystop_ep{episode}_updates{agent.total_updates}.pt")
                agent.save_model(final_save_path_early)
                print(f"Final model saved due to early stopping: {final_save_path_early}")
                break # Exit the training loop
        # --- END ADDED BLOCK ---


    # --- Modified code after the loop ---
    pbar.close()
    writer.close()
    if episode < train_config.num_episodes: # Indicates early stopping occurred
        print(f"T-SAC Training finished early at episode {episode}. Total env steps: {total_env_steps}, Total updates: {agent.total_updates}")
    else: # Training completed all episodes
        print(f"T-SAC Training finished. Total env steps: {total_env_steps}, Total updates: {agent.total_updates}")
        final_save_path = os.path.join(train_config.models_dir, f"tsac_final_ep{train_config.num_episodes}_updates{agent.total_updates}.pt")
        # Avoid saving again if already saved by early stopping
        if not (train_config.enable_early_stopping and 'final_save_path_early' in locals()):
             agent.save_model(final_save_path)

    if run_evaluation:
         print("\nStarting evaluation after training...")
         evaluate_tsac(agent=agent, config=config)

    return agent, episode_rewards


# --- Updated Evaluation Loop (evaluate_tsac) ---
# Evaluation logic doesn't change, but agent instantiation needs to match
def evaluate_tsac(agent: TSAC, config: DefaultConfig):
    """Evaluates the trained T-SAC agent."""
    # ... (Conditional Visualization Import - unchanged) ...
    eval_config = config.evaluation
    world_config = config.world
    vis_config = config.visualization

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
    eval_metrics = []
    success_count = 0
    all_episode_gif_paths = []

    # --- Set Agent to Evaluation Mode ---
    agent.actor.eval(); agent.critic1.eval(); agent.critic2.eval()
    if agent.use_state_normalization and agent.state_normalizer:
        agent.state_normalizer.eval()

    print(f"\nRunning T-SAC Evaluation for {eval_config.num_episodes} episodes...")
    eval_seeds = world_config.seeds
    if len(eval_seeds) != eval_config.num_episodes:
         print(f"Warning: Number of evaluation seeds ({len(eval_seeds)}) doesn't match num_episodes ({eval_config.num_episodes}). Using first {eval_config.num_episodes} seeds or generating if needed.")
         if len(eval_seeds) < eval_config.num_episodes:
              extra_seeds_needed = eval_config.num_episodes - len(eval_seeds)
              eval_seeds.extend([random.randint(0, 2**32 - 1) for _ in range(extra_seeds_needed)])
         eval_seeds = eval_seeds[:eval_config.num_episodes]

    world = World(world_config=world_config)

    for episode in range(eval_config.num_episodes):
        seed_to_use = eval_seeds[episode] if eval_seeds else None
        state = world.reset(seed=seed_to_use)
        episode_reward = 0
        episode_frames = []
        episode_metrics_current = []

        if eval_config.render and vis_available:
            os.makedirs(vis_config.save_dir, exist_ok=True)
            if reset_trajectories: reset_trajectories()
            try:
                fname = f"tsac_eval_ep{episode+1}_frame_000_initial.png"
                initial_frame_file = visualize_world(world, vis_config, filename=fname)
                if initial_frame_file and os.path.exists(initial_frame_file): episode_frames.append(initial_frame_file)
            except Exception as e: print(f"Warn: Vis failed init state ep {episode+1}. E: {e}")

        for step in range(eval_config.max_steps):
            action_normalized = agent.select_action(state, evaluate=True)
            next_state = world.step(action_normalized, training=False, terminal_step=(step == eval_config.max_steps - 1))
            reward = world.reward
            done = world.done
            current_metric = world.performance_metric
            episode_metrics_current.append(current_metric)

            if eval_config.render and vis_available:
                try:
                    fname = f"tsac_eval_ep{episode+1}_frame_{step+1:03d}.png"
                    frame_file = visualize_world(world, vis_config, filename=fname)
                    if frame_file and os.path.exists(frame_file): episode_frames.append(frame_file)
                except Exception as e: print(f"Warn: Vis failed step {step+1} ep {episode+1}. E: {e}")

            state = next_state
            if done: break

        final_metric = world.performance_metric
        eval_rewards.append(episode_reward)
        eval_metrics.append(final_metric)

        success = final_metric >= world_config.success_metric_threshold
        if success: success_count += 1
        status = "Success!" if success else "Failure."
        print(f"  Ep {episode+1}/{eval_config.num_episodes} (Seed:{world.current_seed}): Steps={world.current_step}, Terminated={world.done}, Final Metric: {final_metric:.3f}. {status}")

        if eval_config.render and vis_available and episode_frames and save_gif:
            gif_filename = f"tsac_mapping_eval_episode_{episode+1}_seed{world.current_seed}.gif"
            try:
                print(f"  Saving GIF for T-SAC episode {episode+1} with {len(episode_frames)} frames...")
                gif_path = save_gif(output_filename=gif_filename, vis_config=vis_config, frame_paths=episode_frames, delete_frames=vis_config.delete_frames_after_gif)
                if gif_path: all_episode_gif_paths.append(gif_path)
            except Exception as e: print(f"  Warn: Failed GIF save ep {episode+1}. E: {e}")

    # --- Set Agent back to Training Mode ---
    agent.actor.train(); agent.critic1.train(); agent.critic2.train()
    if agent.use_state_normalization and agent.state_normalizer:
        agent.state_normalizer.train()

    avg_eval_reward = np.mean(eval_rewards) if eval_rewards else 0.0
    std_eval_reward = np.std(eval_rewards) if eval_rewards else 0.0
    avg_eval_metric = np.mean(eval_metrics) if eval_metrics else 0.0
    std_eval_metric = np.std(eval_metrics) if eval_metrics else 0.0
    success_rate = success_count / eval_config.num_episodes if eval_config.num_episodes > 0 else 0.0

    print("\n--- T-SAC Evaluation Summary ---")
    print(f"Episodes: {eval_config.num_episodes}")
    print(f"Average Final Metric (Point Inclusion): {avg_eval_metric:.3f} +/- {std_eval_metric:.3f}")
    print(f"Success Rate (Metric >= {world_config.success_metric_threshold:.2f}): {success_rate:.2%} ({success_count}/{eval_config.num_episodes})")
    print(f"Average Episode Reward: {avg_eval_reward:.3f} +/- {std_eval_reward:.3f}")
    if eval_config.render and vis_available and all_episode_gif_paths: print(f"GIFs saved to: '{os.path.abspath(vis_config.save_dir)}'")
    elif eval_config.render and not vis_available: print("Rendering was enabled but visualization libraries were not found.")
    elif not eval_config.render: print("Rendering disabled.")
    print("--- End T-SAC Evaluation ---\n")

    return eval_rewards, success_rate, avg_eval_metric