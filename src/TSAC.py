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
import random # Added for evaluation seeding

# Local imports
from world import World # Mapping world
from configs import DefaultConfig, ReplayBufferConfig, TSACConfig, WorldConfig, CORE_STATE_DIM, CORE_ACTION_DIM, TRAJECTORY_REWARD_DIM
from torch.utils.tensorboard import SummaryWriter
from typing import Tuple, Optional, List
from utils import RunningMeanStd # Normalization utility

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
        if np.isnan(trajectory).any() or np.isnan(next_trajectory).any():
             print("Warning: Skipping push due to NaN in T-SAC trajectory.")
             return

        self.buffer.append((trajectory, float(action), float(reward), next_trajectory, done))

    def sample(self, batch_size: int) -> Optional[Tuple]:
        if len(self.buffer) < batch_size: return None
        try: batch = random.sample(self.buffer, batch_size)
        except ValueError: return None

        # Trajectories contain normalized states
        trajectory, action, reward, next_trajectory, done = zip(*batch)

        try:
            trajectory_arr = np.array(trajectory, dtype=np.float32) # (b, N, feat_dim)
            action_arr = np.array(action, dtype=np.float32).reshape(-1, 1) # (b, 1)
            reward_arr = np.array(reward, dtype=np.float32) # (b,)
            next_trajectory_arr = np.array(next_trajectory, dtype=np.float32) # (b, N, feat_dim)
            done_arr = np.array(done, dtype=np.float32) # (b,)

            if np.isnan(trajectory_arr).any() or np.isnan(next_trajectory_arr).any():
                 print("Warning: NaN detected in sampled T-SAC batch. Returning None.")
                 return None
        except Exception as e:
            print(f"Error converting TSAC sampled batch: {e}"); return None

        return (trajectory_arr, action_arr, reward_arr, next_trajectory_arr, done_arr)

    def __len__(self):
        return len(self.buffer)

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
        # x shape: (batch, seq_len, embedding_dim)
        # pe shape: (max_len, 1, embedding_dim) -> transpose to (1, max_len, embedding_dim)
        pe_batch_first = self.pe.squeeze(1).unsqueeze(0) # Add batch dim
        # Slice positional encoding to match input sequence length
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
             # print("Warning: NaN/Inf detected in T-SAC Actor log_prob. Clamping.")
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
        # Total length of sequence fed to transformer: 1 (state) + n (actions)
        transformer_input_len = current_seq_len + 1

        # Embed initial state and action sequence
        # State input needs unsqueeze(1) to become (batch, 1, state_dim) before embedding
        state_emb = self.state_embed(initial_state_input).unsqueeze(1) # (batch, 1, embed_dim)
        action_emb = self.action_embed(action_sequence) # (batch, n, embed_dim)

        # Concatenate state and action embeddings
        transformer_input_emb = torch.cat([state_emb, action_emb], dim=1) # (batch, n+1, embed_dim)
        # Add positional encoding
        transformer_input_pos = self.pos_encoder(transformer_input_emb)

        # Generate mask if needed (causal attention mask)
        if self.mask is None or self.mask.size(0) != transformer_input_len:
             self.mask = self._generate_square_subsequent_mask(transformer_input_len, device)

        # Pass through transformer encoder
        transformer_output = self.transformer_encoder(transformer_input_pos, mask=self.mask)

        # Get outputs corresponding to the action inputs (ignore output for s0)
        action_transformer_output = transformer_output[:, 1:, :] # (batch, n, embed_dim)
        # Predict Q-values from action outputs
        q_predictions = self.q_head(action_transformer_output) # (batch, n, 1)

        if torch.isnan(q_predictions).any():
             print("WARNING: TransformerCritic forward produced NaN!")
             q_predictions = torch.nan_to_num(q_predictions, nan=0.0) # Replace NaNs with 0

        return q_predictions

class TSAC:
    """Transformer-based Soft Actor-Critic with gradient averaging."""
    def __init__(self, config: TSACConfig, world_config: WorldConfig, device: torch.device = None):
        self.config = config
        self.world_config = world_config
        self.gamma = config.gamma
        self.tau = config.tau
        self.alpha = config.alpha
        self.auto_tune_alpha = config.auto_tune_alpha
        self.sequence_length = world_config.trajectory_length
        self.state_dim = config.state_dim # Dim of normalized state
        self.action_dim = config.action_dim
        # --- Store agent-level normalization flag ---
        self.use_state_normalization = config.use_state_normalization

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"T-SAC Agent using device: {self.device} (Grad Avg)")
        print(f"T-SAC Max Sequence Length (N): {self.sequence_length}")

        # --- Conditional Agent-Level Normalization ---
        self.state_normalizer = None
        if self.use_state_normalization:
            self.state_normalizer = RunningMeanStd(shape=(self.state_dim,), device=self.device)
            print(f"T-SAC Agent state normalization ENABLED for dim: {self.state_dim} (operates on world's output state)")
        else:
            print(f"T-SAC Agent state normalization DISABLED.")
        # --- End Conditional Normalization ---

        self.actor = Actor(config, world_config).to(self.device)
        self.critic1 = TransformerCritic(config, world_config).to(self.device)
        self.critic2 = TransformerCritic(config, world_config).to(self.device)
        self.critic1_target = TransformerCritic(config, world_config).to(self.device)
        self.critic2_target = TransformerCritic(config, world_config).to(self.device)

        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        for p in self.critic1_target.parameters(): p.requires_grad = False
        for p in self.critic2_target.parameters(): p.requires_grad = False

        actor_lr = config.lr
        critic_lr = config.lr
        if hasattr(config, 'critic_lr') and config.critic_lr is not None:
             critic_lr = config.critic_lr
             print(f"Using separate T-SAC critic LR: {critic_lr}")

        self.actor_optimizer = optim.AdamW(self.actor.parameters(), lr=actor_lr, weight_decay=1e-4)
        self.critic1_optimizer = optim.AdamW(self.critic1.parameters(), lr=critic_lr, weight_decay=1e-4)
        self.critic2_optimizer = optim.AdamW(self.critic2.parameters(), lr=critic_lr, weight_decay=1e-4)

        if self.auto_tune_alpha:
            self.target_entropy = -torch.prod(torch.Tensor([self.action_dim]).to(self.device)).item()
            self.log_alpha = nn.Parameter(torch.tensor(np.log(self.alpha), device=self.device, dtype=torch.float32))
            self.alpha_optimizer = optim.AdamW([self.log_alpha], lr=actor_lr, weight_decay=1e-4)
        else:
            self.log_alpha = torch.tensor(np.log(self.alpha), device=self.device, dtype=torch.float32)

        # Precompute powers of gamma for N-step returns
        self.gamma_powers = torch.pow(self.gamma, torch.arange(self.sequence_length + 1, device=self.device)).unsqueeze(0) # Shape (1, N+1)


    def select_action(self, state: dict, evaluate: bool = False) -> float:
        """Select action (normalized) based on the normalized last state."""
        # state['full_trajectory'] has normalized states
        state_trajectory = state['full_trajectory']
        if np.isnan(state_trajectory).any():
             print("Warning: NaN detected in T-SAC state_trajectory for select_action. Returning 0.")
             return 0.0

        # Get the last normalized basic state from the trajectory
        last_basic_state_normalized = torch.FloatTensor(state_trajectory[-1, :self.state_dim]).to(self.device).unsqueeze(0)

        # --- Conditionally Apply Agent-Level Normalization ---
        if self.use_state_normalization and self.state_normalizer:
             with torch.no_grad():
                self.state_normalizer.eval()
                network_input = self.state_normalizer.normalize(last_basic_state_normalized)
                self.state_normalizer.train()
        else:
             network_input = last_basic_state_normalized # Use world's normalized state
        # --- End Conditional ---

        self.actor.eval()
        with torch.no_grad():
            if evaluate:
                mean, _ = self.actor.forward(network_input)
                action_normalized = torch.tanh(mean)
            else:
                action_normalized, _, _ = self.actor.sample(network_input)
        self.actor.train()

        if torch.isnan(action_normalized).any():
             print("WARNING: NaN detected in T-SAC select_action output! Returning 0.")
             return 0.0

        return action_normalized.detach().cpu().numpy()[0, 0]

    def update_parameters(self, memory: ReplayBuffer, batch_size: int):
        """Perform T-SAC update using gradient averaging over N-step returns."""
        sampled_batch = memory.sample(batch_size)
        if sampled_batch is None: return None

        # state_batch/next_state_batch contain normalized states
        state_batch, action_batch_current, reward_batch, next_state_batch, done_batch = sampled_batch

        state_batch_tensor = torch.FloatTensor(state_batch).to(self.device) # (b, N, feat_dim)
        action_batch_current_tensor = torch.FloatTensor(action_batch_current).to(self.device) # (b, 1) - Action a_N
        reward_batch_tensor = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1) # (b, 1) - Reward r_{N+1}
        next_state_batch_tensor = torch.FloatTensor(next_state_batch).to(self.device) # (b, N, feat_dim) - Contains s_{N+1} effectively
        done_batch_tensor = torch.FloatTensor(done_batch).to(self.device).unsqueeze(1) # (b, 1) - Done flag at step N+1

        # Extract normalized basic states from trajectories in buffer
        # state_batch ~ [s0..sN-1], next_state_batch ~ [s1..sN] (approx, due to how buffer stores)
        # We need s0, s1, ..., sN for targets
        normalized_basic_states_batch = state_batch_tensor[:, :, :self.state_dim] # (b, N, state_dim) ~ [s0..sN-1]
        # Get sN from the last element of the next_state_batch trajectory
        sN_normalized = next_state_batch_tensor[:, -1, :self.state_dim] # (b, state_dim)

        # --- Conditionally Update Agent-Level Normalizer ---
        if self.use_state_normalization and self.state_normalizer:
            # Update with all normalized states observed in the batch
            self.state_normalizer.update(normalized_basic_states_batch.reshape(-1, self.state_dim))
            self.state_normalizer.update(sN_normalized) # Also update with sN
        # --- End Update ---

        # --- Prepare Potentially Re-Normalized Inputs ---
        # initial_state (s0) - normalized by world
        initial_state_normalized = normalized_basic_states_batch[:, 0, :] # (b, state_dim)
        # target_states (s1...sN) - normalized by world
        target_states_normalized = torch.cat([
            normalized_basic_states_batch[:, 1:, :], # s1..sN-1
            sN_normalized.unsqueeze(1)               # sN
        ], dim=1) # (b, N, state_dim) [s1...sN]

        if self.use_state_normalization and self.state_normalizer:
            # Re-normalize states needed for network inputs using agent's RMS
            initial_state_input = self.state_normalizer.normalize(initial_state_normalized) # For critic
            batch_norm, seq_len_norm, state_dim_norm = target_states_normalized.shape
            target_states_input = self.state_normalizer.normalize( # For target actor/critic
                target_states_normalized.reshape(-1, state_dim_norm)
            ).reshape(batch_norm, seq_len_norm, state_dim_norm)
            actor_initial_state_input = initial_state_input # For actor loss (uses initial state s0)
        else:
            # Use world's normalized states directly
            initial_state_input = initial_state_normalized
            target_states_input = target_states_normalized
            actor_initial_state_input = initial_state_normalized
        # --- End Conditional Prepare ---

        # Action sequence (a1...aN) - UNNORMALIZED actions taken
        # Need to extract actions from state_batch and append the current action
        # state_batch[:, i, state_dim] contains action a_i
        # action_batch_current_tensor contains action a_N
        action_sequence_prefix = state_batch_tensor[:, 1:, self.state_dim:(self.state_dim + self.action_dim)] # a1..aN-1 (b, N-1, act_dim)
        action_sequence = torch.cat([action_sequence_prefix, action_batch_current_tensor.unsqueeze(1)], dim=1) # (b, N, act_dim) [a1..aN]

        # Reward sequence (r1...rN) - UNNORMALIZED rewards received
        # Need rewards corresponding to actions a0..aN-1
        # state_batch[:, i, -1] contains reward r_i (received after taking a_{i-1})
        # reward_batch_tensor contains reward r_{N+1} (received after taking a_N)
        rewards_sequence_prefix = state_batch_tensor[:, :, -1] # r1..rN (b, N) - Rewards for transitions s0->s1 ... sN-1 -> sN
        rewards_sequence = rewards_sequence_prefix # Use r1..rN for N-step calculation (b, N)

        current_alpha = self.log_alpha.exp().detach()

        # --- Calculate ALL n-step return targets G^(n) first ---
        targets_G_n = []
        self.actor.eval()
        self.critic1_target.eval()
        self.critic2_target.eval()

        with torch.no_grad():
            # Get actions and log_probs for *all potentially re-normalized* target states s_1...s_N
            batch_size_b, seq_len_n, state_dim_s = target_states_input.shape
            flat_target_states_input = target_states_input.reshape(-1, state_dim_s) # (b*N, state_dim)
            next_actions_flat, next_log_probs_flat, _ = self.actor.sample(flat_target_states_input) # Uses potentially re-normalized state

            next_actions_target = next_actions_flat.reshape(batch_size_b, seq_len_n, self.action_dim) # (b, N, act_dim) [a'1..a'N]
            next_log_probs_target = next_log_probs_flat.reshape(batch_size_b, seq_len_n, 1) # (b, N, 1) [logp(a'1)..logp(a'N)]

            # Get target Q values: Q_tar(s_n, a'_n) using potentially re-normalized target states s_n
            # The critic expects the *initial* state and the *sequence* of actions up to the point of evaluation.
            # This calculation needs rethinking for Transformer critic.
            # Standard SAC/TD3 approach: Calculate V(s_n) = Q_tar(s_n, a'_n) - alpha*logp(a'_n)
            # Q_tar(s_n, a'_n) requires only s_n and a'_n. The transformer critic isn't directly used here.
            # Let's use a simpler MLP critic target like in standard SAC for target calculation.
            # OR, if we must use the Transformer target: Q_tar(s_n, a'_n) would be the prediction
            # from critic_target(s_n, a'_n.unsqueeze(1)).squeeze(1)

            Q1_target_values = []
            Q2_target_values = []
            for n_idx in range(self.sequence_length): # n = 0..N-1 corresponds to target states s_1..s_N
                state_sn_input = target_states_input[:, n_idx, :]    # (b, state_dim) potentially re-normalized s_{n+1}
                action_an_prime = next_actions_target[:, n_idx, :] # (b, act_dim) a'_{n+1}

                # Q_target(s_{n+1}, a'_{n+1}) predicted by the Transformer Target Critic
                # Input: initial state s_{n+1}, action sequence [a'_{n+1}]
                q1_sn_an_prime = self.critic1_target(state_sn_input, action_an_prime.unsqueeze(1)).squeeze(1) # (b, 1)
                q2_sn_an_prime = self.critic2_target(state_sn_input, action_an_prime.unsqueeze(1)).squeeze(1) # (b, 1)

                Q1_target_values.append(q1_sn_an_prime)
                Q2_target_values.append(q2_sn_an_prime)

            Q1_target_all = torch.stack(Q1_target_values, dim=1) # (b, N, 1) [Q1_tar(s1,a'1)..Q1_tar(sN,a'N)]
            Q2_target_all = torch.stack(Q2_target_values, dim=1) # (b, N, 1) [Q2_tar(s1,a'1)..Q2_tar(sN,a'N)]
            Q_target_min_all = torch.min(Q1_target_all, Q2_target_all) # (b, N, 1)

            # V(s_n) = Q_min(s_n, a'_n) - alpha * log_prob(a'_n | s_n)
            V_target_all = Q_target_min_all - current_alpha * next_log_probs_target # (b, N, 1) [V(s1)..V(sN)]

            # Calculate G^(n) = sum_{k=0}^{n-1} (gamma^k * r_{k+1}) + gamma^n * V(s_n) * (1-done_n)
            for n in range(1, self.sequence_length + 1): # n = 1..N
                 # Sum rewards r1...rn
                 reward_sum = torch.sum(rewards_sequence[:, :n] * self.gamma_powers[:, :n], dim=1, keepdim=True)
                 # Get V(s_n) corresponding to this n
                 V_target_sn = V_target_all[:, n-1, :] # V(s1) for n=1, V(sN) for n=N
                 # Done mask is only relevant for the final step (N) corresponding to the sampled 'done' flag
                 done_mask_n = done_batch_tensor if n == self.sequence_length else torch.zeros_like(done_batch_tensor)

                 target_G = reward_sum + self.gamma_powers[:, n] * V_target_sn * (1.0 - done_mask_n)
                 targets_G_n.append(target_G) # List of N targets [G1..GN]

        # --- Critic Update using Gradient Averaging ---
        self.critic1.train(); self.critic2.train()
        self.critic1_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()

        # Get Q predictions for the actual sequence (potentially re-normalized s0, raw a1...aN)
        # Input: initial_state_input (re-norm s0), action_sequence (raw a1..aN)
        Q1_pred_seq = self.critic1(initial_state_input, action_sequence) # (b, N, 1) [Q1(s0,a1)..Q1(s0,a1..aN)]
        Q2_pred_seq = self.critic2(initial_state_input, action_sequence) # (b, N, 1) [Q2(s0,a1)..Q2(s0,a1..aN)]

        total_critic1_loss = 0.0
        total_critic2_loss = 0.0

        # Average the loss gradient over N steps
        for n in range(1, self.sequence_length + 1): # n = 1..N
            # Q-prediction corresponding to taking action a_n (index n-1)
            q1_pred_n = Q1_pred_seq[:, n-1, :]
            q2_pred_n = Q2_pred_seq[:, n-1, :]
            # Target G^(n) calculated earlier
            target_G_n = targets_G_n[n-1].detach()

            loss1_n = F.mse_loss(q1_pred_n, target_G_n)
            loss2_n = F.mse_loss(q2_pred_n, target_G_n)

            total_critic1_loss += loss1_n.item()
            total_critic2_loss += loss2_n.item()

            # Backward pass for each step's loss, scaled by 1/N
            # Retain graph if not the last step in the sequence
            retain_graph = (n < self.sequence_length)
            (loss1_n / self.sequence_length).backward(retain_graph=retain_graph)
            # If critic2 shares layers that were already backpropped through,
            # only need to backward on its unique parts or the final loss.
            # However, separate backward calls are generally safe if optimizers handle it.
            # Need to be careful if graph is fully retained. Let's call backward separately.
            (loss2_n / self.sequence_length).backward(retain_graph=retain_graph)


        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), max_norm=1.0)
        self.critic1_optimizer.step()
        self.critic2_optimizer.step()

        avg_critic1_loss = total_critic1_loss / self.sequence_length
        avg_critic2_loss = total_critic2_loss / self.sequence_length

        # --- Actor and Alpha Update ---
        # Freeze critic parameters for actor update
        for p in self.critic1.parameters(): p.requires_grad = False
        for p in self.critic2.parameters(): p.requires_grad = False

        self.actor.train()
        # Use potentially re-normalized initial state (s0) for actor update
        action_pi_t0, log_prob_pi_t0, _ = self.actor.sample(actor_initial_state_input) # a_pi_0, logp(a_pi_0|s0)
        # Get Q-value for the policy action at the initial state
        # Input: initial_state_input (re-norm s0), action sequence [a_pi_0]
        q1_pi_t0 = self.critic1(initial_state_input, action_pi_t0.unsqueeze(1)).squeeze(1) # Q1(s0, a_pi_0)
        q2_pi_t0 = self.critic2(initial_state_input, action_pi_t0.unsqueeze(1)).squeeze(1) # Q2(s0, a_pi_0)
        q_pi_min_t0 = torch.min(q1_pi_t0, q2_pi_t0)

        current_alpha_val = self.log_alpha.exp() # Use current alpha value
        actor_loss = (current_alpha_val * log_prob_pi_t0 - q_pi_min_t0).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        # Unfreeze critic parameters
        for p in self.critic1.parameters(): p.requires_grad = True
        for p in self.critic2.parameters(): p.requires_grad = True

        # --- Alpha Update ---
        alpha_loss_val = 0.0
        if self.auto_tune_alpha:
            # Use the log_prob calculated during actor update
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

        return {
            'critic1_loss': avg_critic1_loss,
            'critic2_loss': avg_critic2_loss,
            'actor_loss': actor_loss.item(),
            'alpha_loss': alpha_loss_val,
            'alpha': self.alpha
        }

    def save_model(self, path: str):
        """Saves the model state including normalizer."""
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
            'use_state_normalization': self.use_state_normalization, # Save flag
            'device_type': self.device.type
        }
        # --- Conditionally save normalizer ---
        if self.use_state_normalization and self.state_normalizer:
            save_dict['state_normalizer_state_dict'] = self.state_normalizer.state_dict()
        # --- End Conditional ---
        if self.auto_tune_alpha:
            save_dict['log_alpha_value'] = self.log_alpha.data # Save tensor data
            if hasattr(self, 'alpha_optimizer'):
                save_dict['alpha_optimizer_state_dict'] = self.alpha_optimizer.state_dict()
        torch.save(save_dict, path)

    def load_model(self, path: str):
        """Loads the model state including normalizer."""
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

            # --- Load Normalizer State Conditionally ---
            loaded_use_norm = checkpoint.get('use_state_normalization', True) # Default true if missing
            if 'use_state_normalization' in checkpoint and loaded_use_norm != self.use_state_normalization:
                 print(f"Warning: Loaded model normalization setting ({loaded_use_norm}) differs from current config ({self.use_state_normalization}). Using current config setting.")

            if self.use_state_normalization and self.state_normalizer:
                 if 'state_normalizer_state_dict' in checkpoint:
                     try:
                         self.state_normalizer.load_state_dict(checkpoint['state_normalizer_state_dict'])
                         print("Loaded T-SAC state normalizer statistics.")
                     except Exception as e:
                          print(f"Warning: Failed to load T-SAC state normalizer stats: {e}. Using initial values.")
                 else:
                     print("Warning: T-SAC state normalizer statistics not found in checkpoint, but normalization is enabled. Using initial values.")
            elif not self.use_state_normalization and 'state_normalizer_state_dict' in checkpoint:
                  print("Warning: Checkpoint contains T-SAC normalizer stats, but normalization is disabled in current config. Ignoring saved stats.")
            # --- End Load Normalizer ---

            # Load Alpha and its optimizer state if auto-tuning
            if self.auto_tune_alpha:
                if 'log_alpha_value' in checkpoint:
                     # Ensure log_alpha exists and is a Parameter before copying data
                     if not hasattr(self, 'log_alpha') or not isinstance(self.log_alpha, nn.Parameter):
                         self.log_alpha = nn.Parameter(torch.zeros_like(checkpoint['log_alpha_value']))
                     self.log_alpha.data.copy_(checkpoint['log_alpha_value'])
                     if not self.log_alpha.requires_grad: self.log_alpha.requires_grad_(True) # Ensure grad enabled
                else: print("Warn: log_alpha_value not found in checkpoint for auto-tuning alpha.")

                actor_lr = self.config.lr # Use actor LR for alpha optimizer by default
                self.alpha_optimizer = optim.AdamW([self.log_alpha], lr=actor_lr, weight_decay=1e-4)
                if 'alpha_optimizer_state_dict' in checkpoint:
                     try: self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
                     except Exception as e: print(f"Warn: Load alpha optim failed: {e}. Reinitializing optimizer.")
                else: print("Warn: Alpha optim state dict not found.")
                self.alpha = self.log_alpha.exp().item() # Update alpha value
            elif 'log_alpha_value' in checkpoint: # Fixed alpha case
                 # Ensure log_alpha exists and is a tensor
                 if not hasattr(self, 'log_alpha') or not isinstance(self.log_alpha, torch.Tensor):
                      self.log_alpha = torch.tensor(0.0, device=self.device)
                 self.log_alpha.data.copy_(checkpoint['log_alpha_value'].to(self.device))
                 self.alpha = self.log_alpha.exp().item()


            # Load target networks
            self.critic1_target.load_state_dict(self.critic1.state_dict())
            self.critic2_target.load_state_dict(self.critic2.state_dict())
            for p in self.critic1_target.parameters(): p.requires_grad = False
            for p in self.critic2_target.parameters(): p.requires_grad = False

            print(f"T-SAC model loaded successfully from {path}")
            self.actor.train()
            self.critic1.train()
            self.critic2.train()
            self.critic1_target.eval() # Targets should be in eval mode
            self.critic2_target.eval()
            # --- Conditionally set normalizer mode ---
            if self.use_state_normalization and self.state_normalizer:
                 self.state_normalizer.train()
            # --- End Conditional ---

        except KeyError as e: print(f"Error loading T-SAC model: Missing key {e}.")
        except Exception as e: print(f"Error loading T-SAC model from {path}: {e}")


# --- Training Loop (train_tsac) ---
def train_tsac(config: DefaultConfig, use_multi_gpu: bool = False, run_evaluation: bool = True):
    tsac_config = config.tsac
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
            print(f"T-SAC Training using CUDA device: {torch.cuda.current_device()} ({device})")
        except Exception as e:
                print(f"Warning: Error setting CUDA device {cuda_device}: {e}. Falling back to CPU.")
                device = torch.device("cpu")
                print("T-SAC Training using CPU.")
    else:
        device = torch.device("cpu") # Ensure device is 'cpu' if condition fails
        print("T-SAC Training using CPU.")
    # --- End Device Setup ---

    log_dir = os.path.join("runs", f"tsac_pointcloud_{int(time.time())}") # Changed log dir name
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logs: {log_dir}")

    # Agent initialization now uses config flag
    agent = TSAC(config=tsac_config, world_config=world_config, device=device)
    memory = ReplayBuffer(config=buffer_config, world_config=world_config)
    os.makedirs(train_config.models_dir, exist_ok=True)

    # Load Checkpoint
    model_files = [f for f in os.listdir(train_config.models_dir) if f.startswith("tsac_") and f.endswith(".pt")]
    latest_model_path = None
    if model_files:
        try: latest_model_path = max([os.path.join(train_config.models_dir, f) for f in model_files], key=os.path.getmtime)
        except Exception as e: print(f"Could not find latest model: {e}")

    total_steps = 0
    start_episode = 1
    if latest_model_path and os.path.exists(latest_model_path):
        print(f"\nResuming T-SAC training from: {latest_model_path}")
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
        print("\nStarting T-SAC training from scratch.")

    # Training Loop
    episode_rewards = []
    all_losses = {'critic1_loss': [], 'critic2_loss': [], 'actor_loss': [], 'alpha': [], 'alpha_loss': []}
    timing_metrics = {'env_step_time': deque(maxlen=100), 'parameter_update_time': deque(maxlen=100)}

    world = World(world_config=world_config) # Init world outside loop

    pbar = tqdm(range(start_episode, train_config.num_episodes + 1), desc="Training T-SAC Mapping", unit="episode", initial=start_episode-1, total=train_config.num_episodes)

    for episode in pbar:
        state = world.reset() # Reset world (uses internal seeding)
        episode_reward = 0
        episode_steps = 0
        episode_losses_temp = {'critic1_loss': [], 'critic2_loss': [], 'actor_loss': [], 'alpha': [], 'alpha_loss': []}
        updates_made_this_episode = 0
        episode_metrics = [] # Changed from episode_ious

        for step_in_episode in range(train_config.max_steps):
            # select_action uses normalizer conditionally
            action_normalized = agent.select_action(state, evaluate=False)

            step_start_time = time.time()
            next_state = world.step(action_normalized, training=True, terminal_step=(step_in_episode == train_config.max_steps - 1))
            timing_metrics['env_step_time'].append(time.time() - step_start_time)

            reward = world.reward
            done = world.done
            current_metric = world.performance_metric # Changed from current_iou

            # Push transition (state and next_state contain normalized states)
            memory.push(state, action_normalized, reward, next_state, done)

            state = next_state
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
                        timing_metrics['parameter_update_time'].append(time.time() - update_start_time)
                        if losses:
                             # Avoid logging if NaN occurred (already printed warning in update)
                             if not any(np.isnan(v) for k, v in losses.items() if k != 'alpha'):
                                for key, val in losses.items():
                                     if isinstance(val, (float, np.float64)):
                                        episode_losses_temp[key].append(val)
                                updates_made_this_episode += 1
                             else: print(f"INFO: Skipping T-SAC loss log step {total_steps} due to NaN.")
                        else: break # Stop gradient steps if update fails
                    else: break # Stop gradient steps if buffer too small
            if done: break

        # --- Logging ---
        episode_rewards.append(episode_reward)
        avg_losses = {k: np.mean(v) if v else float('nan') for k, v in episode_losses_temp.items()}
        avg_metric = np.mean(episode_metrics) if episode_metrics else 0.0 # Changed from avg_iou

        if updates_made_this_episode > 0:
             if not np.isnan(avg_losses['critic1_loss']): all_losses['critic1_loss'].append(avg_losses['critic1_loss'])
             if not np.isnan(avg_losses['critic2_loss']): all_losses['critic2_loss'].append(avg_losses['critic2_loss'])
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
            if world.current_seed is not None: writer.add_scalar('Environment/Current_Seed', world.current_seed, episode)

            if updates_made_this_episode > 0:
                if not np.isnan(avg_losses['critic1_loss']): writer.add_scalar('Loss/Critic1_AvgEp', avg_losses['critic1_loss'], total_steps)
                if not np.isnan(avg_losses['critic2_loss']): writer.add_scalar('Loss/Critic2_AvgEp', avg_losses['critic2_loss'], total_steps)
                if not np.isnan(avg_losses['actor_loss']): writer.add_scalar('Loss/Actor_AvgEp', avg_losses['actor_loss'], total_steps)
                if agent.auto_tune_alpha and not np.isnan(avg_losses['alpha_loss']): writer.add_scalar('Loss/Alpha_AvgEp', avg_losses['alpha_loss'], total_steps)
                if not np.isnan(avg_losses['alpha']): writer.add_scalar('Alpha/Value', avg_losses['alpha'], total_steps)
            else: writer.add_scalar('Alpha/Value', agent.alpha, total_steps) # Log current alpha if no updates

            # --- Conditionally log normalizer stats ---
            if agent.use_state_normalization and agent.state_normalizer and agent.state_normalizer.count > agent.state_normalizer.epsilon:
                writer.add_scalar('Stats/TSAC_Normalizer_Count', agent.state_normalizer.count.item(), total_steps)
                writer.add_scalar('Stats/TSAC_Normalizer_Mean_Sensor0', agent.state_normalizer.mean[0].item(), total_steps)
                writer.add_scalar('Stats/TSAC_Normalizer_Std_Sensor0', torch.sqrt(agent.state_normalizer.var[0].clamp(min=1e-8)).item(), total_steps)
                if agent.state_dim > 5:
                     writer.add_scalar('Stats/TSAC_Normalizer_Mean_AgentXNorm', agent.state_normalizer.mean[5].item(), total_steps)
                     writer.add_scalar('Stats/TSAC_Normalizer_Std_AgentXNorm', torch.sqrt(agent.state_normalizer.var[5].clamp(min=1e-8)).item(), total_steps)
            # --- End Conditional Log ---

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
            if updates_made_this_episode and not np.isnan(avg_losses['critic1_loss']):
                pbar_postfix['c1_loss'] = f"{avg_losses['critic1_loss']:.2f}"
            pbar.set_postfix(pbar_postfix)


        if episode % save_interval_ep == 0 and episode > 0: # Avoid saving at ep 0
            save_path = os.path.join(train_config.models_dir, f"tsac_ep{episode}_step{total_steps}.pt")
            agent.save_model(save_path)

    pbar.close()
    writer.close()
    print(f"T-SAC Training finished. Total steps: {total_steps}")
    final_save_path = os.path.join(train_config.models_dir, f"tsac_final_ep{train_config.num_episodes}_step{total_steps}.pt")
    agent.save_model(final_save_path)

    if run_evaluation:
         print("\nStarting evaluation after training...")
         evaluate_tsac(agent=agent, config=config) # Pass the possibly modified config

    return agent, episode_rewards


# --- Evaluation Loop (evaluate_tsac) ---
def evaluate_tsac(agent: TSAC, config: DefaultConfig):
    """Evaluates the trained T-SAC agent."""
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
            vis_available = True; print("Visualization enabled.")
        except ImportError:
            print("Vis libs not found. Rendering disabled."); vis_available = False
    else: vis_available = False; print("Rendering disabled by config.")


    eval_rewards = []
    eval_metrics = [] # Changed from eval_ious
    success_count = 0
    all_episode_gif_paths = []

    # --- Set Agent and Normalizer (if exists) to Evaluation Mode ---
    agent.actor.eval(); agent.critic1.eval(); agent.critic2.eval()
    if agent.use_state_normalization and agent.state_normalizer:
        agent.state_normalizer.eval()
    # --- End Set Eval Mode ---

    print(f"\nRunning T-SAC Evaluation for {eval_config.num_episodes} episodes...")
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
                fname = f"tsac_eval_ep{episode+1}_frame_000_initial.png"
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
                    fname = f"tsac_eval_ep{episode+1}_frame_{step+1:03d}.png"
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
            gif_filename = f"tsac_mapping_eval_episode_{episode+1}_seed{world.current_seed}.gif"
            try:
                print(f"  Saving GIF for T-SAC episode {episode+1} with {len(episode_frames)} frames...")
                gif_path = save_gif(output_filename=gif_filename, vis_config=vis_config, frame_paths=episode_frames, delete_frames=vis_config.delete_frames_after_gif)
                if gif_path: all_episode_gif_paths.append(gif_path)
            except Exception as e: print(f"  Warn: Failed GIF save ep {episode+1}. E: {e}")

    # --- Set Agent and Normalizer (if exists) back to Training Mode ---
    agent.actor.train(); agent.critic1.train(); agent.critic2.train()
    if agent.use_state_normalization and agent.state_normalizer:
        agent.state_normalizer.train()
    # --- End Set Train Mode ---

    avg_eval_reward = np.mean(eval_rewards) if eval_rewards else 0.0
    std_eval_reward = np.std(eval_rewards) if eval_rewards else 0.0
    avg_eval_metric = np.mean(eval_metrics) if eval_metrics else 0.0 # Changed from avg_eval_iou
    std_eval_metric = np.std(eval_metrics) if eval_metrics else 0.0 # Changed from std_eval_iou
    success_rate = success_count / eval_config.num_episodes if eval_config.num_episodes > 0 else 0.0

    print("\n--- T-SAC Evaluation Summary ---")
    print(f"Episodes: {eval_config.num_episodes}")
    print(f"Average Final Metric (Point Inclusion): {avg_eval_metric:.3f} +/- {std_eval_metric:.3f}") # Changed label
    print(f"Success Rate (Metric >= {world_config.success_metric_threshold:.2f}): {success_rate:.2%} ({success_count}/{eval_config.num_episodes})")
    print(f"Average Episode Reward: {avg_eval_reward:.3f} +/- {std_eval_reward:.3f}") # Keep reward report
    if eval_config.render and vis_available and all_episode_gif_paths: print(f"GIFs saved to: '{os.path.abspath(vis_config.save_dir)}'")
    elif eval_config.render and not vis_available: print("Rendering was enabled but visualization libraries were not found.")
    elif not eval_config.render: print("Rendering disabled.")
    print("--- End T-SAC Evaluation ---\n")


    return eval_rewards, success_rate, avg_eval_metric # Return metric instead of iou