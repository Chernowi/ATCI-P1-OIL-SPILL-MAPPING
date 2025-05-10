import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import time
from collections import deque, defaultdict
import matplotlib.pyplot as plt
import matplotlib.animation as animation 
from tqdm import tqdm
import random
from typing import Tuple, Optional, Union, Dict, Any, List # Added List
import math # Added

# Local imports
from world import World
from configs import DefaultConfig, PPOConfig, TrainingConfig, CORE_STATE_DIM, WorldConfig
from torch.utils.tensorboard import SummaryWriter
from utils import RunningMeanStd
# Import RNN padding utilities
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

# --- NEW: Recurrent PPO Memory ---
class RecurrentPPOMemory:
    """Memory buffer for Recurrent PPO, storing sequences."""
    def __init__(self, config: PPOConfig, world_config: WorldConfig, device: torch.device):
        self.steps_per_update = config.steps_per_update
        self.state_dim = config.state_dim # Includes heading
        self.use_rnn = config.use_rnn
        self.rnn_type = config.rnn_type
        self.rnn_num_layers = config.rnn_num_layers
        self.rnn_hidden_size = config.rnn_hidden_size
        self.batch_size = config.batch_size # Training minibatch size
        self.device = device

        # Buffers to store full rollouts until steps_per_update is reached
        self.rollout_states: List[np.ndarray] = []
        self.rollout_actions: List[np.ndarray] = []
        self.rollout_log_probs: List[np.ndarray] = []
        self.rollout_values: List[np.ndarray] = []
        self.rollout_rewards: List[np.ndarray] = []
        self.rollout_dones: List[np.ndarray] = []
        self.rollout_actor_hxs: List[Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]] = []
        self.rollout_critic_hxs: List[Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]] = []

        # Storage for completed rollouts ready for batching
        self.memory_states: List[List[np.ndarray]] = []
        self.memory_actions: List[List[np.ndarray]] = []
        self.memory_log_probs: List[List[np.ndarray]] = []
        self.memory_values: List[List[np.ndarray]] = []
        self.memory_rewards: List[List[np.ndarray]] = []
        self.memory_dones: List[List[np.ndarray]] = []
        # Storing hidden states for BPTT start is complex; resetting is common
        # self.memory_actor_hxs_initial: List[...] = []
        # self.memory_critic_hxs_initial: List[...] = []

        self.current_rollout_len = 0
        self.total_samples_in_memory = 0

    def store(self, state_basic: tuple, action: float, log_prob: float, value: float,
              reward: float, done: bool, actor_hxs, critic_hxs):
        """Store one step of experience in the current rollout buffer."""
        # State should be the normalized basic state tuple (incl heading)
        if not isinstance(state_basic, tuple) or len(state_basic) != self.state_dim:
             print(f"Warning (store): Invalid state_basic format. Len={len(state_basic) if isinstance(state_basic, tuple) else 'N/A'}, Expected={self.state_dim}")
             return
        if any(np.isnan(x) for x in state_basic):
             print("Warning (store): NaN detected in state_basic.")
             return # Avoid storing NaN states

        self.rollout_states.append(np.array(state_basic, dtype=np.float32))
        self.rollout_actions.append(np.array([action], dtype=np.float32))
        self.rollout_log_probs.append(np.array([log_prob], dtype=np.float32))
        self.rollout_values.append(np.array([value], dtype=np.float32))
        self.rollout_rewards.append(np.array([reward], dtype=np.float32))
        self.rollout_dones.append(np.array([done], dtype=np.float32))
        # Store hidden states if RNN is used (detach them first)
        # if self.use_rnn:
        #     self.rollout_actor_hxs.append(self._detach_hidden(actor_hxs))
        #     self.rollout_critic_hxs.append(self._detach_hidden(critic_hxs))

        self.current_rollout_len += 1

        # If rollout buffer reaches update size or episode ends, finalize
        if self.current_rollout_len >= self.steps_per_update or done:
            self.finalize_rollout()

    def finalize_rollout(self):
        """Move the completed rollout from temporary buffers to main memory."""
        if not self.rollout_states: return # Nothing to finalize

        self.memory_states.append(list(self.rollout_states))
        self.memory_actions.append(list(self.rollout_actions))
        self.memory_log_probs.append(list(self.rollout_log_probs))
        self.memory_values.append(list(self.rollout_values))
        self.memory_rewards.append(list(self.rollout_rewards))
        self.memory_dones.append(list(self.rollout_dones))
        # Store initial hidden state of the rollout if needed
        # if self.use_rnn and self.rollout_actor_hxs:
        #     self.memory_actor_hxs_initial.append(self.rollout_actor_hxs[0])
        #     self.memory_critic_hxs_initial.append(self.rollout_critic_hxs[0])

        self.total_samples_in_memory += len(self.rollout_states)
        self.clear_rollout_buffers()

    def clear_rollout_buffers(self):
        """Clear the buffers for the current rollout."""
        self.rollout_states.clear()
        self.rollout_actions.clear()
        self.rollout_log_probs.clear()
        self.rollout_values.clear()
        self.rollout_rewards.clear()
        self.rollout_dones.clear()
        self.rollout_actor_hxs.clear()
        self.rollout_critic_hxs.clear()
        self.current_rollout_len = 0

    def clear_memory(self):
        """Clear all stored rollouts."""
        self.memory_states.clear()
        self.memory_actions.clear()
        self.memory_log_probs.clear()
        self.memory_values.clear()
        self.memory_rewards.clear()
        self.memory_dones.clear()
        # self.memory_actor_hxs_initial.clear()
        # self.memory_critic_hxs_initial.clear()
        self.total_samples_in_memory = 0
        self.clear_rollout_buffers() # Also clear any partial rollout

    def _detach_hidden(self, hxs):
        """Detach hidden state tensor or tuple of tensors."""
        if hxs is None: return None
        if isinstance(hxs, torch.Tensor): return hxs.detach()
        if isinstance(hxs, tuple): return tuple(h.detach() for h in hxs)
        return None

    def _pad_sequences(self, sequences: List[List[np.ndarray]], dtype=torch.float32) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pad sequences within a batch and create a mask."""
        seq_tensors = [torch.from_numpy(np.stack(s)).to(self.device) for s in sequences]
        lengths = torch.tensor([len(s) for s in seq_tensors], device=self.device)
        padded_seqs = pad_sequence(seq_tensors, batch_first=True, padding_value=0.0)
        # Create mask (1 for valid steps, 0 for padding)
        mask = torch.arange(padded_seqs.size(1), device=self.device)[None, :] < lengths[:, None]
        return padded_seqs.to(dtype), mask.to(dtype) # Return mask as float for easy multiplication

    def compute_advantages_returns(self, last_value: float, last_done: bool, gamma: float, gae_lambda: float, use_reward_normalization: bool) -> Tuple[List[List[np.ndarray]], List[List[np.ndarray]]]:
        """
        Computes GAE advantages and returns for all stored rollouts.
        Operates across potentially multiple rollouts stored in memory.
        last_value/last_done refer to the state *after* the last stored step.
        Returns lists of advantages and returns, matching the structure of memory_*.
        """
        all_advantages = []
        all_returns = []

        # Combine all rollouts temporarily for normalization (optional but can be better)
        flat_rewards = [r for rollout in self.memory_rewards for r in rollout]
        if use_reward_normalization and len(flat_rewards) > 1:
            rewards_arr = torch.tensor(np.concatenate(flat_rewards), dtype=torch.float32, device=self.device)
            mean = rewards_arr.mean()
            std = rewards_arr.std()
            rewards_arr = (rewards_arr - mean) / (std + 1e-8)
            # Redistribute normalized rewards back to rollouts
            normalized_rewards_list = torch.split(rewards_arr, [len(r) for r in self.memory_rewards])
        else:
            # Use original rewards if not normalizing or insufficient data
            normalized_rewards_list = [torch.tensor(np.concatenate(r), dtype=torch.float32, device=self.device) for r in self.memory_rewards]

        # GAE Calculation per rollout
        for i in range(len(self.memory_rewards)):
            rewards = normalized_rewards_list[i]
            values = torch.tensor(np.concatenate(self.memory_values[i]), dtype=torch.float32, device=self.device)
            dones = torch.tensor(np.concatenate(self.memory_dones[i]), dtype=torch.float32, device=self.device)
            rollout_len = len(rewards)
            advantages = torch.zeros_like(rewards)
            last_gae_lam = 0

            # Determine the value of the state *after* the rollout ended
            # If the last stored step was 'done', the next value is 0
            # Otherwise, use the provided 'last_value' (critic's estimate of V(s_T+1))
            next_value = 0.0 if dones[-1].item() > 0.5 else torch.tensor(last_value, dtype=torch.float32, device=self.device) # Use critic's estimate if not done

            for t in reversed(range(rollout_len)):
                if t == rollout_len - 1:
                    next_non_terminal = 1.0 - dones[t] # Should be 1.0 if we are using external last_value correctly
                    v_next = next_value
                else:
                    next_non_terminal = 1.0 - dones[t] # Mask based on done flag of step t
                    v_next = values[t+1]

                delta = rewards[t] + gamma * v_next * next_non_terminal - values[t]
                last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
                advantages[t] = last_gae_lam

            returns = advantages + values

            # Store as list of numpy arrays (consistent with other memory fields)
            all_advantages.append([adv.cpu().numpy().reshape(1) for adv in advantages])
            all_returns.append([ret.cpu().numpy().reshape(1) for ret in returns])

        return all_advantages, all_returns

    def generate_batches(self, advantages: List[List[np.ndarray]], returns: List[List[np.ndarray]]) -> Optional[Dict[str, Any]]:
        """
        Generates training batches from the stored rollouts.
        Pads sequences within each batch and creates masks.
        Shuffles the *order* of rollouts before batching.
        """
        num_rollouts = len(self.memory_states)
        if num_rollouts == 0: return None

        indices = np.arange(num_rollouts)
        np.random.shuffle(indices) # Shuffle rollouts

        # Create batches of rollouts indices
        batch_indices_list = [indices[i:i + self.batch_size] for i in range(0, num_rollouts, self.batch_size)]

        # Generator for batches
        for batch_indices in batch_indices_list:
            batch_states = [self.memory_states[i] for i in batch_indices]
            batch_actions = [self.memory_actions[i] for i in batch_indices]
            batch_log_probs = [self.memory_log_probs[i] for i in batch_indices]
            batch_advantages = [advantages[i] for i in batch_indices]
            batch_returns = [returns[i] for i in batch_indices]
            # batch_actor_hxs = [self.memory_actor_hxs_initial[i] for i in batch_indices] # If storing initial hxs
            # batch_critic_hxs = [self.memory_critic_hxs_initial[i] for i in batch_indices]

            # Pad sequences and create masks
            padded_states, mask = self._pad_sequences(batch_states)
            padded_actions, _ = self._pad_sequences(batch_actions)
            padded_log_probs, _ = self._pad_sequences(batch_log_probs)
            padded_advantages, _ = self._pad_sequences(batch_advantages)
            padded_returns, _ = self._pad_sequences(batch_returns)

            yield {
                "states": padded_states,
                "actions": padded_actions,
                "old_log_probs": padded_log_probs,
                "advantages": padded_advantages,
                "returns": padded_returns,
                "mask": mask,
                # "actor_hxs_initial": batch_actor_hxs, # Pass initial states if using them
                # "critic_hxs_initial": batch_critic_hxs
            }

    def __len__(self):
        """Returns the total number of transitions stored across all rollouts."""
        return self.total_samples_in_memory


# --- MODIFIED PolicyNetwork ---
class PolicyNetwork(nn.Module):
    """Actor network for PPO. Takes normalized basic_state. Optionally uses RNN."""
    def __init__(self, config: PPOConfig):
        super(PolicyNetwork, self).__init__()
        self.config = config
        self.state_dim = config.state_dim # Includes heading
        self.action_dim = config.action_dim
        self.hidden_dim = config.hidden_dim # MLP hidden dim
        self.log_std_min = config.log_std_min
        self.log_std_max = config.log_std_max
        self.use_rnn = config.use_rnn

        if self.use_rnn:
            self.rnn_hidden_size = config.rnn_hidden_size
            self.rnn_num_layers = config.rnn_num_layers
            rnn_input_dim = self.state_dim # RNN takes normalized state sequence (incl heading)
            if config.rnn_type == 'lstm':
                self.rnn = nn.LSTM(input_size=rnn_input_dim, hidden_size=self.rnn_hidden_size,
                                   num_layers=self.rnn_num_layers, batch_first=True)
            elif config.rnn_type == 'gru':
                self.rnn = nn.GRU(input_size=rnn_input_dim, hidden_size=self.rnn_hidden_size,
                                  num_layers=self.rnn_num_layers, batch_first=True)
            else: raise ValueError(f"Unsupported RNN type: {config.rnn_type}")
            mlp_input_dim = self.rnn_hidden_size # MLP takes RNN output
        else:
            self.rnn = None
            mlp_input_dim = self.state_dim # MLP takes normalized state directly (incl heading)

        # MLP layers after potential RNN
        self.fc1 = nn.Linear(mlp_input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.mean = nn.Linear(self.hidden_dim, self.action_dim)
        self.log_std = nn.Parameter(torch.zeros(1, self.action_dim)) # Learnable log_std

    def forward(self, network_input: torch.Tensor, hidden_state: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None, lengths: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]]:
        """
        Forward pass. Handles sequences and optional padding.
        network_input: (batch, seq_len, state_dim) or (batch, state_dim)
        hidden_state: Previous hidden state for RNN (used for step-by-step or BPTT start)
        lengths: Tensor of sequence lengths for packing (optional, improves efficiency)
        Returns: action_mean_sequence, action_std_sequence, final_hidden_state
        """
        if self.use_rnn and self.rnn:
            # Handle potential padding
            if lengths is not None:
                # Pack sequence -> RNN -> Unpack sequence
                # Ensure lengths are on CPU for pack_padded_sequence
                packed_input = pack_padded_sequence(network_input, lengths.cpu(), batch_first=True, enforce_sorted=False)
                packed_output, final_hidden_state = self.rnn(packed_input, hidden_state)
                rnn_output, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=network_input.size(1))
            else:
                # Process full sequence (assumes no padding or handled by mask later)
                rnn_output, final_hidden_state = self.rnn(network_input, hidden_state)
            # rnn_output shape: (batch, seq_len, rnn_hidden_size)
            mlp_input = rnn_output
        else:
            # MLP: input shape (batch, state_dim) or (batch * seq_len, state_dim)
            # If sequence, apply MLP element-wise
            if network_input.dim() == 3: # (batch, seq_len, state_dim)
                 batch, seq_len, feat_dim = network_input.shape
                 mlp_input = network_input.reshape(-1, feat_dim) # Flatten batch and seq
            else: # (batch, state_dim)
                 mlp_input = network_input
            final_hidden_state = None

        # Apply MLP layers
        x = F.relu(self.fc1(mlp_input))
        x = F.relu(self.fc2(x))
        action_mean = self.mean(x)
        action_log_std = torch.clamp(self.log_std, self.log_std_min, self.log_std_max)
        action_std = action_log_std.exp().expand_as(action_mean)

        # Reshape back if processing sequence with MLP
        if not self.use_rnn and network_input.dim() == 3:
            action_mean = action_mean.reshape(batch, seq_len, -1)
            action_std = action_std.reshape(batch, seq_len, -1)


        return action_mean, action_std, final_hidden_state

    def sample(self, network_input: torch.Tensor, hidden_state: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None) -> Tuple[torch.Tensor, torch.Tensor, Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]]:
        """ Sample action (normalized) for a single step input. Calculates log_prob. """
        # Expects single step input: (batch=1, seq_len=1, state_dim) or (batch=1, state_dim)
        if network_input.dim() == 2: # Add sequence dim if missing
            network_input = network_input.unsqueeze(1)

        mean, std, next_hidden_state = self.forward(network_input, hidden_state)
        # Squeeze seq_len dimension if it was 1
        if mean.size(1) == 1: mean = mean.squeeze(1)
        if std.size(1) == 1: std = std.squeeze(1)

        distribution = Normal(mean, std)
        x_t = distribution.sample() # Sample in unbounded space
        action_normalized = torch.tanh(x_t) # Squash to [-1, 1]

        log_prob_unbounded = distribution.log_prob(x_t)
        clamped_tanh = action_normalized.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        log_det_jacobian = torch.log(1.0 - clamped_tanh.pow(2) + 1e-7)
        log_prob = log_prob_unbounded - log_det_jacobian
        log_prob = log_prob.sum(1, keepdim=True) # Sum across action dimension

        return action_normalized, log_prob, next_hidden_state

    def evaluate(self, network_input: torch.Tensor, action_normalized: torch.Tensor, hidden_state: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None, lengths: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]]:
        """ Evaluate log_prob and entropy for a sequence of states and actions. """
        # network_input: (batch, seq_len, state_dim)
        # action_normalized: (batch, seq_len, action_dim)
        mean, std, final_hidden_state = self.forward(network_input, hidden_state, lengths)
        distribution = Normal(mean, std)

        # Inverse tanh to get the action in the unbounded space (pre-squashing)
        action_tanh = torch.clamp(action_normalized, -1.0 + 1e-6, 1.0 - 1e-6)
        action_original_space = torch.atanh(action_tanh)

        log_prob_unbounded = distribution.log_prob(action_original_space)
        log_det_jacobian = torch.log(1.0 - action_tanh.pow(2) + 1e-7)
        log_prob = log_prob_unbounded - log_det_jacobian
        log_prob = log_prob.sum(2, keepdim=True) # Sum across action dimension

        entropy = distribution.entropy().sum(2, keepdim=True) # Sum entropy across action dim

        return log_prob, entropy, final_hidden_state

    def get_initial_hidden_state(self, batch_size: int, device: torch.device) -> Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        """Return initial hidden state for RNN."""
        if not self.use_rnn: return None
        h_zeros = torch.zeros(self.rnn_num_layers, batch_size, self.rnn_hidden_size, device=device)
        if self.config.rnn_type == 'lstm':
            c_zeros = torch.zeros(self.rnn_num_layers, batch_size, self.rnn_hidden_size, device=device)
            return (h_zeros, c_zeros)
        elif self.config.rnn_type == 'gru':
            return h_zeros
        return None

# --- MODIFIED ValueNetwork ---
class ValueNetwork(nn.Module):
    """Critic network for PPO. Takes normalized basic_state sequence. Optionally uses RNN."""
    def __init__(self, config: PPOConfig):
        super(ValueNetwork, self).__init__()
        self.config = config
        self.state_dim = config.state_dim # Includes heading
        self.hidden_dim = config.hidden_dim # MLP hidden dim
        self.use_rnn = config.use_rnn

        if self.use_rnn:
            self.rnn_hidden_size = config.rnn_hidden_size
            self.rnn_num_layers = config.rnn_num_layers
            rnn_input_dim = self.state_dim # RNN takes normalized state sequence (incl heading)
            if config.rnn_type == 'lstm':
                self.rnn = nn.LSTM(input_size=rnn_input_dim, hidden_size=self.rnn_hidden_size,
                                   num_layers=self.rnn_num_layers, batch_first=True)
            elif config.rnn_type == 'gru':
                self.rnn = nn.GRU(input_size=rnn_input_dim, hidden_size=self.rnn_hidden_size,
                                  num_layers=self.rnn_num_layers, batch_first=True)
            else: raise ValueError(f"Unsupported RNN type: {config.rnn_type}")
            mlp_input_dim = self.rnn_hidden_size # MLP takes RNN output
        else:
            self.rnn = None
            mlp_input_dim = self.state_dim # MLP takes normalized state directly (incl heading)

        # MLP layers after potential RNN
        self.fc1 = nn.Linear(mlp_input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.value = nn.Linear(self.hidden_dim, 1)

    def forward(self, network_input: torch.Tensor, hidden_state: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None, lengths: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]]:
        """
        Forward pass. Handles sequences and optional padding.
        network_input: (batch, seq_len, state_dim) or (batch, state_dim)
        hidden_state: Previous hidden state for RNN
        lengths: Tensor of sequence lengths for packing (optional)
        Returns: value_sequence (or value), final_hidden_state
        """
        # Determine if input is sequential and get dimensions
        is_sequential = network_input.dim() == 3
        batch_size = network_input.shape[0]
        seq_len = network_input.shape[1] if is_sequential else 1
        final_hidden_state = None # Initialize

        if self.use_rnn and self.rnn:
            if not is_sequential:
                # This case should ideally not happen if used correctly
                # print("Warning: RNN ValueNetwork received non-sequential input.")
                # You might want to reshape or raise an error depending on expected usage
                # For now, let's assume it's intended as a single step (batch, features) -> (batch, 1, features)
                network_input = network_input.unsqueeze(1) # Add sequence dim
                is_sequential = True # Treat as sequence now
                seq_len = 1

            # Handle potential padding
            if lengths is not None:
                packed_input = pack_padded_sequence(network_input, lengths.cpu(), batch_first=True, enforce_sorted=False)
                packed_output, final_hidden_state = self.rnn(packed_input, hidden_state)
                rnn_output, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=seq_len) # Use original seq_len
            else:
                rnn_output, final_hidden_state = self.rnn(network_input, hidden_state)
            # rnn_output shape: (batch, seq_len, rnn_hidden_size)
            mlp_input = rnn_output # MLP input comes from RNN output sequence
        else: # MLP case
            if is_sequential:
                 feat_dim = network_input.shape[2]
                 # Flatten batch and sequence dims for MLP processing
                 mlp_input = network_input.reshape(batch_size * seq_len, feat_dim)
            else: # Non-sequential input
                 mlp_input = network_input # Shape (batch, state_dim)
            final_hidden_state = None

        # Apply MLP layers
        x = F.relu(self.fc1(mlp_input))
        x = F.relu(self.fc2(x))
        value = self.value(x) # Shape: (batch*seq_len, 1) if sequential, (batch, 1) otherwise

        # Reshape back if the input was sequential
        if is_sequential:
            # Reshape value from (batch*seq_len, 1) to (batch, seq_len, 1)
            value = value.reshape(batch_size, seq_len, 1)

        return value, final_hidden_state

    def get_initial_hidden_state(self, batch_size: int, device: torch.device) -> Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        """Return initial hidden state for RNN."""
        if not self.use_rnn: return None
        h_zeros = torch.zeros(self.rnn_num_layers, batch_size, self.rnn_hidden_size, device=device)
        if self.config.rnn_type == 'lstm':
            c_zeros = torch.zeros(self.rnn_num_layers, batch_size, self.rnn_hidden_size, device=device)
            return (h_zeros, c_zeros)
        elif self.config.rnn_type == 'gru':
            return h_zeros
        return None

# --- MODIFIED PPO Agent ---
class PPO:
    """Recurrent Proximal Policy Optimization algorithm implementation."""
    def __init__(self, config: PPOConfig, world_config: WorldConfig, device: torch.device = None): # Added world_config
        self.config = config
        self.gamma = config.gamma
        self.gae_lambda = config.gae_lambda
        self.policy_clip = config.policy_clip
        self.n_epochs = config.n_epochs
        self.entropy_coef = config.entropy_coef
        self.value_coef = config.value_coef
        self.state_dim = config.state_dim # Now includes heading
        self.use_rnn = config.use_rnn

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"PPO Agent using device: {self.device}")
        if self.use_rnn: print(f"PPO Agent using Recurrent ({config.rnn_type}) Networks")
        else: print(f"PPO Agent using MLP Networks")

        # State Normalization
        self.use_state_normalization = config.use_state_normalization
        self.state_normalizer = None
        if self.use_state_normalization:
             # Use the updated state_dim
            self.state_normalizer = RunningMeanStd(shape=(self.state_dim,), device=self.device)
            print(f"PPO Agent state normalization ENABLED (dim={self.state_dim}, incl. heading)") # Updated print
        else:
            print("PPO Agent state normalization DISABLED.")

        # Reward Normalization (Flag only, applied in GAE)
        self.use_reward_normalization = config.use_reward_normalization
        if self.use_reward_normalization: print("PPO Agent reward normalization ENABLED (in GAE).")
        else: print("PPO Agent reward normalization DISABLED.")

        self.actor = PolicyNetwork(config).to(self.device)
        self.critic = ValueNetwork(config).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.critic_lr)

        # Use the new Recurrent Memory
        self.memory = RecurrentPPOMemory(config=config, world_config=world_config, device=self.device)

    def select_action(self,
                      state: dict, # Contains 'basic_state' tuple (incl heading)
                      actor_hidden_state: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
                      critic_hidden_state: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
                      evaluate: bool = False
                      ) -> Tuple[float, float, float, Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]], Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]]:
        """
        Select action based on the current normalized basic state (incl heading).
        Returns: action_float, log_prob_float, value_float, next_actor_hidden, next_critic_hidden
        'evaluate=True' means deterministic action (mean), 'evaluate=False' means stochastic action (sample).
        """
        basic_state_normalized_tuple = state['basic_state'] # Includes heading
        state_tensor_normalized = torch.FloatTensor(basic_state_normalized_tuple).to(self.device) # (state_dim,)

        # Normalize state if enabled
        if self.use_state_normalization and self.state_normalizer:
            self.state_normalizer.eval() # Set to eval mode for rollout normalization
            state_tensor_for_net = self.state_normalizer.normalize(state_tensor_normalized.unsqueeze(0)).squeeze(0)
            self.state_normalizer.train() # Set back to train mode
        else:
            state_tensor_for_net = state_tensor_normalized

        # Reshape for network (batch=1, seq_len=1)
        network_input = state_tensor_for_net.unsqueeze(0).unsqueeze(0) # (1, 1, state_dim)

        self.actor.eval()
        self.critic.eval()
        next_actor_hidden, next_critic_hidden = None, None
        log_prob_tensor = torch.tensor([[0.0]], device=self.device) # Default for eval

        with torch.no_grad():
            if evaluate:
                # Get deterministic action (mean) - still need forward pass
                action_mean, _, next_actor_hidden = self.actor.forward(network_input, actor_hidden_state)
                action_normalized = torch.tanh(action_mean.squeeze(1)) # Use mean, shape (1, act_dim)
                # Critic forward pass for value (needed for return signature, even if not used directly in eval action selection)
                value_tensor, next_critic_hidden = self.critic(network_input, critic_hidden_state)
                value_tensor = value_tensor.squeeze(1) # shape (1, 1)
            else:
                # Sample action for training or stochastic evaluation
                action_normalized, log_prob_tensor, next_actor_hidden = self.actor.sample(network_input, actor_hidden_state)
                # Get value prediction
                value_tensor, next_critic_hidden = self.critic(network_input, critic_hidden_state)
                value_tensor = value_tensor.squeeze(1) # shape (1, 1)

        self.actor.train()
        self.critic.train()

        # Detach hidden states for next step
        next_actor_hidden_detached = self.memory._detach_hidden(next_actor_hidden)
        next_critic_hidden_detached = self.memory._detach_hidden(next_critic_hidden)

        action_float = action_normalized.detach().cpu().numpy()[0, 0]
        log_prob_float = log_prob_tensor.detach().cpu().numpy()[0, 0]
        value_float = value_tensor.detach().cpu().numpy()[0, 0]

        return action_float, log_prob_float, value_float, next_actor_hidden_detached, next_critic_hidden_detached

    def store_step(self, state_basic, action, log_prob, value, reward, done, actor_hxs, critic_hxs):
         """Stores a step in the recurrent memory."""
         self.memory.store(state_basic, action, log_prob, value, reward, done, actor_hxs, critic_hxs)

    def update_parameters(self, last_value: float, last_done: bool):
        """Update policy and value networks using Recurrent PPO algorithm."""
        if len(self.memory) < self.config.batch_size: # Check total samples, maybe should check num rollouts?
            # print(f"Skipping PPO update: Not enough samples ({len(self.memory)} < {self.config.batch_size} needed).")
            return None

        # --- Update State Normalizer Stats ---
        if self.use_state_normalization and self.state_normalizer:
            # Combine all states from memory for update
            # all_states_flat will now include the heading dimension
            all_states_flat = np.concatenate([s for rollout in self.memory.memory_states for s in rollout])
            if len(all_states_flat) > 0:
                 self.state_normalizer.update(torch.from_numpy(all_states_flat).to(self.device))
            else: print("Warning: No states in memory for normalizer update.")


        # --- Compute Advantages and Returns ---
        # This now happens *before* batching
        advantages_list, returns_list = self.memory.compute_advantages_returns(
            last_value, last_done, self.gamma, self.gae_lambda, self.use_reward_normalization
        )

        actor_losses, critic_losses, entropies = [], [], []

        # PPO Update Epochs
        for epoch in range(self.n_epochs):
            batch_generator = self.memory.generate_batches(advantages_list, returns_list)
            if batch_generator is None:
                 print("Warning: PPO batch generator returned None. Skipping epoch.")
                 continue

            for batch in batch_generator:
                # Get data for the current batch
                batch_states = batch["states"] # (batch_size, seq_len, state_dim) - incl heading
                batch_actions = batch["actions"] # (batch_size, seq_len, action_dim)
                batch_old_log_probs = batch["old_log_probs"] # (batch_size, seq_len, 1)
                batch_advantages = batch["advantages"] # (batch_size, seq_len, 1)
                batch_returns = batch["returns"] # (batch_size, seq_len, 1)
                mask = batch["mask"] # (batch_size, seq_len) float tensor (1.0 or 0.0)
                batch_size_actual = batch_states.shape[0] # Can be smaller for last batch

                # Normalize advantages per batch (sequence-wise mean/std might be better)
                flat_advantages = batch_advantages[mask > 0.5] # Select only valid steps
                if flat_advantages.numel() > 1:
                    batch_advantages_norm = (batch_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1e-8)
                else:
                    batch_advantages_norm = batch_advantages # Avoid division by zero

                # Normalize states if enabled (already updated normalizer)
                if self.use_state_normalization and self.state_normalizer:
                     batch_seq, T, feat = batch_states.shape
                     norm_states = self.state_normalizer.normalize(batch_states.reshape(-1, feat)).reshape(batch_seq, T, feat)
                else: norm_states = batch_states

                # Get initial hidden states for this batch
                actor_h0 = self.actor.get_initial_hidden_state(batch_size_actual, self.device)
                critic_h0 = self.critic.get_initial_hidden_state(batch_size_actual, self.device)

                # Evaluate current policy on the sequences
                # Hidden states are propagated *inside* the forward/evaluate methods
                new_log_probs, entropy, _ = self.actor.evaluate(norm_states, batch_actions, actor_h0)
                new_values, _ = self.critic(norm_states, critic_h0)

                # --- PPO Loss Calculation (Masked) ---
                mask_expanded = mask.unsqueeze(-1) # Expand mask for broadcasting: (batch, seq_len, 1)

                # Ratio
                ratio = torch.exp(new_log_probs - batch_old_log_probs)

                # Surrogate losses
                surr1 = ratio * batch_advantages_norm
                surr2 = torch.clamp(ratio, 1.0 - self.policy_clip, 1.0 + self.policy_clip) * batch_advantages_norm

                # Actor Loss (apply mask before mean)
                actor_loss = -torch.min(surr1, surr2)
                masked_actor_loss = (actor_loss * mask_expanded).sum() / mask_expanded.sum() # Average over valid steps only

                # Critic Loss (apply mask before mean)
                critic_loss = F.mse_loss(new_values, batch_returns, reduction='none')
                masked_critic_loss = (critic_loss * mask_expanded).sum() / mask_expanded.sum() # Average over valid steps

                # Entropy Loss (apply mask before mean)
                entropy_loss = -entropy
                masked_entropy_loss = (entropy_loss * mask_expanded).sum() / mask_expanded.sum() # Average over valid steps

                # Total Loss
                total_loss = masked_actor_loss + self.value_coef * masked_critic_loss + self.entropy_coef * masked_entropy_loss

                # Optimization Step
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward() # Backpropagation through time happens here
                nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.actor_optimizer.step()
                self.critic_optimizer.step()

                actor_losses.append(masked_actor_loss.item())
                critic_losses.append(masked_critic_loss.item())
                entropies.append(-masked_entropy_loss.item()) # Store positive entropy

        avg_actor_loss = np.mean(actor_losses) if actor_losses else 0.0
        avg_critic_loss = np.mean(critic_losses) if critic_losses else 0.0
        avg_entropy = np.mean(entropies) if entropies else 0.0

        self.memory.clear_memory() # Clear memory after update cycle

        return {'actor_loss': avg_actor_loss, 'critic_loss': avg_critic_loss, 'entropy': avg_entropy}

    def save_model(self, path: str):
        print(f"Saving Recurrent PPO model to {path}...")
        save_dict = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'use_state_normalization': self.use_state_normalization,
            'use_reward_normalization': self.use_reward_normalization,
            'use_rnn': self.use_rnn,
            'device_type': self.device.type
        }
        if self.use_state_normalization and self.state_normalizer:
            save_dict['state_normalizer_state_dict'] = self.state_normalizer.state_dict()
        torch.save(save_dict, path)

    def load_model(self, path: str):
        if not os.path.exists(path):
            print(f"Warning: PPO model file not found: {path}. Skipping loading.")
            return
        print(f"Loading Recurrent PPO model from {path}...")
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        loaded_use_rnn = checkpoint.get('use_rnn', False) # Default False if missing
        if 'use_rnn' in checkpoint and loaded_use_rnn != self.use_rnn:
            print(f"CRITICAL WARNING: Loaded model RNN setting ({loaded_use_rnn}) drastically differs from current config ({self.use_rnn}). Loading weights might fail or lead to unexpected behavior.")
        elif not 'use_rnn' in checkpoint and self.use_rnn:
             print(f"CRITICAL WARNING: Current config uses RNN, but checkpoint does not seem to have RNN flag. Loading weights might fail or lead to unexpected behavior.")

        try:
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        except RuntimeError as e:
             print(f"Error loading model state_dict (possibly due to RNN/state_dim mismatch): {e}") # Added state_dim hint
             print("Skipping loading model weights due to error.")
             return

        loaded_use_state_norm = checkpoint.get('use_state_normalization', False)
        if 'use_state_normalization' in checkpoint and loaded_use_state_norm != self.use_state_normalization:
            print(f"Warning: Loaded model STATE normalization setting ({loaded_use_state_norm}) differs from current config ({self.use_state_normalization}). Using current config setting.")
        if self.use_state_normalization and self.state_normalizer:
            if 'state_normalizer_state_dict' in checkpoint:
                try:
                    # Check shapes before loading
                    loaded_mean_shape = checkpoint['state_normalizer_state_dict']['mean'].shape
                    if loaded_mean_shape != self.state_normalizer.mean.shape:
                         print(f"Warning: Mismatch in loaded PPO state normalizer shape ({loaded_mean_shape}) and current ({self.state_normalizer.mean.shape}). Reinitializing normalizer.")
                    else:
                         self.state_normalizer.load_state_dict(checkpoint['state_normalizer_state_dict'])
                         print(f"Loaded PPO state normalizer statistics (dim={self.state_dim}).") # Updated print
                except Exception as e: print(f"Warning: Failed to load PPO state normalizer stats: {e}.")
            else: print("Warning: PPO state normalizer statistics not found in checkpoint.")

        loaded_use_reward_norm = checkpoint.get('use_reward_normalization', False)
        if 'use_reward_normalization' in checkpoint and loaded_use_reward_norm != self.use_reward_normalization:
             print(f"Warning: Loaded model REWARD normalization setting ({loaded_use_reward_norm}) differs from current config ({self.use_reward_normalization}). Using current config setting.")

        self.actor.train()
        self.critic.train()
        if self.use_state_normalization and self.state_normalizer:
            self.state_normalizer.train()

        print(f"Recurrent PPO model loaded successfully from {path}")


# --- MODIFIED Training Loop (train_ppo) ---
def train_ppo(config: DefaultConfig, use_multi_gpu: bool = False, run_evaluation: bool = True):
    ppo_config = config.ppo
    train_config = config.training
    world_config = config.world
    cuda_device = config.cuda_device

    # --- Log Directory and Config Saving ---
    # (Code unchanged)
    name_prefix = "ppo_rnn_" if config.ppo.use_rnn else "ppo_mlp_"
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

    # --- Device Setup ---
    # (Code unchanged)
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


    # --- Agent initialization (Pass world_config now) ---
    agent = PPO(config=ppo_config, world_config=world_config, device=device)
    os.makedirs(train_config.models_dir, exist_ok=True)

    # --- Checkpoint loading ---
    # (Code mostly unchanged, prefix logic is handled by agent.use_rnn flag)
    model_prefix = "ppo_rnn_" if agent.use_rnn else "ppo_mlp_"
    model_files = [f for f in os.listdir(train_config.models_dir) if f.startswith(model_prefix) and f.endswith(".pt")]
    latest_model_path = None
    # ... (rest of checkpoint finding logic) ...
    if model_files:
        try: latest_model_path = max([os.path.join(train_config.models_dir, f) for f in model_files], key=os.path.getmtime)
        except Exception as e: print(f"Could not find latest model: {e}")

    total_steps = 0
    start_episode = 1
    if latest_model_path and os.path.exists(latest_model_path):
        print(f"\nResuming PPO training from: {latest_model_path}")
        agent.load_model(latest_model_path)
        # Estimate total steps (less critical now, but helpful for progress bar)
        try:
            parts = os.path.basename(latest_model_path).split('_')
            step_part = next((p for p in parts if p.startswith('step')), None)
            if step_part: total_steps = int(step_part.replace('step', '').split('.')[0])
            ep_part = next((p for p in parts if p.startswith('ep')), None)
            if ep_part: start_episode = int(ep_part.replace('ep', '')) + 1
            print(f"Resuming at approx step {total_steps}, episode {start_episode}")
        except Exception as e: print(f"Warn: Could not parse file: {e}."); total_steps = 0; start_episode = 1
    else:
        print("\nStarting Recurrent PPO training from scratch.")


    # Training Loop
    episode_rewards = []
    timing_metrics = {'env_step_time': deque(maxlen=100), 'parameter_update_time': deque(maxlen=100)}
    update_frequency = ppo_config.steps_per_update # This is how often we trigger the update cycle
    pbar = tqdm(range(start_episode, train_config.num_episodes + 1),
                desc="Training Recurrent PPO", unit="episode", initial=start_episode-1, total=train_config.num_episodes)

    world = World(world_config=world_config)
    # learn_steps = 0 # Replaced by memory length check

    # Initialize reward component accumulator
    _world_reset_for_keys = world.reset()
    reward_component_accumulator = {k: [] for k in world.reward_components}
    del _world_reset_for_keys

    # --- Initialize hidden states (start of training) ---
    actor_hxs, critic_hxs = None, None
    if agent.use_rnn:
        actor_hxs = agent.actor.get_initial_hidden_state(1, device)
        critic_hxs = agent.critic.get_initial_hidden_state(1, device)

    state = world.reset() # Get initial state dict

    for episode in pbar:
        # --- Reset hidden states at the start of each episode ---
        if agent.use_rnn:
            actor_hxs = agent.actor.get_initial_hidden_state(1, device)
            critic_hxs = agent.critic.get_initial_hidden_state(1, device)

        episode_reward = 0
        episode_steps = 0
        episode_metrics = []
        episode_reward_components = {k: 0.0 for k in reward_component_accumulator}

        # Rollout phase (collect data up to steps_per_update)
        # The loop structure changes slightly - we run episodes until memory is full
        # This outer loop just controls total training duration / saving / logging
        while agent.memory.current_rollout_len < agent.memory.steps_per_update:
            # Select action using hidden states
            action, log_prob, value, next_actor_hxs, next_critic_hxs = agent.select_action(
                state, # state dict (incl basic_state with heading)
                actor_hidden_state=actor_hxs,
                critic_hidden_state=critic_hxs,
                evaluate=False # Always stochastic during training rollout
            )

            step_start_time = time.time()
            # Reward calculation happens inside world.step regardless of training flag
            next_state_dict = world.step(action, training=True, terminal_step=(episode_steps == train_config.max_steps - 1))
            step_time = time.time() - step_start_time
            timing_metrics['env_step_time'].append(step_time)

            reward = world.reward
            done = world.done
            current_metric = world.performance_metric

            # Accumulate reward components
            for key, val in world.reward_components.items():
                if key in episode_reward_components: episode_reward_components[key] += val

            # Store the transition: state is the one *before* the step
            # Pass the basic_state tuple (incl heading)
            agent.store_step(state['basic_state'], action, log_prob, value, reward, done, actor_hxs, critic_hxs)

            state = next_state_dict # state for the *next* iteration
            actor_hxs = next_actor_hxs # Use the returned next hidden states
            critic_hxs = next_critic_hxs

            episode_reward += reward
            episode_steps += 1
            total_steps += 1 # Increment total environment steps
            episode_metrics.append(current_metric)

            if done:
                # Finalize rollout even if steps_per_update not reached
                agent.memory.finalize_rollout()

                # Store episode results
                episode_rewards.append(episode_reward)
                avg_metric = np.mean(episode_metrics) if episode_metrics else 0.0
                for name, ep_total_value in episode_reward_components.items():
                    if name in reward_component_accumulator: reward_component_accumulator[name].append(ep_total_value)

                # --- Logging (End of Episode within update cycle) ---
                if episode % train_config.log_frequency == 0:
                   # Log less frequently now, maybe based on updates or total steps
                   pass # Moved logging after update step

                # Reset for next episode within the update cycle
                state = world.reset()
                if agent.use_rnn:
                    actor_hxs = agent.actor.get_initial_hidden_state(1, device)
                    critic_hxs = agent.critic.get_initial_hidden_state(1, device)
                episode_reward = 0
                episode_steps = 0
                episode_metrics = []
                episode_reward_components = {k: 0.0 for k in reward_component_accumulator}

                # Check if enough data collected AFTER episode ends
                if agent.memory.total_samples_in_memory >= agent.memory.steps_per_update:
                    break # Break inner loop to proceed to update

            # Check if enough data collected during the episode
            if agent.memory.total_samples_in_memory >= agent.memory.steps_per_update:
                 # Need value of the very last state for GAE
                 with torch.no_grad():
                      # Call select_action with evaluate=True to get value estimate without sampling
                      _, _, last_value, _, _ = agent.select_action(state, actor_hxs, critic_hxs, evaluate=True)
                 break # Break inner loop to proceed to update

        # --- Update Step ---
        # Ensure we have enough data (redundant check, but safe)
        if agent.memory.total_samples_in_memory >= agent.memory.steps_per_update:
             # Get value of the state *after* the last collected step for GAE
             with torch.no_grad():
                  # Need to get value V(s_T+1) where T = steps_per_update
                  # The 'state' variable holds the state after the last stored step
                  # Call select_action with evaluate=True to get value estimate without sampling
                  _, _, last_value, _, _ = agent.select_action(state, actor_hxs, critic_hxs, evaluate=True)
                  last_done = world.done # Use the 'done' flag from the world after the last step

             update_start_time = time.time()
             losses = agent.update_parameters(last_value, last_done)
             update_time = time.time() - update_start_time
             if losses:
                 timing_metrics['parameter_update_time'].append(update_time)
                 writer.add_scalar('Loss/Actor', losses['actor_loss'], total_steps)
                 writer.add_scalar('Loss/Critic', losses['critic_loss'], total_steps)
                 writer.add_scalar('Policy/Entropy', losses['entropy'], total_steps)
             # Memory is cleared inside update_parameters


        # --- Logging (Moved outside inner loop, happens per episode pbar iteration) ---
        current_episode_reward = episode_rewards[-1] if episode_rewards else 0.0 # Get last completed episode reward
        current_avg_metric = np.mean(episode_metrics) if episode_metrics else 0.0 # Use metrics from last completed episode part
        final_metric = world.performance_metric # Metric at the point logging happens

        if episode % train_config.log_frequency == 0:
            log_step = total_steps # Log against env steps
            if timing_metrics['env_step_time']: writer.add_scalar('Time/Environment_Step_ms_Avg100', np.mean(timing_metrics['env_step_time']) * 1000, log_step)
            if timing_metrics['parameter_update_time']: writer.add_scalar('Time/Parameter_Update_ms_Avg100', np.mean(timing_metrics['parameter_update_time']) * 1000, log_step)

            writer.add_scalar('Reward/Episode', current_episode_reward, log_step)
            writer.add_scalar('Steps/Episode', episode_steps, log_step) # Steps in the last part of episode
            writer.add_scalar('Progress/Total_Steps', total_steps, episode)
            writer.add_scalar('Performance/Metric_AvgEpPart', current_avg_metric, log_step) # Avg metric over last part
            writer.add_scalar('Performance/Metric_Current', final_metric, log_step) # Current metric
            writer.add_scalar('Buffer/PPO_Memory_Stored_Transitions', len(agent.memory), log_step)
            if world.current_seed is not None: writer.add_scalar('Environment/Current_Seed', world.current_seed, episode)

            # Log accumulated reward components
            if reward_component_accumulator:
                avg_components_logged_this_interval = {}
                for name, component_list in reward_component_accumulator.items():
                    if component_list:
                        valid_values = [v for v in component_list if not np.isnan(v)]
                        if valid_values:
                            avg_component_value = np.mean(valid_values)
                            writer.add_scalar(f'RewardComponents_AvgEp/{name}', avg_component_value, log_step)
                reward_component_accumulator = {k: [] for k in reward_component_accumulator} # Reset lists

            # Log normalizer stats
            if agent.use_state_normalization and agent.state_normalizer and agent.state_normalizer.count > agent.state_normalizer.epsilon:
                writer.add_scalar('Stats/PPO_Normalizer_Count', agent.state_normalizer.count.item(), log_step)
                writer.add_scalar('Stats/PPO_Normalizer_Mean_Sensor0', agent.state_normalizer.mean[0].item(), log_step)
                writer.add_scalar('Stats/PPO_Normalizer_Std_Sensor0', torch.sqrt(agent.state_normalizer.var[0].clamp(min=1e-8)).item(), log_step)
                if agent.state_dim > 5: # Log X coord norm
                     writer.add_scalar('Stats/PPO_Normalizer_Mean_AgentXNorm', agent.state_normalizer.mean[5].item(), log_step)
                     writer.add_scalar('Stats/PPO_Normalizer_Std_AgentXNorm', torch.sqrt(agent.state_normalizer.var[5].clamp(min=1e-8)).item(), log_step)
                if agent.state_dim > 6: # Log Y coord norm
                     writer.add_scalar('Stats/PPO_Normalizer_Mean_AgentYNorm', agent.state_normalizer.mean[6].item(), log_step)
                     writer.add_scalar('Stats/PPO_Normalizer_Std_AgentYNorm', torch.sqrt(agent.state_normalizer.var[6].clamp(min=1e-8)).item(), log_step)
                if agent.state_dim > 7: # Log Heading norm
                     writer.add_scalar('Stats/PPO_Normalizer_Mean_AgentHeadingNorm', agent.state_normalizer.mean[7].item(), log_step)
                     writer.add_scalar('Stats/PPO_Normalizer_Std_AgentHeadingNorm', torch.sqrt(agent.state_normalizer.var[7].clamp(min=1e-8)).item(), log_step)


            lookback = min(100, len(episode_rewards))
            avg_reward_100 = np.mean(episode_rewards[-lookback:]) if episode_rewards else 0
            writer.add_scalar('Reward/Average_100', avg_reward_100, log_step)

        # Update progress bar
        if episode % 10 == 0:
            lookback = min(10, len(episode_rewards))
            avg_reward_10 = np.mean(episode_rewards[-lookback:]) if episode_rewards else 0
            pbar.set_postfix({
                'avg_rew_10': f'{avg_reward_10:.2f}',
                'steps': total_steps,
                'Metric': f"{final_metric:.3f}"
            })

        # Save model periodically
        if episode % train_config.save_interval == 0 and episode > 0:
            save_prefix = "ppo_rnn" if agent.use_rnn else "ppo_mlp"
            save_path = os.path.join(train_config.models_dir, f"{save_prefix}_ep{episode}_step{total_steps}.pt")
            agent.save_model(save_path)

        # --- Early Stopping Check ---
        # (Code unchanged)
        if train_config.enable_early_stopping and len(episode_rewards) >= train_config.early_stopping_window:
            avg_reward_window = np.mean(episode_rewards[-train_config.early_stopping_window:])
            if avg_reward_window >= train_config.early_stopping_threshold:
                print(f"\nEarly stopping triggered at episode {episode}!")
                save_prefix = "ppo_rnn" if agent.use_rnn else "ppo_mlp"
                final_save_path_early = os.path.join(train_config.models_dir, f"{save_prefix}_earlystop_ep{episode}_step{total_steps}.pt")
                agent.save_model(final_save_path_early)
                print(f"Final model saved due to early stopping: {final_save_path_early}")
                break # Break outer episode loop


    pbar.close()
    writer.close()

    # Final Save
    if episode < train_config.num_episodes:
        print(f"Recurrent PPO Training finished early at episode {episode}. Total steps: {total_steps}")
    else:
        print(f"Recurrent PPO Training finished. Total steps: {total_steps}")
        save_prefix = "ppo_rnn" if agent.use_rnn else "ppo_mlp"
        final_save_path = os.path.join(train_config.models_dir, f"{save_prefix}_final_ep{train_config.num_episodes}_step{total_steps}.pt")
        if not (train_config.enable_early_stopping and 'final_save_path_early' in locals()):
             agent.save_model(final_save_path)

    if run_evaluation:
         print("\nStarting evaluation after training...")
         evaluate_ppo(agent=agent, config=config) # Pass full config

    return agent, episode_rewards


def evaluate_ppo(agent: PPO, config: DefaultConfig, model_path_for_eval: Optional[str] = None):
    eval_config = config.evaluation
    world_config = config.world
    vis_config = config.visualization
    use_stochastic_policy = eval_config.use_stochastic_policy_eval

    vis_available = False
    # visualize_world, reset_trajectories, save_gif are imported from visualization
    
    algo_name = f"ppo_{'rnn' if agent.use_rnn else 'mlp'}"

    if eval_config.render:
        try:
            from visualization import visualize_world, reset_trajectories, save_gif
            vis_available = True
            print("Visualization enabled.")

            if vis_config.output_format == 'mp4':
                try:
                    animation.FFMpegWriterName = "ffmpeg"
                    if not animation.writers.is_available(animation.FFMpegWriterName):
                        print(f"FFMpeg writer not available for PPO. Install ffmpeg and add to PATH. Falling back to GIF.")
                        current_vis_format = 'gif' # Fallback for this run
                    else:
                        print("PPO: Using FFMpeg for MP4 output.")
                        current_vis_format = 'mp4'
                except Exception as e:
                    print(f"PPO: Error checking/setting FFMpeg writer: {e}. Falling back to GIF.")
                    current_vis_format = 'gif'
            else:
                current_vis_format = 'gif'
        except ImportError:
            print("PPO: Visualization libraries (matplotlib/imageio/PIL) not found. Rendering disabled.")
            vis_available = False
            current_vis_format = 'none'
    else:
        print("PPO: Rendering disabled by config.")
        vis_available = False
        current_vis_format = 'none'

    eval_rewards = []
    eval_final_metrics = []
    success_count = 0
    all_episode_media_paths = []

    agent.actor.eval()
    agent.critic.eval()
    if agent.use_state_normalization and agent.state_normalizer:
        agent.state_normalizer.eval()

    policy_mode = "Stochastic" if use_stochastic_policy else "Deterministic"
    print(f"\nRunning PPO {'RNN' if agent.use_rnn else 'MLP'} Evaluation ({policy_mode} Policy) for {eval_config.num_episodes} episodes...")
    if model_path_for_eval: print(f"Using model: {os.path.basename(model_path_for_eval)}")

    eval_seeds = world_config.seeds
    num_eval_episodes = eval_config.num_episodes
    # ... (seed generation/adjustment logic remains the same) ...
    if len(eval_seeds) < num_eval_episodes:
         print(f"Warning: Seed count ({len(eval_seeds)}) < num_episodes ({num_eval_episodes}). Generating additional seeds.")
         eval_seeds.extend([random.randint(0, 2**32 - 1) for _ in range(num_eval_episodes - len(eval_seeds))])
    eval_seeds_to_use = eval_seeds[:num_eval_episodes]

    world = World(world_config=world_config)

    for episode in range(num_eval_episodes):
        seed_to_use = eval_seeds_to_use[episode]
        state = world.reset(seed=seed_to_use)
        episode_reward_sum = 0.0
        episode_steps = 0

        actor_hxs, critic_hxs = None, None
        if agent.use_rnn:
            actor_hxs = agent.actor.get_initial_hidden_state(1, agent.device)
            critic_hxs = agent.critic.get_initial_hidden_state(1, agent.device)

        fig, ax = None, None
        writer_mp4 = None # Explicitly for MP4 writer
        episode_png_frames = [] # For GIF mode

        if eval_config.render and vis_available:
            os.makedirs(vis_config.save_dir, exist_ok=True)
            if reset_trajectories: reset_trajectories()
            
            fig, ax = plt.subplots(figsize=vis_config.figure_size)
            
            # Determine format for *this* episode, in case of FFMpeg setup failure
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
                    print(f"PPO: Error setting up FFMpegWriter for {media_path}: {e}. Falling back to GIF for this episode.")
                    this_episode_format = 'gif' # Fallback for this episode
                    # writer_mp4 remains None
            
            # Initial frame visualization (for both GIF and MP4)
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
                print(f"PPO: Warn: Vis failed for initial state ep {episode+1}. E: {e}")


        for step in range(eval_config.max_steps):
            action_normalized, _, _, next_actor_hxs, next_critic_hxs = agent.select_action(
                state,
                actor_hidden_state=actor_hxs,
                critic_hidden_state=critic_hxs,
                evaluate=(not use_stochastic_policy)
            )
            next_state_dict = world.step(action_normalized, training=False, terminal_step=(step == eval_config.max_steps - 1))
            reward = world.reward
            done = world.done
            episode_steps += 1

            if eval_config.render and vis_available and fig is not None:
                this_episode_format_render_step = current_vis_format
                if writer_mp4 is None and this_episode_format_render_step == 'mp4': # Check if FFMpeg failed setup
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
                    print(f"PPO: Warn: Vis failed step {step+1} ep {episode+1}. E: {e}")

            state = next_state_dict
            actor_hxs = next_actor_hxs
            critic_hxs = next_critic_hxs
            episode_reward_sum += reward
            if done: break

        # --- Episode End & Media Saving ---
        if eval_config.render and vis_available:
            this_episode_format_finish = current_vis_format
            if writer_mp4 is None and this_episode_format_finish == 'mp4':
                 this_episode_format_finish = 'gif'


            if writer_mp4 and this_episode_format_finish == 'mp4':
                try:
                    writer_mp4.finish()
                    # media_path was defined when writer_mp4 was set up
                    print(f"  PPO: MP4 video saved: {media_path}") 
                    all_episode_media_paths.append(media_path)
                except Exception as e:
                    print(f"  PPO: Error finishing MP4 for ep {episode+1}: {e}")
            elif this_episode_format_finish == 'gif' and episode_png_frames:
                gif_filename = f"{algo_name}_eval_{policy_mode.lower()}_ep{episode+1}_seed{seed_to_use}.gif"
                print(f"  PPO: Saving GIF for episode {episode+1} ({policy_mode}) with {len(episode_png_frames)} frames...")
                gif_path = save_gif(output_filename=gif_filename, vis_config=vis_config, frame_paths=episode_png_frames)
                if gif_path: all_episode_media_paths.append(gif_path)
            
            if fig is not None:
                plt.close(fig)

        final_metric = world.performance_metric
        eval_rewards.append(episode_reward_sum)
        eval_final_metrics.append(final_metric)

        success = final_metric >= world_config.success_metric_threshold
        if success: success_count += 1
        status = "Success!" if success else "Failure."
        print(f"  PPO Ep {episode+1}/{num_eval_episodes} (Seed:{seed_to_use}): Steps={episode_steps}, Terminated={world.done}, Final Metric: {final_metric:.3f}, Accumulated Reward: {episode_reward_sum:.2f}. {status}")

    agent.actor.train()
    agent.critic.train()
    if agent.use_state_normalization and agent.state_normalizer:
        agent.state_normalizer.train()

    avg_eval_reward = np.mean(eval_rewards) if eval_rewards else 0.0
    std_eval_reward = np.std(eval_rewards) if eval_rewards else 0.0
    avg_eval_metric = np.mean(eval_final_metrics) if eval_final_metrics else 0.0
    std_eval_metric = np.std(eval_final_metrics) if eval_final_metrics else 0.0
    success_rate = success_count / num_eval_episodes if num_eval_episodes > 0 else 0.0

    print(f"\n--- PPO {'RNN' if agent.use_rnn else 'MLP'} Evaluation Summary ({policy_mode} Policy) ---")
    print(f"Episodes: {num_eval_episodes}")
    print(f"Average Final Metric (Point Inclusion): {avg_eval_metric:.3f} +/- {std_eval_metric:.3f}")
    print(f"Success Rate (Metric >= {world_config.success_metric_threshold:.2f}): {success_rate:.2%} ({success_count}/{num_eval_episodes})")
    print(f"Average Accumulated Episode Reward: {avg_eval_reward:.3f} +/- {std_eval_reward:.3f}")
    if eval_config.render and vis_available and all_episode_media_paths: print(f"PPO: Media saved to: '{os.path.abspath(vis_config.save_dir)}'")
    elif eval_config.render and not vis_available: print("PPO: Rendering was enabled but visualization libraries were not found.")
    elif not eval_config.render: print("PPO: Rendering disabled.")
    print(f"--- End PPO {'RNN' if agent.use_rnn else 'MLP'} Evaluation ({policy_mode} Policy) ---\n")

    return eval_rewards, success_rate, avg_eval_metric
