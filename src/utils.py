import numpy as np
import torch
import math
from typing import Tuple # Added

class RunningMeanStd:
    # Based on https://github.com/openai/baselines/blob/master/baselines/common/running_mean_std.py
    # Added tensor support and device handling
    def __init__(self, epsilon=1e-4, shape=(), device=None):
        """
        Calculates the running mean and std of a data stream.
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

        :param epsilon: helps with arithmetic issues
        :param shape: the shape of the data stream's output
        :param device: The PyTorch device (e.g., 'cpu', 'cuda:0') to store tensors on.
        """
        self.device = device or torch.device("cpu") # Default to CPU if no device specified
        self.mean = torch.zeros(shape, dtype=torch.float32, device=self.device)
        self.var = torch.ones(shape, dtype=torch.float32, device=self.device)
        self.count = torch.tensor(epsilon, dtype=torch.float32, device=self.device)
        self.epsilon = epsilon
        self._is_eval = False # Track mode

    def update(self, x: torch.Tensor):
        """ Adds the incoming data to the running mean and std """
        if self._is_eval: return # Do not update in eval mode

        # Ensure input tensor is on the same device
        x = x.to(self.device)
        if x.dim() == 1: # Handle single instance update
             x = x.unsqueeze(0)
        if x.shape[0] == 0: return # Handle empty batch

        batch_mean = torch.mean(x, dim=0)
        # Use population variance (unbiased=False) for stability in updates
        # Using unbiased=True can lead to issues if batch size is 1
        batch_var = torch.var(x, dim=0, unbiased=False)
        batch_count = torch.tensor(x.shape[0], dtype=torch.float32, device=self.device)

        delta = batch_mean - self.mean # Already on self.device
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        # Combine variances using parallel algorithm
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + torch.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        # Ensure variance doesn't become negative due to floating point errors
        new_var = torch.clamp(new_var, min=0.0)

        # Update state (already on self.device)
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """ Normalizes the data """
        # Ensure input tensor is on the same device
        x = x.to(self.device)
        # Use variance clamped slightly above zero for numerical stability
        normalized_x = (x - self.mean) / torch.sqrt(self.var + self.epsilon)
        # Clamp result to prevent extreme values which can destabilize training
        return torch.clamp(normalized_x, -10.0, 10.0)

    def state_dict(self):
        # Ensure tensors are moved to CPU for saving, common practice
        return {'mean': self.mean.cpu(), 'var': self.var.cpu(), 'count': self.count.cpu()}

    def load_state_dict(self, state_dict):
        # Load tensors and move them back to the correct device
        self.mean = state_dict['mean'].to(self.device)
        self.var = state_dict['var'].to(self.device)
        self.count = state_dict['count'].to(self.device)

    def eval(self):
         self._is_eval = True

    def train(self):
         self._is_eval = False
