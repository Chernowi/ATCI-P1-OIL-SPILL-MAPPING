import numpy as np
import torch
import math

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

        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, dim=0, unbiased=False) # Use population variance for updates
        batch_count = torch.tensor(x.shape[0], dtype=torch.float32, device=self.device)

        delta = batch_mean - self.mean # Already on self.device
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        # Combine variances using parallel algorithm
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + torch.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        # Update state (already on self.device)
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """ Normalizes the data """
        # Ensure input tensor is on the same device
        x = x.to(self.device)
        # Note: Does not update mean/std
        result = (x - self.mean) / torch.sqrt(self.var + self.epsilon)
        return torch.clamp(result, -10.0, 10.0) # Clamp to avoid extreme values

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


def calculate_iou_circles(center1, radius1, center2, radius2) -> float:
    """Calculates the Intersection over Union (IoU) for two circles."""
    # Ensure Location objects are handled correctly
    x1 = center1.x
    y1 = center1.y
    x2 = center2.x
    y2 = center2.y
    r1 = float(radius1)
    r2 = float(radius2)

    # Ensure radii are non-negative
    r1 = max(0.0, r1)
    r2 = max(0.0, r2)

    d_sq = (x1 - x2)**2 + (y1 - y2)**2
    d = math.sqrt(d_sq)

    # Handle edge cases: identical circles, zero radius, containment, no overlap
    if d_sq < 1e-9 and abs(r1 - r2) < 1e-9:
        return 1.0 if r1 > 1e-9 else 0.0 # Identical circles (IoU 1) or two zero-radius points (IoU 0 unless r1=r2=0)
    if r1 == 0.0 and r2 == 0.0:
        return 0.0 # Two distinct zero-radius points

    if d >= r1 + r2 - 1e-9 : # Use tolerance for floating point
        return 0.0  # No overlap

    if d <= abs(r1 - r2) + 1e-9:
        # One circle contains the other
        min_r = min(r1, r2)
        max_r = max(r1, r2)
        if max_r == 0: return 1.0 # Should have been caught by identical check, but for safety
        area_min = math.pi * min_r**2
        area_max = math.pi * max_r**2
        return area_min / area_max if area_max > 0 else 1.0 # IoU is ratio of areas

    # Calculate intersection area for overlapping circles
    try:
        # Clamp arguments for acos to avoid domain errors due to floating point inaccuracies
        arg1 = max(-1.0, min(1.0, (d_sq + r1**2 - r2**2) / (2 * d * r1)))
        arg2 = max(-1.0, min(1.0, (d_sq + r2**2 - r1**2) / (2 * d * r2)))

        alpha = math.acos(arg1)
        beta = math.acos(arg2)
    except ValueError:
        # Should ideally not happen with clamping and prior checks, but as fallback:
        print(f"Warning: acos domain error persists in IoU. d={d}, r1={r1}, r2={r2}")
        # Re-check conditions with tolerance
        if d >= r1 + r2 - 1e-7: return 0.0
        if d <= abs(r1 - r2) + 1e-7:
             min_r, max_r = min(r1, r2), max(r1, r2)
             area_min = math.pi * min_r**2
             area_max = math.pi * max_r**2
             return area_min / area_max if area_max > 0 else 1.0
        return 0.0 # Fallback if still failing

    intersection_area = (r1**2 * alpha - 0.5 * r1**2 * math.sin(2 * alpha) +
                         r2**2 * beta - 0.5 * r2**2 * math.sin(2 * beta))

    # Ensure intersection area is non-negative
    intersection_area = max(0.0, intersection_area)

    # Calculate union area
    area1 = math.pi * r1**2
    area2 = math.pi * r2**2
    union_area = area1 + area2 - intersection_area

    if union_area <= 1e-9: # Avoid division by zero
        # If union is zero, means both areas are zero. If intersection>0 (shouldn't happen), IoU is 1? Assume 0.
        return 1.0 if intersection_area > 1e-9 and abs(area1)<1e-9 and abs(area2)<1e-9 else 0.0


    iou = intersection_area / union_area
    return max(0.0, min(1.0, iou)) # Clamp final result