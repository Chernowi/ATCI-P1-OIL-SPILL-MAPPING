import numpy as np
import math
from typing import List, Tuple, Optional
from world_objects import Location, OilSpillCircle
from configs import MapperConfig
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import warnings

class Mapper:
    """Estimates the oil spill shape based on sensor measurements."""
    def __init__(self, config: MapperConfig):
        self.config = config
        self.oil_points: List[Location] = []
        self.water_points: List[Location] = []
        self.estimated_spill: Optional[OilSpillCircle] = None

    def reset(self):
        """Clears stored points and estimate."""
        self.oil_points = []
        self.water_points = []
        self.estimated_spill = None

    def add_measurement(self, sensor_location: Location, is_oil: bool):
        """Adds a single sensor measurement."""
        if is_oil:
            self.oil_points.append(sensor_location)
        else:
            self.water_points.append(sensor_location)

    def _minimum_enclosing_circle(self, points: np.ndarray) -> Tuple[Tuple[float, float], float]:
        """
        Finds the minimum enclosing circle for a set of 2D points.
        Uses Welzl's algorithm implicitly via standard library optimization.
        Returns center (x, y) and radius.
        """
        if points.shape[0] == 0:
            return (0.0, 0.0), 0.0
        if points.shape[0] == 1:
            return (points[0, 0], points[0, 1]), 0.0

        # Objective function: radius squared
        def objective(params):
            cx, cy = params
            center = np.array([cx, cy])
            distances_sq = np.sum((points - center)**2, axis=1)
            return np.max(distances_sq) # Minimize max squared distance (radius^2)

        # Initial guess: centroid of the points
        centroid = np.mean(points, axis=0)
        initial_guess = centroid

        # Bounds (optional but can help) - based on data range
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        bounds = [(min_coords[0], max_coords[0]), (min_coords[1], max_coords[1])]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning) # Ignore potential overflow/underflow
            result = minimize(objective, initial_guess, method='Nelder-Mead', options={'adaptive': True}) # Simpler method

        if result.success:
            center_x, center_y = result.x
            radius_sq = result.fun
            radius = np.sqrt(max(0, radius_sq)) # Ensure non-negative radius
            return (center_x, center_y), radius
        else:
            # Fallback if optimization fails: use centroid and max distance
            print("Warning: Minimum enclosing circle optimization failed. Using fallback.")
            centroid = np.mean(points, axis=0)
            distances = np.sqrt(np.sum((points - centroid)**2, axis=1))
            radius = np.max(distances) if distances.size > 0 else 0.0
            return (centroid[0], centroid[1]), radius

    def estimate_spill(self):
        """
        Estimates the spill shape (circle) based on collected points.
        Updates self.estimated_spill.
        """
        if len(self.oil_points) < self.config.min_oil_points_for_estimate:
            # Not enough data, keep previous estimate or None
            return

        oil_points_np = np.array([[p.x, p.y] for p in self.oil_points])

        try:
            # Option 1: Minimum Enclosing Circle of oil points
            center_coords, radius = self._minimum_enclosing_circle(oil_points_np)

            # Optional Refinement using water points (simple version):
            # Ensure the circle doesn't contain too many water points far inside.
            # This part can be made more sophisticated.
            if len(self.water_points) >= self.config.min_water_points_for_refinement:
                 water_points_np = np.array([[p.x, p.y] for p in self.water_points])
                 center_np = np.array(center_coords)
                 distances_to_water = np.sqrt(np.sum((water_points_np - center_np)**2, axis=1))
                 # Find water points inside the current estimate
                 water_inside_indices = np.where(distances_to_water < radius - 1e-6)[0] # Add tolerance
                 if len(water_inside_indices) > 0:
                      # Simple shrink: Reduce radius to just exclude the furthest water point inside
                      max_dist_water_inside = np.max(distances_to_water[water_inside_indices])
                      radius = max(0.1, max_dist_water_inside) # Shrink radius, ensure minimum size

            est_center = Location(x=center_coords[0], y=center_coords[1])
            self.estimated_spill = OilSpillCircle(center=est_center, radius=max(0.1, radius)) # Ensure min radius

        except Exception as e:
            print(f"Error during spill estimation: {e}")
            # Keep previous estimate or None if error occurs
            pass

