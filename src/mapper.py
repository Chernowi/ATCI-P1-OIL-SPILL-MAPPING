import numpy as np
import math
from typing import List, Tuple, Optional, Any
from world_objects import Location
from configs import MapperConfig
from scipy.spatial import ConvexHull, Delaunay # Import Delaunay for point-in-hull check
import warnings

class Mapper:
    """
    Estimates the oil spill shape using Convex Hull based on sensor locations
    that detected oil.
    """
    def __init__(self, config: MapperConfig):
        self.config = config
        self.oil_sensor_locations: List[Location] = [] # Locations of sensors that detected oil
        self.water_sensor_locations: List[Location] = [] # Locations of sensors that detected only water
        self.estimated_hull: Optional[ConvexHull] = None
        self.hull_vertices: Optional[np.ndarray] = None # Store vertices [[x1,y1], [x2,y2], ...]

    def reset(self):
        """Clears stored points and estimate."""
        self.oil_sensor_locations = []
        self.water_sensor_locations = []
        self.estimated_hull = None
        self.hull_vertices = None

    def add_measurement(self, sensor_location: Location, is_oil_detected: bool):
        """Adds the location of a sensor based on its detection status."""
        if is_oil_detected:
            # Avoid adding duplicate locations precisely
            if not any(abs(p.x - sensor_location.x) < 1e-6 and abs(p.y - sensor_location.y) < 1e-6 for p in self.oil_sensor_locations):
                self.oil_sensor_locations.append(sensor_location)
        else:
             if not any(abs(p.x - sensor_location.x) < 1e-6 and abs(p.y - sensor_location.y) < 1e-6 for p in self.water_sensor_locations):
                self.water_sensor_locations.append(sensor_location)


    def estimate_spill(self):
        """
        Estimates the spill shape (convex hull) based on collected oil sensor locations.
        Updates self.estimated_hull and self.hull_vertices.
        """
        self.estimated_hull = None # Reset estimate before trying
        self.hull_vertices = None

        if len(self.oil_sensor_locations) < self.config.min_oil_points_for_estimate:
            # Not enough data to form a meaningful hull
            return

        # Convert locations to numpy array for ConvexHull
        oil_points_np = np.array([[p.x, p.y] for p in self.oil_sensor_locations])

        # Ensure we have enough unique points for hull calculation
        unique_oil_points = np.unique(oil_points_np, axis=0)
        if unique_oil_points.shape[0] < 3:
            # ConvexHull requires at least 3 non-collinear points
            # print("Mapper: Not enough unique points for hull.")
            return

        try:
            # Compute the convex hull
            # Use warnings context manager to suppress Qhull precision warnings sometimes seen
            with warnings.catch_warnings():
                warnings.simplefilter("ignore") # Suppress Qhull precision warnings
                hull = ConvexHull(unique_oil_points, qhull_options='QJ') # QJ ensures non-extreme points are included if collinear

            self.estimated_hull = hull
            # Get the vertices of the hull in order
            self.hull_vertices = unique_oil_points[hull.vertices]

        except Exception as e:
            # This can happen with collinear points or other Qhull errors
            # print(f"Mapper: Error during Convex Hull computation: {e}")
            self.estimated_hull = None
            self.hull_vertices = None
            pass # Keep estimate as None if error occurs

    def is_inside_estimate(self, point: Location) -> bool:
        """
        Checks if a given point (Location object) is inside the estimated convex hull.
        Uses Delaunay triangulation method for robustness.
        Returns True if inside or on the boundary, False otherwise.
        """
        if self.estimated_hull is None or self.hull_vertices is None or len(self.hull_vertices) < 3:
            return False # Cannot check if no valid hull exists

        point_np = np.array([point.x, point.y])

        # Use Delaunay triangulation of the hull vertices to check point inclusion
        # This is generally more robust than using equations directly for edge cases
        try:
            # Create Delaunay triangulation from the *hull* vertices
            # Need at least 3 points for Delaunay
            if len(self.hull_vertices) < 3:
                 return False

            # Add a small tolerance to Qhull options if needed
            # For example, 'QJ' (joggle input) can help with precision issues
            delaunay_hull = Delaunay(self.hull_vertices, qhull_options='QJ')

            # find_simplex returns -1 if point is outside the triangulation
            return delaunay_hull.find_simplex(point_np) >= 0
        except Exception as e:
             # Errors can occur if points are perfectly collinear after QJ, etc.
             # print(f"Mapper: Delaunay check failed: {e}")
             return False # Assume outside if check fails

    def get_estimated_area(self) -> float:
        """Returns the area of the estimated convex hull. Returns 0 if no hull."""
        if self.estimated_hull is None:
            return 0.0
        try:
            return self.estimated_hull.volume # In 2D, 'volume' attribute gives area
        except AttributeError:
            return 0.0 # Should have volume attribute if hull exists