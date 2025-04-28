import math
from typing import Tuple

class Velocity():
    """Represents the velocity of an object in 2D space."""
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def is_moving(self) -> bool:
        """Check if any velocity component is non-zero."""
        return self.x != 0 or self.y != 0

    def get_heading(self) -> float:
        """Calculate the heading angle in radians."""
        return math.atan2(self.y, self.x)

    def __str__(self) -> str:
        """String representation of velocity."""
        return f"Vel:(vx:{self.x:.2f}, vy:{self.y:.2f})"

class Location():
    """Represents the location of an object in 2D space."""
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def update(self, velocity: Velocity, dt: float = 1.0):
        """Update location based on velocity and time step."""
        self.x += velocity.x * dt
        self.y += velocity.y * dt

    def distance_to(self, other_loc: 'Location') -> float:
        """Calculate Euclidean distance to another location."""
        return math.sqrt((self.x - other_loc.x)**2 + (self.y - other_loc.y)**2)

    def __str__(self) -> str:
        """String representation of location."""
        return f"Pos:(x:{self.x:.2f}, y:{self.y:.2f})"

class Object():
    """Represents a generic object with location and velocity in 2D."""
    def __init__(self, location: Location, velocity: Velocity = None, name: str = None):
        self.name = name if name else "Unnamed Object"
        self.location = location
        self.velocity = velocity if velocity is not None else Velocity(0.0, 0.0)

    def update_position(self, dt: float = 1.0):
        """Update the object's position based on its velocity and time step."""
        if self.velocity and self.velocity.is_moving():
            self.location.update(self.velocity, dt)

    def get_heading(self) -> float:
         """Get the current heading based on velocity."""
         return self.velocity.get_heading()

    def __str__(self) -> str:
        """String representation of the object."""
        name_str = f"{self.name}: "
        return f"{name_str}{self.location}, {self.velocity}"

# Represent the spill as a simple circle for now
class OilSpillCircle:
    """Represents the true oil spill as a circle."""
    def __init__(self, center: Location, radius: float):
        self.center = center
        self.radius = radius

    def is_inside(self, point: Location) -> bool:
        """Check if a given Location is inside the spill."""
        return self.center.distance_to(point) <= self.radius

    def __str__(self) -> str:
        return f"OilSpill(Center={self.center}, Radius={self.radius:.2f})"

