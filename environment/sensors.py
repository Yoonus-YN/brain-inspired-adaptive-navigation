"""
sensors.py -- Virtual Distance Sensors
=======================================
Three virtual sensors mounted on the robot:

    LEFT sensor  : looks 90deg left of robot facing
    FRONT sensor : looks straight ahead
    RIGHT sensor : looks 90deg right of robot facing

Each sensor performs ray-casting on the grid and returns
the number of free cells until it hits an obstacle or wall.

Readings are returned as:
  - Raw integer cell counts (0 = obstacle immediately adjacent)
  - Normalised float in [0, 1] (0 = obstacle, 1 = max range)

Usage:
    from environment.sensors import SensorArray
    sensors = SensorArray(max_range=10)
    readings = sensors.read(robot, world)
    # -> {'left': 2, 'front': 5, 'right': 8}
    normalised = sensors.read_normalised(robot, world)
    # -> array([0.2, 0.5, 0.8])
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from typing import Dict, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from environment.robot import Robot
    from environment.grid_world import GridWorld

from environment.robot import DIRECTION_DELTA


class SensorArray:
    """
    Three-sensor array: LEFT, FRONT, RIGHT.

    Parameters
    ----------
    max_range : int
        Maximum number of cells each sensor ray travels.
    """

    SENSOR_NAMES = ["left", "front", "right"]

    def __init__(self, max_range: int = 10):
        self.max_range = max_range

    # -- Core reading ----------------------------------------------------------

    def read(self,
             robot: "Robot",
             world: "GridWorld") -> Dict[str, int]:
        """
        Perform ray-casting for all three sensors.

        Parameters
        ----------
        robot : Robot
        world : GridWorld

        Returns
        -------
        dict with keys 'left', 'front', 'right' -> raw cell counts (int).
        """
        left_orient, front_orient, right_orient = robot.relative_directions()
        directions = {
            "left":  left_orient,
            "front": front_orient,
            "right": right_orient,
        }

        readings = {}
        for name, orient in directions.items():
            readings[name] = self._cast_ray(robot.pos, orient, world)
        return readings

    def read_normalised(self,
                        robot: "Robot",
                        world: "GridWorld") -> np.ndarray:
        """
        Read all sensors and return normalised array.

        Returns
        -------
        np.ndarray of shape (3,), values in [0, 1].
        Order: [left, front, right].
        """
        raw = self.read(robot, world)
        return np.array([
            raw["left"]  / self.max_range,
            raw["front"] / self.max_range,
            raw["right"] / self.max_range,
        ], dtype=np.float32)

    def read_all(self,
                 robot: "Robot",
                 world: "GridWorld") -> Tuple[np.ndarray, np.ndarray]:
        """
        Return both raw and normalised readings.

        Returns
        -------
        (raw_array, normalised_array) each of shape (3,).
        """
        raw_dict = self.read(robot, world)
        raw = np.array([raw_dict["left"], raw_dict["front"], raw_dict["right"]],
                       dtype=np.float32)
        return raw, raw / self.max_range

    # -- Ray casting -----------------------------------------------------------

    def _cast_ray(self,
                  pos: Tuple[int, int],
                  orientation: int,
                  world: "GridWorld") -> int:
        """
        Cast a ray from `pos` in the given orientation.

        Returns the number of free cells until obstacle/wall
        (not including the obstacle cell itself).
        """
        dr, dc = DIRECTION_DELTA[orientation]
        row, col = pos

        for dist in range(1, self.max_range + 1):
            r = row + dr * dist
            c = col + dc * dist
            if world.is_obstacle(r, c):
                return dist - 1   # cells of free space before obstacle
        return self.max_range     # ray reached max range without hitting anything

    # -- Diagnostics -----------------------------------------------------------

    def describe(self,
                 robot: "Robot",
                 world: "GridWorld") -> str:
        """Return a human-readable sensor status string."""
        raw = self.read(robot, world)
        norm = self.read_normalised(robot, world)
        lines = ["Sensor Readings:"]
        for i, name in enumerate(self.SENSOR_NAMES):
            bar = "#" * raw[name] + "." * (self.max_range - raw[name])
            lines.append(f"  {name.capitalize():6s}: {raw[name]:3d}/{self.max_range} "
                         f"|{bar}| {norm[i]:.2f}")
        return "\n".join(lines)


# -- Standalone test ------------------------------------------------------------
if __name__ == "__main__":
    from environment.grid_world import GridWorld
    from environment.robot import Robot, EAST
    from environment.obstacles import load_preset

    world = GridWorld(20, 20, seed=42)
    load_preset("moderate", world)

    robot = Robot(start=(1, 1), orientation=EAST)
    sensors = SensorArray(max_range=10)

    print(robot)
    print()
    print(sensors.describe(robot, world))
    print()
    raw, norm = sensors.read_all(robot, world)
    print(f"Raw:        {raw}")
    print(f"Normalised: {norm}")
