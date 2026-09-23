"""
robot.py -- Virtual Mobile Robot
================================
Represents the virtual robot in the 2D grid world.

The robot has:
  - Position  : (row, col)
  - Orientation: 0=NORTH, 1=EAST, 2=SOUTH, 3=WEST
  - Actions   : TURN_LEFT, MOVE_FORWARD, TURN_RIGHT

Movement convention
-------------------
  NORTH -> row decreases
  SOUTH -> row increases
  EAST  -> col increases
  WEST  -> col decreases

Usage:
    from environment.robot import Robot, Action
    robot = Robot(start=(1, 1), orientation=0)
    robot.step(Action.MOVE_FORWARD, world)
"""

from enum import IntEnum
from typing import Tuple, List, Optional, TYPE_CHECKING
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np

if TYPE_CHECKING:
    from environment.grid_world import GridWorld


# -- Actions --------------------------------------------------------------------

class Action(IntEnum):
    """Discrete actions available to the robot."""
    TURN_LEFT     = 0   # Rotate 90deg counter-clockwise, then move forward
    MOVE_FORWARD  = 1   # Move one cell in current direction
    TURN_RIGHT    = 2   # Rotate 90deg clockwise, then move forward

    @classmethod
    def names(cls) -> List[str]:
        return ["LEFT", "FORWARD", "RIGHT"]


# -- Orientation constants -------------------------------------------------------

NORTH = 0
EAST  = 1
SOUTH = 2
WEST  = 3

# Direction deltas: (Deltarow, Deltacol) for each orientation
DIRECTION_DELTA = {
    NORTH: (-1,  0),
    EAST:  ( 0, +1),
    SOUTH: (+1,  0),
    WEST:  ( 0, -1),
}

ORIENTATION_NAMES = {NORTH: "N", EAST: "E", SOUTH: "S", WEST: "W"}


# -- Robot ----------------------------------------------------------------------

class Robot:
    """
    Virtual mobile robot on a 2D grid.

    Parameters
    ----------
    start : (row, col)
        Initial position.
    orientation : int
        Initial facing direction. 0=NORTH, 1=EAST, 2=SOUTH, 3=WEST.
    """

    def __init__(self,
                 start: Tuple[int, int] = (1, 1),
                 orientation: int = EAST):
        self.start_pos   = start
        self.start_orient = orientation

        self.pos         = start          # (row, col)
        self.orientation = orientation    # 0..3
        self.collided    = False
        self.reached_goal = False

        # History for trajectory analysis
        self.trajectory: List[Tuple[int, int]] = [start]
        self.action_history: List[int] = []
        self.steps: int = 0
        self.collision_count: int = 0

    # -- Core step -------------------------------------------------------------

    def step(self,
             action: int,
             world: "GridWorld") -> Tuple[bool, bool]:
        """
        Execute one action.

        Actions are interpreted as:
          TURN_LEFT  (0) -> turn left, then attempt forward move
          MOVE_FORWARD(1) -> move forward in current direction
          TURN_RIGHT (2) -> turn right, then attempt forward move

        Parameters
        ----------
        action : int or Action
        world  : GridWorld

        Returns
        -------
        (collision, goal_reached) : (bool, bool)
        """
        action = int(action)
        self.action_history.append(action)
        self.steps += 1

        # 1. Update orientation
        if action == Action.TURN_LEFT:
            self.orientation = (self.orientation - 1) % 4
        elif action == Action.TURN_RIGHT:
            self.orientation = (self.orientation + 1) % 4
        # MOVE_FORWARD keeps orientation

        # 2. Compute target cell
        dr, dc = DIRECTION_DELTA[self.orientation]
        new_row = self.pos[0] + dr
        new_col = self.pos[1] + dc

        # 3. Collision check
        if world.is_obstacle(new_row, new_col):
            self.collided = True
            self.collision_count += 1
            self.trajectory.append(self.pos)  # stay in place
            return True, False

        # 4. Move
        self.collided = False
        self.pos = (new_row, new_col)
        self.trajectory.append(self.pos)

        # 5. Goal check
        if self.pos == world.goal:
            self.reached_goal = True
            return False, True

        return False, False

    # -- Reset -----------------------------------------------------------------

    def reset(self) -> None:
        """Reset the robot to its initial state."""
        self.pos          = self.start_pos
        self.orientation  = self.start_orient
        self.collided     = False
        self.reached_goal = False
        self.trajectory   = [self.start_pos]
        self.action_history = []
        self.steps        = 0
        self.collision_count = 0

    # -- Utility ---------------------------------------------------------------

    def facing_vector(self) -> Tuple[int, int]:
        """Return (Deltarow, Deltacol) for current orientation."""
        return DIRECTION_DELTA[self.orientation]

    def relative_directions(self) -> Tuple[int, int, int]:
        """
        Return orientations for (left_of_robot, front, right_of_robot).

        Returns
        -------
        (left_orient, front_orient, right_orient) as 0..3 values.
        """
        front = self.orientation
        left  = (self.orientation - 1) % 4
        right = (self.orientation + 1) % 4
        return left, front, right

    def distance_to_goal(self, goal: Tuple[int, int]) -> float:
        """Euclidean distance from current position to goal."""
        return float(np.sqrt((self.pos[0] - goal[0])**2 +
                              (self.pos[1] - goal[1])**2))

    def angle_to_goal(self, goal: Tuple[int, int]) -> float:
        """
        Angle (radians) between facing direction and goal direction.
        0 = facing goal directly, pi = facing away.
        """
        dr = goal[0] - self.pos[0]
        dc = goal[1] - self.pos[1]
        goal_angle = float(np.arctan2(dc, -dr))  # -dr: NORTH = up = +y

        facing_dr, facing_dc = DIRECTION_DELTA[self.orientation]
        facing_angle = float(np.arctan2(facing_dc, -facing_dr))

        diff = goal_angle - facing_angle
        # Wrap to [-pi, pi]
        diff = (diff + np.pi) % (2 * np.pi) - np.pi
        return diff

    def get_state_vector(self, goal: Tuple[int, int]) -> np.ndarray:
        """
        Return a compact state vector (used by RL agent).
        Values are NOT yet normalised -- the RL environment handles that.

        Returns
        -------
        np.ndarray of shape (3,) : [pos_row, pos_col, orientation]
        """
        return np.array([self.pos[0], self.pos[1], self.orientation],
                        dtype=np.float32)

    def __repr__(self) -> str:
        orient_str = ORIENTATION_NAMES[self.orientation]
        return (f"Robot(pos={self.pos}, facing={orient_str}, "
                f"steps={self.steps}, collisions={self.collision_count})")


if __name__ == "__main__":
    from environment.grid_world import GridWorld

    world = GridWorld(10, 10, seed=42)
    robot = Robot(start=(1, 1), orientation=EAST)
    print("Initial:", robot)
    coll, goal = robot.step(Action.MOVE_FORWARD, world)
    print("After forward:", robot, f"collision={coll}, goal={goal}")
    robot.step(Action.TURN_RIGHT, world)
    print("After turn right:", robot)
