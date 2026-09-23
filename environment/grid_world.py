"""
grid_world.py -- 2D Grid World Environment
==========================================
The core simulation environment for BIAN-SNN.

Grid cell types:
    0  = FREE space
    1  = OBSTACLE / WALL
    2  = ROBOT position (for rendering only)
    3  = GOAL position  (for rendering only)

Coordinate convention:
    (row, col) -- row 0 is the TOP of the grid.
    Directions: NORTH = row-1, SOUTH = row+1,
                WEST  = col-1, EAST  = col+1.

Usage:
    from environment.grid_world import GridWorld
    world = GridWorld(width=20, height=20, seed=42)
    world.place_obstacles(n=15)
    world.set_start((1, 1))
    world.set_goal((18, 18))
    world.render()
"""

from typing import List, Tuple, Optional
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

# Cell type constants
FREE     = 0
OBSTACLE = 1
ROBOT    = 2
GOAL     = 3


class GridWorld:
    """
    Lightweight 2D grid-based navigation environment.

    Parameters
    ----------
    width : int
        Number of columns in the grid.
    height : int
        Number of rows in the grid.
    seed : int, optional
        Random seed for reproducible obstacle placement.
    """

    def __init__(self, width: int = 20, height: int = 20, seed: int = 42):
        self.width  = width
        self.height = height
        self.seed   = seed
        self.rng    = np.random.default_rng(seed)

        # Base grid -- all free initially
        self._grid: np.ndarray = np.zeros((height, width), dtype=np.int8)

        # Build border walls
        self._build_walls()

        # Start / goal (may be overridden)
        self.start: Tuple[int, int] = (1, 1)
        self.goal:  Tuple[int, int] = (height - 2, width - 2)

        # Track all obstacle cell positions (excluding border)
        self._obstacle_cells: List[Tuple[int, int]] = []

    # -- Construction ----------------------------------------------------------

    def _build_walls(self) -> None:
        """Set the border cells to OBSTACLE."""
        self._grid[0, :]  = OBSTACLE
        self._grid[-1, :] = OBSTACLE
        self._grid[:, 0]  = OBSTACLE
        self._grid[:, -1] = OBSTACLE

    def place_obstacles(self, n: int = 15,
                        forbidden: Optional[List[Tuple[int, int]]] = None) -> None:
        """
        Randomly place `n` obstacle cells inside the grid.

        Parameters
        ----------
        n : int
            Number of obstacles to place.
        forbidden : list of (row, col), optional
            Cells that must remain free (e.g. start, goal).
        """
        if forbidden is None:
            forbidden = [self.start, self.goal]

        inner_cells = [
            (r, c)
            for r in range(1, self.height - 1)
            for c in range(1, self.width - 1)
            if (r, c) not in forbidden
        ]

        chosen = self.rng.choice(len(inner_cells),
                                 size=min(n, len(inner_cells)),
                                 replace=False)
        self._obstacle_cells = []
        for idx in chosen:
            r, c = inner_cells[idx]
            self._grid[r, c] = OBSTACLE
            self._obstacle_cells.append((r, c))

    def add_obstacle(self, row: int, col: int) -> None:
        """Manually add a single obstacle cell."""
        if self._in_bounds(row, col):
            self._grid[row, col] = OBSTACLE
            self._obstacle_cells.append((row, col))

    def remove_obstacle(self, row: int, col: int) -> None:
        """Remove an obstacle cell (make it free)."""
        if self._in_bounds(row, col):
            self._grid[row, col] = FREE
            if (row, col) in self._obstacle_cells:
                self._obstacle_cells.remove((row, col))

    def set_start(self, pos: Tuple[int, int]) -> None:
        """Set the robot start position."""
        self.start = pos

    def set_goal(self, pos: Tuple[int, int]) -> None:
        """Set the goal position."""
        self.goal = pos

    def reset(self) -> None:
        """Reset to free grid and rebuild walls (keeps start/goal/seed)."""
        self._grid = np.zeros((self.height, self.width), dtype=np.int8)
        self._build_walls()
        self._obstacle_cells = []

    # -- Queries ---------------------------------------------------------------

    def is_obstacle(self, row: int, col: int) -> bool:
        """Return True if (row, col) is an obstacle or out of bounds."""
        if not self._in_bounds(row, col):
            return True
        return bool(self._grid[row, col] == OBSTACLE)

    def is_free(self, row: int, col: int) -> bool:
        """Return True if (row, col) is traversable."""
        return not self.is_obstacle(row, col)

    def get_grid(self) -> np.ndarray:
        """Return a copy of the raw grid array."""
        return self._grid.copy()

    def get_render_grid(self, robot_pos: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Return grid copy with robot and goal marked for visualisation.

        Parameters
        ----------
        robot_pos : (row, col), optional
            Current robot position.  Uses self.start if None.
        """
        grid = self._grid.copy()
        gr, gc = self.goal
        grid[gr, gc] = GOAL
        rr, rc = robot_pos if robot_pos else self.start
        grid[rr, rc] = ROBOT
        return grid

    def _in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.height and 0 <= col < self.width

    # -- Rendering -------------------------------------------------------------

    def render(self,
               robot_pos: Optional[Tuple[int, int]] = None,
               path: Optional[List[Tuple[int, int]]] = None,
               title: str = "BIAN-SNN Grid World",
               ax: Optional[plt.Axes] = None,
               show: bool = True) -> plt.Axes:
        """
        Render the grid world using Matplotlib.

        Parameters
        ----------
        robot_pos : (row, col), optional
        path : list of (row, col), optional
            Sequence of cells to draw as a path line.
        title : str
        ax : plt.Axes, optional
            Existing axes to draw on.
        show : bool
            Whether to call plt.show().

        Returns
        -------
        plt.Axes
        """
        grid = self.get_render_grid(robot_pos)

        cmap = ListedColormap([
            "#F0F4F8",   # 0 FREE     -- light grey-blue
            "#2D3A4A",   # 1 OBSTACLE -- dark slate
            "#00BFA5",   # 2 ROBOT    -- teal
            "#FFB300",   # 3 GOAL     -- amber
        ])

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
            fig.patch.set_facecolor("#1A2332")

        ax.set_facecolor("#1A2332")
        ax.imshow(grid, cmap=cmap, vmin=0, vmax=3,
                  interpolation="nearest", aspect="equal")

        # Grid lines
        ax.set_xticks(np.arange(-0.5, self.width, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, self.height, 1), minor=True)
        ax.grid(which="minor", color="#2D3A4A", linewidth=0.5)
        ax.tick_params(which="both", bottom=False, left=False,
                       labelbottom=False, labelleft=False)

        # Draw path
        if path and len(path) > 1:
            path_rows = [p[0] for p in path]
            path_cols = [p[1] for p in path]
            ax.plot(path_cols, path_rows,
                    color="#FF6B6B", linewidth=2.0,
                    linestyle="--", alpha=0.85, zorder=3)

        # Robot label
        rr, rc = robot_pos if robot_pos else self.start
        ax.text(rc, rr, "R", ha="center", va="center",
                fontsize=12, zorder=5)

        # Goal label
        gr, gc = self.goal
        ax.text(gc, gr, "G", ha="center", va="center",
                fontsize=12, zorder=5)

        # Legend
        legend_patches = [
            mpatches.Patch(color="#F0F4F8", label="Free"),
            mpatches.Patch(color="#2D3A4A", label="Obstacle"),
            mpatches.Patch(color="#00BFA5", label="Robot"),
            mpatches.Patch(color="#FFB300", label="Goal"),
        ]
        if path:
            legend_patches.append(
                mpatches.Patch(color="#FF6B6B", label="Path"))

        ax.legend(handles=legend_patches,
                  loc="upper right", framealpha=0.8,
                  fontsize=8, facecolor="#2D3A4A",
                  labelcolor="white")

        ax.set_title(title, color="white", fontsize=13, pad=10)
        return ax

    # -- String representation -------------------------------------------------

    def __repr__(self) -> str:
        return (f"GridWorld(width={self.width}, height={self.height}, "
                f"seed={self.seed}, obstacles={len(self._obstacle_cells)})")

    def to_ascii(self, robot_pos: Optional[Tuple[int, int]] = None) -> str:
        """Return an ASCII art string of the grid."""
        lines = []
        grid = self.get_render_grid(robot_pos)
        char_map = {FREE: ".", OBSTACLE: "#", ROBOT: "R", GOAL: "G"}
        for row in grid:
            lines.append(" ".join(char_map[c] for c in row))
        return "\n".join(lines)


# -- Standalone test ------------------------------------------------------------
if __name__ == "__main__":
    world = GridWorld(width=20, height=20, seed=42)
    world.place_obstacles(n=15)
    world.set_start((1, 1))
    world.set_goal((18, 18))

    print(world)
    print(world.to_ascii())

    world.render(title="Grid World Test", show=False)
    plt.tight_layout()
    out_dir = os.path.join(os.path.dirname(__file__), "..", "results", "figures")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "grid_world_demo.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight", facecolor="#1A2332")
    print(f"Saved demo figure -> {out_path}")
    plt.close("all")
