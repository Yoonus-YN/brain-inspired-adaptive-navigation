"""
obstacles.py -- Obstacle Generation Utilities
=============================================
Helper functions for creating structured and random obstacle layouts
in the grid world.

Usage:
    from environment.obstacles import ObstacleGenerator
    gen = ObstacleGenerator(seed=42)
    gen.rectangular_block(world, top_left=(5,5), width=3, height=4)
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from typing import List, Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from environment.grid_world import GridWorld


class ObstacleGenerator:
    """
    Generates obstacle configurations for a GridWorld.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def random_obstacles(self,
                         world: "GridWorld",
                         n: int = 15,
                         forbidden: Optional[List[Tuple[int, int]]] = None
                         ) -> List[Tuple[int, int]]:
        """
        Place `n` randomly-positioned obstacle cells in the world.

        Parameters
        ----------
        world : GridWorld
            The world to add obstacles to.
        n : int
            Number of obstacles.
        forbidden : list of (row, col), optional
            Cells that must stay free.

        Returns
        -------
        list of (row, col) -- placed obstacle positions.
        """
        if forbidden is None:
            forbidden = [world.start, world.goal]

        inner = [
            (r, c)
            for r in range(1, world.height - 1)
            for c in range(1, world.width - 1)
            if (r, c) not in forbidden and not world.is_obstacle(r, c)
        ]

        n = min(n, len(inner))
        indices = self.rng.choice(len(inner), size=n, replace=False)
        placed = []
        for idx in indices:
            r, c = inner[idx]
            world.add_obstacle(r, c)
            placed.append((r, c))
        return placed

    def rectangular_block(self,
                           world: "GridWorld",
                           top_left: Tuple[int, int],
                           width: int,
                           height: int) -> List[Tuple[int, int]]:
        """
        Place a solid rectangular block of obstacles.

        Parameters
        ----------
        world : GridWorld
        top_left : (row, col)
        width : int -- number of columns
        height : int -- number of rows

        Returns
        -------
        list of (row, col) placed positions.
        """
        r0, c0 = top_left
        placed = []
        for r in range(r0, r0 + height):
            for c in range(c0, c0 + width):
                if world._in_bounds(r, c):
                    world.add_obstacle(r, c)
                    placed.append((r, c))
        return placed

    def corridor_maze(self, world: "GridWorld") -> List[Tuple[int, int]]:
        """
        Create a simple corridor maze pattern -- horizontal walls
        with alternating gaps.

        Returns
        -------
        list of (row, col) placed positions.
        """
        placed = []
        gap_col_even = world.width - 2
        gap_col_odd  = 1

        for r in range(3, world.height - 2, 4):
            for c in range(1, world.width - 1):
                # Even walls have gap on the RIGHT
                gap = gap_col_even if (r // 4) % 2 == 0 else gap_col_odd
                if c != gap and world.is_free(r, c):
                    if (r, c) != world.start and (r, c) != world.goal:
                        world.add_obstacle(r, c)
                        placed.append((r, c))
        return placed


# Pre-built environment configurations for quick experiments
def load_preset(name: str, world: "GridWorld") -> None:
    """
    Load a named obstacle preset into the world.

    Presets
    -------
    'empty'     : No obstacles (border walls only)
    'sparse'    : 10 random obstacles
    'moderate'  : 20 random obstacles
    'dense'     : 35 random obstacles
    'corridor'  : Corridor maze pattern
    'blocks'    : Three rectangular blocks
    """
    gen = ObstacleGenerator(seed=world.seed)

    if name == "empty":
        pass  # no obstacles

    elif name == "sparse":
        gen.random_obstacles(world, n=10)

    elif name == "moderate":
        gen.random_obstacles(world, n=20)

    elif name == "dense":
        gen.random_obstacles(world, n=35)

    elif name == "corridor":
        gen.corridor_maze(world)

    elif name == "blocks":
        gen.rectangular_block(world, top_left=(4, 4),  width=3, height=4)
        gen.rectangular_block(world, top_left=(4, 12), width=3, height=4)
        gen.rectangular_block(world, top_left=(12, 8), width=4, height=3)

    else:
        raise ValueError(f"Unknown preset: '{name}'. "
                         f"Choose from: empty, sparse, moderate, dense, "
                         f"corridor, blocks")


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from environment.grid_world import GridWorld

    presets = ["empty", "sparse", "moderate", "corridor", "blocks"]
    fig, axes = plt.subplots(1, len(presets), figsize=(20, 4))
    fig.patch.set_facecolor("#1A2332")

    for ax, name in zip(axes, presets):
        w = GridWorld(20, 20, seed=42)
        load_preset(name, w)
        w.render(title=name.capitalize(), ax=ax, show=False)

    plt.suptitle("Obstacle Presets", color="white", fontsize=14)
    plt.tight_layout()
    out_dir = os.path.join(os.path.dirname(__file__), "..", "results", "figures")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "obstacle_presets.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight", facecolor="#1A2332")
    print(f"Saved presets figure -> {out_path}")
    plt.close("all")
