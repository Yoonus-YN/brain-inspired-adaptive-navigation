"""
astar.py -- A* Pathfinding Algorithm
=====================================
Implements A* on the 2D grid world using Manhattan distance
as the heuristic.

This provides the conventional navigation baseline against
which SNN and RL approaches are compared.

A* core equation:
    f(n) = g(n) + h(n)
    g(n) = cost from start to current node
    h(n) = estimated cost from n to goal (Manhattan)
    f(n) = total estimated cost

Usage:
    from navigation.astar import AStarNavigator
    nav = AStarNavigator()
    path, info = nav.find_path(world, start=(1,1), goal=(18,18))
    # path -> list of (row, col) from start to goal, or [] if no path
    # info -> dict with search statistics
"""

import heapq
import time
from typing import List, Tuple, Dict, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from environment.grid_world import GridWorld


# -- Priority queue entry -------------------------------------------------------

class _PQEntry:
    """Comparison wrapper for the priority queue."""
    __slots__ = ("f", "pos")

    def __init__(self, f: float, pos: Tuple[int, int]):
        self.f   = f
        self.pos = pos

    def __lt__(self, other: "_PQEntry") -> bool:
        return self.f < other.f


# -- A* Navigator --------------------------------------------------------------

class AStarNavigator:
    """
    A* pathfinder for the 2D grid world.

    Parameters
    ----------
    allow_diagonal : bool
        If True, diagonal moves are allowed (cost sqrt2).
        Default False -- 4-directional grid movement only.
    """

    # 4-directional neighbours
    _NEIGHBOURS_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # 8-directional (diagonal)
    _NEIGHBOURS_8 = _NEIGHBOURS_4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    def __init__(self, allow_diagonal: bool = False):
        self.allow_diagonal = allow_diagonal
        self._neighbours = (self._NEIGHBOURS_8 if allow_diagonal
                            else self._NEIGHBOURS_4)

    # -- Public API ------------------------------------------------------------

    def find_path(self,
                  world: "GridWorld",
                  start: Optional[Tuple[int, int]] = None,
                  goal:  Optional[Tuple[int, int]] = None
                  ) -> Tuple[List[Tuple[int, int]], Dict]:
        """
        Run A* from `start` to `goal`.

        Parameters
        ----------
        world : GridWorld
        start : (row, col) -- defaults to world.start
        goal  : (row, col) -- defaults to world.goal

        Returns
        -------
        path : list of (row, col) from start to goal (inclusive).
               Empty list if no path exists.
        info : dict with search statistics.
        """
        start = start or world.start
        goal  = goal  or world.goal

        t0 = time.perf_counter()

        # g[n] -- cost from start
        g: Dict[Tuple[int, int], float] = {start: 0.0}

        # parent map for path reconstruction
        parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}

        # open set as min-heap of (f, PQEntry)
        open_heap: List[_PQEntry] = []
        heapq.heappush(open_heap, _PQEntry(self._h(start, goal), start))

        closed: set = set()
        nodes_expanded = 0

        while open_heap:
            entry = heapq.heappop(open_heap)
            current = entry.pos

            if current in closed:
                continue
            closed.add(current)
            nodes_expanded += 1

            # -- Goal reached --
            if current == goal:
                path = self._reconstruct(parent, goal)
                elapsed = time.perf_counter() - t0
                return path, {
                    "success":        True,
                    "path_length":    len(path) - 1,
                    "nodes_expanded": nodes_expanded,
                    "time_s":         elapsed,
                }

            # -- Expand neighbours --
            for dr, dc in self._neighbours:
                nb = (current[0] + dr, current[1] + dc)

                if nb in closed or world.is_obstacle(nb[0], nb[1]):
                    continue

                move_cost = 1.414 if (dr != 0 and dc != 0) else 1.0
                tentative_g = g[current] + move_cost

                if tentative_g < g.get(nb, float("inf")):
                    g[nb]      = tentative_g
                    parent[nb] = current
                    f = tentative_g + self._h(nb, goal)
                    heapq.heappush(open_heap, _PQEntry(f, nb))

        # -- No path found --
        elapsed = time.perf_counter() - t0
        return [], {
            "success":        False,
            "path_length":    0,
            "nodes_expanded": nodes_expanded,
            "time_s":         elapsed,
        }

    def find_next_action(self,
                         world: "GridWorld",
                         robot_pos: Tuple[int, int],
                         robot_orient: int,
                         goal: Optional[Tuple[int, int]] = None
                         ) -> int:
        """
        Return the Action index (0=LEFT, 1=FORWARD, 2=RIGHT) for the
        next step toward the goal.

        This enables A* to control the robot step-by-step using
        the same action interface as SNN and RL.

        Parameters
        ----------
        world        : GridWorld
        robot_pos    : (row, col)
        robot_orient : int (NORTH=0, EAST=1, SOUTH=2, WEST=3)
        goal         : (row, col) -- defaults to world.goal

        Returns
        -------
        int -- Action index.
        """
        from environment.robot import DIRECTION_DELTA, NORTH, EAST, SOUTH, WEST

        goal = goal or world.goal
        path, info = self.find_path(world, start=robot_pos, goal=goal)

        if not info["success"] or len(path) < 2:
            return 1  # FORWARD (stuck -- best default)

        next_cell = path[1]
        dr = next_cell[0] - robot_pos[0]
        dc = next_cell[1] - robot_pos[1]

        # Determine required orientation for next cell
        target_orient = None
        for orient, (odr, odc) in DIRECTION_DELTA.items():
            if odr == dr and odc == dc:
                target_orient = orient
                break

        if target_orient is None:
            return 1  # FORWARD fallback

        turn = (target_orient - robot_orient) % 4
        if turn == 0:
            return 1   # FORWARD
        elif turn == 1:
            return 2   # TURN_RIGHT
        elif turn == 3:
            return 0   # TURN_LEFT
        else:
            # 180deg -- turn right (arbitrary choice)
            return 2

    # -- Heuristic -------------------------------------------------------------

    @staticmethod
    def _h(pos: Tuple[int, int], goal: Tuple[int, int]) -> float:
        """Manhattan distance heuristic (admissible for 4-directional grids)."""
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

    # -- Path reconstruction ---------------------------------------------------

    @staticmethod
    def _reconstruct(parent: Dict, goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        path = []
        node: Optional[Tuple[int, int]] = goal
        while node is not None:
            path.append(node)
            node = parent[node]
        path.reverse()
        return path


# -- Standalone test ------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from environment.grid_world import GridWorld
    from environment.obstacles import load_preset

    world = GridWorld(20, 20, seed=42)
    load_preset("moderate", world)

    nav = AStarNavigator()
    path, info = nav.find_path(world)

    print("A* Result:")
    for k, v in info.items():
        print(f"  {k}: {v}")
    if path:
        print(f"  path preview: {path[:5]} ...")

    world.render(path=path, title="A* Path")
    plt.tight_layout()
    plt.show()
