"""
environment.py -- Gymnasium-Compatible RL Environment
=====================================================
Wraps the BIAN-SNN grid world into a standard Gymnasium
(OpenAI Gym) environment interface.

State space (5 dimensions, all normalised to [0, 1]):
    [d_L, d_F, d_R, dist_goal, angle_goal_norm]
    d_L, d_F, d_R : normalised sensor readings
    dist_goal      : Euclidean distance to goal / max_distance
    angle_goal_norm: (angle + pi) / (2pi)   -> [0, 1]

Action space: Discrete(3)
    0 = TURN_LEFT then move forward
    1 = MOVE_FORWARD
    2 = TURN_RIGHT then move forward

Usage:
    from reinforcement_learning.environment import NavEnv
    env = NavEnv()
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(action)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from environment.grid_world import GridWorld
from environment.robot import Robot, EAST
from environment.sensors import SensorArray
from environment.obstacles import load_preset, ObstacleGenerator
from reinforcement_learning.rewards import RewardFunction, RewardConfig


class NavEnv(gym.Env):
    """
    Navigation RL environment.

    Parameters
    ----------
    grid_size    : int   -- grid dimensions (square)
    n_obstacles  : int   -- number of random obstacles per episode
    max_steps    : int   -- step limit per episode
    preset       : str   -- obstacle preset ('random', 'sparse', etc.)
                          'random' regenerates obstacles each episode
    seed         : int   -- base random seed
    reward_config: RewardConfig, optional
    render_mode  : str or None -- 'human' for matplotlib rendering
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self,
                 grid_size:     int            = 20,
                 n_obstacles:   int            = 15,
                 max_steps:     int            = 500,
                 preset:        str            = "random",
                 seed:          int            = 42,
                 reward_config: Optional[RewardConfig] = None,
                 render_mode:   Optional[str]  = None):
        super().__init__()
        self.grid_size   = grid_size
        self.n_obstacles = n_obstacles
        self.max_steps   = max_steps
        self.preset      = preset
        self.base_seed   = seed
        self.render_mode = render_mode

        self.reward_fn = RewardFunction(reward_config)
        self.sensors   = SensorArray(max_range=10)

        # -- Spaces --
        # Observation: 5 floats in [0, 1]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(5,), dtype=np.float32)

        # Action: LEFT=0, FORWARD=1, RIGHT=2
        self.action_space = spaces.Discrete(3)

        # Internals (initialised in reset)
        self.world:   Optional[GridWorld] = None
        self.robot:   Optional[Robot]     = None
        self._step_count = 0
        self._episode_seed = seed
        self._rng = np.random.default_rng(seed)

        # Max possible distance (diagonal of grid)
        self._max_dist = float(np.sqrt(2) * grid_size)

    # -- Gym interface ---------------------------------------------------------

    def reset(self,
              seed:    Optional[int]  = None,
              options: Optional[Dict] = None
              ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._episode_seed = seed
            self._rng = np.random.default_rng(seed)

        # Build world
        self.world = GridWorld(self.grid_size, self.grid_size,
                               seed=int(self._rng.integers(0, 9999)))
        start = (1, 1)
        goal  = (self.grid_size - 2, self.grid_size - 2)
        self.world.set_start(start)
        self.world.set_goal(goal)

        if self.preset == "random":
            gen = ObstacleGenerator(seed=self.world.seed)
            gen.random_obstacles(self.world, n=self.n_obstacles)
        else:
            load_preset(self.preset, self.world)

        self.robot = Robot(start=start, orientation=EAST)
        self._step_count = 0

        obs = self._get_obs()
        return obs, self._get_info()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        assert self.robot is not None and self.world is not None, \
            "Call reset() before step()."

        prev_dist = self.robot.distance_to_goal(self.world.goal)
        collision, goal_reached = self.robot.step(action, self.world)
        curr_dist = self.robot.distance_to_goal(self.world.goal)

        reward = self.reward_fn.compute(
            collision  = collision,
            goal       = goal_reached,
            prev_dist  = prev_dist,
            curr_dist  = curr_dist,
        )

        self._step_count += 1
        terminated = goal_reached or collision
        truncated  = self._step_count >= self.max_steps

        obs  = self._get_obs()
        info = self._get_info()
        info.update({
            "collision":     collision,
            "goal_reached":  goal_reached,
            "steps":         self._step_count,
        })

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(6, 6))
            self.world.render(robot_pos=self.robot.pos,
                              path=self.robot.trajectory,
                              ax=ax, show=False)
            plt.tight_layout()
            plt.pause(0.05)
            plt.close(fig)

    def close(self):
        import matplotlib
        matplotlib.pyplot.close("all")

    # -- Observation builder ---------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        """Build the 5-dimensional observation vector."""
        _, sensor_norm = self.sensors.read_all(self.robot, self.world)
        dist = self.robot.distance_to_goal(self.world.goal) / self._max_dist
        angle = self.robot.angle_to_goal(self.world.goal)
        angle_norm = (angle + np.pi) / (2 * np.pi)

        return np.array([
            sensor_norm[0],  # d_L
            sensor_norm[1],  # d_F
            sensor_norm[2],  # d_R
            float(np.clip(dist, 0.0, 1.0)),
            float(np.clip(angle_norm, 0.0, 1.0)),
        ], dtype=np.float32)

    def _get_info(self) -> Dict[str, Any]:
        return {
            "robot_pos":   self.robot.pos if self.robot else None,
            "step_count":  self._step_count,
        }


if __name__ == "__main__":
    env = NavEnv(grid_size=20, n_obstacles=15, max_steps=200, seed=42)
    obs, info = env.reset()
    print("Observation space:", env.observation_space)
    print("Action space:     ", env.action_space)
    print("Initial obs:      ", obs)

    total_reward = 0.0
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        total_reward += reward
        print(f"  step={step+1}  action={action}  reward={reward:+.2f}  "
              f"pos={info['robot_pos']}  done={term or trunc}")
        if term or trunc:
            break

    print(f"\nTotal reward: {total_reward:.2f}")
    env.close()
