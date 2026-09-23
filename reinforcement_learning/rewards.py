"""
rewards.py -- Reward Function
=============================
Defines the reward signal for reinforcement learning navigation.

Event                 | Reward
--------------------- | -------
Reach goal            | +100
Collision             | -100
Move toward goal      | +2
Move safely (no event)| +1
Move away from goal   | -1

All values are initial experimental parameters -- adjust them
in config.py or override per experiment.

Usage:
    from reinforcement_learning.rewards import RewardFunction
    rf = RewardFunction()
    reward = rf.compute(collision=False, goal=True, prev_dist=5.0, curr_dist=4.0)
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class RewardConfig:
    """Reward shaping parameters."""
    goal:      float = +100.0
    collision: float = -100.0
    toward:    float = +2.0
    safe:      float = +1.0
    away:      float = -1.0
    time_penalty: float = -0.1   # small penalty per step to encourage speed


class RewardFunction:
    """
    Computes navigation reward for each environment step.

    Parameters
    ----------
    config : RewardConfig, optional
        Override default reward values.
    """

    def __init__(self, config: Optional[RewardConfig] = None):
        self.cfg = config or RewardConfig()

    def compute(self,
                collision:  bool,
                goal:       bool,
                prev_dist:  float,
                curr_dist:  float) -> float:
        """
        Compute step reward.

        Parameters
        ----------
        collision : bool  -- did robot collide this step?
        goal      : bool  -- did robot reach the goal?
        prev_dist : float -- distance to goal before step
        curr_dist : float -- distance to goal after step

        Returns
        -------
        float -- reward value
        """
        if goal:
            return self.cfg.goal

        if collision:
            return self.cfg.collision

        # Distance-based shaping
        if curr_dist < prev_dist:
            return self.cfg.toward + self.cfg.time_penalty
        elif curr_dist > prev_dist:
            return self.cfg.away + self.cfg.time_penalty
        else:
            return self.cfg.safe + self.cfg.time_penalty

    def describe(self) -> str:
        lines = [
            "Reward Configuration:",
            f"  Goal reached  : {self.cfg.goal:+.1f}",
            f"  Collision     : {self.cfg.collision:+.1f}",
            f"  Toward goal   : {self.cfg.toward:+.1f}",
            f"  Safe step     : {self.cfg.safe:+.1f}",
            f"  Away from goal: {self.cfg.away:+.1f}",
            f"  Time penalty  : {self.cfg.time_penalty:+.2f}",
        ]
        return "\n".join(lines)
