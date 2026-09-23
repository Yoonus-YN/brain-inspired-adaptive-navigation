"""Unit tests for the reinforcement learning module."""
import numpy as np
import pytest

from reinforcement_learning.environment import NavEnv
from reinforcement_learning.rewards import RewardFunction, RewardConfig
from reinforcement_learning.agent import DQNAgent


def test_reward_calculator():
    cfg = RewardConfig(goal=100.0, collision=-50.0, toward=2.0, away=-1.0, time_penalty=-0.1)
    rf = RewardFunction(config=cfg)

    # Moving closer to goal
    r1 = rf.compute(collision=False, goal=False, prev_dist=5.0, curr_dist=4.0)
    assert r1 > 0  # toward bonus + safe - time_penalty

    # Collision reward
    r_coll = rf.compute(collision=True, goal=False, prev_dist=5.0, curr_dist=5.0)
    assert r_coll == -50.0

    # Goal reached
    r_goal = rf.compute(collision=False, goal=True, prev_dist=1.0, curr_dist=0.0)
    assert r_goal == 100.0


def test_gym_env():
    env = NavEnv(grid_size=10, n_obstacles=3, max_steps=50, seed=42)

    obs, info = env.reset()
    assert isinstance(obs, np.ndarray)
    assert len(obs) == 5

    # Step in environment
    next_obs, reward, terminated, truncated, info = env.step(action=0)
    assert isinstance(next_obs, np.ndarray)
    assert isinstance(reward, (int, float))
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "collision" in info
    assert "goal_reached" in info


def test_dqn_agent_step():
    agent = DQNAgent(state_dim=5, action_dim=3, lr=1e-3, gamma=0.99,
                     epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.99,
                     buffer_capacity=1000, batch_size=16)

    state = np.zeros(5, dtype=np.float32)
    action = agent.select_action(state)
    assert action in [0, 1, 2]

    # Store transitions
    for _ in range(32):
        s = np.random.rand(5).astype(np.float32)
        a = np.random.randint(0, 3)
        r = float(np.random.randn())
        s_next = np.random.rand(5).astype(np.float32)
        done = False
        agent.store(s, a, r, s_next, done)

    # Perform a training step
    loss = agent.train_step()
    assert loss is not None
    assert loss >= 0.0
