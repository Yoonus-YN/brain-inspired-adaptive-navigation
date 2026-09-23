"""Unit tests for the navigation module."""
import os
import tempfile
import pytest
import numpy as np

from environment.grid_world import GridWorld
from navigation.astar import AStarNavigator
from navigation.metrics import MetricsTracker, TrialResult


def test_astar_simple_path():
    world = GridWorld(10, 10)
    world.set_start((1, 1))
    world.set_goal((1, 5))

    nav = AStarNavigator()
    path, info = nav.find_path(world)

    assert info["success"] is True
    assert len(path) == 5
    assert path[0] == (1, 1)
    assert path[-1] == (1, 5)


def test_astar_with_obstacles():
    world = GridWorld(10, 10)
    world.set_start((1, 1))
    world.set_goal((3, 1))
    # Place obstacle directly between start and goal
    world.add_obstacle(2, 1)

    nav = AStarNavigator()
    path, info = nav.find_path(world)

    assert info["success"] is True
    assert (2, 1) not in path  # Should avoid the obstacle
    assert path[0] == (1, 1)
    assert path[-1] == (3, 1)


def test_astar_unreachable_goal():
    world = GridWorld(10, 10)
    world.set_start((1, 1))
    world.set_goal((8, 8))
    # Completely box in the goal
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            world.add_obstacle(8 + dx, 8 + dy)

    nav = AStarNavigator()
    path, info = nav.find_path(world)

    assert info["success"] is False
    assert path == []


def test_metrics_tracker():
    tracker = MetricsTracker(method="A*")
    tracker.record(
        trial_id=1,
        success=True,
        steps=10,
        collisions=0,
        optimal_steps=10,
        nav_time_s=0.05,
        seed=42
    )
    summary = tracker.summary()

    assert summary["n_trials"] == 1
    assert summary["success_rate"] == 1.0
    assert summary["steps_mean"] == 10.0

    # Test CSV save and reload
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "test_metrics.csv")
        tracker.save_csv(csv_path)
        assert os.path.exists(csv_path)

        loaded = MetricsTracker.load_csv(csv_path, method="A*")
        assert len(loaded.trials) == 1
        assert loaded.trials[0].steps == 10
