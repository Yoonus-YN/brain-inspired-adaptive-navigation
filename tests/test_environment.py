"""Unit tests for the environment module."""
import numpy as np
import pytest

from environment.grid_world import GridWorld, FREE, OBSTACLE
from environment.robot import Robot, Action, NORTH, EAST, SOUTH, WEST
from environment.sensors import SensorArray
from environment.obstacles import ObstacleGenerator, load_preset


def test_grid_world_initialization():
    world = GridWorld(10, 10, seed=42)
    assert world.width == 10
    assert world.height == 10
    # Outer boundaries should be obstacles
    for x in range(10):
        assert world.is_obstacle(0, x)
        assert world.is_obstacle(9, x)
    for y in range(10):
        assert world.is_obstacle(y, 0)
        assert world.is_obstacle(y, 9)
    # Center should be free
    assert world.is_free(5, 5)


def test_grid_world_start_goal():
    world = GridWorld(10, 10)
    world.set_start((1, 1))
    world.set_goal((8, 8))
    assert world.start == (1, 1)
    assert world.goal == (8, 8)
    assert world.start == (1, 1)


def test_robot_movement_and_turning():
    world = GridWorld(10, 10)
    robot = Robot(start=(2, 2), orientation=EAST)
    assert robot.orientation == EAST
    assert robot.pos == (2, 2)

    # Turn left from EAST -> NORTH
    collision, goal = robot.step(Action.TURN_LEFT, world)
    assert robot.orientation == NORTH
    assert not collision

    # Step forward NORTH -> (1, 2)
    collision, goal = robot.step(Action.MOVE_FORWARD, world)
    assert robot.pos == (0, 2) or robot.pos == (1, 2)
    assert robot.steps == 2


def test_sensor_array_reading():
    world = GridWorld(10, 10)
    world.set_start((1, 1))
    world.set_goal((8, 8))
    robot = Robot(start=(1, 1), orientation=EAST)
    sensors = SensorArray(max_range=8)

    readings = sensors.read(robot, world)
    assert "left" in readings
    assert "front" in readings
    assert "right" in readings

    norm_readings = sensors.read_normalised(robot, world)
    assert isinstance(norm_readings, np.ndarray)
    assert len(norm_readings) == 3
    for nr in norm_readings:
        assert 0.0 <= nr <= 1.0


def test_obstacle_generator_and_presets():
    world = GridWorld(15, 15, seed=123)
    world.set_start((1, 1))
    world.set_goal((13, 13))
    gen = ObstacleGenerator(seed=123)
    added = gen.random_obstacles(world, n=5)
    assert len(added) > 0

    preset_world = GridWorld(15, 15, seed=42)
    load_preset("sparse", preset_world)
    assert preset_world.width == 15
    assert preset_world.height == 15
    assert preset_world.start is not None
    assert preset_world.goal is not None
