"""
visualize_live.py -- Live Visual Navigation Demo
================================================
Shows step-by-step visual animation in the terminal of the robot navigating
the grid world using either A*, SNN, or RL.

Usage:
    python experiments/visualize_live.py --method astar
    python experiments/visualize_live.py --method snn
    python experiments/visualize_live.py --method rl
"""

import sys
import os
import time
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from environment.grid_world import GridWorld
from environment.robot import Robot, Action, EAST
from environment.sensors import SensorArray
from environment.obstacles import load_preset
from navigation.astar import AStarNavigator
from neuroscience.snn_controller import SNNController
from reinforcement_learning.environment import NavEnv
from reinforcement_learning.agent import DQNAgent


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def render_live_frame(world: GridWorld, robot: Robot, sensors: SensorArray, step: int, action_name: str, method: str):
    clear_screen()
    print("=" * 60)
    print(f"  BIAN-SNN LIVE VISUAL NAVIGATION DEMO [{method.upper()}]")
    print("=" * 60)
    print(f"Step: {step:3d}  | Action: {action_name:10s} | Pos: {robot.pos} | Collisions: {robot.collision_count}")
    print("-" * 60)
    
    # ASCII Grid
    ascii_grid = world.to_ascii(robot_pos=robot.pos)
    print(ascii_grid)
    print("-" * 60)
    
    # Sensors
    print(sensors.describe(robot, world))
    print("=" * 60)
    print("Legend: [R] Robot  [G] Goal  [#] Wall/Obstacle  [.] Free Space")


def run_live_astar(delay: float = 0.25):
    world = GridWorld(15, 15, seed=42)
    world.set_start((1, 1))
    world.set_goal((13, 13))
    load_preset("sparse", world)
    
    robot = Robot(start=world.start, orientation=EAST)
    sensors = SensorArray(max_range=10)
    
    nav = AStarNavigator()
    path, info = nav.find_path(world, start=world.start, goal=world.goal)
    
    if not info["success"]:
        print("A* could not find a path to goal.")
        return

    render_live_frame(world, robot, sensors, step=0, action_name="START", method="A*")
    time.sleep(delay)
    
    for i in range(1, len(path)):
        target_pos = path[i]
        # Move robot directly to next cell in path
        robot.pos = target_pos
        robot.trajectory.append(target_pos)
        robot.steps += 1
        
        reached = (robot.pos == world.goal)
        action_str = f"STEP -> {target_pos}"
        render_live_frame(world, robot, sensors, step=i, action_name=action_str, method="A*")
        
        if reached:
            print("\n*** GOAL REACHED SUCCESSFULLY BY A*! ***\n")
            break
        time.sleep(delay)


def run_live_snn(max_steps: int = 50, delay: float = 0.25):
    world = GridWorld(15, 15, seed=42)
    world.set_start((1, 1))
    world.set_goal((13, 13))
    load_preset("sparse", world)
    
    robot = Robot(start=world.start, orientation=EAST)
    sensors = SensorArray(max_range=10)
    controller = SNNController(seed=42)
    
    action_names = ["TURN_LEFT", "FORWARD", "TURN_RIGHT"]
    
    render_live_frame(world, robot, sensors, step=0, action_name="START", method="SNN")
    time.sleep(delay)
    
    for step in range(1, max_steps + 1):
        norm_readings = sensors.read_normalised(robot, world)
        action, motor_spikes = controller.decide(norm_readings)
        coll, goal_reached = robot.step(action, world)
        
        act_name = f"{action_names[action]} (spikes={motor_spikes.astype(int)})"
        render_live_frame(world, robot, sensors, step=step, action_name=act_name, method="SNN")
        
        if goal_reached:
            print("\n*** GOAL REACHED BY SNN! ***\n")
            break
        if coll:
            print("\n[!] Collision detected with wall/obstacle.\n")
            time.sleep(delay * 2)
            break
            
        time.sleep(delay)


def run_live_rl(max_steps: int = 50, delay: float = 0.25):
    env = NavEnv(grid_size=15, n_obstacles=8, max_steps=max_steps, seed=42)
    obs, info = env.reset()
    
    agent = DQNAgent(state_dim=5, action_dim=3)
    checkpoint_path = os.path.join(os.path.dirname(__file__), "..", "reinforcement_learning", "checkpoints", "dqn_nav.pth")
    if os.path.exists(checkpoint_path):
        agent.load(checkpoint_path)
        print(f"Loaded trained DQN model from {checkpoint_path}")
    else:
        print("Using initialized DQN agent (no pre-trained checkpoint found).")
        
    action_names = ["TURN_LEFT", "FORWARD", "TURN_RIGHT"]
    
    render_live_frame(env.world, env.robot, env.sensors, step=0, action_name="START", method="RL")
    time.sleep(delay)
    
    for step in range(1, max_steps + 1):
        action = agent.select_action(obs, evaluate=True)
        obs, reward, term, trunc, info = env.step(action)
        
        act_name = f"{action_names[action]} (r={reward:+.1f})"
        render_live_frame(env.world, env.robot, env.sensors, step=step, action_name=act_name, method="RL")
        
        if info.get("goal_reached"):
            print("\n*** GOAL REACHED BY RL! ***\n")
            break
        if info.get("collision"):
            print("\n[!] Collision detected.\n")
            time.sleep(delay * 2)
            break
        if term or trunc:
            break
            
        time.sleep(delay)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live visual navigation demo.")
    parser.add_argument("--method", choices=["astar", "snn", "rl"], default="astar",
                        help="Navigation method to visualize (astar, snn, or rl)")
    parser.add_argument("--delay", type=float, default=0.25,
                        help="Seconds delay between steps (default: 0.25)")
    args = parser.parse_args()
    
    if args.method == "astar":
        run_live_astar(delay=args.delay)
    elif args.method == "snn":
        run_live_snn(delay=args.delay)
    elif args.method == "rl":
        run_live_rl(delay=args.delay)
