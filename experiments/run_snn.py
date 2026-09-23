"""
run_snn.py -- SNN Navigation Experiments
=========================================
Runs the SNN controller on multiple environments and records metrics.

Output:
  results/tables/snn_results.csv
  results/figures/snn_trajectories.png
  results/figures/snn_spike_activity.png

Usage:
    python experiments/run_snn.py
    python experiments/run_snn.py --trials 30 --window 50
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm

from config import (GRID_WIDTH, GRID_HEIGHT, NUM_OBSTACLES,
                    RANDOM_SEED, NUM_TRIALS, FIGURES_DIR, TABLES_DIR,
                    SNN_WINDOW_MS, SNN_HIDDEN_NEURONS)
from environment.grid_world import GridWorld
from environment.robot import Robot, EAST
from environment.sensors import SensorArray
from environment.obstacles import ObstacleGenerator
from navigation.astar import AStarNavigator   # for path efficiency reference
from navigation.metrics import MetricsTracker, EpisodeTimer
from neuroscience.snn_controller import SNNController


def run_snn_experiment(
        n_trials:    int   = NUM_TRIALS,
        grid_size:   int   = GRID_WIDTH,
        n_obstacles: int   = NUM_OBSTACLES,
        max_steps:   int   = 500,
        base_seed:   int   = RANDOM_SEED,
        window_ms:   float = SNN_WINDOW_MS,
        n_hidden:    int   = SNN_HIDDEN_NEURONS,
        verbose:     bool  = False,
) -> MetricsTracker:
    """
    Run SNN controller on `n_trials` environments.

    Returns MetricsTracker with all trial results.
    """
    tracker  = MetricsTracker(method="SNN")
    nav      = AStarNavigator()   # used only for optimal path reference
    sensors  = SensorArray(max_range=10)
    ctrl     = SNNController(n_hidden=n_hidden, window_ms=window_ms,
                              seed=base_seed)

    for trial in tqdm(range(n_trials), desc="SNN trials", unit="trial"):
        seed  = base_seed + trial
        world = GridWorld(grid_size, grid_size, seed=seed)
        start = (1, 1)
        goal  = (grid_size - 2, grid_size - 2)
        world.set_start(start)
        world.set_goal(goal)
        gen = ObstacleGenerator(seed=seed)
        gen.random_obstacles(world, n=n_obstacles)

        # Optimal reference
        _, opt_info = nav.find_path(world, start, goal)
        optimal_steps = opt_info.get("path_length", 0) if opt_info["success"] else 0

        robot = Robot(start=start, orientation=EAST)
        ctrl.reset_stats()

        with EpisodeTimer() as timer:
            for _ in range(max_steps):
                _, sensor_norm = sensors.read_all(robot, world)
                action, _ = ctrl.decide(sensor_norm)
                collision, goal_reached = robot.step(action, world)
                if collision or goal_reached:
                    break

        tracker.record(
            trial_id      = trial,
            success       = robot.reached_goal,
            steps         = robot.steps,
            collisions    = robot.collision_count,
            optimal_steps = optimal_steps,
            nav_time_s    = timer.elapsed_s,
            spike_count   = ctrl.total_spikes,
            seed          = seed,
            env_config    = f"{grid_size}x{grid_size}_obs{n_obstacles}",
        )

        if verbose:
            print(f"Trial {trial:3d}: success={robot.reached_goal}, "
                  f"steps={robot.steps}, "
                  f"spikes={ctrl.total_spikes}, "
                  f"collisions={robot.collision_count}")

    return tracker


def plot_sample_trajectories(n_samples: int = 6,
                              grid_size: int = GRID_WIDTH,
                              n_obstacles: int = NUM_OBSTACLES,
                              base_seed: int = RANDOM_SEED) -> None:
    """Plot sample SNN navigation trajectories."""
    sensors = SensorArray(max_range=10)
    ctrl    = SNNController(seed=base_seed)

    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor("#1A2332")
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.3)

    for i in range(n_samples):
        seed  = base_seed + i
        world = GridWorld(grid_size, grid_size, seed=seed)
        world.set_start((1, 1))
        world.set_goal((grid_size - 2, grid_size - 2))
        gen = ObstacleGenerator(seed=seed)
        gen.random_obstacles(world, n=n_obstacles)

        robot = Robot(start=(1, 1), orientation=EAST)
        ctrl.reset_stats()

        for _ in range(500):
            _, sensor_norm = sensors.read_all(robot, world)
            action, _ = ctrl.decide(sensor_norm)
            collision, goal = robot.step(action, world)
            if collision or goal:
                break

        ax = fig.add_subplot(gs[i // 3, i % 3])
        status = "[OK]" if robot.reached_goal else "[X]"
        world.render(robot_pos=robot.pos,
                     path=robot.trajectory,
                     title=f"Trial {i+1} {status}  steps={robot.steps}",
                     ax=ax, show=False)

    plt.suptitle("SNN Controller -- Sample Trajectories",
                 color="white", fontsize=14, y=1.01)

    os.makedirs(FIGURES_DIR, exist_ok=True)
    out = os.path.join(FIGURES_DIR, "snn_trajectories.png")
    plt.savefig(out, dpi=120, bbox_inches="tight", facecolor="#1A2332")
    print(f"Saved -> {out}")
    plt.close("all")


def plot_spike_activity(base_seed: int = RANDOM_SEED) -> None:
    """Visualise SNN spike activity on one trial."""
    sensors = SensorArray(max_range=10)
    ctrl    = SNNController(seed=base_seed)

    world = GridWorld(GRID_WIDTH, GRID_HEIGHT, seed=base_seed)
    world.set_start((1, 1))
    world.set_goal((GRID_WIDTH - 2, GRID_HEIGHT - 2))
    gen = ObstacleGenerator(seed=base_seed)
    gen.random_obstacles(world, n=NUM_OBSTACLES)

    robot = Robot(start=(1, 1), orientation=EAST)

    spike_history   = []
    sensor_history  = []
    action_history  = []

    for _ in range(100):
        _, sensor_norm = sensors.read_all(robot, world)
        action, motor_spikes = ctrl.decide(sensor_norm)
        collision, goal = robot.step(action, world)
        spike_history.append(motor_spikes.copy())
        sensor_history.append(sensor_norm.copy())
        action_history.append(action)
        if collision or goal:
            break

    spike_history  = np.array(spike_history)
    sensor_history = np.array(sensor_history)

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    fig.patch.set_facecolor("#1A2332")
    steps = np.arange(len(spike_history))

    # Motor spike counts
    ax = axes[0]
    ax.set_facecolor("#1A2332")
    for i, (name, col) in enumerate(
            zip(["LEFT", "FORWARD", "RIGHT"],
                ["#FF6B6B", "#00BFA5", "#7C4DFF"])):
        ax.plot(steps, spike_history[:, i], color=col, label=name, linewidth=2)
    ax.set_ylabel("Motor spikes", color="white")
    ax.set_title("Motor Neuron Activity per Step", color="white")
    ax.tick_params(colors="white")
    ax.legend(facecolor="#2D3A4A", labelcolor="white", fontsize=9)

    # Sensor readings
    ax = axes[1]
    ax.set_facecolor("#1A2332")
    for i, (name, col) in enumerate(
            zip(["Left sensor", "Front sensor", "Right sensor"],
                ["#FF6B6B", "#FFB300", "#00BFA5"])):
        ax.plot(steps, sensor_history[:, i], color=col, label=name, linewidth=2)
    ax.set_ylabel("Sensor (norm)", color="white")
    ax.set_title("Sensor Readings", color="white")
    ax.tick_params(colors="white")
    ax.legend(facecolor="#2D3A4A", labelcolor="white", fontsize=9)

    # Actions
    ax = axes[2]
    ax.set_facecolor("#1A2332")
    action_colors = {0: "#FF6B6B", 1: "#00BFA5", 2: "#7C4DFF"}
    for s, a in enumerate(action_history):
        ax.bar(s, 1, color=action_colors[a], width=0.8)
    ax.set_ylabel("Action", color="white")
    ax.set_xlabel("Step", color="white")
    ax.set_title("Actions (Red=LEFT, Teal=FORWARD, Purple=RIGHT)", color="white")
    ax.set_ylim(0, 1.2)
    ax.tick_params(colors="white")

    plt.tight_layout()

    os.makedirs(FIGURES_DIR, exist_ok=True)
    out = os.path.join(FIGURES_DIR, "snn_spike_activity.png")
    plt.savefig(out, dpi=120, bbox_inches="tight", facecolor="#1A2332")
    print(f"Saved -> {out}")
    plt.close("all")


# -- Entry point ----------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SNN navigation experiments")
    parser.add_argument("--trials",    type=int,   default=NUM_TRIALS)
    parser.add_argument("--grid",      type=int,   default=GRID_WIDTH)
    parser.add_argument("--obstacles", type=int,   default=NUM_OBSTACLES)
    parser.add_argument("--window",    type=float, default=SNN_WINDOW_MS)
    parser.add_argument("--hidden",    type=int,   default=SNN_HIDDEN_NEURONS)
    parser.add_argument("--seed",      type=int,   default=RANDOM_SEED)
    parser.add_argument("--verbose",   action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print(f"  BIAN-SNN -- SNN Controller Experiment")
    print(f"  Grid: {args.grid}x{args.grid}  Obstacles: {args.obstacles}")
    print(f"  Window: {args.window}ms  Hidden: {args.hidden}")
    print(f"  Trials: {args.trials}  Seed: {args.seed}")
    print("=" * 60)

    tracker = run_snn_experiment(
        n_trials    = args.trials,
        grid_size   = args.grid,
        n_obstacles = args.obstacles,
        window_ms   = args.window,
        n_hidden    = args.hidden,
        base_seed   = args.seed,
        verbose     = args.verbose,
    )

    tracker.print_summary()

    os.makedirs(TABLES_DIR, exist_ok=True)
    tracker.save_csv(os.path.join(TABLES_DIR, "snn_results.csv"))

    plot_sample_trajectories(n_samples=6, grid_size=args.grid,
                             n_obstacles=args.obstacles, base_seed=args.seed)
    plot_spike_activity(base_seed=args.seed)

    print("\nSNN experiment complete.")
