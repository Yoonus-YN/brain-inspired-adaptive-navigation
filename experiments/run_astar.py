"""
run_astar.py -- A* Baseline Navigation Experiments
===================================================
Runs A* on multiple random environments and records metrics.

Output:
  results/tables/astar_results.csv
  results/figures/astar_trajectories.png
  results/figures/astar_metrics.png

Usage:
    python experiments/run_astar.py
    python experiments/run_astar.py --trials 50 --grid 20 --obstacles 15
"""

import sys
import os
import argparse
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")          # non-interactive backend -- no plt.show() blocking
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm

from config import (GRID_WIDTH, GRID_HEIGHT, NUM_OBSTACLES,
                    RANDOM_SEED, NUM_TRIALS, FIGURES_DIR, TABLES_DIR, LOGS_DIR)
from environment.grid_world import GridWorld
from environment.robot import Robot, EAST
from environment.sensors import SensorArray
from environment.obstacles import ObstacleGenerator
from navigation.astar import AStarNavigator
from navigation.metrics import MetricsTracker, EpisodeTimer


def run_astar_experiment(
        n_trials:    int = NUM_TRIALS,
        grid_size:   int = GRID_WIDTH,
        n_obstacles: int = NUM_OBSTACLES,
        max_steps:   int = 500,
        base_seed:   int = RANDOM_SEED,
        verbose:     bool = False,
) -> MetricsTracker:
    """
    Run A* on `n_trials` random environments.

    Returns
    -------
    MetricsTracker with all trial results.
    """
    tracker = MetricsTracker(method="A*")
    nav     = AStarNavigator(allow_diagonal=False)

    for trial in tqdm(range(n_trials), desc="A* trials", unit="trial"):
        seed  = base_seed + trial
        world = GridWorld(grid_size, grid_size, seed=seed)
        start = (1, 1)
        goal  = (grid_size - 2, grid_size - 2)
        world.set_start(start)
        world.set_goal(goal)

        gen = ObstacleGenerator(seed=seed)
        gen.random_obstacles(world, n=n_obstacles)

        robot = Robot(start=start, orientation=EAST)

        # Compute full A* path first (optimal reference)
        opt_path, opt_info = nav.find_path(world, start, goal)

        if not opt_info["success"]:
            # No path exists for this environment -- record failure
            tracker.record(
                trial_id=trial, success=False, steps=0,
                collisions=0, optimal_steps=0, seed=seed,
                env_config=f"{grid_size}x{grid_size}_obs{n_obstacles}")
            continue

        optimal_steps = opt_info["path_length"]

        # Simulate step-by-step robot following A* decisions
        with EpisodeTimer() as timer:
            for step in range(max_steps):
                action = nav.find_next_action(world, robot.pos,
                                              robot.orientation, goal)
                collision, goal_reached = robot.step(action, world)

                if collision:
                    # A* should not collide; record if it does (edge case)
                    break
                if goal_reached:
                    break

        success = robot.reached_goal
        tracker.record(
            trial_id      = trial,
            success       = success,
            steps         = robot.steps,
            collisions    = robot.collision_count,
            optimal_steps = optimal_steps,
            nav_time_s    = timer.elapsed_s,
            seed          = seed,
            env_config    = f"{grid_size}x{grid_size}_obs{n_obstacles}",
        )

        if verbose:
            eff = min(optimal_steps / robot.steps, 1.0) if robot.steps > 0 else 0
            print(f"Trial {trial:3d}: success={success}, "
                  f"steps={robot.steps}, eff={eff:.3f}, "
                  f"t={timer.elapsed_s*1000:.1f}ms")

    return tracker


def plot_sample_trajectories(n_samples: int = 6,
                              grid_size: int = GRID_WIDTH,
                              n_obstacles: int = NUM_OBSTACLES,
                              base_seed: int = RANDOM_SEED) -> None:
    """Plot sample A* trajectories for visualisation."""
    nav = AStarNavigator()
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

        path, info = nav.find_path(world)
        ax = fig.add_subplot(gs[i // 3, i % 3])
        success_str = "[OK]" if info["success"] else "[X]"
        world.render(path=path,
                     title=f"Trial {i+1} {success_str}  steps={info['path_length']}",
                     ax=ax, show=False)

    plt.suptitle("A* Baseline -- Sample Trajectories",
                 color="white", fontsize=14, y=1.01)

    os.makedirs(FIGURES_DIR, exist_ok=True)
    out = os.path.join(FIGURES_DIR, "astar_trajectories.png")
    plt.savefig(out, dpi=120, bbox_inches="tight",
                facecolor="#1A2332")
    print(f"Saved -> {out}")
    plt.close("all")


def plot_metrics(tracker: MetricsTracker) -> None:
    """Plot metric distributions from A* results."""
    df = tracker.to_dataframe()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.patch.set_facecolor("#1A2332")
    colors = ["#00BFA5", "#7C4DFF", "#FFB300"]

    metrics = [
        ("steps",           "Steps to Goal",      "Steps"),
        ("path_efficiency", "Path Efficiency",     "Efficiency (0--1)"),
        ("nav_time_s",      "Navigation Time",     "Seconds"),
    ]

    for ax, (col, title, ylabel), color in zip(axes, metrics, colors):
        ax.set_facecolor("#1A2332")
        data = df[df["success"]][col].dropna()
        ax.hist(data, bins=15, color=color, alpha=0.85, edgecolor="#1A2332")
        ax.axvline(data.mean(), color="white", linestyle="--",
                   linewidth=1.5, label=f"u={data.mean():.2f}")
        ax.set_title(title, color="white", fontsize=11)
        ax.set_xlabel(ylabel, color="white")
        ax.set_ylabel("Frequency", color="white")
        ax.tick_params(colors="white")
        ax.legend(fontsize=8, facecolor="#2D3A4A", labelcolor="white")

    plt.suptitle(
        f"A* Metrics -- Success Rate: {df['success'].mean()*100:.1f}%  "
        f"(n={len(df)})",
        color="white", fontsize=13)
    plt.tight_layout()

    os.makedirs(FIGURES_DIR, exist_ok=True)
    out = os.path.join(FIGURES_DIR, "astar_metrics.png")
    plt.savefig(out, dpi=120, bbox_inches="tight", facecolor="#1A2332")
    print(f"Saved -> {out}")
    plt.close("all")


# -- Entry point ----------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run A* navigation experiments")
    parser.add_argument("--trials",    type=int, default=NUM_TRIALS)
    parser.add_argument("--grid",      type=int, default=GRID_WIDTH)
    parser.add_argument("--obstacles", type=int, default=NUM_OBSTACLES)
    parser.add_argument("--seed",      type=int, default=RANDOM_SEED)
    parser.add_argument("--verbose",   action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print(f"  BIAN-SNN -- A* Baseline Experiment")
    print(f"  Grid: {args.grid}x{args.grid}  Obstacles: {args.obstacles}")
    print(f"  Trials: {args.trials}  Seed: {args.seed}")
    print("=" * 60)

    tracker = run_astar_experiment(
        n_trials    = args.trials,
        grid_size   = args.grid,
        n_obstacles = args.obstacles,
        base_seed   = args.seed,
        verbose     = args.verbose,
    )

    tracker.print_summary()

    # Save results
    os.makedirs(TABLES_DIR, exist_ok=True)
    csv_path = os.path.join(TABLES_DIR, "astar_results.csv")
    tracker.save_csv(csv_path)

    # Plots
    plot_sample_trajectories(n_samples=6, grid_size=args.grid,
                             n_obstacles=args.obstacles, base_seed=args.seed)
    plot_metrics(tracker)

    print("\nA* experiment complete.")
