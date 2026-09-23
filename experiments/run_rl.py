"""
run_rl.py -- Reinforcement Learning Training and Evaluation
===========================================================
Trains a DQN agent on the navigation environment, evaluates it,
and saves results.

Output:
  reinforcement_learning/checkpoints/dqn_nav.pth
  results/tables/rl_results.csv
  results/figures/rl_reward_curve.png
  results/figures/rl_trajectories.png

Usage:
    python experiments/run_rl.py
    python experiments/run_rl.py --episodes 500 --eval-trials 30
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

from config import (GRID_WIDTH, GRID_HEIGHT, NUM_OBSTACLES, RANDOM_SEED,
                    RL_TRAIN_EPISODES, RL_EVAL_EPISODES, NUM_TRIALS,
                    FIGURES_DIR, TABLES_DIR, CHECKPOINTS_DIR,
                    LEARNING_RATE, GAMMA, EPSILON_START, EPSILON_END,
                    EPSILON_DECAY, REPLAY_BUFFER_SIZE, BATCH_SIZE,
                    TARGET_UPDATE_FREQ)
from reinforcement_learning.environment import NavEnv
from reinforcement_learning.agent import DQNAgent
from navigation.astar import AStarNavigator   # for optimal reference
from navigation.metrics import MetricsTracker, EpisodeTimer
from environment.grid_world import GridWorld
from environment.obstacles import ObstacleGenerator


def train_dqn(
        n_episodes:  int   = RL_TRAIN_EPISODES,
        grid_size:   int   = GRID_WIDTH,
        n_obstacles: int   = NUM_OBSTACLES,
        max_steps:   int   = 500,
        seed:        int   = RANDOM_SEED,
        verbose:     bool  = False,
) -> tuple:
    """
    Train DQN agent.

    Returns
    -------
    (agent, reward_history, success_history)
    """
    env = NavEnv(grid_size=grid_size, n_obstacles=n_obstacles,
                 max_steps=max_steps, seed=seed)

    agent = DQNAgent(
        state_dim       = 5,
        action_dim      = 3,
        lr              = LEARNING_RATE,
        gamma           = GAMMA,
        epsilon_start   = EPSILON_START,
        epsilon_end     = EPSILON_END,
        epsilon_decay   = EPSILON_DECAY,
        buffer_capacity = REPLAY_BUFFER_SIZE,
        batch_size      = BATCH_SIZE,
        target_update   = TARGET_UPDATE_FREQ,
        seed            = seed,
    )

    reward_history  = []
    success_history = []
    loss_history    = []

    print(f"\nTraining DQN for {n_episodes} episodes...")
    for ep in tqdm(range(n_episodes), desc="Training", unit="ep"):
        obs, _ = env.reset(seed=seed + ep)
        total_reward = 0.0
        losses       = []

        for _ in range(max_steps):
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.store(obs, action, reward, next_obs, done)
            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)

            obs = next_obs
            total_reward += reward

            if done:
                break

        agent.end_episode()
        reward_history.append(total_reward)
        success_history.append(float(info.get("goal_reached", False)))
        if losses:
            loss_history.append(np.mean(losses))

        if verbose and (ep + 1) % 50 == 0:
            recent_sr = np.mean(success_history[-50:]) * 100
            recent_r  = np.mean(reward_history[-50:])
            print(f"  Ep {ep+1:4d} | eps={agent.epsilon:.3f} | "
                  f"reward={recent_r:+.1f} | success={recent_sr:.1f}%")

    env.close()
    return agent, reward_history, success_history, loss_history


def evaluate_dqn(
        agent:       DQNAgent,
        n_trials:    int = RL_EVAL_EPISODES,
        grid_size:   int = GRID_WIDTH,
        n_obstacles: int = NUM_OBSTACLES,
        max_steps:   int = 500,
        base_seed:   int = RANDOM_SEED + 1000,
) -> MetricsTracker:
    """
    Evaluate trained DQN agent (greedy, no exploration).

    Returns MetricsTracker with evaluation results.
    """
    tracker = MetricsTracker(method="RL")
    nav     = AStarNavigator()
    env     = NavEnv(grid_size=grid_size, n_obstacles=n_obstacles,
                     max_steps=max_steps, seed=base_seed)

    print(f"\nEvaluating DQN for {n_trials} trials (eps=0)...")
    for trial in tqdm(range(n_trials), desc="Evaluating", unit="trial"):
        seed = base_seed + trial
        obs, _ = env.reset(seed=seed)

        # Compute optimal reference
        world_ref = GridWorld(grid_size, grid_size, seed=seed)
        world_ref.set_start((1, 1))
        world_ref.set_goal((grid_size - 2, grid_size - 2))
        gen = ObstacleGenerator(seed=seed)
        gen.random_obstacles(world_ref, n=n_obstacles)
        _, opt_info = nav.find_path(world_ref)
        optimal_steps = opt_info.get("path_length", 0) if opt_info["success"] else 0

        total_reward = 0.0
        steps = 0
        goal_reached = False
        collision    = False

        with EpisodeTimer() as timer:
            for _ in range(max_steps):
                action = agent.select_greedy_action(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                steps        += 1
                goal_reached  = info.get("goal_reached", False)
                collision     = info.get("collision", False)
                if terminated or truncated:
                    break

        tracker.record(
            trial_id      = trial,
            success       = goal_reached,
            steps         = steps,
            collisions    = int(collision),
            optimal_steps = optimal_steps,
            nav_time_s    = timer.elapsed_s,
            reward_total  = total_reward,
            seed          = seed,
            env_config    = f"{grid_size}x{grid_size}_obs{n_obstacles}",
        )

    env.close()
    return tracker


def plot_training_curves(reward_history, success_history,
                          loss_history, window: int = 20) -> None:
    """Plot DQN training reward and success rate curves."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor("#1A2332")

    def smooth(data, w):
        if len(data) < w:
            return np.array(data)
        return np.convolve(data, np.ones(w) / w, mode="valid")

    episodes = np.arange(len(reward_history))

    # Reward
    ax = axes[0]
    ax.set_facecolor("#1A2332")
    ax.plot(episodes, reward_history, color="#2D3A4A", alpha=0.4, linewidth=0.8)
    sm = smooth(reward_history, window)
    ax.plot(np.arange(len(sm)) + window // 2, sm, color="#00BFA5", linewidth=2)
    ax.set_xlabel("Episode", color="white")
    ax.set_ylabel("Total Reward", color="white")
    ax.set_title("Training Reward", color="white")
    ax.tick_params(colors="white")

    # Success rate
    ax = axes[1]
    ax.set_facecolor("#1A2332")
    sm_sr = smooth(success_history, window)
    ax.plot(np.arange(len(sm_sr)) + window // 2, sm_sr * 100,
            color="#FFB300", linewidth=2)
    ax.set_xlabel("Episode", color="white")
    ax.set_ylabel("Success Rate (%)", color="white")
    ax.set_title(f"Success Rate (smoothed {window}ep)", color="white")
    ax.tick_params(colors="white")
    ax.set_ylim(0, 105)

    # Loss
    ax = axes[2]
    ax.set_facecolor("#1A2332")
    if loss_history:
        loss_episodes = np.arange(len(loss_history))
        sm_loss = smooth(loss_history, window)
        ax.plot(loss_episodes, loss_history, color="#2D3A4A", alpha=0.4, linewidth=0.8)
        ax.plot(np.arange(len(sm_loss)) + window // 2, sm_loss,
                color="#FF6B6B", linewidth=2)
    ax.set_xlabel("Episode", color="white")
    ax.set_ylabel("MSE Loss", color="white")
    ax.set_title("Q-Network Loss", color="white")
    ax.tick_params(colors="white")

    plt.suptitle("DQN Training Curves", color="white", fontsize=14)
    plt.tight_layout()

    os.makedirs(FIGURES_DIR, exist_ok=True)
    out = os.path.join(FIGURES_DIR, "rl_reward_curve.png")
    plt.savefig(out, dpi=120, bbox_inches="tight", facecolor="#1A2332")
    print(f"Saved -> {out}")
    plt.close("all")


# -- Entry point ----------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate DQN agent")
    parser.add_argument("--episodes",     type=int, default=RL_TRAIN_EPISODES)
    parser.add_argument("--eval-trials",  type=int, default=RL_EVAL_EPISODES)
    parser.add_argument("--grid",         type=int, default=GRID_WIDTH)
    parser.add_argument("--obstacles",    type=int, default=NUM_OBSTACLES)
    parser.add_argument("--seed",         type=int, default=RANDOM_SEED)
    parser.add_argument("--no-train",     action="store_true",
                        help="Skip training, load from checkpoint")
    parser.add_argument("--verbose",      action="store_true")
    args = parser.parse_args()

    ckpt_path = os.path.join(CHECKPOINTS_DIR, "dqn_nav.pth")

    print("=" * 60)
    print(f"  BIAN-SNN -- RL (DQN) Experiment")
    print(f"  Grid: {args.grid}x{args.grid}  Obstacles: {args.obstacles}")
    print(f"  Train episodes: {args.episodes}  Eval trials: {args.eval_trials}")
    print(f"  Seed: {args.seed}")
    print("=" * 60)

    if args.no_train and os.path.exists(ckpt_path):
        agent = DQNAgent(state_dim=5, action_dim=3, seed=args.seed)
        agent.load(ckpt_path)
        reward_history = success_history = loss_history = []
    else:
        agent, reward_history, success_history, loss_history = train_dqn(
            n_episodes  = args.episodes,
            grid_size   = args.grid,
            n_obstacles = args.obstacles,
            seed        = args.seed,
            verbose     = args.verbose,
        )
        os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
        agent.save(ckpt_path)
        plot_training_curves(reward_history, success_history, loss_history)

    # Evaluate
    tracker = evaluate_dqn(
        agent       = agent,
        n_trials    = args.eval_trials,
        grid_size   = args.grid,
        n_obstacles = args.obstacles,
        base_seed   = args.seed + 1000,
    )

    tracker.print_summary()

    os.makedirs(TABLES_DIR, exist_ok=True)
    tracker.save_csv(os.path.join(TABLES_DIR, "rl_results.csv"))

    print("\nRL experiment complete.")
