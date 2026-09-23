"""
compare_models.py -- Multi-Method Comparison
=============================================
Loads saved results from A*, SNN, and RL experiments and
produces comparison tables and figures.

Usage:
    # Run all three experiments first, then:
    python experiments/compare_models.py

Output:
  results/figures/comparison_bar.png
  results/figures/comparison_radar.png
  results/tables/comparison_summary.csv
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from config import FIGURES_DIR, TABLES_DIR
from navigation.metrics import MetricsTracker, compare_trackers


# -- Load results ---------------------------------------------------------------

def load_results() -> dict:
    """
    Load CSV results for each method.

    Returns dict: {'A*': MetricsTracker, 'SNN': MetricsTracker, 'RL': MetricsTracker}
    """
    method_files = {
        "A*":  os.path.join(TABLES_DIR, "astar_results.csv"),
        "SNN": os.path.join(TABLES_DIR, "snn_results.csv"),
        "RL":  os.path.join(TABLES_DIR, "rl_results.csv"),
    }

    trackers = {}
    for method, filepath in method_files.items():
        if os.path.exists(filepath):
            trackers[method] = MetricsTracker.load_csv(filepath, method=method)
            print(f"Loaded {method}: {len(trackers[method].trials)} trials")
        else:
            print(f"WARNING: {filepath} not found -- run experiment first.")

    return trackers


# -- Comparison plots -----------------------------------------------------------

METHOD_COLORS = {
    "A*":  "#00BFA5",
    "SNN": "#7C4DFF",
    "RL":  "#FFB300",
}


def plot_bar_comparison(trackers: dict) -> None:
    """Grouped bar chart comparing all methods on key metrics."""
    methods = list(trackers.keys())
    summaries = {m: trackers[m].summary() for m in methods}

    metrics = [
        ("success_rate",          "Success Rate",      "Rate (0--1)"),
        ("steps_mean",            "Avg Steps",         "Steps"),
        ("collisions_mean",       "Avg Collisions",    "Collisions"),
        ("path_efficiency_mean",  "Path Efficiency",   "Efficiency (0--1)"),
        ("nav_time_s_mean",       "Avg Nav Time",      "Seconds"),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(20, 5))
    fig.patch.set_facecolor("#1A2332")

    for ax, (key, title, ylabel) in zip(axes, metrics):
        ax.set_facecolor("#1A2332")
        vals = [summaries[m].get(key, 0) for m in methods]
        bars = ax.bar(methods, vals,
                      color=[METHOD_COLORS.get(m, "#AAAAAA") for m in methods],
                      edgecolor="#1A2332", linewidth=0.5, width=0.5)

        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(vals) * 0.01,
                    f"{val:.3f}", ha="center", va="bottom",
                    color="white", fontsize=8)

        ax.set_title(title, color="white", fontsize=10)
        ax.set_ylabel(ylabel, color="white", fontsize=9)
        ax.tick_params(colors="white")
        ax.set_ylim(0, max(vals) * 1.2 if max(vals) > 0 else 1)

    plt.suptitle("Navigation Method Comparison: A* vs SNN vs RL",
                 color="white", fontsize=14)
    plt.tight_layout()

    os.makedirs(FIGURES_DIR, exist_ok=True)
    out = os.path.join(FIGURES_DIR, "comparison_bar.png")
    plt.savefig(out, dpi=120, bbox_inches="tight", facecolor="#1A2332")
    print(f"Saved -> {out}")
    plt.close("all")


def plot_radar_comparison(trackers: dict) -> None:
    """Radar (spider) chart for normalised multi-metric comparison."""
    methods = list(trackers.keys())
    summaries = {m: trackers[m].summary() for m in methods}

    # Metrics to include (higher=better -- invert collisions and time)
    radar_metrics = [
        "success_rate",
        "path_efficiency_mean",
        "inv_collisions",      # inverted
        "inv_time",            # inverted
    ]
    labels = ["Success Rate", "Path Efficiency",
              "Low Collisions", "Speed"]

    # Compute values (normalise 0-1 across methods)
    raw_vals = {}
    for m in methods:
        s = summaries[m]
        coll = s.get("collisions_mean", 1.0)
        t    = s.get("nav_time_s_mean", 1.0)
        raw_vals[m] = [
            s.get("success_rate", 0),
            s.get("path_efficiency_mean", 0),
            1.0 / (1.0 + coll),   # invert: fewer collisions = higher score
            1.0 / (1.0 + t * 100),
        ]

    n_metrics = len(labels)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # close polygon

    fig, ax = plt.subplots(figsize=(7, 7),
                            subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("#1A2332")
    ax.set_facecolor("#1A2332")

    for method in methods:
        vals = raw_vals[method] + raw_vals[method][:1]
        color = METHOD_COLORS.get(method, "#AAAAAA")
        ax.plot(angles, vals, color=color, linewidth=2, label=method)
        ax.fill(angles, vals, color=color, alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, color="white", size=10)
    ax.set_ylim(0, 1)
    ax.tick_params(colors="white")
    ax.set_yticklabels([])
    ax.spines["polar"].set_color("#2D3A4A")
    ax.grid(color="#2D3A4A", linewidth=0.5)

    legend = ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1),
                        facecolor="#2D3A4A", labelcolor="white", fontsize=10)

    ax.set_title("Normalised Performance Radar",
                 color="white", fontsize=13, pad=20)

    out = os.path.join(FIGURES_DIR, "comparison_radar.png")
    plt.savefig(out, dpi=120, bbox_inches="tight", facecolor="#1A2332")
    print(f"Saved -> {out}")
    plt.close("all")


def print_comparison_table(trackers: dict) -> pd.DataFrame:
    """Print and return a comparison DataFrame."""
    tracker_list = list(trackers.values())
    df = compare_trackers(tracker_list)
    display_cols = [
        "success_rate",
        "steps_mean", "steps_std",
        "collisions_mean",
        "path_efficiency_mean",
        "nav_time_s_mean",
        "n_trials",
    ]
    display_cols = [c for c in display_cols if c in df.columns]
    df_display = df[display_cols].round(4)
    print("\n" + "="*70)
    print("  COMPARISON TABLE")
    print("="*70)
    print(df_display.to_string())
    print("="*70)

    os.makedirs(TABLES_DIR, exist_ok=True)
    out = os.path.join(TABLES_DIR, "comparison_summary.csv")
    df_display.to_csv(out)
    print(f"\nSaved -> {out}")
    return df_display


# -- Entry point ----------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("  BIAN-SNN -- Comparison: A* vs SNN vs RL")
    print("=" * 60)

    trackers = load_results()

    if not trackers:
        print("\nNo results found. Run experiments first:")
        print("  python experiments/run_astar.py")
        print("  python experiments/run_snn.py")
        print("  python experiments/run_rl.py")
        sys.exit(0)

    # Print individual summaries
    for method, tracker in trackers.items():
        tracker.print_summary()

    # Comparison table
    df = print_comparison_table(trackers)

    # Comparison plots (only if 2+ methods available)
    if len(trackers) >= 2:
        plot_bar_comparison(trackers)

    if len(trackers) >= 2:
        plot_radar_comparison(trackers)

    print("\nComparison complete.")
