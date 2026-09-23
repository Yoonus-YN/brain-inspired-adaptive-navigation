"""
metrics.py -- Navigation Evaluation Metrics
===========================================
Computes and records performance metrics for navigation experiments.

Metrics tracked:
  success_rate      : fraction of trials reaching the goal
  path_length       : number of steps taken
  collision_count   : number of collision events
  path_efficiency   : optimal_length / actual_length  (0..1, higher=better)
  nav_time_s        : wall-clock seconds for one episode
  steps_to_goal     : steps taken when goal was reached (else max_steps)
  reward_total      : total reward accumulated (RL only)

Usage:
    from navigation.metrics import MetricsTracker
    tracker = MetricsTracker()
    tracker.record(trial_id=0, success=True, steps=45,
                   collisions=1, optimal_steps=30, time_s=0.02)
    df = tracker.to_dataframe()
    tracker.summary()
"""

import time
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd


# -- Trial result dataclass -----------------------------------------------------

@dataclass
class TrialResult:
    """Result of a single navigation trial."""
    trial_id:        int
    method:          str     = "unknown"
    success:         bool    = False
    steps:           int     = 0
    collisions:      int     = 0
    optimal_steps:   int     = 0     # A* reference (0 = not computed)
    path_efficiency: float   = 0.0   # optimal/actual, capped at 1.0
    nav_time_s:      float   = 0.0
    reward_total:    float   = 0.0   # RL only
    spike_count:     int     = 0     # SNN only
    seed:            int     = 0
    env_config:      str     = ""

    def __post_init__(self) -> None:
        if self.steps > 0 and self.optimal_steps > 0:
            self.path_efficiency = min(
                self.optimal_steps / self.steps, 1.0)


# -- Metrics Tracker ------------------------------------------------------------

class MetricsTracker:
    """
    Accumulates trial results and computes summary statistics.

    Parameters
    ----------
    method : str
        Label for this navigation method (e.g. 'A*', 'SNN', 'RL').
    """

    def __init__(self, method: str = "unknown"):
        self.method  = method
        self.trials: List[TrialResult] = []

    def record(self,
               trial_id:      int,
               success:       bool,
               steps:         int,
               collisions:    int     = 0,
               optimal_steps: int     = 0,
               nav_time_s:    float   = 0.0,
               reward_total:  float   = 0.0,
               spike_count:   int     = 0,
               seed:          int     = 0,
               env_config:    str     = "") -> TrialResult:
        """
        Record one trial result.

        Returns
        -------
        TrialResult -- the recorded result.
        """
        result = TrialResult(
            trial_id      = trial_id,
            method        = self.method,
            success       = success,
            steps         = steps,
            collisions    = collisions,
            optimal_steps = optimal_steps,
            nav_time_s    = nav_time_s,
            reward_total  = reward_total,
            spike_count   = spike_count,
            seed          = seed,
            env_config    = env_config,
        )
        self.trials.append(result)
        return result

    def to_dataframe(self) -> pd.DataFrame:
        """Return all trial results as a pandas DataFrame."""
        return pd.DataFrame([asdict(t) for t in self.trials])

    def summary(self) -> Dict[str, Any]:
        """
        Compute aggregate statistics across all recorded trials.

        Returns
        -------
        dict with mean/std/min/max for each numeric metric.
        """
        if not self.trials:
            return {}

        df = self.to_dataframe()
        num_cols = ["steps", "collisions", "path_efficiency",
                    "nav_time_s", "reward_total", "spike_count"]

        stats: Dict[str, Any] = {
            "method":       self.method,
            "n_trials":     len(self.trials),
            "success_rate": float(df["success"].mean()),
        }

        for col in num_cols:
            if col in df.columns:
                stats[f"{col}_mean"] = float(df[col].mean())
                stats[f"{col}_std"]  = float(df[col].std())
                stats[f"{col}_min"]  = float(df[col].min())
                stats[f"{col}_max"]  = float(df[col].max())

        return stats

    def print_summary(self) -> None:
        """Print a formatted summary table to stdout."""
        s = self.summary()
        if not s:
            print("No trials recorded.")
            return

        print(f"\n{'='*55}")
        print(f"  Method : {s['method']}")
        print(f"  Trials : {s['n_trials']}")
        print(f"  Success rate : {s['success_rate']*100:.1f}%")
        print(f"  {'Metric':<22} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
        print(f"  {'-'*53}")
        metrics = ["steps", "collisions", "path_efficiency",
                   "nav_time_s", "reward_total"]
        for m in metrics:
            mn = s.get(f"{m}_mean")
            if mn is None:
                continue
            print(f"  {m:<22} "
                  f"{mn:>8.3f} "
                  f"{s[f'{m}_std']:>8.3f} "
                  f"{s[f'{m}_min']:>8.3f} "
                  f"{s[f'{m}_max']:>8.3f}")
        print(f"{'='*55}")

    def reset(self) -> None:
        """Clear all recorded trials."""
        self.trials = []

    def save_csv(self, filepath: str) -> None:
        """Save results to CSV file."""
        df = self.to_dataframe()
        df.to_csv(filepath, index=False)
        print(f"Saved {len(df)} trials -> {filepath}")

    @classmethod
    def load_csv(cls, filepath: str, method: str = "unknown") -> "MetricsTracker":
        """Load results from a CSV file."""
        tracker = cls(method=method)
        df = pd.read_csv(filepath)
        for _, row in df.iterrows():
            result = TrialResult(**{
                k: row[k] for k in TrialResult.__dataclass_fields__
                if k in row
            })
            tracker.trials.append(result)
        return tracker


# -- Comparison helper ---------------------------------------------------------

def compare_trackers(trackers: List[MetricsTracker]) -> pd.DataFrame:
    """
    Produce a single comparison DataFrame from multiple trackers.

    Parameters
    ----------
    trackers : list of MetricsTracker

    Returns
    -------
    pd.DataFrame -- one row per method.
    """
    rows = [t.summary() for t in trackers]
    return pd.DataFrame(rows).set_index("method")


# -- Episode timer context manager ---------------------------------------------

class EpisodeTimer:
    """
    Simple context manager for timing navigation episodes.

    Usage:
        with EpisodeTimer() as timer:
            ... run episode ...
        print(timer.elapsed_s)
    """
    def __enter__(self):
        self._start = time.perf_counter()
        self.elapsed_s = 0.0
        return self

    def __exit__(self, *args):
        self.elapsed_s = time.perf_counter() - self._start


if __name__ == "__main__":
    # Quick smoke test
    tracker = MetricsTracker(method="A*")
    for i in range(5):
        tracker.record(trial_id=i, success=True, steps=40 + i,
                       collisions=0, optimal_steps=35, nav_time_s=0.01)
    tracker.record(trial_id=5, success=False, steps=200, collisions=3,
                   optimal_steps=35, nav_time_s=0.05)
    tracker.print_summary()
