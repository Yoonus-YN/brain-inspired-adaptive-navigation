"""
sensory_layer.py -- Sensor -> Spike Encoding
============================================
Converts robot sensor distance readings into spike trains
that drive the input layer of the SNN.

Encoding strategy: Rate coding (Poisson)
  - High sensor reading (far obstacle) -> high firing rate
  - Low sensor reading (near obstacle)  -> low firing rate

Each sensor drives a small population of neurons (default 10 per sensor),
all receiving the same encoded rate.

Usage:
    from neuroscience.sensory_layer import SensoryEncoder
    encoder = SensoryEncoder(neurons_per_sensor=10, max_rate_hz=100.0)
    spikes = encoder.encode(sensor_readings_norm, dt_ms=1.0)
    # spikes: np.ndarray shape (30,)  [left_pop | front_pop | right_pop]
"""

import numpy as np
from typing import Optional


class SensoryEncoder:
    """
    Encodes normalised sensor readings [0, 1] into Poisson spike trains
    for a population of LIF input neurons.

    Architecture:
        LEFT sensor  -> neurons [0   : N_per]
        FRONT sensor -> neurons [N_per : 2*N_per]
        RIGHT sensor -> neurons [2*N_per : 3*N_per]

    Parameters
    ----------
    neurons_per_sensor : int   -- number of neurons per sensor channel
    max_rate_hz        : float -- maximum firing rate (when sensor = 1.0)
    dt                 : float -- simulation timestep (ms)
    seed               : int   -- RNG seed for spike generation
    invert             : bool  -- if True, high reading -> high firing rate
                                 (default True: far=safe=high rate)
    """

    NUM_SENSORS = 3   # LEFT, FRONT, RIGHT

    def __init__(self,
                 neurons_per_sensor: int   = 10,
                 max_rate_hz:        float = 100.0,
                 dt:                 float = 1.0,
                 seed:               int   = 42,
                 invert:             bool  = True):
        self.neurons_per_sensor = neurons_per_sensor
        self.max_rate_hz        = max_rate_hz
        self.dt                 = dt
        self.invert             = invert
        self.rng                = np.random.default_rng(seed)

        self.n_total = self.NUM_SENSORS * neurons_per_sensor

    def encode(self,
               sensor_norm: np.ndarray,
               window_ms:   float = 50.0) -> np.ndarray:
        """
        Encode a single normalised sensor reading vector into spike counts
        over a time window.

        Parameters
        ----------
        sensor_norm : np.ndarray shape (3,) -- normalised readings in [0, 1]
                      Order: [left, front, right]
        window_ms   : float -- integration window in ms

        Returns
        -------
        np.ndarray shape (n_total,) -- spike count per neuron over window.
        """
        assert sensor_norm.shape == (self.NUM_SENSORS,), \
            f"Expected shape (3,), got {sensor_norm.shape}"

        rates = self._readings_to_rates(sensor_norm)  # shape (3,)
        spikes = np.zeros(self.n_total, dtype=np.float32)

        for s_idx in range(self.NUM_SENSORS):
            rate_hz  = rates[s_idx]
            p_spike  = rate_hz * (window_ms / 1000.0) / (window_ms / self.dt)
            # Poisson: p_spike is probability per timestep
            n_steps  = int(window_ms / self.dt)
            pop_start = s_idx * self.neurons_per_sensor
            pop_end   = pop_start + self.neurons_per_sensor

            # Each neuron independently samples spikes
            random_vals = self.rng.uniform(0, 1,
                                           size=(n_steps, self.neurons_per_sensor))
            count = (random_vals < p_spike).sum(axis=0)
            spikes[pop_start:pop_end] = count.astype(np.float32)

        return spikes

    def encode_to_current(self,
                           sensor_norm: np.ndarray,
                           current_scale: float = 3.0) -> np.ndarray:
        """
        Convert sensor readings directly to input currents for LIF neurons.
        Simpler than Poisson encoding -- useful for fast inference.

        Parameters
        ----------
        sensor_norm   : np.ndarray shape (3,)
        current_scale : float -- scales [0,1] -> [0, current_scale] (nA)

        Returns
        -------
        np.ndarray shape (n_total,) -- input currents (nA)
        """
        rates_norm = self._readings_to_rates_norm(sensor_norm)  # (3,)
        currents = np.zeros(self.n_total, dtype=np.float32)
        for s_idx in range(self.NUM_SENSORS):
            start = s_idx * self.neurons_per_sensor
            end   = start + self.neurons_per_sensor
            currents[start:end] = rates_norm[s_idx] * current_scale
        return currents

    def _readings_to_rates(self, sensor_norm: np.ndarray) -> np.ndarray:
        """Convert normalised readings to firing rates (Hz)."""
        rates_norm = self._readings_to_rates_norm(sensor_norm)
        return rates_norm * self.max_rate_hz

    def _readings_to_rates_norm(self, sensor_norm: np.ndarray) -> np.ndarray:
        """Normalised rates in [0, 1]."""
        if self.invert:
            # Far obstacle -> high activity (safe path -> active neurons)
            return sensor_norm.copy()
        else:
            # Near obstacle -> high activity (danger -> active)
            return 1.0 - sensor_norm

    @property
    def population_labels(self):
        """Labels for each neuron population."""
        labels = []
        for s in ["L", "F", "R"]:
            labels.extend([f"{s}{i}" for i in range(self.neurons_per_sensor)])
        return labels

    def __repr__(self) -> str:
        return (f"SensoryEncoder(n_per_sensor={self.neurons_per_sensor}, "
                f"n_total={self.n_total}, max_rate={self.max_rate_hz}Hz)")


# -- Standalone test ------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    encoder = SensoryEncoder(neurons_per_sensor=10, max_rate_hz=100.0,
                              dt=1.0, seed=42)
    print(encoder)

    # Simulate: obstacle on the left, clear right
    readings = np.array([0.2, 0.7, 0.9])   # [left, front, right]
    print(f"\nSensor readings (norm): {readings}")
    print(f"  left  0.2 -> near obstacle")
    print(f"  front 0.7 -> moderate")
    print(f"  right 0.9 -> clear")

    spikes = encoder.encode(readings, window_ms=50.0)
    currents = encoder.encode_to_current(readings)

    print(f"\nSpike counts per neuron (50ms window):")
    print(f"  LEFT  pop: {spikes[:10].astype(int)}")
    print(f"  FRONT pop: {spikes[10:20].astype(int)}")
    print(f"  RIGHT pop: {spikes[20:].astype(int)}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor("#1A2332")

    colors = ["#FF6B6B", "#00BFA5", "#7C4DFF"]
    labels = ["LEFT", "FRONT", "RIGHT"]

    ax = axes[0]
    ax.set_facecolor("#1A2332")
    ax.bar(range(30), spikes, color=[colors[i//10] for i in range(30)])
    ax.set_xlabel("Neuron index", color="white")
    ax.set_ylabel("Spike count (50ms)", color="white")
    ax.set_title("Sensory Encoding -- Spike Counts", color="white")
    ax.tick_params(colors="white")
    for i, (lbl, col) in enumerate(zip(labels, colors)):
        ax.axvspan(i*10-0.5, (i+1)*10-0.5, alpha=0.1, color=col)
        ax.text(i*10+4.5, max(spikes)*0.9, lbl, color=col, fontsize=9)

    ax = axes[1]
    ax.set_facecolor("#1A2332")
    ax.bar(range(30), currents, color=[colors[i//10] for i in range(30)])
    ax.set_xlabel("Neuron index", color="white")
    ax.set_ylabel("Input current (nA)", color="white")
    ax.set_title("Sensory Encoding -- Currents", color="white")
    ax.tick_params(colors="white")

    plt.tight_layout()
    plt.show()
