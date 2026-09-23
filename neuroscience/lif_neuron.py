"""
lif_neuron.py -- Leaky Integrate-and-Fire Neuron (Pure NumPy)
=============================================================
Implements the LIF neuron model:

    tau_m . dV/dt = -(V - V_rest) + R.I(t)

When V >= V_threshold, the neuron fires (spike) and V is reset to V_reset.

This module provides:
  - LIFNeuron       : single neuron simulation
  - LIFPopulation   : vectorised population (N neurons, same parameters)
  - run_simulation  : convenience function for running a fixed duration

All computation uses NumPy (no Brian2 required here).
Brian2 is used separately in the notebooks for detailed neuroscience demos.

Usage:
    from neuroscience.lif_neuron import LIFNeuron, LIFPopulation

    neuron = LIFNeuron()
    V_trace, spike_train = neuron.simulate(I_ext=2.5, duration_ms=200.0)

    pop = LIFPopulation(n=10)
    I = np.random.uniform(1.0, 3.0, size=10)
    V_traces, spike_trains = pop.simulate(I_ext=I, duration_ms=100.0)
"""

import numpy as np
from typing import Tuple, Optional

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# -- Single LIF Neuron ---------------------------------------------------------

class LIFNeuron:
    """
    Single Leaky Integrate-and-Fire neuron.

    Parameters
    ----------
    tau_m       : float -- membrane time constant (ms)
    v_rest      : float -- resting potential (mV)
    v_threshold : float -- spike threshold (mV)
    v_reset     : float -- post-spike reset potential (mV)
    r_membrane  : float -- membrane resistance (MOmega)
    dt          : float -- simulation timestep (ms)
    """

    def __init__(self,
                 tau_m:       float = 20.0,
                 v_rest:      float = -70.0,
                 v_threshold: float = -55.0,
                 v_reset:     float = -75.0,
                 r_membrane:  float = 10.0,
                 dt:          float = 1.0):
        self.tau_m       = tau_m
        self.v_rest      = v_rest
        self.v_threshold = v_threshold
        self.v_reset     = v_reset
        self.r_membrane  = r_membrane
        self.dt          = dt

        # State
        self.V = v_rest

    def reset_state(self) -> None:
        """Reset membrane potential to resting value."""
        self.V = self.v_rest

    def step(self, I_ext: float) -> bool:
        """
        Advance simulation by one timestep (self.dt ms).

        Parameters
        ----------
        I_ext : float -- external input current (nA)

        Returns
        -------
        bool -- True if a spike was emitted this timestep.
        """
        dV = (-(self.V - self.v_rest) + self.r_membrane * I_ext) / self.tau_m
        self.V += dV * self.dt

        if self.V >= self.v_threshold:
            self.V = self.v_reset
            return True
        return False

    def simulate(self,
                 I_ext: float,
                 duration_ms: float,
                 I_noise_std: float = 0.0
                 ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate for `duration_ms` milliseconds with constant current.

        Parameters
        ----------
        I_ext       : float -- constant external current (nA)
        duration_ms : float -- simulation duration (ms)
        I_noise_std : float -- Gaussian noise std added to current each step

        Returns
        -------
        V_trace     : np.ndarray shape (T,) -- membrane potential over time (mV)
        spike_train : np.ndarray shape (T,) -- 1 where spike, 0 otherwise
        """
        T = int(duration_ms / self.dt)
        V_trace    = np.zeros(T, dtype=np.float32)
        spike_train = np.zeros(T, dtype=np.int8)

        self.reset_state()
        rng = np.random.default_rng(seed=0)

        for t in range(T):
            noise = rng.normal(0, I_noise_std) if I_noise_std > 0 else 0.0
            fired = self.step(I_ext + noise)
            V_trace[t]     = self.V
            spike_train[t] = int(fired)

        return V_trace, spike_train

    def firing_rate(self, I_ext: float, duration_ms: float = 1000.0) -> float:
        """
        Compute mean firing rate (Hz) for constant input current.

        Parameters
        ----------
        I_ext       : float -- input current (nA)
        duration_ms : float -- integration window (ms)

        Returns
        -------
        float -- spikes per second.
        """
        _, spikes = self.simulate(I_ext, duration_ms)
        return float(spikes.sum() / (duration_ms / 1000.0))

    def f_I_curve(self,
                  I_range: Optional[np.ndarray] = None,
                  duration_ms: float = 1000.0
                  ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the frequency--current (f-I) curve.

        Parameters
        ----------
        I_range     : 1-D array of input currents to test
        duration_ms : simulation duration per point

        Returns
        -------
        (I_range, firing_rates) both as np.ndarray.
        """
        if I_range is None:
            I_range = np.linspace(0.0, 5.0, 50)
        rates = np.array([self.firing_rate(I, duration_ms) for I in I_range])
        return I_range, rates


# -- LIF Population ------------------------------------------------------------

class LIFPopulation:
    """
    Vectorised population of N LIF neurons sharing the same parameters.
    All N neurons are simulated in parallel using NumPy.

    Parameters
    ----------
    n           : int   -- number of neurons
    tau_m       : float
    v_rest      : float
    v_threshold : float
    v_reset     : float
    r_membrane  : float
    dt          : float
    """

    def __init__(self,
                 n:           int   = 10,
                 tau_m:       float = 20.0,
                 v_rest:      float = -70.0,
                 v_threshold: float = -55.0,
                 v_reset:     float = -75.0,
                 r_membrane:  float = 10.0,
                 dt:          float = 1.0):
        self.n           = n
        self.tau_m       = tau_m
        self.v_rest      = v_rest
        self.v_threshold = v_threshold
        self.v_reset     = v_reset
        self.r_membrane  = r_membrane
        self.dt          = dt

        # State: shape (N,)
        self.V = np.full(n, v_rest, dtype=np.float32)

    def reset_state(self) -> None:
        """Reset all membrane potentials."""
        self.V[:] = self.v_rest

    def step(self, I_ext: np.ndarray) -> np.ndarray:
        """
        Advance all N neurons by one timestep.

        Parameters
        ----------
        I_ext : np.ndarray shape (N,) -- input currents

        Returns
        -------
        np.ndarray shape (N,) -- boolean spike mask (0 or 1, int8)
        """
        dV = (-(self.V - self.v_rest) + self.r_membrane * I_ext) / self.tau_m
        self.V += dV * self.dt

        spikes = (self.V >= self.v_threshold).astype(np.int8)
        self.V[spikes.astype(bool)] = self.v_reset
        return spikes

    def simulate(self,
                 I_ext: np.ndarray,
                 duration_ms: float
                 ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate population for `duration_ms` ms with constant per-neuron currents.

        Parameters
        ----------
        I_ext       : np.ndarray shape (N,) -- constant currents per neuron
        duration_ms : float

        Returns
        -------
        V_traces    : np.ndarray shape (T, N) -- membrane potentials
        spike_matrix: np.ndarray shape (T, N) -- spike matrix (0/1)
        """
        T = int(duration_ms / self.dt)
        V_traces     = np.zeros((T, self.n), dtype=np.float32)
        spike_matrix = np.zeros((T, self.n), dtype=np.int8)

        self.reset_state()
        for t in range(T):
            spikes          = self.step(I_ext)
            V_traces[t]     = self.V.copy()
            spike_matrix[t] = spikes

        return V_traces, spike_matrix

    def simulate_dynamic(self,
                          I_sequence: np.ndarray
                          ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate with time-varying input currents.

        Parameters
        ----------
        I_sequence : np.ndarray shape (T, N) -- input current at each timestep

        Returns
        -------
        V_traces    : np.ndarray shape (T, N)
        spike_matrix: np.ndarray shape (T, N)
        """
        T = I_sequence.shape[0]
        V_traces     = np.zeros((T, self.n), dtype=np.float32)
        spike_matrix = np.zeros((T, self.n), dtype=np.int8)

        self.reset_state()
        for t in range(T):
            spikes          = self.step(I_sequence[t])
            V_traces[t]     = self.V.copy()
            spike_matrix[t] = spikes

        return V_traces, spike_matrix

    def spike_counts(self, spike_matrix: np.ndarray) -> np.ndarray:
        """Total spikes per neuron from a spike matrix."""
        return spike_matrix.sum(axis=0)

    def mean_firing_rates(self,
                           spike_matrix: np.ndarray,
                           duration_ms:  float) -> np.ndarray:
        """Mean firing rate (Hz) per neuron."""
        return self.spike_counts(spike_matrix) / (duration_ms / 1000.0)


# -- Standalone test / demo -----------------------------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Single neuron demo
    neuron = LIFNeuron(tau_m=20, v_rest=-70, v_threshold=-55, v_reset=-75,
                       r_membrane=10, dt=1.0)

    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    fig.patch.set_facecolor("#1A2332")

    # 1. Voltage trace
    V, spk = neuron.simulate(I_ext=2.5, duration_ms=200.0)
    t_ms = np.arange(len(V))

    ax = axes[0]
    ax.set_facecolor("#1A2332")
    ax.plot(t_ms, V, color="#00BFA5", linewidth=1.2, label="V(t)")
    ax.axhline(neuron.v_threshold, color="#FFB300", linestyle="--",
               linewidth=1, label=f"Threshold ({neuron.v_threshold} mV)")
    ax.axhline(neuron.v_rest, color="#64B5F6", linestyle=":",
               linewidth=1, label=f"Rest ({neuron.v_rest} mV)")
    spike_times = t_ms[spk.astype(bool)]
    ax.scatter(spike_times, [neuron.v_threshold]*len(spike_times),
               color="#FF6B6B", zorder=5, s=40, label="Spikes")
    ax.set_ylabel("V (mV)", color="white")
    ax.set_title("LIF Neuron -- Membrane Potential", color="white")
    ax.tick_params(colors="white")
    ax.legend(fontsize=8, facecolor="#2D3A4A", labelcolor="white")

    # 2. Spike raster
    ax = axes[1]
    ax.set_facecolor("#1A2332")
    ax.vlines(spike_times, 0, 1, color="#FF6B6B", linewidth=1.5)
    ax.set_ylabel("Spike", color="white")
    ax.set_title("Spike Train", color="white")
    ax.tick_params(colors="white")
    ax.set_ylim(-0.1, 1.1)

    # 3. f-I curve
    ax = axes[2]
    ax.set_facecolor("#1A2332")
    I_arr, rates = neuron.f_I_curve(np.linspace(0, 5, 40), duration_ms=1000)
    ax.plot(I_arr, rates, color="#7C4DFF", linewidth=2)
    ax.set_xlabel("Input Current I (nA)", color="white")
    ax.set_ylabel("Firing Rate (Hz)", color="white")
    ax.set_title("f--I Curve", color="white")
    ax.tick_params(colors="white")

    plt.tight_layout()
    plt.suptitle("LIF Neuron Dynamics", color="white", fontsize=14, y=1.01)
    plt.show()
