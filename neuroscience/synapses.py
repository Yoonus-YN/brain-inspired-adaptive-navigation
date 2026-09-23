"""
synapses.py -- Synaptic Connections
====================================
Implements synaptic weight matrices and spike propagation
between neuron populations.

Provides:
  - SynapseMatrix : dense weight matrix connecting pre -> post populations
  - compute_synaptic_input : propagate spike vector through weights
  - stdp_update : Spike-Timing-Dependent Plasticity weight update (skeleton)

Usage:
    from neuroscience.synapses import SynapseMatrix
    syn = SynapseMatrix(n_pre=30, n_post=20, seed=42)
    I_post = syn.propagate(pre_spikes)   # shape (n_post,)
"""

import numpy as np
from typing import Optional


class SynapseMatrix:
    """
    Dense synaptic weight matrix connecting N_pre -> N_post neurons.

    Parameters
    ----------
    n_pre       : int   -- number of pre-synaptic neurons
    n_post      : int   -- number of post-synaptic neurons
    weight_scale: float -- scale factor for random weight initialisation
    weight_min  : float -- minimum weight (can be negative for inhibitory)
    weight_max  : float -- maximum weight
    seed        : int   -- random seed
    """

    def __init__(self,
                 n_pre:        int,
                 n_post:       int,
                 weight_scale: float = 0.5,
                 weight_min:   float = 0.0,
                 weight_max:   float = 1.0,
                 seed:         int   = 42):
        self.n_pre   = n_pre
        self.n_post  = n_post
        self.seed    = seed
        self.rng     = np.random.default_rng(seed)

        # Weight matrix W[i, j] = weight from pre_i -> post_j
        # Shape: (n_pre, n_post)
        self.W = self.rng.uniform(
            weight_min,
            weight_max * weight_scale,
            size=(n_pre, n_post)
        ).astype(np.float32)

    def propagate(self, pre_spikes: np.ndarray) -> np.ndarray:
        """
        Compute post-synaptic input current from pre-synaptic spike vector.

        I_post[j] = Sum_i (W[i,j] * pre_spikes[i])

        Parameters
        ----------
        pre_spikes : np.ndarray shape (n_pre,) -- spike vector (0 or 1)

        Returns
        -------
        np.ndarray shape (n_post,) -- weighted input currents.
        """
        return pre_spikes.astype(np.float32) @ self.W  # (n_post,)

    def set_weights(self, W: np.ndarray) -> None:
        """Manually set the weight matrix."""
        assert W.shape == (self.n_pre, self.n_post), \
            f"Expected shape ({self.n_pre}, {self.n_post}), got {W.shape}"
        self.W = W.astype(np.float32)

    def scale_weights(self, factor: float) -> None:
        """Multiply all weights by a scalar factor."""
        self.W *= factor

    def clip_weights(self,
                     w_min: float = 0.0,
                     w_max: float = 1.0) -> None:
        """Clip all weights to [w_min, w_max]."""
        np.clip(self.W, w_min, w_max, out=self.W)

    def __repr__(self) -> str:
        return (f"SynapseMatrix(n_pre={self.n_pre}, n_post={self.n_post}, "
                f"mean_w={self.W.mean():.4f}, std_w={self.W.std():.4f})")


# -- STDP Update (Hebbian spike-timing plasticity) ------------------------------

def stdp_update(W:            np.ndarray,
                pre_spikes:   np.ndarray,
                post_spikes:  np.ndarray,
                pre_trace:    np.ndarray,
                post_trace:   np.ndarray,
                A_plus:       float = 0.01,
                A_minus:      float = 0.012,
                tau_plus:     float = 20.0,
                tau_minus:    float = 20.0,
                dt:           float = 1.0,
                w_min:        float = 0.0,
                w_max:        float = 1.0
                ) -> tuple:
    """
    One timestep of Spike-Timing-Dependent Plasticity (STDP).

    LTP (Long-Term Potentiation):  post fires after pre  -> W increases
    LTD (Long-Term Depression):    pre fires after post   -> W decreases

    Trace-based STDP rule:
      pre_trace[i]  += dt * (-pre_trace[i]/tau_plus)  + pre_spikes[i]
      post_trace[j] += dt * (-post_trace[j]/tau_minus) + post_spikes[j]

      DeltaW[i,j] = A_plus  * pre_trace[i]  * post_spikes[j]  (LTP)
               - A_minus * post_trace[j] * pre_spikes[i]   (LTD)

    Parameters
    ----------
    W            : np.ndarray (n_pre, n_post) -- weight matrix to update in place
    pre_spikes   : np.ndarray (n_pre,)  -- current pre spikes (0/1)
    post_spikes  : np.ndarray (n_post,) -- current post spikes (0/1)
    pre_trace    : np.ndarray (n_pre,)  -- eligibility trace for pre neurons
    post_trace   : np.ndarray (n_post,) -- eligibility trace for post neurons
    A_plus       : float -- LTP learning rate
    A_minus      : float -- LTD learning rate
    tau_plus     : float -- LTP trace time constant (ms)
    tau_minus    : float -- LTD trace time constant (ms)
    dt           : float -- timestep (ms)
    w_min, w_max : float -- weight bounds

    Returns
    -------
    (pre_trace, post_trace) -- updated eligibility traces.
    """
    pre_spikes  = pre_spikes.astype(np.float32)
    post_spikes = post_spikes.astype(np.float32)

    # Decay traces
    pre_trace  += dt * (-pre_trace  / tau_plus)  + pre_spikes
    post_trace += dt * (-post_trace / tau_minus) + post_spikes

    # Weight update: outer product for LTP and LTD
    # W[i,j] += A_plus  * pre_trace[i]  * post_spikes[j]
    #          - A_minus * post_trace[j] * pre_spikes[i]
    dW = (A_plus  * np.outer(pre_trace,  post_spikes)
        - A_minus * np.outer(pre_spikes, post_trace))
    W += dW
    np.clip(W, w_min, w_max, out=W)

    return pre_trace, post_trace


# -- Standalone test ------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    syn = SynapseMatrix(n_pre=30, n_post=20, weight_scale=0.5, seed=42)
    print(syn)

    # Random spike vector
    rng = np.random.default_rng(0)
    pre_spikes = (rng.uniform(0, 1, 30) > 0.7).astype(np.float32)
    I_post = syn.propagate(pre_spikes)

    print(f"Pre spikes:  {pre_spikes.astype(int)}")
    print(f"Post input:  {I_post.round(3)}")

    # Visualise weight matrix
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#1A2332")
    ax.set_facecolor("#1A2332")
    im = ax.imshow(syn.W.T, aspect="auto", cmap="plasma")
    plt.colorbar(im, ax=ax, label="Weight")
    ax.set_xlabel("Pre-synaptic neuron", color="white")
    ax.set_ylabel("Post-synaptic neuron", color="white")
    ax.set_title("Synaptic Weight Matrix", color="white")
    ax.tick_params(colors="white")
    plt.tight_layout()
    plt.show()
