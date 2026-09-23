"""
snn_controller.py -- SNN Navigation Controller
==============================================
Full pipeline from sensor readings -> SNN -> robot action.

Architecture:
    Sensor readings (3)
         v
    Sensory layer (30 input neurons, 10 per sensor)
         v  [W_in: 30 -> 20]
    Hidden LIF population (20 neurons)
         v  [W_out: 20 -> 3]
    Motor output neurons (3) -> [LEFT, FORWARD, RIGHT]
         v
    Winner-take-all -> Action

The winning action is determined by which motor neuron
accumulates the most spikes over a 50 ms window.

Usage:
    from neuroscience.snn_controller import SNNController
    controller = SNNController(seed=42)
    action = controller.decide(sensor_norm, world, robot)
    # action: 0=LEFT, 1=FORWARD, 2=RIGHT
"""

import numpy as np
from typing import Optional, Tuple, TYPE_CHECKING

from neuroscience.lif_neuron import LIFPopulation
from neuroscience.synapses import SynapseMatrix
from neuroscience.sensory_layer import SensoryEncoder

if TYPE_CHECKING:
    from environment.grid_world import GridWorld
    from environment.robot import Robot


class SNNController:
    """
    Three-layer SNN navigation controller.

    Layer 0 -- Sensory (input)
        30 neurons: 10 per sensor (LEFT, FRONT, RIGHT)

    Layer 1 -- Hidden
        Configurable hidden neuron population.

    Layer 2 -- Motor output
        3 neurons: one per action (LEFT=0, FORWARD=1, RIGHT=2)

    Parameters
    ----------
    neurons_per_sensor : int   -- input neurons per sensor channel
    n_hidden           : int   -- hidden population size
    max_sensor_range   : int   -- for sensor normalisation
    max_rate_hz        : float -- peak sensory firing rate (Hz)
    window_ms          : float -- spike counting window (ms)
    dt                 : float -- timestep (ms)
    seed               : int
    """

    ACTION_NAMES = ["LEFT", "FORWARD", "RIGHT"]

    def __init__(self,
                 neurons_per_sensor: int   = 10,
                 n_hidden:           int   = 20,
                 max_sensor_range:   int   = 10,
                 max_rate_hz:        float = 100.0,
                 window_ms:          float = 50.0,
                 dt:                 float = 1.0,
                 weight_scale:       float = 0.5,
                 seed:               int   = 42):
        self.neurons_per_sensor = neurons_per_sensor
        self.n_input  = 3 * neurons_per_sensor
        self.n_hidden = n_hidden
        self.n_output = 3
        self.window_ms = window_ms
        self.dt        = dt
        self.seed      = seed

        # -- Sensory encoder
        self.encoder = SensoryEncoder(
            neurons_per_sensor=neurons_per_sensor,
            max_rate_hz=max_rate_hz,
            dt=dt,
            seed=seed,
        )

        # -- Hidden LIF population
        self.hidden_pop = LIFPopulation(
            n           = n_hidden,
            tau_m       = 20.0,
            v_rest      = -70.0,
            v_threshold = -55.0,
            v_reset     = -75.0,
            r_membrane  = 10.0,
            dt          = dt,
        )

        # -- Output LIF population (motor neurons)
        self.output_pop = LIFPopulation(
            n           = self.n_output,
            tau_m       = 15.0,
            v_rest      = -70.0,
            v_threshold = -55.0,
            v_reset     = -75.0,
            r_membrane  = 10.0,
            dt          = dt,
        )

        # -- Synaptic weight matrices
        self.W_in  = SynapseMatrix(self.n_input,  n_hidden,
                                   weight_scale=weight_scale, seed=seed)
        self.W_out = SynapseMatrix(n_hidden, self.n_output,
                                   weight_scale=max(weight_scale, 1.0),
                                   weight_min=0.1,
                                   weight_max=2.0,
                                   seed=seed+1)

        # -- State for recording
        self.last_spike_counts: Optional[np.ndarray] = None
        self.last_action:       Optional[int]         = None
        self.total_spikes:      int = 0
        self.step_count:        int = 0
        self._rng = np.random.default_rng(seed)

    # -- Core decision ---------------------------------------------------------

    def decide(self, sensor_norm: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        Run one SNN inference step and return an action.

        Parameters
        ----------
        sensor_norm : np.ndarray shape (3,) -- [left, front, right] in [0,1]

        Returns
        -------
        action      : int -- 0=LEFT, 1=FORWARD, 2=RIGHT
        motor_spikes: np.ndarray shape (3,) -- spike counts per motor neuron
        """
        T = int(self.window_ms / self.dt)

        # 1. Convert sensor readings to input currents
        I_input = self.encoder.encode_to_current(sensor_norm)  # (n_input,)

        # 2. Run forward pass over T timesteps
        motor_spike_counts = np.zeros(self.n_output, dtype=np.float32)
        hidden_spike_accum  = np.zeros(self.n_hidden, dtype=np.float32)
        self.hidden_pop.reset_state()
        self.output_pop.reset_state()

        for _ in range(T):
            # Input -> Hidden
            I_hidden = self.W_in.propagate(I_input)
            hidden_spikes = self.hidden_pop.step(I_hidden)      # (n_hidden,) 0/1
            hidden_spike_accum += hidden_spikes.astype(np.float32)

            # Hidden -> Output: use accumulated current proportional to
            # hidden firing to ensure output neurons receive enough drive
            I_output = self.W_out.propagate(
                self.hidden_pop.V - self.hidden_pop.v_rest      # normalised voltage drive
            )
            output_spikes = self.output_pop.step(I_output)      # (n_output,) 0/1

            motor_spike_counts += output_spikes.astype(np.float32)

        # 3. Action selection
        if motor_spike_counts.sum() == 0:
            # No spikes at all -- random action (shouldn't happen after fix)
            action = int(self._rng.integers(0, self.n_output))
        else:
            # Softmax over spike counts for probabilistic selection
            # This avoids the tie-breaking bias of argmax(equal values) = 0
            counts = motor_spike_counts.astype(np.float64)
            temperature = max(counts.max(), 1.0)
            probs = np.exp((counts - counts.max()) / temperature)
            probs /= probs.sum()
            action = int(self._rng.choice(self.n_output, p=probs))

        self.last_spike_counts = motor_spike_counts
        self.last_action       = action
        self.total_spikes     += int(motor_spike_counts.sum())
        self.step_count       += 1

        return action, motor_spike_counts

    # -- Utility methods -------------------------------------------------------

    def reset_stats(self) -> None:
        """Reset spike counters and step tracker."""
        self.total_spikes = 0
        self.step_count   = 0
        self.last_spike_counts = None
        self.last_action       = None

    def mean_spikes_per_step(self) -> float:
        """Average spike count per decision step."""
        if self.step_count == 0:
            return 0.0
        return self.total_spikes / self.step_count

    def action_name(self, action: int) -> str:
        return self.ACTION_NAMES[action]

    def describe_last_decision(self) -> str:
        """Human-readable string for the last decision."""
        if self.last_spike_counts is None:
            return "No decision made yet."
        counts = self.last_spike_counts
        lines = ["SNN Decision:"]
        for i, name in enumerate(self.ACTION_NAMES):
            bar = "#" * int(counts[i]) + "." * max(0, 10 - int(counts[i]))
            marker = " <- SELECTED" if i == self.last_action else ""
            lines.append(f"  {name:8s}: {int(counts[i]):3d} spikes  |{bar}|{marker}")
        return "\n".join(lines)

    def set_weights(self,
                    W_in:  Optional[np.ndarray] = None,
                    W_out: Optional[np.ndarray] = None) -> None:
        """Manually override weight matrices."""
        if W_in  is not None: self.W_in.set_weights(W_in)
        if W_out is not None: self.W_out.set_weights(W_out)

    def __repr__(self) -> str:
        return (f"SNNController("
                f"input={self.n_input}, "
                f"hidden={self.n_hidden}, "
                f"output={self.n_output}, "
                f"window={self.window_ms}ms)")


# -- Standalone demo ------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from environment.grid_world import GridWorld
    from environment.robot import Robot, EAST
    from environment.sensors import SensorArray
    from environment.obstacles import load_preset

    # Setup
    world = GridWorld(20, 20, seed=42)
    load_preset("moderate", world)
    robot = Robot(start=(1, 1), orientation=EAST)
    sensors = SensorArray(max_range=10)
    controller = SNNController(seed=42)

    print(controller)
    print()

    # Run 10 steps
    trajectory = [robot.pos]
    spike_history = []

    for step in range(15):
        _, sensor_norm = sensors.read_all(robot, world)
        action, motor_spikes = controller.decide(sensor_norm)
        collision, goal = robot.step(action, world)
        trajectory.append(robot.pos)
        spike_history.append(motor_spikes.copy())

        status = "COLLISION" if collision else ("GOAL!" if goal else "")
        print(f"Step {step+1:2d}: sensors={sensor_norm.round(2)} "
              f"-> {controller.ACTION_NAMES[action]:8s} "
              f"spikes={motor_spikes.astype(int)} {status}")
        if goal:
            break

    print()
    print(controller.describe_last_decision())
    print(f"\nTotal spikes: {controller.total_spikes}")
    print(f"Mean spikes/step: {controller.mean_spikes_per_step():.1f}")

    # Visualise
    spike_history = np.array(spike_history)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#1A2332")

    world.render(robot_pos=robot.pos, path=trajectory,
                 title="SNN Navigation", ax=axes[0], show=False)

    ax = axes[1]
    ax.set_facecolor("#1A2332")
    colors = ["#FF6B6B", "#00BFA5", "#7C4DFF"]
    for i, (name, col) in enumerate(zip(["LEFT", "FORWARD", "RIGHT"], colors)):
        ax.plot(spike_history[:, i], color=col, label=name, linewidth=2)
    ax.set_xlabel("Step", color="white")
    ax.set_ylabel("Motor spike count", color="white")
    ax.set_title("Motor Neuron Activity", color="white")
    ax.tick_params(colors="white")
    ax.legend(facecolor="#2D3A4A", labelcolor="white")

    plt.tight_layout()
    plt.show()
