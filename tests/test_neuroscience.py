"""Unit tests for the neuroscience module."""
import numpy as np
import pytest

from neuroscience.lif_neuron import LIFNeuron, LIFPopulation
from neuroscience.sensory_layer import SensoryEncoder
from neuroscience.synapses import SynapseMatrix
from neuroscience.snn_controller import SNNController


def test_lif_neuron_subthreshold():
    neuron = LIFNeuron()
    # Inject subthreshold current
    spiked = neuron.step(I_ext=0.5)
    assert not spiked
    assert neuron.V > -70.0  # Depolarized from resting


def test_lif_neuron_spike_and_reset():
    neuron = LIFNeuron()
    spiked = False
    # Inject strong current
    for _ in range(50):
        if neuron.step(I_ext=5.0):
            spiked = True
            break
    assert spiked
    assert neuron.V == neuron.v_reset


def test_lif_population():
    pop = LIFPopulation(n=10)
    I = np.full(10, 5.0)
    spikes = pop.step(I)
    assert len(spikes) == 10
    assert len(pop.V) == 10


def test_sensory_encoder():
    encoder = SensoryEncoder(neurons_per_sensor=10, max_rate_hz=100.0)
    readings = np.array([0.5, 0.8, 0.2])
    spikes = encoder.encode(readings, window_ms=50.0)
    assert len(spikes) == 30
    assert np.all(spikes >= 0.0)

    currents = encoder.encode_to_current(readings)
    assert len(currents) == 30
    assert np.all(currents >= 0.0)


def test_synapse_matrix():
    syn = SynapseMatrix(n_pre=5, n_post=3, weight_scale=0.2, seed=42)
    assert syn.W.shape == (5, 3)
    pre_act = np.ones(5)
    out_curr = syn.propagate(pre_act)
    assert len(out_curr) == 3
    assert np.all(out_curr > 0.0)


def test_snn_controller_action():
    controller = SNNController(neurons_per_sensor=10, n_hidden=15, seed=42)
    sensor_norm = np.array([0.3, 0.9, 0.5])
    action, motor_spikes = controller.decide(sensor_norm)

    assert action in [0, 1, 2]
    assert len(motor_spikes) == 3
