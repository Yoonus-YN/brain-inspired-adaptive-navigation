# 🧠 BIAN-SNN
## Brain-Inspired Adaptive Navigation Using Spiking Neural Networks and Reinforcement Learning

> **University AI / Computational Neuroscience / Robotics Project**
> Medium-scale | Software-only | Python | Zero-budget

---

## Overview

BIAN-SNN investigates whether a virtual mobile robot can navigate a 2D environment using a **brain-inspired Spiking Neural Network (SNN)** — and how that compares to conventional AI approaches.

```
Virtual Environment
        ↓
Virtual Sensors (LEFT · FRONT · RIGHT)
        ↓
Spiking Neural Network (LIF neurons)
        ↓
Motor Decision (LEFT / FORWARD / RIGHT)
        ↓
Robot Movement → Environment Feedback
```

Three navigation approaches are implemented and compared:

| Method | Description |
|--------|-------------|
| **A*** | Conventional shortest-path algorithm (baseline) |
| **SNN** | Brain-inspired Leaky Integrate-and-Fire network |
| **RL**  | Deep Q-Network reinforcement learning agent |

---

## Project Structure

```
brain-inspired-adaptive-navigation/
│
├── config.py                    ← All parameters (seed, grid, SNN, RL)
├── requirements.txt
│
├── environment/
│   ├── grid_world.py            ← 2D grid world
│   ├── robot.py                 ← Virtual mobile robot
│   ├── obstacles.py             ← Obstacle generation
│   └── sensors.py               ← Ray-casting distance sensors
│
├── navigation/
│   ├── astar.py                 ← A* pathfinding
│   └── metrics.py               ← Evaluation metrics tracker
│
├── neuroscience/
│   ├── lif_neuron.py            ← LIF neuron + population (NumPy)
│   ├── synapses.py              ← Synaptic weights + STDP
│   ├── sensory_layer.py         ← Sensor → spike encoding
│   └── snn_controller.py        ← Full SNN navigation controller
│
├── reinforcement_learning/
│   ├── environment.py           ← Gymnasium-compatible NavEnv
│   ├── agent.py                 ← DQN agent (PyTorch)
│   └── rewards.py               ← Reward function
│
├── experiments/
│   ├── run_astar.py             ← A* experiment
│   ├── run_snn.py               ← SNN experiment
│   ├── run_rl.py                ← RL training + evaluation
│   └── compare_models.py        ← Side-by-side comparison
│
├── notebooks/
│   ├── 01_environment.ipynb     ← Grid world interactive demo
│   ├── 02_astar.ipynb           ← A* visualisation
│   ├── 03_lif.ipynb             ← LIF neuron dynamics
│   ├── 04_snn.ipynb             ← SNN controller demo
│   ├── 05_rl.ipynb              ← RL training & evaluation
│   └── 06_comparison.ipynb      ← Final comparison
│
├── data/                        ← Raw experiment data
├── results/
│   ├── figures/                 ← Generated plots
│   ├── tables/                  ← CSV results
│   └── logs/                    ← Experiment logs
└── docs/
    ├── proposal/
    ├── report/
    └── presentation/
```

---

## Quick Start

### 1. Set up environment

```bash
# Create a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the baseline (A*)

```bash
python experiments/run_astar.py
```

### 3. Run SNN navigation

```bash
python experiments/run_snn.py
```

### 4. Train and evaluate RL agent

```bash
python experiments/run_rl.py --episodes 500
```

### 5. Compare all methods

```bash
python experiments/compare_models.py
```

### 6. Run unit tests

```bash
python -m pytest tests/ -v
```

### 7. Open notebooks

```bash
jupyter notebook notebooks/
```

---

## LIF Neuron Model

```
τ_m · dV/dt = -(V - V_rest) + R·I(t)

When V ≥ V_threshold → SPIKE → V_reset
```

Default parameters:

| Parameter | Value |
|-----------|-------|
| τ_m | 20 ms |
| V_rest | −70 mV |
| V_threshold | −55 mV |
| V_reset | −75 mV |
| R | 10 MΩ |
| dt | 1 ms |

---

## SNN Architecture

```
Sensor readings (3)
       ↓  rate encoding
Input neurons (30)  — 10 per sensor
       ↓  W_in [30 → 20]
Hidden LIF population (20 neurons)
       ↓  W_out [20 → 3]
Motor neurons (3)
       ↓  winner-take-all
Action: LEFT | FORWARD | RIGHT
```

---

## RL State & Action Space

**State** `S = [d_L, d_F, d_R, dist_goal, angle_goal]` — all normalised to `[0, 1]`

**Actions** `A = {LEFT=0, FORWARD=1, RIGHT=2}`

**Reward shaping:**

| Event | Reward |
|-------|--------|
| Reach goal | +100 |
| Collision | −100 |
| Move toward goal | +2 |
| Move safely | +1 |
| Move away from goal | −1 |

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Success rate | % of trials reaching the goal |
| Steps | Steps taken per episode |
| Collision count | Collisions per trial |
| Path efficiency | Optimal steps / actual steps |
| Navigation time | Wall-clock seconds |
| Spike count | Total SNN spikes (SNN only) |
| Reward | Cumulative episode reward (RL only) |

---

## Hardware Requirements

| Resource | Minimum |
|----------|---------|
| RAM | 4 GB (8 GB recommended) |
| Python | 3.9+ |
| GPU | Optional (CUDA auto-detected) |
| OS | Windows / Linux / macOS |

All simulations are lightweight and designed for an **8 GB RAM laptop**.

---

## Configuration

All parameters are centralised in [`config.py`](config.py).
Key settings:

```python
RANDOM_SEED    = 42       # reproducibility
GRID_WIDTH     = 20       # environment size
NUM_OBSTACLES  = 15       # obstacle density
MAX_STEPS      = 500      # episode limit
TAU_M          = 20.0     # LIF time constant (ms)
SNN_WINDOW_MS  = 50.0     # spike counting window
RL_TRAIN_EPISODES = 500   # DQN training episodes
```

---

## Development Roadmap

| Phase | Week | Status |
|-------|------|--------|
| Environment | 2 | ✅ |
| Robot + Sensors | 3–4 | ✅ |
| A* Baseline | 5 | ✅ |
| LIF Neuron | 6 | ✅ |
| SNN Controller | 7–8 | ✅ |
| RL Agent | 9–10 | ✅ |
| Experiments | 11–12 | ⬜ |
| Analysis | 13 | ⬜ |
| Report | 14 | ⬜ |

---

## Citation / Academic Use

If using this project as a reference for academic work, cite:

```
BIAN-SNN: Brain-Inspired Adaptive Navigation for a Virtual Mobile Robot
Using Spiking Neural Networks and Reinforcement Learning.
University project — computational neuroscience + AI + robotics.
```

---

## License

For academic use only. No commercial use permitted.
