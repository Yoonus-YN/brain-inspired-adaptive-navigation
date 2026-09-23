"""
BIAN-SNN: Brain-Inspired Adaptive Navigation
============================================
Central configuration file.
All major parameters are defined here for reproducibility.
To run experiments with different settings, modify this file
or override parameters at the call site.
"""

# ---------------------------------------------
# Reproducibility
# ---------------------------------------------
RANDOM_SEED = 42

# ---------------------------------------------
# Environment / Grid World
# ---------------------------------------------
GRID_WIDTH  = 20          # columns
GRID_HEIGHT = 20          # rows
NUM_OBSTACLES = 15        # number of random obstacle cells
CELL_SIZE = 30            # pixels per cell (for rendering)

# ---------------------------------------------
# Robot
# ---------------------------------------------
ROBOT_START = (1, 1)      # (row, col)  -- 0-indexed
GOAL_POS    = (18, 18)    # (row, col)  -- 0-indexed
MAX_STEPS   = 500         # episode step limit

# ---------------------------------------------
# Sensors
# ---------------------------------------------
MAX_SENSOR_RANGE = 10     # cells
NUM_SENSORS = 3           # LEFT, FRONT, RIGHT

# ---------------------------------------------
# LIF Neuron
# ---------------------------------------------
TAU_M        = 20.0       # ms  -- membrane time constant
V_REST       = -70.0      # mV  -- resting potential
V_THRESHOLD  = -55.0      # mV  -- spike threshold
V_RESET      = -75.0      # mV  -- post-spike reset
R_MEMBRANE   = 10.0       # MOmega  -- membrane resistance
DT           = 1.0        # ms  -- simulation timestep
SIM_DURATION = 100.0      # ms  -- default simulation window

# ---------------------------------------------
# SNN Controller
# ---------------------------------------------
SNN_INPUT_NEURONS  = 30   # sensory population (10 per sensor)
SNN_HIDDEN_NEURONS = 20   # hidden layer population
SNN_OUTPUT_NEURONS = 3    # one per action: LEFT, FORWARD, RIGHT
SNN_WEIGHT_SCALE   = 0.5  # initial random weight scale
SNN_WINDOW_MS      = 50.0 # ms -- spike counting window per step

# ---------------------------------------------
# Reinforcement Learning
# ---------------------------------------------
RL_STATE_DIM      = 5     # [d_L, d_F, d_R, dist_goal, angle_goal]
RL_ACTION_DIM     = 3     # LEFT=0, FORWARD=1, RIGHT=2

# DQN hyperparameters
LEARNING_RATE     = 1e-3
GAMMA             = 0.99
EPSILON_START     = 1.0
EPSILON_END       = 0.05
EPSILON_DECAY     = 0.995
REPLAY_BUFFER_SIZE = 10_000
BATCH_SIZE        = 64
TARGET_UPDATE_FREQ = 10   # episodes

# Training
RL_TRAIN_EPISODES = 500
RL_EVAL_EPISODES  = 50

# ---------------------------------------------
# Reward values
# ---------------------------------------------
REWARD_GOAL      = +100.0
REWARD_COLLISION = -100.0
REWARD_STEP      = +1.0
REWARD_TOWARD    = +2.0
REWARD_AWAY      = -1.0

# ---------------------------------------------
# Experiments
# ---------------------------------------------
NUM_TRIALS       = 30     # trials per method per environment config
EXPERIMENT_SEEDS = list(range(NUM_TRIALS))

# ---------------------------------------------
# Paths
# ---------------------------------------------
import os
PROJECT_ROOT   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR       = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR    = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR    = os.path.join(RESULTS_DIR, "figures")
TABLES_DIR     = os.path.join(RESULTS_DIR, "tables")
LOGS_DIR       = os.path.join(RESULTS_DIR, "logs")
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "reinforcement_learning", "checkpoints")
