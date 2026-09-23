"""
agent.py -- Deep Q-Network (DQN) Agent
=======================================
Implements a DQN agent for the navigation environment.

Architecture:
    State (5) -> FC(64) -> ReLU -> FC(64) -> ReLU -> FC(3)
    (Q-values for LEFT, FORWARD, RIGHT)

Features:
  - Experience replay buffer
  - Target network (periodic soft/hard update)
  - Epsilon-greedy exploration with decay
  - GPU support (if NVIDIA RTX is available)

Usage:
    from reinforcement_learning.agent import DQNAgent
    agent = DQNAgent(state_dim=5, action_dim=3)
    action = agent.select_action(state)
    agent.store(state, action, reward, next_state, done)
    loss = agent.train_step()
    agent.save("checkpoints/dqn_nav.pth")
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import os
from typing import Optional, Tuple


# -- Q-Network -----------------------------------------------------------------

class QNetwork(nn.Module):
    """
    Feedforward Q-network.

    Parameters
    ----------
    state_dim  : int -- observation dimension
    action_dim : int -- number of discrete actions
    hidden_dim : int -- hidden layer size
    """

    def __init__(self,
                 state_dim:  int = 5,
                 action_dim: int = 3,
                 hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -- Replay Buffer -------------------------------------------------------------

class ReplayBuffer:
    """
    Circular experience replay buffer.

    Parameters
    ----------
    capacity : int -- maximum number of transitions stored
    """

    def __init__(self, capacity: int = 10_000):
        self.buffer: deque = deque(maxlen=capacity)

    def push(self,
             state:      np.ndarray,
             action:     int,
             reward:     float,
             next_state: np.ndarray,
             done:       bool) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states,      dtype=np.float32),
            np.array(actions,     dtype=np.int64),
            np.array(rewards,     dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones,       dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


# -- DQN Agent -----------------------------------------------------------------

class DQNAgent:
    """
    Deep Q-Network agent.

    Parameters
    ----------
    state_dim       : int   -- observation size
    action_dim      : int   -- number of discrete actions
    hidden_dim      : int   -- Q-network hidden layer size
    lr              : float -- learning rate
    gamma           : float -- discount factor
    epsilon_start   : float -- initial exploration rate
    epsilon_end     : float -- minimum exploration rate
    epsilon_decay   : float -- multiplicative decay per episode
    buffer_capacity : int   -- replay buffer size
    batch_size      : int   -- training batch size
    target_update   : int   -- episodes between target network hard updates
    seed            : int   -- random seed
    device          : str   -- 'auto', 'cuda', or 'cpu'
    """

    def __init__(self,
                 state_dim:       int   = 5,
                 action_dim:      int   = 3,
                 hidden_dim:      int   = 64,
                 lr:              float = 1e-3,
                 gamma:           float = 0.99,
                 epsilon_start:   float = 1.0,
                 epsilon_end:     float = 0.05,
                 epsilon_decay:   float = 0.995,
                 buffer_capacity: int   = 10_000,
                 batch_size:      int   = 64,
                 target_update:   int   = 10,
                 seed:            int   = 42,
                 device:          str   = "auto"):
        self.action_dim    = action_dim
        self.gamma         = gamma
        self.epsilon       = epsilon_start
        self.epsilon_end   = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size    = batch_size
        self.target_update = target_update

        # Device selection
        if device == "auto":
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Seeding
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Networks
        self.policy_net = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimiser
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn   = nn.MSELoss()

        # Replay buffer
        self.buffer = ReplayBuffer(buffer_capacity)

        # Tracking
        self.episode_count = 0
        self.train_steps   = 0

        print(f"DQNAgent | device={self.device} | "
              f"params={sum(p.numel() for p in self.policy_net.parameters()):,}")

    # -- Action selection ------------------------------------------------------

    def select_action(self, state: np.ndarray) -> int:
        """
        Epsilon-greedy action selection.

        Parameters
        ----------
        state : np.ndarray shape (state_dim,)

        Returns
        -------
        int -- action index
        """
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32,
                             device=self.device).unsqueeze(0)
            q = self.policy_net(s)
            return int(q.argmax().item())

    def select_greedy_action(self, state: np.ndarray) -> int:
        """Always pick greedy (no exploration) -- use for evaluation."""
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32,
                             device=self.device).unsqueeze(0)
            return int(self.policy_net(s).argmax().item())

    # -- Experience storage ----------------------------------------------------

    def store(self,
              state:      np.ndarray,
              action:     int,
              reward:     float,
              next_state: np.ndarray,
              done:       bool) -> None:
        """Store a transition in the replay buffer."""
        self.buffer.push(state, action, reward, next_state, done)

    # -- Training --------------------------------------------------------------

    def train_step(self) -> Optional[float]:
        """
        Sample a minibatch and perform one gradient update.

        Returns
        -------
        float -- loss value, or None if buffer too small.
        """
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(
            self.batch_size)

        states      = torch.tensor(states,      device=self.device)
        actions     = torch.tensor(actions,     device=self.device)
        rewards     = torch.tensor(rewards,     device=self.device)
        next_states = torch.tensor(next_states, device=self.device)
        dones       = torch.tensor(dones,       device=self.device)

        # Current Q-values
        q_current = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()

        # Target Q-values (using target network)
        with torch.no_grad():
            q_next = self.target_net(next_states).max(1).values
            q_target = rewards + self.gamma * q_next * (1 - dones)

        loss = self.loss_fn(q_current, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        self.train_steps += 1
        return float(loss.item())

    def end_episode(self) -> None:
        """Call at end of each episode: decay epsilon, update target network."""
        self.episode_count += 1
        self.epsilon = max(self.epsilon_end,
                          self.epsilon * self.epsilon_decay)

        if self.episode_count % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    # -- Save / Load -----------------------------------------------------------

    def save(self, filepath: str) -> None:
        """Save policy network weights and agent state."""
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        torch.save({
            "policy_net":    self.policy_net.state_dict(),
            "target_net":    self.target_net.state_dict(),
            "optimizer":     self.optimizer.state_dict(),
            "epsilon":       self.epsilon,
            "episode_count": self.episode_count,
            "train_steps":   self.train_steps,
        }, filepath)
        print(f"Saved agent -> {filepath}")

    def load(self, filepath: str) -> None:
        """Load agent state from file."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon       = checkpoint["epsilon"]
        self.episode_count = checkpoint["episode_count"]
        self.train_steps   = checkpoint["train_steps"]
        print(f"Loaded agent <- {filepath} "
              f"(ep={self.episode_count}, eps={self.epsilon:.3f})")

    def __repr__(self) -> str:
        return (f"DQNAgent(eps={self.epsilon:.3f}, "
                f"episodes={self.episode_count}, "
                f"buffer={len(self.buffer)})")
