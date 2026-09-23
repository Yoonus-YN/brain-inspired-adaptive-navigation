"""
server.py -- BIAN-SNN Interactive Web Dashboard Server
======================================================
Serves the live interactive visualization dashboard and provides
REST APIs to interact with the Python simulation models in real-time.
"""

import sys
import os
import json
from typing import Dict, Any

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from flask import Flask, jsonify, request, send_from_directory, send_file
import numpy as np

from environment.grid_world import GridWorld, FREE, OBSTACLE
from environment.robot import Robot, Action, EAST, NORTH, SOUTH, WEST, ORIENTATION_NAMES
from environment.sensors import SensorArray
from environment.obstacles import load_preset, ObstacleGenerator
from navigation.astar import AStarNavigator
from neuroscience.snn_controller import SNNController
from reinforcement_learning.environment import NavEnv
from reinforcement_learning.agent import DQNAgent

app = Flask(__name__, static_folder="static")

# -- Global Simulation State --
class SimulationEngine:
    def __init__(self, grid_size: int = 15, preset: str = "sparse", seed: int = 42):
        self.grid_size = grid_size
        self.preset = preset
        self.seed = seed
        self.reset(grid_size, preset, seed)

    def reset(self, grid_size: int = 15, preset: str = "sparse", seed: int = 42):
        self.grid_size = grid_size
        self.preset = preset
        self.seed = seed
        
        self.world = GridWorld(grid_size, grid_size, seed=seed)
        self.world.set_start((1, 1))
        self.world.set_goal((grid_size - 2, grid_size - 2))
        
        if preset != "empty":
            try:
                load_preset(preset, self.world)
            except Exception:
                ObstacleGenerator(seed).random_obstacles(self.world, n=10)

        self.robot = Robot(start=self.world.start, orientation=EAST)
        self.sensors = SensorArray(max_range=10)
        
        # Controllers
        self.astar_nav = AStarNavigator()
        self.astar_path = []
        self.astar_step_idx = 0
        self._compute_astar()

        self.snn = SNNController(seed=seed)
        
        self.rl_agent = DQNAgent(state_dim=5, action_dim=3)
        chk_path = os.path.join(REPO_ROOT, "reinforcement_learning", "checkpoints", "dqn_nav.pth")
        if os.path.exists(chk_path):
            try:
                self.rl_agent.load(chk_path)
            except Exception:
                pass

        self.last_step_info: Dict[str, Any] = {}

    def _compute_astar(self):
        path, info = self.astar_nav.find_path(self.world, start=self.robot.pos, goal=self.world.goal)
        self.astar_path = path
        self.astar_step_idx = 0
        return info

    def get_state(self) -> Dict[str, Any]:
        raw_sensors = self.sensors.read(self.robot, self.world)
        norm_sensors = self.sensors.read_normalised(self.robot, self.world).tolist()
        
        grid_matrix = self.world.get_grid().tolist()
        
        return {
            "width": self.world.width,
            "height": self.world.height,
            "start": list(self.world.start) if self.world.start else None,
            "goal": list(self.world.goal) if self.world.goal else None,
            "grid": grid_matrix,
            "robot": {
                "pos": list(self.robot.pos),
                "orientation": self.robot.orientation,
                "orient_name": ORIENTATION_NAMES.get(self.robot.orientation, "E"),
                "steps": self.robot.steps,
                "collisions": self.robot.collision_count,
                "reached_goal": (self.robot.pos == self.world.goal),
                "trajectory": [list(p) for p in self.robot.trajectory],
            },
            "sensors": {
                "raw": raw_sensors,
                "norm": norm_sensors,
                "max_range": self.sensors.max_range,
            },
            "last_step": self.last_step_info,
            "astar_path": [list(p) for p in self.astar_path],
        }

    def step(self, method: str = "astar") -> Dict[str, Any]:
        if self.robot.pos == self.world.goal:
            return {"done": True, "reason": "goal_reached"}

        action = Action.MOVE_FORWARD
        extra_info = {}

        if method == "astar":
            if not self.astar_path or self.robot.pos not in self.astar_path:
                self._compute_astar()

            if self.astar_path:
                try:
                    curr_idx = self.astar_path.index(self.robot.pos)
                    if curr_idx + 1 < len(self.astar_path):
                        next_pos = self.astar_path[curr_idx + 1]
                        dr = next_pos[0] - self.robot.pos[0]
                        dc = next_pos[1] - self.robot.pos[1]
                        
                        target_orient = self.robot.orientation
                        if dr == -1 and dc == 0:
                            target_orient = NORTH
                        elif dr == 1 and dc == 0:
                            target_orient = SOUTH
                        elif dr == 0 and dc == 1:
                            target_orient = EAST
                        elif dr == 0 and dc == -1:
                            target_orient = WEST

                        if self.robot.orientation == target_orient:
                            action = Action.MOVE_FORWARD
                        elif (self.robot.orientation + 1) % 4 == target_orient:
                            action = Action.TURN_RIGHT
                        else:
                            action = Action.TURN_LEFT
                        
                        extra_info["target_waypoint"] = list(next_pos)
                except ValueError:
                    self._compute_astar()

        elif method == "snn":
            norm_readings = self.sensors.read_normalised(self.robot, self.world)
            action_int, motor_spikes = self.snn.decide(norm_readings)
            action = Action(action_int)
            extra_info["motor_spikes"] = motor_spikes.tolist()
            extra_info["hidden_voltages"] = self.snn.hidden_pop.V.tolist()
            extra_info["motor_voltages"] = self.snn.output_pop.V.tolist()

        elif method == "rl":
            norm_readings = self.sensors.read_normalised(self.robot, self.world)
            max_dist = float(np.hypot(self.world.width, self.world.height))
            dist = float(self.robot.distance_to_goal(self.world.goal)) / max_dist
            angle = float(self.robot.angle_to_goal(self.world.goal))
            angle_norm = (angle + np.pi) / (2 * np.pi)
            
            state_vec = np.array([
                norm_readings[0], norm_readings[1], norm_readings[2],
                np.clip(dist, 0.0, 1.0), np.clip(angle_norm, 0.0, 1.0)
            ], dtype=np.float32)
            
            action_int = self.rl_agent.select_action(state_vec, evaluate=True)
            action = Action(action_int)
            extra_info["state_vector"] = state_vec.tolist()

        collided, reached_goal = self.robot.step(action, self.world)
        
        self.last_step_info = {
            "method": method,
            "action": action.name,
            "action_int": int(action),
            "collided": collided,
            "reached_goal": reached_goal,
            "extra": extra_info,
        }
        
        return self.get_state()

    def toggle_obstacle(self, row: int, col: int) -> bool:
        if (row, col) == self.world.start or (row, col) == self.world.goal:
            return False
        if row <= 0 or row >= self.world.height - 1 or col <= 0 or col >= self.world.width - 1:
            return False
        
        if self.world.is_obstacle(row, col):
            self.world.remove_obstacle(row, col)
        else:
            self.world.add_obstacle(row, col)
            
        self._compute_astar()
        return True


engine = SimulationEngine()


# -- Routes --

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/<path:path>")
def static_proxy(path):
    return send_from_directory(app.static_folder, path)

@app.route("/api/state", methods=["GET"])
def api_state():
    return jsonify(engine.get_state())

@app.route("/api/step", methods=["POST"])
def api_step():
    data = request.get_json() or {}
    method = data.get("method", "astar")
    return jsonify(engine.step(method))

@app.route("/api/reset", methods=["POST"])
def api_reset():
    data = request.get_json() or {}
    grid_size = int(data.get("grid_size", 15))
    preset = data.get("preset", "sparse")
    seed = int(data.get("seed", 42))
    engine.reset(grid_size, preset, seed)
    return jsonify(engine.get_state())

@app.route("/api/toggle_obstacle", methods=["POST"])
def api_toggle_obstacle():
    data = request.get_json() or {}
    r = int(data.get("row", 0))
    c = int(data.get("col", 0))
    success = engine.toggle_obstacle(r, c)
    return jsonify({"success": success, "state": engine.get_state()})

@app.route("/api/figures", methods=["GET"])
def api_figures():
    fig_dir = os.path.join(REPO_ROOT, "results", "figures")
    files = [f for f in os.listdir(fig_dir) if f.endswith(".png")]
    return jsonify({"figures": files})

@app.route("/api/figures/<filename>")
def serve_figure(filename):
    fig_dir = os.path.join(REPO_ROOT, "results", "figures")
    return send_file(os.path.join(fig_dir, filename), mimetype="image/png")

@app.route("/api/summary", methods=["GET"])
def api_summary():
    csv_path = os.path.join(REPO_ROOT, "results", "tables", "comparison_summary.csv")
    if os.path.exists(csv_path):
        import pandas as pd
        df = pd.read_csv(csv_path)
        return jsonify(df.to_dict(orient="records"))
    return jsonify([])


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n============================================================")
    print(f"  BIAN-SNN Web Dashboard Server Running on:")
    print(f"  http://localhost:{port}")
    print(f"============================================================\n")
    app.run(host="0.0.0.0", port=port, debug=False)
