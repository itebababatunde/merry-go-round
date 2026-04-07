"""
simulator.py — Main simulation loop for the MGR paper reproduction.

The Simulator class wires together all Phase 1-3 modules into a single
runnable experiment. It supports three methods:

  'mgr'      — Full MGR algorithm (Algorithm 1) + CLF-CBF QP safety filter
  'clf_cbf'  — CLF-CBF QP only, no deadlock prevention (baseline)
  'orca'     — ORCA baseline (Phase 5; raises NotImplementedError here)

Double-buffer pattern:
    All control inputs are computed from the *current* (pre-step) robot
    positions before any robot moves. This ensures decentralised correctness:
    every robot sees the same world snapshot when making its decision.

Right-hand rule (paper §III-E):
    Obstacle avoidance is handled as a post-QP ω override, NOT as QP
    constraints. When a robot is within DELTA_COMM of an obstacle surface,
    its angular velocity is blended toward the clockwise tangent of the
    nearest obstacle. The blending strength scales linearly with proximity
    (0 at DELTA_COMM, 1 at the surface).
"""

import math
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from experiments.config import DT, T_MAX, DELTA_COMM, D_SAFE, W_MAX

from src.robot import RobotMode
from src.controllers.goal_controller import goal_control
from src.controllers.mgr_controller import mgr_control
from src.controllers.clf_cbf_qp import clf_cbf_qp
from src.mgr.roundabout_mgr import run_mgr_update
from src.mgr.escape import is_escapable, escape_robot
from src.simulation.metrics import compute_metrics

# Finite-difference step for SDF gradient in right-hand rule
_RHR_EPS = 1e-3
# Heading gain for right-hand rule angular correction
_RHR_K_TURN = 2.0


def _wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def _sdf_gradient(obs, pos: np.ndarray) -> np.ndarray:
    """Central finite-difference gradient of obs.sdf at pos."""
    ex = np.array([_RHR_EPS, 0.0])
    ey = np.array([0.0, _RHR_EPS])
    gx = (obs.sdf(pos + ex) - obs.sdf(pos - ex)) / (2 * _RHR_EPS)
    gy = (obs.sdf(pos + ey) - obs.sdf(pos - ey)) / (2 * _RHR_EPS)
    return np.array([gx, gy])


def _right_hand_rule(robot, w: float, obstacles: list) -> float:
    """
    Post-QP ω override implementing the paper's right-hand rule (§III-E).

    Finds the nearest obstacle within DELTA_COMM. If found, blends the QP
    output ω toward the clockwise tangent of that obstacle's surface. The
    blend strength scales linearly: 0 at DELTA_COMM, 1 at the surface.

    Parameters
    ----------
    robot : Robot
        The robot being controlled. Reads .pos, .theta.
    w : float
        Angular velocity from QP (before override).
    obstacles : list
        Obstacle objects with .sdf(point) method.

    Returns
    -------
    float
        Possibly overridden angular velocity, clipped to ±W_MAX.
    """
    nearest_d = float('inf')
    nearest_obs = None
    for obs in obstacles:
        d = obs.sdf(robot.pos)
        if d < nearest_d:
            nearest_d = d
            nearest_obs = obs

    if nearest_obs is None or nearest_d >= DELTA_COMM:
        return w

    # Outward surface normal via SDF gradient
    grad = _sdf_gradient(nearest_obs, robot.pos)
    grad_norm = np.linalg.norm(grad)
    if grad_norm < 1e-8:
        return w
    n_hat = grad / grad_norm  # unit outward normal

    # Clockwise (right-hand) tangent: rotate n_hat by -90°
    t_hat = np.array([n_hat[1], -n_hat[0]])

    # Desired heading to track the CW tangent
    phi_rhr = math.atan2(t_hat[1], t_hat[0])
    heading_err = _wrap_angle(phi_rhr - robot.theta)
    w_rhr = float(np.clip(_RHR_K_TURN * heading_err, -W_MAX, W_MAX))

    # Blend: strength → 1 as robot approaches surface
    strength = float(np.clip(1.0 - nearest_d / DELTA_COMM, 0.0, 1.0))
    w_blended = (1.0 - strength) * w + strength * w_rhr
    return float(np.clip(w_blended, -W_MAX, W_MAX))


class Simulator:
    """
    Runs a single simulation instance to completion or T_MAX timeout.

    Parameters
    ----------
    env : Environment
        Workspace with obstacles and env_type.
    robots : list of Robot
        Pre-initialised robots (positions, goals set externally).
    method : str
        'mgr', 'clf_cbf', or 'orca'.
    record_every : int
        Record a state snapshot every this many steps (default 5 = 0.25 s).
    """

    def __init__(self, env, robots: list, method: str = 'mgr',
                 record_every: int = 5):
        if method not in ('mgr', 'clf_cbf', 'orca'):
            raise ValueError(f"Unknown method '{method}'. Use 'mgr', 'clf_cbf', or 'orca'.")
        if method == 'orca':
            raise NotImplementedError("ORCA baseline is implemented in Phase 5.")

        self.env = env
        self.robots = robots
        self.method = method
        self.record_every = record_every

        self._history: list[dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> dict:
        """
        Execute the simulation loop.

        Returns
        -------
        dict
            Metrics dict from compute_metrics plus a 'min_dist' safety key.
        """
        robots = self.robots
        obstacles = self.env.obstacles
        env_type = self.env.env_type
        use_mgr = (self.method == 'mgr')

        roundabouts: dict = {}
        qp_info_map: dict = {}
        next_id: int = 0

        n_steps = int(T_MAX / DT)
        min_dist_overall = float('inf')
        t = 0.0

        for step in range(n_steps):
            t = step * DT
            active = [r for r in robots if not r.arrived]
            if not active:
                break

            # ----------------------------------------------------------
            # Step 1 — MGR update (Algorithm 1, lines 1–27)
            # ----------------------------------------------------------
            if use_mgr:
                next_id = run_mgr_update(
                    active, roundabouts, obstacles, qp_info_map, next_id
                )

            # ----------------------------------------------------------
            # Step 2 — Escape checks (Algorithm 1, lines 28–30)
            # ----------------------------------------------------------
            if use_mgr:
                for r in active:
                    if r.mode != RobotMode.MGR:
                        continue
                    C = roundabouts.get(r.roundabout_id)
                    if C and is_escapable(r, C, active, obstacles, env_type):
                        escape_robot(r, C)

            # ----------------------------------------------------------
            # Step 3 — Compute controls (double-buffer)
            # All QP solves use pre-step positions; no robot has moved yet.
            # ----------------------------------------------------------
            pending: dict = {}
            new_qp_info: dict = {}

            for r in active:
                neighbors = [
                    n for n in active
                    if n.id != r.id
                    and np.linalg.norm(r.pos - n.pos) <= DELTA_COMM
                ]

                # Reference velocity from mode-appropriate controller
                if r.mode == RobotMode.GOAL:
                    v_des, w_des = goal_control(r)
                else:
                    C = roundabouts.get(r.roundabout_id)
                    if C is None:
                        # Roundabout lost (pruned) — fall back to GOAL
                        r.mode = RobotMode.GOAL
                        r.roundabout_id = None
                        v_des, w_des = goal_control(r)
                    else:
                        v_des, w_des = mgr_control(r, C)

                # Safety filter
                v, w, info = clf_cbf_qp(r, neighbors, obstacles, v_des, w_des)
                new_qp_info[r.id] = info

                # Right-hand rule ω override for obstacle avoidance
                w = _right_hand_rule(r, w, obstacles)

                pending[r.id] = (v, w)

            qp_info_map = new_qp_info

            # ----------------------------------------------------------
            # Step 4 — Apply all controls simultaneously
            # ----------------------------------------------------------
            for r in active:
                v, w = pending[r.id]
                r.apply_control(v, w, DT)

            # ----------------------------------------------------------
            # Step 5 — Arrival checks
            # ----------------------------------------------------------
            for r in active:
                r.check_arrival(t)

            # ----------------------------------------------------------
            # Step 6 — Safety tracking + snapshot recording
            # ----------------------------------------------------------
            active_now = [r for r in robots if not r.arrived]
            if len(active_now) >= 2:
                for i in range(len(active_now)):
                    for j in range(i + 1, len(active_now)):
                        d = float(np.linalg.norm(
                            active_now[i].pos - active_now[j].pos
                        ))
                        if d < min_dist_overall:
                            min_dist_overall = d

            if step % self.record_every == 0:
                self._history.append(self._snapshot(t, roundabouts))

        # Final snapshot
        self._history.append(self._snapshot(t, roundabouts))

        metrics = compute_metrics(robots, t)
        metrics['min_dist'] = min_dist_overall
        metrics['method'] = self.method
        return metrics

    def get_history(self) -> list:
        """
        Return recorded state snapshots for visualization.

        Each snapshot is a dict:
            {
              't': float,
              'robots': [{'id', 'pos', 'theta', 'mode', 'arrived'}, ...],
              'roundabouts': [{'id', 'center', 'radius', 'members'}, ...],
            }
        """
        return self._history

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _snapshot(self, t: float, roundabouts: dict) -> dict:
        return {
            't': t,
            'robots': [
                {
                    'id':      r.id,
                    'pos':     r.pos.tolist(),
                    'theta':   r.theta,
                    'mode':    r.mode.name,
                    'arrived': r.arrived,
                }
                for r in self.robots
            ],
            'roundabouts': [
                {
                    'id':      C.id,
                    'center':  C.center.tolist(),
                    'radius':  C.radius,
                    'members': list(C.members),
                }
                for C in roundabouts.values()
            ],
        }
