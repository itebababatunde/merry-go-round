"""
orca_baseline.py — ORCA baseline using the rvo2 library.

ORCA operates holonomically: robots move as 2D point masses with direct
velocity control. There is no unicycle kinematics conversion. This matches
the paper's treatment of ORCA as a holonomic baseline (§V-A).

Obstacle conversion (paper §V):
  - CircularObstacle  → 16-gon polygon via obs.to_polygon_vertices(n=16)
  - RectangularObstacle → 4-vertex polygon via obs.to_polygon_vertices()
  - Workspace boundary → 4 thin wall polygons to keep robots inside

rvo2 parameters (all from config or standard defaults):
  timeStep       = DT           = 0.05 s
  neighborDist   = DELTA_COMM   = 1.0 m
  maxNeighbors   = 10
  timeHorizon    = 2.0 s        (robot–robot look-ahead, = T_DEADLOCK)
  timeHorizonObst= 0.5 s        (obstacle look-ahead, standard default)
  radius         = ROBOT_RADIUS = 0.2 m
  maxSpeed       = V_MAX        = 0.8 m/s

The OrcaSimulator.run() return value is format-identical to Simulator.run()
so Phase 6's batch runner can use both classes interchangeably.
"""

import numpy as np
import rvo2

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from experiments.config import (
    DT, T_MAX, V_MAX, ROBOT_RADIUS, DELTA_COMM, EPSILON_GOAL, D_SAFE, WORKSPACE
)
from src.simulation.metrics import compute_metrics

# rvo2 fixed parameters
_MAX_NEIGHBORS   = 10
_TIME_HORIZON    = 2.0    # s — robot–robot look-ahead
_TIME_HORIZON_OB = 0.5    # s — obstacle look-ahead
_WALL_THICKNESS  = 0.05   # m — workspace boundary wall half-width


def _build_wall_polygons(size: float = WORKSPACE) -> list:
    """
    Return 4 thin rectangular polygon vertex lists enclosing the workspace.

    rvo2 needs CCW vertex order for obstacle polygons that act as solid walls.
    Each wall is a thin strip just outside the workspace boundary.
    """
    t = _WALL_THICKNESS
    s = size
    # Bottom, top, left, right — CCW vertices for each strip
    return [
        [(-t, -t), (s + t, -t), (s + t, 0.0),  (-t, 0.0)],   # bottom
        [(-t, s),  (s + t, s),  (s + t, s + t), (-t, s + t)], # top
        [(-t, -t), (0.0, -t),   (0.0, s + t),   (-t, s + t)], # left
        [(s, -t),  (s + t, -t), (s + t, s + t), (s, s + t)],  # right
    ]


class OrcaSimulator:
    """
    ORCA baseline simulator using the rvo2 library.

    Parameters
    ----------
    env : Environment
        Workspace with obstacles (CircularObstacle / RectangularObstacle) and env_type.
    robots : list of Robot
        Pre-initialised robots. Only .pos and .goal are used; theta is ignored
        (holonomic motion, no heading).
    record_every : int
        Snapshot interval in steps (default 5 = 0.25 s).
    """

    def __init__(self, env, robots: list, record_every: int = 5):
        self.env          = env
        self.robots       = robots
        self.record_every = record_every
        self._history: list[dict] = []

    # ------------------------------------------------------------------
    # Public API (matches Simulator interface)
    # ------------------------------------------------------------------

    def run(self) -> dict:
        """
        Execute the ORCA simulation loop.

        Returns
        -------
        dict — same format as Simulator.run():
            success_rate, arrival_rate, makespan, mean_time,
            n_arrived, n_total, t_elapsed, min_dist, method
        """
        robots   = self.robots
        N        = len(robots)
        n_steps  = int(T_MAX / DT)

        # ----------------------------------------------------------
        # Build rvo2 simulator
        # ----------------------------------------------------------
        rvo_sim = rvo2.PyRVOSimulator(
            DT,
            DELTA_COMM,
            _MAX_NEIGHBORS,
            _TIME_HORIZON,
            _TIME_HORIZON_OB,
            ROBOT_RADIUS,
            V_MAX,
        )

        # Add agents (one per robot, in same order)
        for r in robots:
            rvo_sim.addAgent(
                tuple(r.pos),
                DELTA_COMM,
                _MAX_NEIGHBORS,
                _TIME_HORIZON,
                _TIME_HORIZON_OB,
                ROBOT_RADIUS,
                V_MAX,
                (0.0, 0.0),
            )

        # Add obstacle polygons from environment
        obs_count = 0
        for obs in self.env.obstacles:
            verts = obs.to_polygon_vertices() if hasattr(obs, 'to_polygon_vertices') else None
            if verts is None:
                continue
            # to_polygon_vertices returns np.ndarray rows; convert to tuple list
            poly = [tuple(v) for v in verts]
            rvo_sim.addObstacle(poly)
            obs_count += 1

        # Add workspace boundary walls
        for wall_verts in _build_wall_polygons(self.env.size):
            rvo_sim.addObstacle(wall_verts)

        rvo_sim.processObstacles()

        # ----------------------------------------------------------
        # Simulation loop
        # ----------------------------------------------------------
        arrived_flags = [False] * N
        arrival_times = [None] * N
        min_dist_overall = float('inf')
        t = 0.0

        for step in range(n_steps):
            t = step * DT

            # Set preferred velocities
            all_done = True
            for i, r in enumerate(robots):
                if arrived_flags[i]:
                    rvo_sim.setAgentPrefVelocity(i, (0.0, 0.0))
                    continue
                all_done = False
                to_goal = r.goal - np.array(rvo_sim.getAgentPosition(i))
                dist = float(np.linalg.norm(to_goal))
                if dist < EPSILON_GOAL:
                    rvo_sim.setAgentPrefVelocity(i, (0.0, 0.0))
                else:
                    pref = (to_goal / dist) * V_MAX
                    rvo_sim.setAgentPrefVelocity(i, (float(pref[0]), float(pref[1])))

            if all_done:
                break

            rvo_sim.doStep()

            # Read back positions and check arrivals
            active_positions = []
            for i, r in enumerate(robots):
                pos = np.array(rvo_sim.getAgentPosition(i))
                vel = np.array(rvo_sim.getAgentVelocity(i))
                r.pos      = pos
                r.velocity = vel

                if not arrived_flags[i]:
                    active_positions.append(pos)
                    dist_to_goal = float(np.linalg.norm(pos - r.goal))
                    if dist_to_goal < EPSILON_GOAL:
                        arrived_flags[i] = True
                        arrival_times[i] = t
                        r.arrived      = True
                        r.arrival_time = t

            # Track minimum pairwise distance among active robots
            active_pos = [np.array(rvo_sim.getAgentPosition(i))
                          for i in range(N) if not arrived_flags[i]]
            if len(active_pos) >= 2:
                for ii in range(len(active_pos)):
                    for jj in range(ii + 1, len(active_pos)):
                        d = float(np.linalg.norm(active_pos[ii] - active_pos[jj]))
                        if d < min_dist_overall:
                            min_dist_overall = d

            # Record snapshot
            if step % self.record_every == 0:
                self._history.append(self._snapshot(t, arrived_flags))

        # Final snapshot
        self._history.append(self._snapshot(t, arrived_flags))

        # Mark Robot objects as arrived
        for i, r in enumerate(robots):
            if arrived_flags[i] and not r.arrived:
                r.arrived      = True
                r.arrival_time = arrival_times[i]

        metrics = compute_metrics(robots, t)
        metrics['min_dist'] = min_dist_overall
        metrics['method']   = 'orca'
        return metrics

    def get_history(self) -> list:
        """
        Return recorded state snapshots (same format as Simulator.get_history()).

        Note: theta is always 0.0 for holonomic ORCA robots (no heading state).
        """
        return self._history

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _snapshot(self, t: float, arrived_flags: list) -> dict:
        return {
            't': t,
            'robots': [
                {
                    'id':      r.id,
                    'pos':     r.pos.tolist(),
                    'theta':   0.0,          # holonomic — no heading
                    'mode':    'GOAL',
                    'arrived': arrived_flags[i],
                }
                for i, r in enumerate(self.robots)
            ],
            'roundabouts': [],  # ORCA has no roundabouts
        }
