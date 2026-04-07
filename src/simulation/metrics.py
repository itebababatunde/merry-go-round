"""
metrics.py — Simulation outcome metrics (paper §V-A).

Four metrics match the paper exactly:

  success_rate  — 1.0 if ALL robots arrive within T_MAX, else 0.0
  arrival_rate  — fraction of robots that arrive (n_arrived / N)
  makespan      — time when the LAST robot arrives (None if success=0)
  mean_time     — average arrival time of robots that did arrive (None if none arrived)

Helper utilities for safety verification:

  min_pairwise_distance  — minimum inter-robot distance at the current instant
  collision_occurred     — True if any pair is closer than d_safe
"""

import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from experiments.config import D_SAFE


def compute_metrics(robots: list, t_elapsed: float) -> dict:
    """
    Compute the four paper metrics from a completed (or timed-out) simulation.

    Parameters
    ----------
    robots : list of Robot
        All robots in the simulation (arrived and not-arrived).
    t_elapsed : float
        Wall-clock simulation time at end of run (seconds).

    Returns
    -------
    dict with keys:
        'success_rate' : float   — 1.0 or 0.0
        'arrival_rate' : float   — in [0, 1]
        'makespan'     : float | None
        'mean_time'    : float | None
        'n_arrived'    : int
        'n_total'      : int
        't_elapsed'    : float
    """
    n_total = len(robots)
    arrived = [r for r in robots if r.arrived]
    n_arrived = len(arrived)

    success_rate = 1.0 if n_arrived == n_total else 0.0
    arrival_rate = n_arrived / n_total if n_total > 0 else 0.0

    if success_rate == 1.0:
        makespan = float(max(r.arrival_time for r in arrived))
    else:
        makespan = None

    if n_arrived > 0:
        mean_time = float(np.mean([r.arrival_time for r in arrived]))
    else:
        mean_time = None

    return {
        'success_rate': success_rate,
        'arrival_rate': arrival_rate,
        'makespan':     makespan,
        'mean_time':    mean_time,
        'n_arrived':    n_arrived,
        'n_total':      n_total,
        't_elapsed':    t_elapsed,
    }


def min_pairwise_distance(robots: list) -> float:
    """
    Return the minimum Euclidean distance between any two robots.

    Used during testing to confirm the safety barrier was never violated.
    Returns inf if fewer than 2 robots are provided.
    """
    active = [r for r in robots if not r.arrived]
    if len(active) < 2:
        return float('inf')
    min_d = float('inf')
    for i in range(len(active)):
        for j in range(i + 1, len(active)):
            d = float(np.linalg.norm(active[i].pos - active[j].pos))
            if d < min_d:
                min_d = d
    return min_d


def collision_occurred(robots: list, d_safe: float = D_SAFE) -> bool:
    """
    Return True if any pair of active robots is closer than d_safe.

    Used as a sanity check during unit testing.
    """
    return min_pairwise_distance(robots) < d_safe - 1e-4
