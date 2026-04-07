"""
deadlock.py — Deadlock detection predicates (Algorithm 1, lines 6–9).

Two functions:

  is_goal_checking(ri, rj)
      Returns True when both robots are already near their goals — skip
      deadlock detection for this pair (paper Eq. 12).

  is_deadlock_candidate(ri, rj, deadlock_flags_i=None)
      Returns True when a deadlock between ri and rj is detected or predicted
      via two conditions (paper §IV-A):

      (a) ‖xi − xj‖ ≤ 2·rsafe  (Eq. 10) — already at the safety barrier.
      (b) min ‖x'i − x'j‖ ≤ kD·rsafe  (Eq. 11, CORRECTED) — predicted
          minimum distance over [0, T] is within kD·rsafe of each other.
          Note: paper uses rsafe (0.22 m), NOT dsafe (0.44 m).

      An additional guard: if rj.id appears in deadlock_flags_i (degenerate
      Lg_h from the QP, signalling a head-on singularity), treat as a deadlock
      candidate immediately.
"""

import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from experiments.config import D_SAFE, R_SAFE, K_D, T_DEADLOCK, EPSILON_GOAL
from src.robot import RobotMode


def is_goal_checking(ri, rj) -> bool:
    """
    Return True if both robots are near their goals in GOAL mode.

    When this is True, deadlock detection for the pair is skipped — the
    robots are about to converge naturally without forming a roundabout.

    Paper Eq. 12:
        ‖xi − gi‖ ≤ ε  AND  ‖xj − gj‖ ≤ ε
        AND  ai.mode = aj.mode = GOAL
    """
    return (
        ri.dist_to_goal() <= EPSILON_GOAL
        and rj.dist_to_goal() <= EPSILON_GOAL
        and ri.mode == RobotMode.GOAL
        and rj.mode == RobotMode.GOAL
    )


def is_deadlock_candidate(ri, rj, deadlock_flags_i=None) -> bool:
    """
    Return True if robots ri and rj are in or predicted to enter deadlock.

    Parameters
    ----------
    ri, rj : Robot
        The two robots being evaluated. Reads .pos, .velocity.
    deadlock_flags_i : set of int, optional
        Neighbor IDs flagged as degenerate by clf_cbf_qp (‖Lg_h‖ < 1e-4).
        If rj.id is in this set, return True immediately.

    Returns
    -------
    bool
        True if a deadlock is detected or predicted.
    """
    # Head-on degeneracy guard: QP gradient vanished — treat as deadlock
    if deadlock_flags_i and rj.id in deadlock_flags_i:
        return True

    dp = ri.pos - rj.pos
    dist = float(np.linalg.norm(dp))

    # Condition (a) — already at the safety barrier (paper Eq. 10)
    if dist <= D_SAFE + 1e-3:
        return True

    # Condition (b) — predicted minimum distance ≤ kD·rsafe (paper Eq. 11)
    # Threshold: K_D · R_SAFE (NOT D_SAFE — paper uses rsafe, kD ∈ [1,2))
    dv = ri.velocity - rj.velocity
    dv_norm_sq = float(dv @ dv)

    if dv_norm_sq < 1e-16:
        # Robots moving in parallel or stationary — no collision predicted
        return False

    t_star = float(np.clip(-(dp @ dv) / dv_norm_sq, 0.0, T_DEADLOCK))
    min_dist_sq = float(np.dot(dp + dv * t_star, dp + dv * t_star))
    threshold_sq = (K_D * R_SAFE) ** 2

    return min_dist_sq <= threshold_sq
