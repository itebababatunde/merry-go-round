"""
roundabout_mgr.py — Roundabout creation, validation, adjustment, and joining.

Implements Algorithm 1 (lines 10–27) from the paper:

  find_center(ri, rj)          → midpoint candidate for roundabout center
  is_mgr_valid(C, obstacles)   → checks obstacle clearance (ISMGRVALID)
  adjust_mgr(C, obstacles)     → grid search for nearest valid center (ADJUST_MGR)
  create_mgr(ri, rj, ...)      → build a new valid Roundabout (CREATE_MGR)
  join_mgr(robot, C)           → add robot to roundabout, set mode = MGR
  run_mgr_update(...)          → top-level per-step call (Algorithm 1 faithful)

Processing order: pairs sorted by ascending predicted minimum distance so the
most urgent deadlock is resolved first. This prevents inconsistent assignments
when multiple pairs overlap.
"""

import math
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from experiments.config import (
    MGR_RADIUS, DELTA_COMM, DELTA_C, WORKSPACE
)
from src.robot import RobotMode
from src.mgr.roundabout import Roundabout
from src.mgr.deadlock import is_deadlock_candidate, is_goal_checking


# ---------------------------------------------------------------------------
# Core primitives
# ---------------------------------------------------------------------------

def find_center(ri, rj) -> np.ndarray:
    """
    Return the midpoint of the two robots as the initial roundabout center.

    Paper §IV-A: "We use the midpoint between ai and aj for simplicity."
    """
    return (ri.pos + rj.pos) / 2.0


def is_mgr_valid(C: Roundabout, obstacles: list,
                 workspace: float = WORKSPACE) -> bool:
    """
    Return True if roundabout C has sufficient clearance from all obstacles
    and lies within the workspace (paper ISMGRVALID).

    Paper condition: dC ≥ C.r + k·C.n  for every obstacle O,
    where dC = distance from C.c to the closest point of O.
    Equivalently: obs.sdf(C.center) ≥ effective_clearance().

    Also rejects centers that are too close to workspace boundaries.
    """
    clr = C.effective_clearance()
    for obs in obstacles:
        if obs.sdf(C.center) < clr:
            return False
    # Workspace boundary check
    for i in range(2):
        if C.center[i] < clr or C.center[i] > workspace - clr:
            return False
    return True


def adjust_mgr(C: Roundabout, obstacles: list) -> Roundabout | None:
    """
    Search for the nearest valid center within 10·C.r of the current C.c.

    Paper §IV-B footnote: "searching a radius of 10·C.r around C.c proved
    sufficient." Grid step = C.r / 2 gives a 41×41 search grid.

    "The valid cell with the lowest index among those having the closest
    distance to the current C.c is selected as the new C.c." — implemented by
    iterating in row-major order (dx outer, dy inner) and using strict < for
    distance comparison.

    Returns the Roundabout with updated center, or None if no valid cell found.
    """
    step = C.radius / 2.0
    half = 10.0 * C.radius
    offsets = np.arange(-half, half + step * 0.5, step)

    original_center = C.center.copy()
    best_center = None
    best_dist = float('inf')

    for dx in offsets:
        for dy in offsets:
            trial_center = original_center + np.array([dx, dy])
            C_trial = Roundabout(
                id=C.id,
                center=trial_center,
                radius=C.radius,
                members=list(C.members),
            )
            if is_mgr_valid(C_trial, obstacles):
                d = float(np.linalg.norm(trial_center - original_center))
                if d < best_dist:
                    best_dist = d
                    best_center = trial_center.copy()

    if best_center is None:
        return None

    C.center = best_center
    return C


def create_mgr(ri, rj, obstacles: list, next_id: int) -> Roundabout | None:
    """
    Create a new roundabout for the deadlocked pair (ri, rj).

    Returns a valid Roundabout with ri and rj as initial members, or None
    if no valid center can be found.
    """
    center = find_center(ri, rj)
    C = Roundabout(id=next_id, center=center, radius=MGR_RADIUS, members=[])
    if not is_mgr_valid(C, obstacles):
        C = adjust_mgr(C, obstacles)
    return C   # None if adjust_mgr found no valid cell


def join_mgr(robot, C: Roundabout) -> None:
    """
    Add robot to roundabout C and switch it to MGR mode.

    Paper Algorithm 1 JOIN_MGR (line 15/22): sets ai.mode = MGR.
    If the robot is already a member (e.g., called twice), it is not
    duplicated in the members list.
    """
    if robot.id not in C.members:
        C.members.append(robot.id)
    robot.mode = RobotMode.MGR
    robot.roundabout_id = C.id


# ---------------------------------------------------------------------------
# Top-level per-step update (Algorithm 1)
# ---------------------------------------------------------------------------

def _predicted_min_dist_sq(ri, rj) -> float:
    """Return squared predicted minimum distance for sorting pairs by urgency."""
    from experiments.config import T_DEADLOCK
    dp = ri.pos - rj.pos
    dv = ri.velocity - rj.velocity
    dv_norm_sq = float(dv @ dv)
    if dv_norm_sq < 1e-16:
        return float(np.dot(dp, dp))
    t_star = float(np.clip(-(dp @ dv) / dv_norm_sq, 0.0, T_DEADLOCK))
    diff = dp + dv * t_star
    return float(np.dot(diff, diff))


def run_mgr_update(
    active_robots: list,
    roundabouts: dict,
    obstacles: list,
    qp_info_map: dict,
    next_id: int = 0,
) -> int:
    """
    Run one step of Algorithm 1 for all active robots.

    Must be called BEFORE QP controls are computed each timestep.

    Parameters
    ----------
    active_robots : list of Robot
        All robots that have not yet arrived.
    roundabouts : dict {int: Roundabout}
        Live roundabout registry (modified in-place).
    obstacles : list
        Static obstacle objects.
    qp_info_map : dict {robot_id: info_dict}
        Info dicts from the previous QP solve (contains 'deadlock_flags').
    next_id : int
        Next roundabout ID to assign.

    Returns
    -------
    int
        Updated next_id (incremented each time a new roundabout is created).
    """
    robot_map = {r.id: r for r in active_robots}

    # ------------------------------------------------------------------
    # Step A — RECEIVE_MGR (Algorithm 1, lines 1–3)
    # If a neighbor is in MGR mode and this robot is not yet in that
    # roundabout, join it immediately (broadcast propagation).
    # ------------------------------------------------------------------
    for ri in active_robots:
        if ri.mode == RobotMode.MGR:
            continue   # already in a roundabout
        for rj in active_robots:
            if rj.id == ri.id or rj.mode != RobotMode.MGR:
                continue
            if np.linalg.norm(ri.pos - rj.pos) > DELTA_COMM:
                continue
            C = roundabouts.get(rj.roundabout_id)
            if C is None:
                continue
            # Validate (and adjust) before joining
            if not is_mgr_valid(C, obstacles):
                C = adjust_mgr(C, obstacles)
            if C is not None:
                join_mgr(ri, C)
                break   # robot joins the first valid neighbour roundabout

    # ------------------------------------------------------------------
    # Step B — Deadlock detection (Algorithm 1, lines 4–27)
    # Only GOAL-mode robots check for deadlocks with their neighbours.
    # ------------------------------------------------------------------
    # Collect candidate pairs
    goal_robots = [r for r in active_robots if r.mode == RobotMode.GOAL]
    candidate_pairs = []
    seen = set()

    for ri in goal_robots:
        flags_i = qp_info_map.get(ri.id, {}).get('deadlock_flags', set())
        for rj in goal_robots:
            if rj.id <= ri.id:
                continue
            key = (ri.id, rj.id)
            if key in seen:
                continue
            seen.add(key)
            if np.linalg.norm(ri.pos - rj.pos) > DELTA_COMM:
                continue
            flags_j = qp_info_map.get(rj.id, {}).get('deadlock_flags', set())
            combined_flags = flags_i | {rid for rid in flags_j if rid == ri.id}
            if is_deadlock_candidate(ri, rj, combined_flags):
                candidate_pairs.append((ri, rj))

    # Sort by ascending predicted min-distance (most urgent first)
    candidate_pairs.sort(key=lambda pair: _predicted_min_dist_sq(*pair))

    for ri, rj in candidate_pairs:
        # Re-check: either robot may have joined a roundabout in this loop
        if ri.mode == RobotMode.MGR or rj.mode == RobotMode.MGR:
            # If one is already MGR, the other should join that roundabout
            mgr_robot = ri if ri.mode == RobotMode.MGR else rj
            goal_robot = rj if ri.mode == RobotMode.MGR else ri
            if goal_robot.mode == RobotMode.GOAL:
                C = roundabouts.get(mgr_robot.roundabout_id)
                if C is not None:
                    if not is_mgr_valid(C, obstacles):
                        C = adjust_mgr(C, obstacles)
                    if C is not None:
                        join_mgr(goal_robot, C)
            continue

        if is_goal_checking(ri, rj):
            continue

        c = find_center(ri, rj)

        # Check if an existing roundabout is within DELTA_C of c
        nearby = [
            C for C in roundabouts.values()
            if np.linalg.norm(C.center - c) <= DELTA_C
        ]

        if nearby:
            # Join the nearest existing roundabout
            C = min(nearby, key=lambda C: np.linalg.norm(C.center - c))
            if not is_mgr_valid(C, obstacles):
                C = adjust_mgr(C, obstacles)
            if C is not None:
                join_mgr(ri, C)
                join_mgr(rj, C)
        else:
            # Create a new roundabout
            C = create_mgr(ri, rj, obstacles, next_id)
            if C is not None:
                roundabouts[C.id] = C
                next_id += 1
                join_mgr(ri, C)
                join_mgr(rj, C)

    # ------------------------------------------------------------------
    # Step C — Prune empty roundabouts
    # ------------------------------------------------------------------
    empty_ids = [cid for cid, C in roundabouts.items() if C.n_members == 0]
    for cid in empty_ids:
        del roundabouts[cid]

    return next_id
