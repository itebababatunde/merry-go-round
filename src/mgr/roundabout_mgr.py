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
    MGR_RADIUS, DELTA_COMM, DELTA_C, WORKSPACE, D_SAFE as _D_SAFE
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


def _bisector_center(ri, rj, obstacles: list, n_members: int) -> np.ndarray | None:
    """
    Search the perpendicular bisector of ri-rj for a valid roundabout center.

    Every bisector point is equidistant from ri and rj → angular separation
    is exactly 180° by construction. Preferred over ADJUST_MGR when it succeeds
    because the close, symmetric center gives better escape geometry.

    Searches ±10·C.r in steps of C.r/2 (41 candidates, same radius as ADJUST_MGR).
    """
    midpoint = find_center(ri, rj)
    direction = rj.pos - ri.pos
    length = float(np.linalg.norm(direction))
    perp = np.array([1.0, 0.0]) if length < 1e-6 else np.array([-direction[1], direction[0]]) / length

    step = MGR_RADIUS / 2.0
    half = 10.0 * MGR_RADIUS
    offsets = np.arange(-half, half + step * 0.5, step)
    dummy_members = list(range(n_members))

    best_center, best_dist = None, float('inf')
    for t in offsets:
        trial_center = midpoint + t * perp
        C_trial = Roundabout(id=0, center=trial_center, radius=MGR_RADIUS, members=dummy_members)
        if is_mgr_valid(C_trial, obstacles):
            d = abs(t)
            if d < best_dist:
                best_dist = d
                best_center = trial_center.copy()
    return best_center


def create_mgr(ri, rj, obstacles: list, next_id: int) -> Roundabout | None:
    """
    Create a new roundabout for the deadlocked pair (ri, rj).

    Paper CREATE_MGR: FIND_CENTER then ADJUST_MGR if center is invalid.
    We prefer the perpendicular bisector search over ADJUST_MGR when it succeeds:
    bisector points are equidistant from both robots (180° sep by construction)
    and stay close to the pair, giving better escape geometry. ADJUST_MGR is
    the fallback for cases where the bisector direction is obstacle-blocked.

    Two-pass strategy to handle tight corridors:
      Pass 1 (strict): n=2 clearance = 0.5 m — orbit path has ROBOT_RADIUS margin
      Pass 2 (fallback): n=0 clearance = 0.3 m — for narrow corridors (rect15/circ15)

    Returns a Roundabout with empty members list, or None if no valid center found.
    """
    center = find_center(ri, rj)

    # Pass 1 — try midpoint, then bisector, then full ADJUST_MGR
    C = Roundabout(id=next_id, center=center, radius=MGR_RADIUS, members=[ri.id, rj.id])
    if not is_mgr_valid(C, obstacles):
        bisector_c = _bisector_center(ri, rj, obstacles, n_members=2)
        if bisector_c is not None:
            C.center = bisector_c
        else:
            C = adjust_mgr(C, obstacles)
    if C is not None:
        C.members = []
        return C

    # Pass 2 — same order but with n=0 clearance
    C = Roundabout(id=next_id, center=center, radius=MGR_RADIUS, members=[])
    if not is_mgr_valid(C, obstacles):
        bisector_c = _bisector_center(ri, rj, obstacles, n_members=0)
        if bisector_c is not None:
            C.center = bisector_c
        else:
            C = adjust_mgr(C, obstacles)
    return C


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
# Angular separation helper
# ---------------------------------------------------------------------------

def _angle_from_center(robot, center: np.ndarray) -> float:
    """Return the angle (radians) from center to robot.pos."""
    v = robot.pos - center
    return math.atan2(v[1], v[0])


def _angular_gap_ok(robot, C: Roundabout, robot_map: dict) -> bool:
    """
    Return True if robot's angle from C.center gives sufficient angular
    separation from every existing member to avoid orbit collisions.

    Min gap = 2·arcsin(D_SAFE / (2·C.radius)) — the angle at which two
    co-members on the orbit circle would be exactly D_SAFE apart.
    This replaces the previous 2π/(n+1) formula, which was far too strict
    when the orbit radius is large relative to D_SAFE (e.g. radius=1.5 m
    gives only ~17° required, not 180° for n=1).
    """
    from experiments.config import D_SAFE
    # Minimum physically-safe angular separation on the orbit ring.
    # Formula: 2·arcsin(D_SAFE / (2·C.radius)) — angle at which two co-members
    # on the ring are exactly D_SAFE apart.
    # NOTE: use C.radius directly, NOT max(C.radius, D_SAFE). The max() cap was
    # added to avoid domain errors but gives 60° for C.r=0.3m when the correct
    # ring spacing requires 94°, causing collisions when robots converge to ring.
    # C.radius >= D_SAFE/2 is guaranteed by MGR_RADIUS=0.3 > D_SAFE/2=0.22.
    min_gap = 2.0 * math.asin(min(D_SAFE / (2.0 * C.radius), 1.0))
    ri_angle = _angle_from_center(robot, C.center)
    for mid in C.members:
        m = robot_map.get(mid)
        if m is None:
            continue
        m_angle = _angle_from_center(m, C.center)
        diff = abs(math.atan2(
            math.sin(ri_angle - m_angle),
            math.cos(ri_angle - m_angle)
        ))
        if diff < min_gap:
            return False
    return True


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
    # Cooldown / MGR-step bookkeeping
    # Decrement escape cooldowns so recently-escaped robots are not
    # immediately re-recruited by RECEIVE_MGR.
    # Track how many steps each robot has been in MGR mode so that
    # robots stuck alone in a roundabout can be force-released.
    # ------------------------------------------------------------------
    _MGR_TIMEOUT_STEPS      = 300    # 15 s — orbit timeout; robots joining late still escape before T_MAX
    _MGR_ALONE_TIMEOUT_STEPS = 60    # 3 s — fast release when alone
    _FORCE_RELEASE_COOLDOWN  = 300   # 15 s — long enough to escape orbit zone in dense obstacle envs

    for r in active_robots:
        if r.escape_cooldown > 0:
            r.escape_cooldown -= 1
        if r.mode == RobotMode.MGR:
            r.mgr_step_count += 1
            C = roundabouts.get(r.roundabout_id)
            if C is not None:
                alone = (C.n_members == 1)
                timeout = _MGR_ALONE_TIMEOUT_STEPS if alone else _MGR_TIMEOUT_STEPS
                if r.mgr_step_count >= timeout:
                    # Force-release robot stuck in roundabout too long.
                    # Set escape_cooldown so RECEIVE_MGR does not immediately
                    # re-recruit this robot, which would reset the counter and
                    # create an infinite 60-s cycle with zero net progress.
                    if r.id in C.members:
                        C.members.remove(r.id)
                    r.mode = RobotMode.GOAL
                    r.roundabout_id = None
                    r.escape_perp_count = 0
                    r.mgr_step_count = 0
                    r.escape_cooldown = _FORCE_RELEASE_COOLDOWN

    # ------------------------------------------------------------------
    # Step A — RECEIVE_MGR (Algorithm 1, lines 1–3)
    # If a neighbor is in MGR mode and this robot is not yet in that
    # roundabout, join it immediately (broadcast propagation).
    # ------------------------------------------------------------------
    for ri in active_robots:
        if ri.mode == RobotMode.MGR:
            continue   # already in a roundabout
        if ri.escape_cooldown > 0:
            continue   # recently escaped — let it move away first
        for rj in active_robots:
            if rj.id == ri.id or rj.mode != RobotMode.MGR:
                continue
            if np.linalg.norm(ri.pos - rj.pos) > DELTA_COMM:
                continue
            C = roundabouts.get(rj.roundabout_id)
            if C is None:
                continue
            # Only join if robot is close enough to the orbit circle to converge
            # within a reasonable time. Threshold: center distance ≤ C.radius + DELTA_COMM.
            if np.linalg.norm(ri.pos - C.center) > C.radius + DELTA_COMM:
                continue
            # Angular gap check: prevent joining too close to an existing co-member's
            # orbit angle, which can cause near-collisions during simultaneous convergence
            # (CBF degenerates when Lg_h→0 near coincident orbit positions).
            # Override: ri's goal path passes through the roundabout disk (clearance ≤
            # radius + D_SAFE). These robots are permanently deflected by orbiting robots
            # yet never detected as GOAL-GOAL deadlocks — they must join to make progress.
            ri_to_goal = ri.goal - ri.pos
            ri_goal_len = float(np.linalg.norm(ri_to_goal))
            if ri_goal_len > 1e-6:
                t_proj = float(np.clip(
                    np.dot(C.center - ri.pos, ri_to_goal) / (ri_goal_len ** 2), 0.0, 1.0
                ))
                path_clearance = float(np.linalg.norm(ri.pos + t_proj * ri_to_goal - C.center))
            else:
                path_clearance = float(np.linalg.norm(ri.pos - C.center))
            path_blocked = path_clearance <= C.radius + _D_SAFE

            if not path_blocked and not _angular_gap_ok(ri, C, robot_map):
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
    # Collect candidate pairs — skip recently-escaped robots so they have time
    # to navigate away from the roundabout zone before entering a new orbit.
    goal_robots = [r for r in active_robots
                   if r.mode == RobotMode.GOAL and r.escape_cooldown == 0]
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
            # Join the nearest existing roundabout (paper Alg. 1 lines 11–14).
            # Angular gap check retained for EXISTING roundabouts to prevent
            # multiple robots piling onto one orbit at close angular positions,
            # which can cause CBF failure during simultaneous convergence.
            # For newly-created roundabouts (else branch), robots join directly.
            C = min(nearby, key=lambda C: np.linalg.norm(C.center - c))
            if not is_mgr_valid(C, obstacles):
                C = adjust_mgr(C, obstacles)
            if C is not None:
                ri_ok = _angular_gap_ok(ri, C, robot_map)
                if ri_ok:
                    C.members.append(ri.id)
                    rj_ok = _angular_gap_ok(rj, C, robot_map)
                    C.members.remove(ri.id)
                else:
                    rj_ok = False
                if ri_ok and rj_ok:
                    join_mgr(ri, C)
                    join_mgr(rj, C)
                elif ri_ok:
                    join_mgr(ri, C)
                elif rj_ok:
                    join_mgr(rj, C)
        else:
            # No nearby roundabout — create a new one (paper Alg. 1 lines 15–18).
            # No angular separation gate — paper Algorithm 1 joins unconditionally.
            # Safety: angular gap correction in mgr_control slows the following robot
            # to near-zero while the leading robot races CCW, rapidly increasing the
            # gap. By the time the slower robot reaches the orbit, the leading robot
            # has orbited far enough to ensure physical separation ≥ D_SAFE.
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
