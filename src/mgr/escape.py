"""
escape.py — Escape condition for MGR mode (Algorithm 1, lines 28–30).

A robot in MGR mode escapes when two conditions are simultaneously met
for at least 2 consecutive timesteps (hysteresis):

  1. Perpendicularity: the vector from the robot to the roundabout center
     (vic = C.c − xi) is nearly orthogonal to the vector from the robot
     to its goal (vig = gi − xi). Paper §IV-A: "If vic is orthogonal to vig."
     Implementation: |cos(angle)| < 0.15 (≈ 81° of perpendicular).

  2. Sector clearance: the outward escape sector is free of other robots
     and obstacles. Paper §IV-A: sector centered at C.c, spanning ±δθ
     around the outward direction, extending to ‖C.c − xi‖ + δsensing
     from C.c where δsensing = δcomm = 1.0 m (paper §III-E).

The perpendicularity threshold 0.15 and 2-step hysteresis are implementation
choices not stated explicitly in the paper.
"""

import math
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from experiments.config import (
    DELTA_COMM, DELTA_THETA_OBS, DELTA_THETA_FREE, D_SAFE
)
from src.robot import RobotMode

# |cos(angle)| threshold for perpendicularity check.
# 0.15 ≈ cos(81°) — robot's outward direction is within ≈9° of goal direction.
_PERP_COS_THRESH = 0.15

# Number of consecutive timesteps perpendicularity must hold before escape.
_HYSTERESIS_STEPS = 2

# Number of angular samples used to check if an obstacle blocks the sector.
_OBS_SECTOR_SAMPLES = 8


def _wrap_angle(angle: float) -> float:
    """Wrap angle to [-π, π]."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def is_escapable(robot, C, all_robots: list,
                 obstacles: list, env_type: str) -> bool:
    """
    Return True if robot can safely escape its roundabout this timestep.

    Updates robot.escape_perp_count in-place (hysteresis counter).

    Parameters
    ----------
    robot : Robot
        The MGR-mode robot being evaluated. .escape_perp_count is updated.
    C : Roundabout
        The roundabout the robot is currently in.
    all_robots : list of Robot
        All robots in the simulation (to check sector clearance).
    obstacles : list
        Static obstacle objects with .sdf(point) method.
    env_type : str
        One of 'free', 'swap', 'circ15', 'rect15'. Determines δθ.

    Returns
    -------
    bool
        True if both escape conditions are satisfied for ≥ HYSTERESIS_STEPS.
    """
    # ------------------------------------------------------------------
    # Condition 1 — Perpendicularity (with hysteresis)
    # Paper: vic ⊥ vig  (C.c − xi  orthogonal to  gi − xi)
    # ------------------------------------------------------------------
    # Outward radial from roundabout center to robot
    outward = robot.pos - C.center
    dist_ic = float(np.linalg.norm(outward))   # ‖xi − C.c‖, used for sector_radius
    v_ig    = robot.goal - robot.pos            # robot → goal
    norm_ig = float(np.linalg.norm(v_ig))

    if dist_ic < 1e-6 or norm_ig < 1e-6:
        # Degenerate: robot at center or at goal — do not escape yet
        robot.escape_perp_count = 0
        return False

    # vic = CCW tangential velocity direction at the robot's current orbit position.
    # Rotate outward radial 90° CCW: [x, y] → [-y, x], then normalise.
    # The paper's "vic ⊥ vig" means the robot's orbital velocity is perpendicular
    # to its goal direction — i.e., the goal lies in the outward radial direction,
    # the natural exit point of the orbit.
    v_ic = np.array([-outward[1], outward[0]]) / dist_ic   # unit CCW tangent

    cos_angle = abs(float(v_ic @ v_ig) / norm_ig)
    perp_ok = cos_angle < _PERP_COS_THRESH

    if perp_ok:
        robot.escape_perp_count += 1
    else:
        robot.escape_perp_count = 0

    if robot.escape_perp_count < _HYSTERESIS_STEPS:
        return False

    # ------------------------------------------------------------------
    # Condition 2 — Sector clearance
    # Paper: sector centered at C.c, spans ±δθ in outward direction,
    #        extends to ‖C.c − xi‖ + δsensing (δsensing = δcomm = 1.0 m).
    # ------------------------------------------------------------------
    # Outward direction: from C.c through robot position
    outward = robot.pos - C.center
    outward_angle = math.atan2(outward[1], outward[0])

    # Sector parameters
    delta_theta = (DELTA_THETA_FREE if env_type in ('free', 'swap')
                   else DELTA_THETA_OBS)
    sector_radius = C.radius + DELTA_COMM  # C.r + δsensing (paper: uses ring radius, not actual dist)

    # Pre-escape proximity safety check: only block escape if a non-co-member
    # robot is within D_SAFE AND is inside the escape sector (i.e., directly
    # in the escape path). Co-members are excluded since CBF between co-members
    # is already suppressed. For dense scenarios (N=40/60 swap), robots from
    # adjacent roundabouts may always be within D_SAFE in the perpendicular
    # direction, but NOT in the escape sector — blocking escape in that case
    # causes permanent deadlock (robots never escape).
    co_member_ids = set(C.members) if C is not None else set()
    for other in all_robots:
        if other.id == robot.id or other.arrived:
            continue
        if other.id in co_member_ids:
            continue   # co-members already skip CBF; don't block escape
        if float(np.linalg.norm(other.pos - robot.pos)) >= D_SAFE:
            continue
        # Near robot: only block if it is also inside the escape sector
        d_vec = other.pos - C.center
        d_dist = float(np.linalg.norm(d_vec))
        if d_dist <= sector_radius:
            d_angle = math.atan2(d_vec[1], d_vec[0])
            if abs(_wrap_angle(d_angle - outward_angle)) <= delta_theta:
                return False   # near robot is in the escape sector — block

    # Check other robots in the escape sector.
    for other in all_robots:
        if other.id == robot.id or other.arrived:
            continue
        d_vec = other.pos - C.center
        d_dist = float(np.linalg.norm(d_vec))
        if d_dist > sector_radius:
            continue
        d_angle = math.atan2(d_vec[1], d_vec[0])
        if abs(_wrap_angle(d_angle - outward_angle)) <= delta_theta:
            return False   # another robot blocks the escape sector

    # Check obstacles: sample _OBS_SECTOR_SAMPLES points at sector_radius
    # across the full sector width. If any sample is within D_SAFE of an
    # obstacle surface, the sector is blocked.
    for obs in obstacles:
        for k in range(_OBS_SECTOR_SAMPLES):
            frac = k / max(_OBS_SECTOR_SAMPLES - 1, 1)
            sample_angle = outward_angle + delta_theta * (2 * frac - 1)
            pt = C.center + sector_radius * np.array([
                math.cos(sample_angle), math.sin(sample_angle)
            ])
            if obs.sdf(pt) < D_SAFE:
                return False   # obstacle too close in the sector

    return True


_ESCAPE_COOLDOWN_STEPS = 200  # ≈ 10 s at DT=0.05 — prevents immediate re-join in dense environments


def escape_robot(robot, C) -> None:
    """
    Transition robot out of MGR mode back to GOAL mode.

    Paper Algorithm 1 ESCAPE_MGR (line 29): ai.mode ← GOAL.
    Also cleans up the roundabout membership and resets the hysteresis counter.
    Sets escape_cooldown so RECEIVE_MGR does not immediately re-recruit this
    robot into an adjacent roundabout before it has moved away.
    """
    if robot.id in C.members:
        C.members.remove(robot.id)
    robot.mode = RobotMode.GOAL
    robot.roundabout_id = None
    robot.escape_perp_count = 0
    robot.escape_cooldown = _ESCAPE_COOLDOWN_STEPS
    robot.mgr_step_count = 0
