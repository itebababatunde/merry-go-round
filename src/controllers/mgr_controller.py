"""
mgr_controller.py — Orbital velocity computation for MGR mode (Eq. 9).

The robot follows a counterclockwise circular orbit around the roundabout
center at the target radius C.r, with a radial correction term that
continuously nudges the robot back to the correct orbit radius.

Eq. 9 decomposition:
    θ_i    = atan2(pos.y − C.center.y, pos.x − C.center.x)

    v_tan  = V_MAX · [−sin θ_i, cos θ_i]          ← CCW tangential
    v_rad  = (KP_RAD / n) · (r_curr − C.r) · inward_unit · V_MAX
             # positive r_err → outside orbit → push inward (negative radial)

    v_2d   = v_tan + v_rad          (world-frame 2D vector)
    speed  = min(‖v_2d‖, V_MAX)
    v_2d   = normalise(v_2d) · speed

    Convert to unicycle:
    φ_des  = atan2(v_2d.y, v_2d.x)
    v_des  = speed
    ω_des  = clip(K_TURN · wrap(φ_des − θ), ±W_MAX)

The output (v_des, w_des) is passed to the CLF-CBF QP safety filter
exactly like the GOAL mode output.
"""

import math
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from experiments.config import V_MAX, W_MAX, KP_RAD, D_SAFE

# Angular velocity gain for heading alignment in MGR mode.
# 2.0 rad/s per rad of heading error — same as K_ALPHA in GOAL mode.
_K_TURN = 2.0


def _wrap_angle(angle: float) -> float:
    """Wrap angle to [-π, π]."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def mgr_control(robot, roundabout, robot_map: dict = None) -> tuple[float, float]:
    """
    Compute desired (v_des, w_des) for counterclockwise orbital motion.

    The output is the reference velocity for the CLF-CBF QP safety filter;
    it is NOT yet safety-constrained with respect to other robots.

    Parameters
    ----------
    robot : Robot
        Current robot state (pos, theta are read).
    roundabout : object with attributes
        .center  : np.ndarray shape (2,) — roundabout center [cx, cy]
        .radius  : float                  — target orbit radius
        .members : list                   — member robot IDs (len = n)
    robot_map : dict, optional
        Mapping from robot id → Robot for all active robots. When provided,
        the angular gap correction is applied: if a co-member is too close
        ahead in the CCW orbit direction, v_des is scaled down to prevent
        catching up (tangential collision). Standard CBF h = ||pi-pj||² has
        dh/dt ≈ 0 for pure tangential motion, so this proactive controller-
        level correction is the only mechanism that prevents angular gap
        collapse between co-members orbiting the same ring.

    Returns
    -------
    v_des : float
        Desired linear velocity in [0, V_MAX] m/s.
    w_des : float
        Desired angular velocity in [−W_MAX, +W_MAX] rad/s.
    """
    cx, cy = roundabout.center

    # Vector from roundabout center to robot
    dx = robot.pos[0] - cx
    dy = robot.pos[1] - cy
    r_curr = math.hypot(dx, dy)

    # Angle of robot relative to roundabout center
    theta_i = math.atan2(dy, dx)

    # ------------------------------------------------------------------
    # Tangential component — counterclockwise unit vector
    # d/dθ [cos θ, sin θ] = [−sin θ, cos θ]
    # ------------------------------------------------------------------
    v_tan = V_MAX * np.array([-math.sin(theta_i), math.cos(theta_i)])

    # ------------------------------------------------------------------
    # Radial component — correction toward target orbit radius.
    # r_err > 0 means robot is outside the orbit → push inward.
    # r_err < 0 means robot is inside the orbit  → push outward.
    # Note: KP_RAD is NOT divided by n here. Dividing by n caused convergence
    # times > T_MAX when roundabouts had 4+ members, leaving robots frozen.
    # ------------------------------------------------------------------
    if r_curr > 1e-6:
        inward_unit = np.array([-dx, -dy]) / r_curr   # points toward center
    else:
        inward_unit = np.zeros(2)

    r_err = r_curr - roundabout.radius        # positive = outside orbit
    v_rad = KP_RAD * r_err * inward_unit * V_MAX

    # ------------------------------------------------------------------
    # Combine and clip to V_MAX
    # ------------------------------------------------------------------
    v_2d = v_tan + v_rad
    speed = np.linalg.norm(v_2d)
    if speed > 1e-8:
        v_2d = v_2d / speed * min(speed, V_MAX)
        speed = min(speed, V_MAX)
    else:
        # Degenerate: robot is exactly at center — default to tangential
        v_2d = v_tan
        speed = V_MAX

    # ------------------------------------------------------------------
    # Convert 2D world-frame velocity to unicycle (v_scalar, ω_des)
    # ------------------------------------------------------------------
    phi_des = math.atan2(v_2d[1], v_2d[0])
    heading_err = _wrap_angle(phi_des - robot.theta)

    v_des = float(speed)
    w_des = float(np.clip(_K_TURN * heading_err, -W_MAX, W_MAX))

    # ------------------------------------------------------------------
    # Angular gap correction — prevent catching up to a co-member ahead.
    # The CBF h = ‖pi−pj‖² has dh/dt ≈ 0 for pure tangential circular
    # motion, so it cannot prevent angular gap collapse. This correction
    # proactively scales v_des toward 0 as the gap to any ahead co-member
    # shrinks below the safety buffer, acting as a following controller.
    # ------------------------------------------------------------------
    if robot_map is not None and len(roundabout.members) > 1:
        # Minimum safe angular separation for chord distance = D_SAFE
        min_safe_angle = 2.0 * math.asin(
            min(D_SAFE / (2.0 * max(roundabout.radius, D_SAFE * 0.01)), 1.0)
        )
        buffer_angle = min_safe_angle * 1.5   # start slowing at 1.5× the safety limit

        theta_i = math.atan2(dy, dx)          # robot's current orbit angle
        v_scale = 1.0
        for member_id in roundabout.members:
            if member_id == robot.id:
                continue
            nb = robot_map.get(member_id)
            if nb is None or nb.arrived:
                continue
            nb_dx = nb.pos[0] - cx
            nb_dy = nb.pos[1] - cy
            if math.hypot(nb_dx, nb_dy) < 1e-6:
                continue
            theta_j = math.atan2(nb_dy, nb_dx)
            # gap_ccw > 0: j is ahead of i in CCW direction by gap_ccw radians
            gap_ccw = _wrap_angle(theta_j - theta_i)
            if 0.0 < gap_ccw < buffer_angle:
                # j is ahead and close — slow down proportionally
                alpha = gap_ccw / buffer_angle   # 0 at contact, 1 at buffer
                v_scale = min(v_scale, alpha)

        v_des = float(v_des * v_scale)

    return v_des, w_des
