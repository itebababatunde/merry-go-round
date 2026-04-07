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
from experiments.config import V_MAX, W_MAX, KP_RAD

# Angular velocity gain for heading alignment in MGR mode.
# 2.0 rad/s per rad of heading error — same as K_ALPHA in GOAL mode.
_K_TURN = 2.0


def _wrap_angle(angle: float) -> float:
    """Wrap angle to [-π, π]."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def mgr_control(robot, roundabout) -> tuple[float, float]:
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

    Returns
    -------
    v_des : float
        Desired linear velocity in [0, V_MAX] m/s.
    w_des : float
        Desired angular velocity in [−W_MAX, +W_MAX] rad/s.
    """
    cx, cy = roundabout.center
    n = max(len(roundabout.members), 1)   # avoid divide-by-zero

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
    # Radial component — correction toward target orbit radius
    # r_err > 0 means robot is outside the orbit → push inward (toward center)
    # r_err < 0 means robot is inside the orbit  → push outward (away from center)
    # ------------------------------------------------------------------
    if r_curr > 1e-6:
        inward_unit = np.array([-dx, -dy]) / r_curr   # points toward center
    else:
        inward_unit = np.zeros(2)

    r_err = r_curr - roundabout.radius        # positive = outside orbit
    v_rad = (KP_RAD / n) * r_err * (-inward_unit) * V_MAX
    # Note: -inward_unit = outward direction.
    # r_err > 0 → outside → v_rad points inward (correct, since -(+) × outward = inward)
    # Rewriting: push = -(r_err) × outward = r_err × inward
    v_rad = (KP_RAD / n) * r_err * inward_unit * V_MAX

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

    return v_des, w_des
