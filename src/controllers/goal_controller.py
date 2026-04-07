"""
goal_controller.py — Unicycle feedback law for GOAL mode.

Produces a desired velocity (v_des, w_des) that is then passed into the
CLF-CBF QP safety filter. This is the standard distance/heading error
feedback controller for a unicycle robot (reference [16] in the paper).

Control law:
    ρ     = ‖pos − goal‖
    φ     = atan2(Δy, Δx)          — angle to goal in world frame
    α     = wrap(φ − θ)            — heading error

    v_des = min(K_RHO · ρ, V_MAX)
    ω_des = clip(K_ALPHA · α, ±W_MAX)
"""

import math
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from experiments.config import V_MAX, W_MAX, K_RHO, K_ALPHA
from src.robot import Robot


def _wrap_angle(angle: float) -> float:
    """Wrap angle to [-π, π]."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def goal_control(robot: Robot) -> tuple[float, float]:
    """
    Compute desired (v_des, w_des) to drive the robot toward its goal.

    This output is NOT yet safety-filtered — it is the reference velocity
    that the CLF-CBF QP will track as closely as safety allows.

    Parameters
    ----------
    robot : Robot
        Current robot state (pos, theta, goal are read).

    Returns
    -------
    v_des : float
        Desired linear velocity in [0, V_MAX] m/s.
    w_des : float
        Desired angular velocity in [−W_MAX, +W_MAX] rad/s.
    """
    dx = robot.goal[0] - robot.pos[0]
    dy = robot.goal[1] - robot.pos[1]
    rho = math.hypot(dx, dy)

    # Desired heading toward goal
    phi = math.atan2(dy, dx)

    # Heading error (wrapped to [-π, π])
    alpha = _wrap_angle(phi - robot.theta)

    v_des = min(K_RHO * rho, V_MAX)
    w_des = float(np.clip(K_ALPHA * alpha, -W_MAX, W_MAX))

    return v_des, w_des
