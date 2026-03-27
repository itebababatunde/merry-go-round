"""
robot.py — Robot state representation and unicycle kinematics.

Each robot is modelled as a unicycle:
    ẋ     = v · cos θ
    ẏ     = v · sin θ
    θ̇     = ω

where v is linear velocity and ω is angular velocity.

The robot can be in one of two navigation modes:
    GOAL  — moving directly toward its goal using feedback control + QP
    MGR   — following a roundabout as part of deadlock prevention
"""

import math
import numpy as np
from enum import Enum, auto

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from experiments.config import V_MAX, W_MAX, EPSILON_GOAL


class RobotMode(Enum):
    GOAL = auto()   # normal goal-directed navigation
    MGR  = auto()   # participating in a Merry-Go-Round roundabout


class Robot:
    """
    Represents a single unicycle robot.

    Attributes
    ----------
    id : int
        Unique robot identifier.
    pos : np.ndarray, shape (2,)
        Current 2D position [x, y] in metres.
    theta : float
        Current heading angle in radians, wrapped to [-π, π].
    goal : np.ndarray, shape (2,)
        Goal position [gx, gy] in metres.
    velocity : np.ndarray, shape (2,)
        Current world-frame velocity [vx, vy] derived from last control input.
        Used by the deadlock predictor (Eq. 11).
    mode : RobotMode
        Navigation mode: GOAL or MGR.
    roundabout_id : int or None
        ID of the roundabout this robot is currently part of (MGR mode only).
    escape_perp_count : int
        Consecutive timesteps the perpendicularity escape condition has been met.
        Used for hysteresis in is_escapable().
    arrived : bool
        True once the robot has reached its goal.
    arrival_time : float or None
        Simulation time (s) when the robot arrived.
    """

    def __init__(self, robot_id: int, pos, theta: float, goal):
        self.id = robot_id
        self.pos   = np.array(pos,  dtype=float)
        self.theta = float(theta)
        self.goal  = np.array(goal, dtype=float)

        self.velocity = np.zeros(2, dtype=float)

        self.mode          = RobotMode.GOAL
        self.roundabout_id = None
        self.escape_perp_count = 0

        self.arrived      = False
        self.arrival_time = None

    # ------------------------------------------------------------------
    # Kinematics
    # ------------------------------------------------------------------

    def apply_control(self, v: float, w: float, dt: float) -> None:
        """
        Integrate unicycle kinematics one timestep forward.

        The equations are:
            x(t+dt) = x(t) + v · cos θ(t) · dt
            y(t+dt) = y(t) + v · sin θ(t) · dt
            θ(t+dt) = θ(t) + ω · dt              (then wrapped to [-π, π])

        Controls are clipped to their physical limits before integration.

        Parameters
        ----------
        v  : linear velocity command (m/s)
        w  : angular velocity command (rad/s)
        dt : timestep (s)
        """
        v = float(np.clip(v, -V_MAX, V_MAX))
        w = float(np.clip(w, -W_MAX, W_MAX))

        self.pos[0] += v * math.cos(self.theta) * dt
        self.pos[1] += v * math.sin(self.theta) * dt
        self.theta   = _wrap_angle(self.theta + w * dt)

        # Store world-frame velocity for deadlock prediction
        self.velocity = np.array([v * math.cos(self.theta),
                                   v * math.sin(self.theta)])

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def state(self) -> np.ndarray:
        """Full state vector [x, y, θ]."""
        return np.array([self.pos[0], self.pos[1], self.theta])

    def dist_to_goal(self) -> float:
        """Euclidean distance to goal position."""
        return float(np.linalg.norm(self.pos - self.goal))

    def check_arrival(self, t: float) -> bool:
        """
        Mark robot as arrived if within EPSILON_GOAL of goal.
        Returns True if newly arrived this call.
        """
        if not self.arrived and self.dist_to_goal() < EPSILON_GOAL:
            self.arrived      = True
            self.arrival_time = t
            return True
        return False

    def __repr__(self) -> str:
        return (f"Robot(id={self.id}, pos={self.pos}, θ={self.theta:.3f} rad, "
                f"mode={self.mode.name}, arrived={self.arrived})")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _wrap_angle(angle: float) -> float:
    """Wrap an angle to the interval [-π, π]."""
    return (angle + math.pi) % (2 * math.pi) - math.pi
