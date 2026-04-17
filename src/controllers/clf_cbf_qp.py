"""
clf_cbf_qp.py — CLF-CBF Quadratic Program safety filter (Eq. 8 in the paper).

Decision variable:  u = [v, ω, δ]ᵀ
    v : linear velocity (m/s)
    ω : angular velocity (rad/s)
    δ : CLF slack — allows the goal-attraction constraint to be softened
        when it conflicts with safety constraints (δ ≥ 0)

CVXOPT minimises  ½ uᵀ P u + qᵀ u  subject to  G u ≤ h.

    P = diag(2, 2, 1)              — from paper H; CVXOPT absorbs the ½
    q = [−2v_des, −2ω_des, 0]ᵀ   — from paper F

Constraints assembled row by row:

  (1) CLF (soft):
        Lg_V · u + λ·V ≤ δ
        → [Lg_Vv, 0, −1] · u ≤ −λ·V
        where V = ‖p − g‖², Lg_Vv = 2(x−gx)cosθ + 2(y−gy)sinθ

  (2) CBF — inter-robot (hard, one row per neighbor):
        Lg_h · u + β·h ≥ 0
        → [−Lg_hv, 0, 0] · u ≤ β·h
        where h = ‖pi−pj‖²−D_SAFE², Lg_hv = 2(xi−xj)cosθ + 2(yi−yj)sinθ

  (3) Velocity box constraints:
        v  ≤  V_MAX,  −v ≤  V_MAX
        ω  ≤  W_MAX,  −ω ≤  W_MAX
        −δ ≤  0                    (δ ≥ 0)

NOTE — obstacle avoidance: the paper (§III-E) uses a right-hand rule heuristic
to navigate around static obstacles, NOT obstacle CBF constraints in the QP.
The `obstacles` parameter is kept in the signature for Phase 4 compatibility
but no constraint rows are added here. The right-hand rule is applied in the
simulator as a separate ω override.

Head-on degeneracy guard:
    If ‖Lg_h_ij‖ < 1e-4 for any robot pair, the gradient vanishes and the
    CBF gives no directional safety information. The pair is flagged in the
    returned info dict so the MGR deadlock detector can act on it.

Infeasibility fallback:
    If CVXOPT returns a non-optimal status, set v = 0 and apply pure
    heading correction ω = clip(2·wrap(φ_goal − θ), ±W_MAX).
"""

import math
import numpy as np
import cvxopt
from cvxopt import matrix, solvers

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from experiments.config import (
    V_MAX, W_MAX, D_SAFE, GAMMA_CLF, ALPHA_CBF, H_DIAG
)
from src.robot import Robot

# Suppress CVXOPT solver output
solvers.options['show_progress'] = False
solvers.options['abstol'] = 1e-7
solvers.options['reltol'] = 1e-6
solvers.options['feastol'] = 1e-7

# Heading gain used in the infeasibility fallback
_FALLBACK_HEADING_GAIN = 2.0

# Threshold below which ‖Lg_h‖ is considered degenerate (head-on singularity)
_LGH_NORM_THRESH = 1e-4


def _wrap_angle(angle: float) -> float:
    """Wrap angle to [-π, π]."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def clf_cbf_qp(
    robot: Robot,
    neighbors: list,
    obstacles: list,
    v_des: float,
    w_des: float,
) -> tuple[float, float, dict]:
    """
    Solve the CLF-CBF QP and return a safety-filtered control input.

    Parameters
    ----------
    robot : Robot
        The controlled robot. Reads pos, theta, goal.
    neighbors : list of Robot
        Robots within DELTA_COMM communication range (already filtered).
    obstacles : list
        Obstacle objects with .sdf(pos) method.
    v_des : float
        Desired linear velocity from goal_control or mgr_control.
    w_des : float
        Desired angular velocity from goal_control or mgr_control.

    Returns
    -------
    v : float
        Safety-filtered linear velocity.
    w : float
        Safety-filtered angular velocity.
    info : dict
        'delta'          : float      — CLF slack value
        'deadlock_flags' : set[int]   — IDs of neighbors whose Lg_h is degenerate
        'feasible'       : bool       — False if fallback was used
    """
    x, y = robot.pos
    theta = robot.theta
    gx, gy = robot.goal
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    # ------------------------------------------------------------------
    # Objective:  P = diag(2, 2, 1),  q = [−2v_des, −2w_des, 0]
    # ------------------------------------------------------------------
    P_np = np.diag([float(H_DIAG[0]), float(H_DIAG[1]), float(H_DIAG[2])])
    q_np = np.array([-2.0 * v_des, -2.0 * w_des, 0.0])

    # ------------------------------------------------------------------
    # Build inequality constraints row by row: G_rows · u ≤ h_vals
    # ------------------------------------------------------------------
    G_rows = []
    h_vals = []

    # --- (1) CLF constraint (soft, uses slack δ at index 2) ---
    V_clf = (x - gx) ** 2 + (y - gy) ** 2     # = ‖p − g‖²
    Lg_V_v = 2.0 * (x - gx) * cos_t + 2.0 * (y - gy) * sin_t
    # Clamp Lg_V_v to min(Lg_V_v, 0) in the CLF row.
    # When heading TOWARD goal (Lg_V_v < 0): normal CLF — encourages v > 0 to
    #   speed convergence; larger v reduces required δ slack.
    # When heading AWAY from goal (Lg_V_v > 0): row becomes [0, 0, -1] ≤ -λV,
    #   i.e., δ ≥ λV absorbs the violation but v is unrestricted. Without
    #   this clamp, the QP forces v → 0 whenever Lg_V_v * λ * V_clf ≈ 2*v_des
    #   (which happens exactly when the RHR has turned the robot to face away from
    #   the goal near an obstacle), creating a permanent deadlock fixed point.
    Lg_V_v_clf = min(Lg_V_v, 0.0)
    # Row: [Lg_V_v_clf, 0, −1] · u ≤ −λ·V
    G_rows.append([Lg_V_v_clf, 0.0, -1.0])
    h_vals.append(-GAMMA_CLF * V_clf)

    # --- (2) CBF — inter-robot collision avoidance ---
    deadlock_flags = set()
    for nb in neighbors:
        dpx = x - nb.pos[0]
        dpy = y - nb.pos[1]
        dist_sq = dpx ** 2 + dpy ** 2
        h_ij = dist_sq - D_SAFE ** 2

        Lg_h_v = 2.0 * dpx * cos_t + 2.0 * dpy * sin_t
        Lg_h_norm = abs(Lg_h_v)   # ω has no contribution (Lg_h_w = 0)

        if Lg_h_norm < _LGH_NORM_THRESH:
            deadlock_flags.add(nb.id)

        # Row: [−Lg_h_v, 0, 0] · u ≤ β·h_ij − neighbor-velocity term
        # Full dh/dt = 2(pi-pj)·(vi-vj). QP controls vi only; move vj to RHS:
        #   -2(pi-pj)·vi ≤ α*h - 2(pi-pj)·vj
        # nb_contribution is always used — it is essential for co-members
        # orbiting CCW where Lg_h_v_i can be negative (co-member just ahead).
        # Without nb_contribution the QP forces v≤0, stopping the orbit.
        # nb_contribution from the co-member's own tangential velocity
        # relaxes the constraint and allows both to orbit freely.
        #
        # h_cbf uses the FULL h_ij (including negative values). When h < 0
        # (robots already inside D_SAFE and neighbor is stopped, nb≈0), the
        # constraint forces active separation:
        #   heading toward neighbor (Lg_h_v < 0): v ≤ negative (go backward)
        #   heading away from neighbor (Lg_h_v > 0): v ≥ positive (go forward)
        # Without this, stopped robots with h<0 have RHS=0 and the QP returns
        # v=0, leaving the robots permanently stuck in collision.
        nb_vel = nb.velocity  # world-frame [vx, vy]; may be predicted orbit vel
        nb_contribution = 2.0 * dpx * nb_vel[0] + 2.0 * dpy * nb_vel[1]
        h_cbf = ALPHA_CBF * h_ij if h_ij >= 0 else 0.0
        G_rows.append([-Lg_h_v, 0.0, 0.0])
        h_vals.append(h_cbf - nb_contribution)

    # Obstacle avoidance is handled by a right-hand rule in the simulator
    # (paper §III-E), NOT by QP constraint rows here.

    # --- (3) Velocity box constraints ---
    # v ≤ V_MAX  →  [1, 0, 0] ≤ V_MAX
    G_rows.append([1.0, 0.0, 0.0]);  h_vals.append(V_MAX)
    # −v ≤ V_MAX  →  [−1, 0, 0] ≤ V_MAX
    G_rows.append([-1.0, 0.0, 0.0]); h_vals.append(V_MAX)
    # ω ≤ W_MAX
    G_rows.append([0.0, 1.0, 0.0]);  h_vals.append(W_MAX)
    # −ω ≤ W_MAX
    G_rows.append([0.0, -1.0, 0.0]); h_vals.append(W_MAX)
    # δ ≥ 0  →  −δ ≤ 0
    G_rows.append([0.0, 0.0, -1.0]); h_vals.append(0.0)

    # ------------------------------------------------------------------
    # Convert to CVXOPT matrices (column-major, 'd' = double)
    # ------------------------------------------------------------------
    G_np = np.array(G_rows, dtype=float)
    h_np = np.array(h_vals, dtype=float)

    P_cvx = matrix(P_np, tc='d')
    q_cvx = matrix(q_np, tc='d')
    G_cvx = matrix(G_np, tc='d')
    h_cvx = matrix(h_np, tc='d')

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------
    try:
        sol = solvers.qp(P_cvx, q_cvx, G_cvx, h_cvx)
        if sol['status'] == 'optimal':
            u = np.array(sol['x']).flatten()
            v_out = float(np.clip(u[0], -V_MAX, V_MAX))
            w_out = float(np.clip(u[1], -W_MAX, W_MAX))
            delta = float(u[2])
            return v_out, w_out, {
                'delta': delta,
                'deadlock_flags': deadlock_flags,
                'feasible': True,
            }
    except Exception:
        pass

    # ------------------------------------------------------------------
    # Infeasibility fallback: stop and re-align heading toward goal
    # ------------------------------------------------------------------
    phi_goal = math.atan2(gy - y, gx - x)
    w_fallback = float(np.clip(
        _FALLBACK_HEADING_GAIN * _wrap_angle(phi_goal - theta),
        -W_MAX, W_MAX
    ))
    return 0.0, w_fallback, {
        'delta': 0.0,
        'deadlock_flags': deadlock_flags,
        'feasible': False,
    }
