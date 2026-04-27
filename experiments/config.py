"""
config.py — Single source of truth for all paper hyperparameters.

Every other module imports constants from here. No magic numbers elsewhere.
All values are taken directly from the paper (arXiv:2503.05848v1) unless
marked ESTIMATED where the paper does not state the value explicitly.
"""

import math

# ---------------------------------------------------------------------------
# Physical robot parameters
# ---------------------------------------------------------------------------
ROBOT_RADIUS = 0.2          # m — physical robot body radius
R_SAFE = 0.22               # m — safety margin per robot (slightly > ROBOT_RADIUS)
D_SAFE = 2 * R_SAFE         # m = 0.44 m — minimum allowed inter-robot distance

V_MAX = 0.8                 # m/s — maximum linear velocity
W_MAX = math.pi / 2         # rad/s — maximum angular velocity (π/2)

# ---------------------------------------------------------------------------
# Workspace
# ---------------------------------------------------------------------------
WORKSPACE = 16.0            # m — side length of the square workspace W

# ---------------------------------------------------------------------------
# QP controller parameters  (Eq. 8 in the paper)
# ---------------------------------------------------------------------------
# Hessian diagonal H = diag(2, 2, 1) — includes slack variable δ as 3rd entry
H_DIAG = [2.0, 2.0, 1.0]

# CLF: V(x) = (p − g)ᵀ I (p − g),  decay rate γ(V) = λ·V
GAMMA_CLF = 1.0             # λ

# CBF: h_ij = ‖pi − pj‖² − D_SAFE²,  class-K coefficient α(h) = β·h
ALPHA_CBF = 5.0             # β

# ---------------------------------------------------------------------------
# GOAL mode feedback controller gains  (reference [16] in the paper)
# ---------------------------------------------------------------------------
K_RHO   = 1.0               # proportional gain on distance error
K_ALPHA = 2.0               # proportional gain on heading error

# ---------------------------------------------------------------------------
# MGR algorithm parameters
# ---------------------------------------------------------------------------
K_D = 1.0                   # deadlock prediction multiplier (Eq. 11); 1 ≤ K_D < 2

# ESTIMATED: paper states a time horizon T but does not give its value.
# At V_MAX=0.8 m/s, T=2.0 s gives a 1.6 m look-ahead (1.6× DELTA_COMM).
# Reduce to 1.0 s if MGR over-triggers on normal passing manoeuvres.
T_DEADLOCK = 2.0            # s — deadlock prediction horizon

DELTA_COMM = 1.0            # m — communication / sensing range δ_comm
DELTA_DEADLOCK = DELTA_COMM  # m — deadlock detection range (= δ_comm; robots face
                              #     goal at spawn so no pre-deflection offset)

DELTA_C      = 2.0          # m — roundabout proximity threshold δ_c (obstacle envs)
DELTA_C_SWAP = 16.0         # m — workspace-wide δ_c for swap: all head-on pairs
                             #     join the first-created roundabout regardless of y

MGR_RADIUS = 0.3            # m — initial roundabout radius C.r (paper §V-A: "C.r = 0.3 m")
K_INCREMENT = 0.1           # m — radius increment per extra member robot

KP_RAD = 0.05               # proportional radial gain k_p (Eq. 9)

# Escape sector half-angle δ_θ (Fig. 4 in paper)
DELTA_THETA_OBS  = math.pi / 6    # rad — environments with obstacles
DELTA_THETA_FREE = math.pi / 12   # rad — obstacle-free environments

# ESTIMATED: extra sensing radius beyond the roundabout used for escape check.
DELTA_SENSING = 0.5         # m

# Near-goal threshold ε (must be < R_SAFE per the paper)
EPSILON_GOAL = R_SAFE - 0.01      # m ≈ 0.21 m

# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------
# DT = 0.05 s (20 Hz): at V_MAX, robots move 0.04 m/step = 9 % of D_SAFE.
# This provides sufficient temporal resolution for CBF constraint continuity.
DT   = 0.05                 # s — simulation timestep
T_MAX = 120.0               # s — 2-minute timeout per instance

N_INSTANCES = 20            # number of random instances per (env, N) combination

# ---------------------------------------------------------------------------
# Experiment configurations  (robot counts per environment)
# ---------------------------------------------------------------------------
ROBOT_COUNTS = {
    "free":   [20, 40, 60, 80],
    "circ15": [20, 40, 60],
    "rect15": [20, 40, 60],
    "swap":   [20, 40, 60],
}
