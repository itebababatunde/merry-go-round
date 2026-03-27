# Phase 1 Summary — Foundation

## Files Created

```
merry-go-round/
├── requirements.txt
├── experiments/
│   └── config.py
└── src/
    ├── robot.py
    └── environment.py
```

---

## What Was Built

### `experiments/config.py` — Single Source of Truth

All paper hyperparameters live in one place. Every other module imports from
here — no magic numbers scattered across files. Two values the paper does not
state explicitly are marked `ESTIMATED` with justification comments:

| Constant | Value | Source |
|----------|-------|--------|
| `ROBOT_RADIUS` | 0.2 m | Paper §V-A |
| `R_SAFE` | 0.22 m | Paper §V-A |
| `D_SAFE` | 0.44 m | = 2 × R_SAFE |
| `V_MAX` | 0.8 m/s | Paper §V-A |
| `W_MAX` | π/2 rad/s | Paper §V-A |
| `WORKSPACE` | 16.0 m | Paper §V-A |
| `H_DIAG` | [2, 2, 1] | Paper Eq. 8 |
| `GAMMA_CLF` (λ) | 1.0 | Paper §V-A |
| `ALPHA_CBF` (β) | 5.0 | Paper §V-A |
| `K_D` | 1.0 | Paper Eq. 11 |
| `T_DEADLOCK` | 2.0 s | **ESTIMATED** — paper doesn't state T; at V_MAX=0.8 m/s this gives a 1.6 m look-ahead (1.6× comm range). Exposed as a tunable CLI argument. |
| `DELTA_COMM` | 1.0 m | Paper §V-A |
| `DELTA_C` | 2.0 m | Paper §V-A |
| `MGR_RADIUS` | 0.3 m | Paper §V-A |
| `K_INCREMENT` | 0.1 m | Paper §V-A |
| `KP_RAD` | 0.05 | Paper §V-A |
| `DELTA_THETA_OBS` | π/6 rad | Paper §V-A |
| `DELTA_THETA_FREE` | π/12 rad | Paper §V-A |
| `DELTA_SENSING` | 0.5 m | **ESTIMATED** — paper references δ_sensing but gives no value. |
| `EPSILON_GOAL` | R_SAFE − 0.01 ≈ 0.21 m | Paper §III-E (must be < R_SAFE) |
| `DT` | 0.05 s (20 Hz) | Not stated; at V_MAX robots move 0.04 m/step = 9% of D_SAFE — sufficient for CBF continuity |
| `T_MAX` | 120.0 s | Paper §V-A (2-minute limit) |
| `N_INSTANCES` | 20 | Paper §V-A |

---

### `src/robot.py` — Unicycle Kinematics

**What it models:**

Each robot is a unicycle — a common simplified model for wheeled robots. It has
three state variables: position (x, y) and heading angle θ. The two control
inputs are linear velocity v (how fast it moves forward) and angular velocity ω
(how fast it turns).

**The equations integrated each timestep (`apply_control`):**

```
x(t + dt)  =  x(t)  +  v · cos θ(t) · dt
y(t + dt)  =  y(t)  +  v · sin θ(t) · dt
θ(t + dt)  =  θ(t)  +  ω · dt           → then wrapped to [−π, π]
```

Why these equations? The robot always moves in the direction it is facing (θ).
`cos θ` and `sin θ` project that direction onto the x and y axes. Angular
velocity changes the heading for the next step.

**Key design decisions:**

- Controls are clipped to `[−V_MAX, V_MAX]` and `[−W_MAX, W_MAX]` before
  integration — the physical robot cannot exceed these limits.
- `self.velocity` stores `[v·cosθ, v·sinθ]` after each step. This world-frame
  velocity vector is what the deadlock predictor (Phase 3) uses to predict
  future robot positions.
- Angle wrapping keeps θ in `[−π, π]` at all times. Without this, θ would grow
  unboundedly during long turns, causing `cos`/`sin` to still work correctly but
  making heading comparisons meaningless.

**`RobotMode` enum:**

```python
class RobotMode(Enum):
    GOAL = auto()   # normal goal-directed navigation
    MGR  = auto()   # participating in a Merry-Go-Round roundabout
```

This two-state mode flag drives which controller is active each timestep
(Phase 2 and 4).

---

### `src/environment.py` — Workspace and Obstacle Generation

**Four environments matching the paper (Section V-A):**

| Type | Obstacles | Robot counts |
|------|-----------|-------------|
| `free` | None | 20–120 |
| `circ15` | Circular, 15% coverage | 20–100 |
| `rect15` | Rectangular, 15% coverage | 20–80 |
| `swap` | None; forced head-on | 20–60 |

**Obstacle generation (circ15 / rect15):**

Target area = 16² × 0.15 = **38.4 m²**. Obstacles are rejection-sampled one
at a time:
1. Draw random size and position.
2. Reject if it overlaps an existing obstacle (with a D_SAFE clearance gap
   between them).
3. Reject if the center is too close to a workspace boundary.
4. Accept and accumulate area until coverage ≥ 15%.

Circular obstacles: radius ∈ [0.3, 0.7] m.
Rectangular obstacles: half-width ∈ [0.2, 0.5] m, half-height ∈ [0.2, 0.5] m.

**Start / goal sampling:**

Rejection-sample N positions such that:
- Each is ≥ D_SAFE from every obstacle surface.
- Each is ≥ 2·R_SAFE from the workspace boundary.
- Any two positions in the same set (starts or goals) are ≥ D_SAFE apart.

**Swap environment:**

Starts are placed in a narrow band near the left wall (x ≈ 0.44–0.50 m).
Goals are the mirror image about x = 8.0 m (right wall, x ≈ 15.5 m).
This guarantees every robot must cross ~15 m directly through the workspace,
creating guaranteed head-on encounters — the hardest possible test for
deadlock prevention.

**Signed Distance Function (`sdf`):**

Both obstacle types implement `sdf(point)`:
- Returns a **positive** value = distance to the nearest obstacle surface
  when the point is outside.
- Returns a **negative** value when the point is inside the obstacle
  (collision).

This will be used in Phase 2 (CBF obstacle constraints) and Phase 3 (MGR
roundabout validity: the roundabout center must be ≥ C.r from all obstacles).

---

## Sanity Check Results

```
=== Robot kinematics ===
  Straight-line: pos=[0.04 0.] after v=0.8, w=0.0, dt=0.05  ✓
  Angular step:  θ=0.03927 rad = (π/4)·0.05                  ✓
  Angle wrap:    θ=−3.10 rad after crossing +π                ✓

=== Obstacle coverage ===
  circ15: 55 obstacles, coverage=15.00%  ✓
  rect15: 84 obstacles, coverage=15.01%  ✓

=== Swap scenario ===
  Robot 0: start x=0.486  goal x=15.514  gap=15.03 m  ✓
  Robot 1: start x=0.492  goal x=15.508  gap=15.02 m  ✓
  Robot 2: start x=0.446  goal x=15.554  gap=15.11 m  ✓
  Robot 3: start x=0.486  goal x=15.514  gap=15.03 m  ✓
  Robot 4: start x=0.462  goal x=15.538  gap=15.08 m  ✓
  Robot 5: start x=0.479  goal x=15.521  gap=15.04 m  ✓
  All gaps ≈ 15 m  ✓

=== Start/goal separation (free, N=20) ===
  starts: min separation = 0.5311 m  (≥ D_SAFE = 0.44 m)  ✓
  goals:  min separation = 0.6111 m  (≥ D_SAFE = 0.44 m)  ✓

All Phase 1 checks PASSED ✓
```

---

## What Phase 2 Will Build

The QP-based CLF-CBF controller — the mathematical core of the paper (Eq. 8).
This takes the robot state and a desired velocity, and returns a
safety-constrained control input that:
- Drives the robot toward its goal (Control Lyapunov Function constraint).
- Keeps all pairwise inter-robot distances ≥ D_SAFE (Control Barrier
  Function constraint).
- Solves a Quadratic Program via CVXOPT at every timestep.
