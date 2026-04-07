# Phase 2 Summary — Core QP Controller

## Files Created

```
merry-go-round/
└── src/
    └── controllers/
        ├── goal_controller.py      # Unicycle feedback law for GOAL mode
        ├── mgr_controller.py       # Orbital velocity computation for MGR mode (Eq. 9)
        └── clf_cbf_qp.py           # CVXOPT QP safety filter — CLF-CBF formulation (Eq. 8)
```

---

## What Was Built

### `src/controllers/goal_controller.py` — GOAL Mode Feedback Law

Produces a desired velocity `(v_des, ω_des)` that drives the robot toward its goal. This
is a standard unicycle polar-coordinate feedback controller (reference [16] in the paper).
The output is **not safety-filtered** — it is the reference velocity passed into the QP.

**Control law:**

```
ρ     = ‖pos − goal‖                   — distance to goal
φ     = atan2(Δy, Δx)                  — angle to goal in world frame
α     = wrap(φ − θ)                    — heading error (wrapped to [−π, π])

v_des = min(K_RHO · ρ, V_MAX)         — K_RHO = 1.0
ω_des = clip(K_ALPHA · α, ±W_MAX)     — K_ALPHA = 2.0
```

**Design decisions:**
- `v_des` scales linearly with distance and saturates at `V_MAX` — robot slows naturally
  as it approaches the goal.
- `ω_des` scales linearly with heading error and saturates at `W_MAX` — fast re-orientation
  without angular oscillation.
- No integral or derivative terms — the QP handles safety and the MGR handles deadlock.

---

### `src/controllers/mgr_controller.py` — MGR Orbital Control (Eq. 9)

Implements the Merry-Go-Round orbital controller. The robot moves counterclockwise
around the roundabout center at target radius `C.r`, with a proportional radial correction
that continuously nudges it back onto the orbit.

**Velocity decomposition (world frame):**

```
θ_i    = atan2(pos.y − C.center.y, pos.x − C.center.x)  — angle of robot w.r.t. center

v_tan  = V_MAX · [−sin θ_i, cos θ_i]                     — CCW tangential at full speed

r_err  = ‖pos − C.center‖ − C.r                          — signed radial error
v_rad  = (KP_RAD / n) · r_err · inward_unit · V_MAX      — proportional radial correction

v_2d   = v_tan + v_rad                                    — combined 2D velocity (world frame)
speed  = min(‖v_2d‖, V_MAX)
v_2d   = normalise(v_2d) · speed
```

**Convert to unicycle:**

```
φ_des  = atan2(v_2d.y, v_2d.x)
v_des  = speed
ω_des  = clip(K_TURN · wrap(φ_des − θ), ±W_MAX)         — K_TURN = 2.0
```

**Design decisions:**
- The radial gain is divided by `n` (number of roundabout members) so that the effective
  correction weakens as more robots join — prevents over-correction and oscillation in
  large roundabouts.
- `KP_RAD = 0.05` from config — intentionally small so the tangential component dominates
  and the orbit stays smooth.
- Output is fed into the CLF-CBF QP like GOAL mode, so inter-robot safety is always
  maintained even during orbital motion.

---

### `src/controllers/clf_cbf_qp.py` — CLF-CBF Quadratic Program (Eq. 8)

The mathematical core of the paper. Runs as the **final safety filter** for every robot
every timestep, regardless of whether it is in GOAL or MGR mode.

#### Decision Variable

`u = [v, ω, δ]ᵀ`

| Variable | Role |
|----------|------|
| `v` | Linear velocity output (m/s) |
| `ω` | Angular velocity output (rad/s) |
| `δ` | CLF slack — allows goal-attraction to be softened when it conflicts with safety |

#### CVXOPT QP Form: minimise ½ uᵀPu + qᵀu

```
P = diag(2, 2, 1)              — H from paper; CVXOPT absorbs the ½ factor
q = [−2·v_des, −2·ω_des, 0]ᵀ — F from paper
```
Minimising this drives `(v, ω)` as close as possible to `(v_des, ω_des)` while keeping `δ` small.

#### Lie Derivatives for Unicycle Kinematics

The unicycle has `f(x) = 0` (no drift) and:

```
g(x) = [[cos θ,  0],
         [sin θ,  0],
         [  0,    1]]
```

This gives — for any smooth scalar function `h(x)` depending only on position:

```
Lf_h = 0               (no drift term)
Lg_h = ∂h/∂x · cosθ + ∂h/∂y · sinθ      (only v enters; ω does not affect position directly)
```

#### Constraints (all as Gu ≤ h in CVXOPT convention)

**1. CLF — goal attraction (soft, allows slack δ):**

```
V = ‖p − goal‖²
Lg_V = 2(x−gx)cosθ + 2(y−gy)sinθ

Row: [Lg_V, 0, −1] · u ≤ −λ·V     (λ = GAMMA_CLF = 1.0)
```

Why this form: the standard CLF condition is `Lg_V·u ≤ −λ·V + δ`, rearranged to
`Lg_V·u − δ ≤ −λ·V`. The slack `δ` appears with coefficient −1 in the constraint
row so increasing `δ` relaxes the constraint.

**2. CBF — inter-robot collision avoidance (hard, one row per neighbor):**

```
h_ij = ‖pi − pj‖² − D_SAFE²
Lg_h = 2(xi−xj)cosθ + 2(yi−yj)sinθ

Row: [−Lg_h, 0, 0] · u ≤ β·h_ij     (β = ALPHA_CBF = 5.0)
```

Negated because the CBF condition `Lg_h·u + β·h ≥ 0` becomes `−Lg_h·u ≤ β·h`.

**3. CBF — obstacle avoidance (hard, one row per obstacle):**

```
h_obs = sdf(pos) − D_SAFE
∇sdf  ≈ central finite difference (ε = 1e-3 m)
Lg_h_obs = ∇sdf · [cosθ, sinθ]

Row: [−Lg_h_obs, 0, 0] · u ≤ β·h_obs
```

**4. Velocity box constraints:**

```
±v ≤ V_MAX,   ±ω ≤ W_MAX,   −δ ≤ 0
```

#### Head-On Degeneracy Guard

When two robots face each other exactly head-on, `Lg_h_ij = 2·dpx·cosθ + 2·dpy·sinθ → 0`
because `dp` is anti-parallel to the heading direction. The CBF constraint row becomes a
near-zero row, giving the solver no directional safety information. The controller detects
this with:

```python
if |Lg_h_ij| < 1e-4:
    deadlock_flags.add(neighbor.id)
```

These flags are returned in `info['deadlock_flags']` so Phase 3 (MGR deadlock detection)
can act on them without waiting for the predicted minimum distance to be computed.

#### Infeasibility Fallback

If CVXOPT returns status ≠ `'optimal'` (rare but possible in extreme crowding):

```python
v = 0.0
ω = clip(2.0 · wrap(φ_goal − θ), ±W_MAX)   # pure heading correction
```

The robot stops and realigns toward its goal; `info['feasible'] = False` is logged.

---

## Sanity Check Results

### 2-Robot Head-On Test (no MGR)

Two robots start 6 m apart, facing each other directly, both commanded toward the other's
starting position. Only the CLF-CBF QP is active — no MGR layer.

```
Step       t     dist      h_12      v1     w1      d1      v2     w2      d2  feasible
    0    0.00   6.0000  35.80640   0.800  0.000 26.4000   0.800  0.000 26.4000  yes
   10    0.50   5.2000  26.84640   0.800  0.000 22.4000   0.800  0.000 22.4000  yes
   30    1.50   3.6000  12.76640   0.800  0.000 15.3600   0.800  0.000 15.3600  yes
   60    3.00   1.2000   1.24640   0.800  0.000  7.2000   0.800  0.000  7.2000  yes
  100    5.00   0.4400   0.00000   0.000  0.000 10.3684   0.000  0.000 10.3684  yes
  150    7.50   0.4400   0.00000   0.000  0.000 10.3684   0.000  0.000 10.3684  yes
  199    9.95   0.4400   0.00000   0.000  0.000 10.3684   0.000  0.000 10.3684  yes

Min separation:         0.4400 m  (= D_SAFE)          ✓
Min h_ij across all steps: 0.000000  (≥ 0 required)   ✓
QP always feasible:    True                            ✓
```

**Interpretation:**
- Steps 0–60: robots approach at `V_MAX = 0.8 m/s`, QP imposes no constraint yet (h_12 >> 0).
- Step 60–100: QP begins decelerating both robots as `h_12 → 0`.
- Step 100+: robots stop exactly at `D_SAFE = 0.44 m` separation, `h_12 ≈ 0`, `v = 0`.
- Neither robot reaches its goal — this is **expected**. The symmetric deadlock is exactly
  the scenario Phase 3 (MGR) is designed to resolve.
- CLF slack `δ ≈ 10.37` at the deadlock state: the goal-attraction constraint is fully
  relaxed (δ large) to keep the safety constraint satisfied.

### MGR Orbital Controller Test

Robot placed on a roundabout (C.r = 0.3 m), simulated for 20 s:

```
Orbit radius stats: mean=0.3064 m   min=0.3026 m   max=0.3077 m   (target = 0.300 m)
```

Robot maintains orbit within ±0.008 m (< 3% of orbit radius) of the target. ✓

---

## Key Implementation Decisions

| Decision | Reason |
|----------|--------|
| δ is a soft slack only on the CLF, not on CBFs | Safety (collision avoidance) is never relaxed; only goal-attraction can soften |
| No obstacle CBF rows in QP | Paper §III-E explicitly states obstacle avoidance uses a right-hand rule, NOT QP constraints. The QP only constrains robot-robot pairs (Eq. 8). The `obstacles` parameter is kept in the signature for Phase 4 compatibility. |
| `KP_RAD / n` radial gain scaling | Prevents over-correction as more robots join a roundabout; keeps orbit smooth |
| `clf_cbf_qp` returns `info` dict | Exposes `deadlock_flags` and `feasible` status for Phase 3 and diagnostics |
| CVXOPT `show_progress = False` | Suppresses per-iteration output during batch experiments |

## Post-Checkpoint Correction (paper review)

An initial implementation of `clf_cbf_qp.py` incorrectly added obstacle CBF constraint rows
to the QP using SDF gradients. This was corrected after re-reading the paper:

> "For handling obstacles, we use a right-hand rule which makes robots encountering static
> obstacles move clockwise around obstacles." — paper §III-E

The corrected QP contains only:
1. CLF constraint (soft, with slack δ)
2. Inter-robot CBF constraints (hard, one per neighbor)
3. Velocity box constraints

Obstacle avoidance is handled separately by a right-hand rule in the simulator (Phase 4).

---

## What Phase 3 Will Build

The MGR deadlock prevention layer:

- **`src/mgr/deadlock.py`** — `is_deadlock_candidate` (predicted min-distance check) and
  `is_goal_checking` (near-goal exemption).
- **`src/mgr/roundabout.py`** — `Roundabout` dataclass (center, radius, members).
- **`src/mgr/roundabout_mgr.py`** — `FIND_CENTER`, `ISMGRVALID`, `ADJUST_MGR`, `JOIN_MGR`.
- **`src/mgr/escape.py`** — `is_escapable` (perpendicularity + sector clearance).

Phase 3 will use the `deadlock_flags` returned by `clf_cbf_qp` and the frozen velocity
vectors in `robot.velocity` to trigger and manage roundabouts. The 2-robot head-on scenario
from Phase 2 will finally resolve: both robots will form a roundabout, orbit past each other,
escape, and reach their goals.
