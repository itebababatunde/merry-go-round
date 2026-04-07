# Phase 3 Summary — MGR Algorithm

## Files Created

```
merry-go-round/
└── src/
    ├── controllers/
    │   └── clf_cbf_qp.py           # CORRECTED: obstacle CBF rows removed (paper §III-E)
    └── mgr/
        ├── roundabout.py           # Roundabout dataclass
        ├── deadlock.py             # is_deadlock_candidate, is_goal_checking
        ├── roundabout_mgr.py       # find_center, is_mgr_valid, adjust_mgr,
        │                           # create_mgr, join_mgr, run_mgr_update
        └── escape.py               # is_escapable, escape_robot
```

---

## Paper Checkpoint Corrections Applied in This Phase

Three critical deviations from the paper were found and fixed before writing Phase 3:

| # | File | Deviation | Fix |
|---|------|-----------|-----|
| 1 | `clf_cbf_qp.py` | Obstacle CBF rows in QP — paper uses right-hand rule, not QP constraints (§III-E) | Removed obstacle CBF block entirely |
| 2 | `deadlock.py` | Threshold was `K_D · D_SAFE = 0.44 m` — paper Eq. 11 says `kD · rsafe = 0.22 m` | Changed to `K_D · R_SAFE` |
| 3 | `escape.py` | Sector radius used `DELTA_SENSING = 0.5 m` — paper §III-E says δsensing = δcomm = 1.0 m | Changed to `DELTA_COMM` |

---

## What Was Built

### `src/mgr/roundabout.py` — Roundabout Dataclass

A roundabout C is a temporary circular reference path. Robots orbit counterclockwise
at radius C.r from center C.c until an escape condition is met.

```python
@dataclass
class Roundabout:
    id: int
    center: np.ndarray    # C.c — shape (2,)
    radius: float         # C.r = 0.3 m initially
    members: list         # robot IDs currently orbiting (C.n = len(members))

    def n_members(self) -> int:     # property
    def effective_clearance(self):  # C.r + k·C.n  (paper ISMGRVALID condition)
```

**`effective_clearance()`** encodes the paper's validity condition: the roundabout center
must be at least `C.r + k·C.n` metres from any obstacle, where k = K_INCREMENT = 0.1 m.
As more robots join, the required clearance grows to accommodate their outer orbits (Fig. 2a).

---

### `src/mgr/deadlock.py` — Deadlock Detection (Algorithm 1, lines 6–9)

#### `is_goal_checking(ri, rj) -> bool`

Skips deadlock detection when both robots are already near their goals in GOAL mode.
Prevents spurious roundabout creation when robots are converging to nearby destinations.

**Paper Eq. 12:**
```
‖xi − gi‖ ≤ ε  AND  ‖xj − gj‖ ≤ ε  AND  ri.mode = rj.mode = GOAL
```

#### `is_deadlock_candidate(ri, rj, deadlock_flags_i=None) -> bool`

Two conditions — either triggers:

**Condition (a) — already at safety barrier (Eq. 10):**
```
‖xi − xj‖ ≤ D_SAFE + 1e-3
```
Catches robots that are already stopped face-to-face at the QP's safety barrier.

**Condition (b) — predicted minimum distance ≤ kD·rsafe (Eq. 11, CORRECTED):**
```
dp = ri.pos − rj.pos,   dv = ri.velocity − rj.velocity
t* = clip(−dp·dv / ‖dv‖², 0, T_DEADLOCK)
min_dist = ‖dp + dv·t*‖

return min_dist ≤ K_D · R_SAFE     ← 0.22 m with kD=1
```
This is the closed-form analytic minimum of ‖dp + dv·t‖² over `t ∈ [0, T]`. The threshold
`K_D · R_SAFE` (not D_SAFE) means condition (b) fires preemptively — before robots reach
the safety barrier — since kD ∈ [1, 2) keeps the threshold strictly below D_SAFE.

**Head-on degeneracy guard:** if `rj.id ∈ deadlock_flags_i` (CBF gradient degenerate from
the QP), treat as deadlock immediately. This catches the singular case where both robots
face exactly head-on and the Lie derivative ‖Lg_h‖ → 0.

---

### `src/mgr/roundabout_mgr.py` — Algorithm 1 Implementation

#### `find_center(ri, rj)`
Midpoint of the two robots. Paper: *"We use the midpoint between ai and aj for simplicity."*

#### `is_mgr_valid(C, obstacles)`
Checks that `obs.sdf(C.center) ≥ C.effective_clearance()` for every obstacle, and that
the center is inside the workspace with the same margin. This is paper ISMGRVALID.

#### `adjust_mgr(C, obstacles) -> Roundabout | None`
Grid search for the nearest valid center when the candidate is invalid (paper ADJUST_MGR).

- **Grid:** step = C.r / 2, half-width = 10·C.r → 41×41 grid (matches paper §IV-B footnote).
- **Selection rule:** "valid cell with lowest index among those closest to C.c" — implemented
  by iterating in row-major order (dx outer, dy inner) with strict `<` for distance.
- Returns None if no valid cell found in the search window.

#### `create_mgr(ri, rj, obstacles, next_id)`
Combines `find_center` + `is_mgr_valid` + `adjust_mgr` into one call. Returns None if
no valid center can be found after adjustment.

#### `join_mgr(robot, C)`
Appends robot to `C.members` (idempotent), sets `robot.mode = MGR`, sets `robot.roundabout_id`.

#### `run_mgr_update(active_robots, roundabouts, obstacles, qp_info_map, next_id)`

Faithful implementation of Algorithm 1, called **once per timestep before QP controls**:

```
Step A — RECEIVE_MGR (lines 1–3):
  Each GOAL-mode robot checks if any neighbor within δcomm is in MGR mode.
  If so, it joins that neighbor's roundabout (broadcast propagation).
  This must run before deadlock detection so broadcast takes priority.

Step B — Deadlock detection (lines 4–27):
  Collect all GOAL-mode pairs within δcomm.
  Sort by ascending predicted min-distance (most urgent pair first).
  For each deadlock pair:
    If a roundabout exists within δc = 2m of the computed center:
      → validate (adjust if needed) → JOIN existing roundabout
    Else:
      → CREATE new roundabout → JOIN both robots

Step C — Prune empty roundabouts.
```

**Sort by urgency:** sorting pairs by ascending predicted min-distance ensures the most
imminent deadlock is resolved first, preventing inconsistent assignments when three or more
robots interact simultaneously.

---

### `src/mgr/escape.py` — Escape Condition (Algorithm 1, lines 28–30)

#### `is_escapable(robot, C, all_robots, obstacles, env_type) -> bool`

A robot escapes when two conditions hold simultaneously for ≥ 2 consecutive timesteps.

**Condition 1 — Perpendicularity (paper §IV-A):**

```
vic = C.center − robot.pos     # robot → roundabout center
vig = robot.goal − robot.pos   # robot → goal

cos_angle = |vic · vig| / (‖vic‖ · ‖vig‖)
perp_ok   = cos_angle < 0.15   # ≈ within 81° of perpendicular
```

The geometric intuition: when `vic ⊥ vig`, the robot is at the "side" of the roundabout
where continuing straight would take it toward its goal. The 0.15 threshold and 2-step
hysteresis are implementation choices not stated in the paper; they prevent premature
escape from transient near-perpendicularity during orbit.

**Condition 2 — Sector clearance (paper §IV-A, Fig. 4, CORRECTED):**

```
outward_dir   = (robot.pos − C.center) / ‖robot.pos − C.center‖
outward_angle = atan2(outward_dir.y, outward_dir.x)
δθ            = π/12 (free/swap) or π/6 (obstacle envs)
sector_radius = ‖robot.pos − C.center‖ + δcomm      ← δcomm = 1.0 m

For each other robot: check if inside the sector (angle within δθ AND dist ≤ sector_radius)
For each obstacle:    sample 8 points across the sector arc; check sdf < D_SAFE
```

The sector is centered at **C.c** (not the robot), spans ±δθ around the outward direction,
and extends `orbit_radius + 1.0 m` from C.c — matching the paper exactly.

#### `escape_robot(robot, C)`
Removes robot from C.members, sets mode = GOAL, clears roundabout_id and resets
escape_perp_count to 0.

---

## Verification Results

### 2-Robot Head-On Test with MGR Active

```
Setup: Robot 0 at [5, 8] heading East → goal [11, 8]
       Robot 1 at [11, 8] heading West → goal [5, 8]
       Free environment, T_MAX = 120 s
```

```
t=3.15s   Robots 0 & 1 join roundabout 0 at [8.0, 8.0]  (GOAL → MGR)
t=6.85s   Robot 0 ESCAPING roundabout 0
t=6.85s   Robot 1 ESCAPING roundabout 0
t=6.90s   Both robots back in GOAL mode  (MGR → GOAL)
t=10.95s  Robot 0 ARRIVED at goal
t=10.95s  Robot 1 ARRIVED at goal
```

**Mode transition sequence (both robots):** `GOAL → MGR → GOAL` ✓

**Metrics:**

| Check | Value | Status |
|-------|-------|--------|
| Min pairwise distance | 0.4400 m | ≥ D_SAFE ✓ |
| Both robots arrived | Yes | ✓ |
| Safety barrier violated | No | ✓ |
| Roundabouts pruned after escape | Yes | ✓ |

**Timeline interpretation:**
- t = 0 – 3.15 s: robots approach at V_MAX, QP decelerates them as h_ij → 0.
- t = 3.15 s: condition (a) triggers (‖pos‖ ≤ D_SAFE), roundabout created at midpoint [8, 8].
- t = 3.15 – 6.85 s: both robots orbit counterclockwise. Perpendicularity condition
  accumulates over several steps until the 2-step hysteresis is satisfied and the escape
  sector is clear.
- t = 6.85 – 10.95 s: robots escape in GOAL mode and navigate directly to their goals.

---

## Algorithm 1 Step-by-Step (Paper Faithful)

```
Input: robot ai, its state (xi, vi, gi), nearby robots A within δcomm,
       time horizon T, roundabout set C, obstacles O

1:  if RECEIVE_MGR(C):                    ← neighbor broadcast received
2:      ai.mode ← JOIN_MGR(ai, C)
3:  else:
4:      for each aj ∈ A:
5:          {xj, vj, gj} ← RECEIVE_STATE(aj)
6:          if ISDEADLOCK_CANDIDATE(xi, vi, xj, vj, T):
7:              if ISGOAL_CHECKING(ai, aj, ...): continue
8:              c ← FIND_CENTER(ai, aj)           ← midpoint
9:              if ∃C ∈ C within δc of c:
10:                 if not ISMGRVALID(C, O): C ← ADJUST_MGR(C, O)
11:                 ai.mode ← JOIN_MGR(ai, C)
12:                 SEND_MGR(aj, C)
13:             else:
14:                 C ← CREATE_MGR(C)
15:                 if not ISMGRVALID(C, O): C ← ADJUST_MGR(C, O)
16:                 ai.mode ← JOIN_MGR(ai, C)
17:                 SEND_MGR(aj, C)
18: if ISESCAPABLE(ai) and ai.mode = MGR:
19:     ai.mode ← ESCAPE_MGR(ai)
```

**Key design choices not stated in paper (disclosed for report):**

| Choice | Value | Rationale |
|--------|-------|-----------|
| Perpendicularity threshold | \|cos θ\| < 0.15 | ≈81° tolerance prevents premature escape |
| Hysteresis steps | 2 | Filters transient perpendicularity during orbit |
| Grid step in ADJUST_MGR | C.r / 2 | 41×41 grid matches paper footnote |
| Pair processing order | ascending predicted min-dist | Resolves most urgent deadlock first |

---

## What Phase 4 Will Build

The simulation loop and metrics:

- **`src/simulation/simulator.py`** — Main loop implementing the double-buffer pattern
  (compute all controls before applying any), integrating `run_mgr_update`, the right-hand
  rule for obstacle avoidance (paper §III-E), and arrival checking.
- **`src/simulation/metrics.py`** — Success rate, arrival rate, makespan, mean time.
