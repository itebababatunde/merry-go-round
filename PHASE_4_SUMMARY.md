# Phase 4 Summary — Simulation Loop & Metrics

## Files Created

```
merry-go-round/
└── src/
    └── simulation/
        ├── simulator.py    # Simulator class — main loop, double-buffer, right-hand rule
        └── metrics.py      # compute_metrics, min_pairwise_distance, collision_occurred
```

---

## What Was Built

### `src/simulation/metrics.py` — Four Paper Metrics

Implements the four metrics defined in paper §V-A verbatim.

| Metric | Definition | Code |
|--------|------------|------|
| **Success rate** | 1.0 if ALL N robots arrive within T_MAX, else 0.0 | `1.0 if n_arrived == n_total else 0.0` |
| **Arrival rate** | Fraction of robots that arrive | `n_arrived / n_total` |
| **Makespan** | Time when the LAST robot arrives (None if success=0) | `max(r.arrival_time for r in arrived)` |
| **Mean time** | Average arrival time of robots that did arrive | `mean(r.arrival_time for r in arrived)` |

**Design note on mean time:** The paper notes that when arrival rates are low, mean time only
includes "easy" goals and is therefore artificially low. Computing it only over robots that
actually arrived (not penalising non-arrivals) matches the paper's treatment exactly.

**Helper functions:**

```python
min_pairwise_distance(robots) -> float
    # Minimum Euclidean distance between any two active robots.
    # Returns inf if fewer than 2 active robots.
    # Used during verification to confirm no safety violation occurred.

collision_occurred(robots, d_safe=D_SAFE) -> bool
    # True if any pair is closer than d_safe (with 1e-4 tolerance).
    # Used as a binary safety check in unit tests.
```

---

### `src/simulation/simulator.py` — Main Loop

#### Class interface

```python
class Simulator:
    def __init__(self, env, robots, method='mgr', record_every=5)
    def run(self) -> dict       # Execute loop; returns metrics + 'min_dist' + 'method'
    def get_history(self) -> list[dict]   # State snapshots for visualization
```

**Parameters:**
- `env` — `Environment` instance (provides `.obstacles`, `.env_type`)
- `robots` — list of `Robot` objects (pre-initialised with starts/goals)
- `method` — `'mgr'` | `'clf_cbf'` | `'orca'` (orca raises NotImplementedError — Phase 5)
- `record_every` — snapshot interval in steps; default 5 = one snapshot every 0.25 s

---

#### The Six-Step Loop (per timestep)

```
for step in 0 .. T_MAX/DT:
    t = step × DT
    active = [r for r in robots if not r.arrived]
    if not active: break

    ① [MGR only]  run_mgr_update(active, roundabouts, obstacles, qp_info_map, next_id)
    ② [MGR only]  Escape checks: is_escapable → escape_robot
    ③             Double-buffer: compute ALL (v, w) from frozen pre-step positions
    ④             Apply simultaneously: r.apply_control(v, w, DT) for all active robots
    ⑤             Arrival checks: r.check_arrival(t)
    ⑥             Record snapshot every `record_every` steps
```

---

#### Step ③ — Double-Buffer (Decentralisation Correctness)

**The problem:** Each robot's QP reads its neighbours' current positions. If robot A applied
its control before robot B's QP ran, B would see A's *new* position — violating the
decentralised simultaneous-decision model in which all robots observe the same world state
and act concurrently.

**The solution:** Separate the compute phase from the apply phase:

```python
# Phase A — Compute (all robots read from frozen pre-step positions)
pending = {}
for r in active:
    v_des, w_des = goal_control(r)           # or mgr_control(r, C)
    v, w, info   = clf_cbf_qp(r, neighbors, obstacles, v_des, w_des)
    w            = _right_hand_rule(r, w, obstacles)
    pending[r.id] = (v, w)

# Phase B — Apply (all robots move simultaneously)
for r in active:
    r.apply_control(*pending[r.id], DT)
```

This is the correct discrete-time analogue of the continuous-time decentralised system
described in the paper.

---

#### Step ③ — Right-Hand Rule for Obstacle Avoidance (paper §III-E)

The paper states: *"For handling obstacles, we use a right-hand rule which makes robots
encountering static obstacles move clockwise around obstacles."*

This is implemented as a **post-QP ω override** blended by proximity to the nearest obstacle.

```
1. Find nearest obstacle obs and its SDF distance d = obs.sdf(r.pos).
2. If d ≥ DELTA_COMM (1.0 m): no effect — return QP ω unchanged.
3. Compute outward surface normal n̂ via central finite-difference of SDF.
4. Clockwise tangent: t̂ = [n̂.y, −n̂.x]       (rotate n̂ by −90°)
5. Desired heading:   φ_rhr = atan2(t̂.y, t̂.x)
6. Heading correction: w_rhr = clip(2.0 · wrap(φ_rhr − θ), ±W_MAX)
7. Blend strength:    strength = 1 − d / DELTA_COMM   (0 at DELTA_COMM, 1 at surface)
8. Output:            w = (1 − strength)·w_qp + strength·w_rhr
```

**Design decisions:**
- Only ω is overridden — v is unchanged. The QP has already set a safe linear speed.
- Only the nearest obstacle dominates — avoids competing ω corrections from multiple obstacles.
- Blending (not hard switching) gives smooth behaviour near the sensing boundary.
- SDF gradient is recomputed each step using the same central finite-difference used in the
  escape sector check — no new infrastructure needed.

---

#### Method Flag Behaviour

| `method` | Step ① MGR update | Step ② Escape | Controller in Step ③ |
|----------|--------------------|---------------|----------------------|
| `'mgr'` | `run_mgr_update` | `is_escapable` | `goal_control` or `mgr_control` + QP |
| `'clf_cbf'` | skipped | skipped | `goal_control` + QP only |
| `'orca'` | skipped | skipped | `NotImplementedError` (Phase 5) |

For `'clf_cbf'`, all robots remain in `GOAL` mode throughout. The QP alone provides safety.
This reproduces the paper's CLF-CBF baseline exactly.

---

#### State Snapshot Format (for Phase 7 visualisation)

Recorded every `record_every` steps (default: every 5th step = 0.25 s interval).
At N=120 robots over 120 s: 480 snapshots × 120 robots = ~57,600 entries — manageable in memory.

```python
{
    't': float,
    'robots': [
        {'id': int, 'pos': [x, y], 'theta': float,
         'mode': 'GOAL' | 'MGR', 'arrived': bool}
        ...
    ],
    'roundabouts': [
        {'id': int, 'center': [cx, cy], 'radius': float, 'members': [int, ...]}
        ...
    ],
}
```

---

## Verification Results

### 5-Robot Free Environment Demo (seed=42, method='mgr')

**Robot placements:**

| Robot | Start | Goal |
|-------|-------|------|
| 0 | [12.14, 7.08] | [6.05, 14.45] |
| 1 | [13.42, 10.98] | [10.18, 12.88] |
| 2 | [1.86, 15.19] | [7.14, 3.88] |
| 3 | [11.95, 12.33] | [8.83, 1.40] |
| 4 | [2.38, 7.25] | [12.95, 9.99] |

**Per-robot arrival times:**

| Robot | Arrival time |
|-------|-------------|
| 0 | 12.95 s |
| 1 | 5.75 s |
| 2 | 16.10 s |
| 3 | 15.05 s |
| 4 | 13.95 s |

**Metrics — MGR:**

| Metric | Value | Status |
|--------|-------|--------|
| Success rate | 1.00 | ✓ |
| Arrival rate | 1.00 | ✓ |
| Makespan | 16.10 s | — |
| Mean time | 12.76 s | — |
| Min pairwise dist | 0.7849 m | ≥ D_SAFE = 0.44 m ✓ |
| Snapshots recorded | 66 | every 5 steps ✓ |

**Metrics — CLF-CBF baseline (same instance):**

| Metric | Value | Status |
|--------|-------|--------|
| Success rate | 1.00 | ✓ |
| Arrival rate | 1.00 | ✓ |
| Min pairwise dist | 0.7849 m | ≥ D_SAFE ✓ |

Both methods succeeded on this low-density (N=5, Free) instance, which is expected —
deadlocks become common at higher densities and in obstacle environments.

---

## Key Design Decisions

| Decision | Reason |
|----------|--------|
| Double-buffer (compute all, then apply all) | Decentralisation correctness — all robots observe same world snapshot |
| Right-hand rule as post-QP ω blend (not QP constraint) | Paper §III-E; keeps QP lean (robot-robot CBFs only) |
| Nearest-obstacle-only RHR | Avoids conflicting ω corrections from multiple nearby obstacles |
| `method` flag on `Simulator` | Single class handles all three methods; Phase 6 batch runner just changes the flag |
| `record_every=5` default | 0.25 s interval × 120 s = 480 snapshots — low memory, sufficient for smooth animation |
| Roundabout-lost fallback | If `roundabout_id` points to a pruned roundabout, robot falls back to GOAL mode silently |

---

## What Phase 5 Will Build

Baselines:

- **`src/baselines/clf_cbf_only.py`** — Already handled by `Simulator(method='clf_cbf')`;
  no new file strictly needed. May add a thin wrapper for CLI convenience.
- **`src/baselines/orca_baseline.py`** — `rvo2` wrapper. Converts circular obstacles to
  16-gon approximations, rectangular obstacles to 4-vertex polygons. Operates holonomically
  (no unicycle conversion), matching the paper's treatment of ORCA as a holonomic baseline.
  Plugs into the same metrics infrastructure.

---

## Post-Phase Accuracy Fixes (applied after Phase 7 completion)

The following fixes were applied to the simulation loop during accuracy validation.

### Fix 1 — Robot map pre-construction for angular gap correction

**Problem:** The angular gap check in `mgr_controller.py` needs a dict mapping robot ID → robot
object to look up co-member positions. This was being constructed inside the controller call on
every step, scanning all active robots repeatedly (O(N²) per timestep).

**Fix:** The simulator now pre-constructs `robot_map = {r.id: r for r in active}` once per
timestep and passes it to `run_mgr_update` and the escape checker. All angular gap lookups use
this pre-built map.

### Fix 2 — Cold-start fix for first-step QP

**Problem:** On step 0, robots had no prior QP solution and `robot.velocity` was zero. The
deadlock predictor used `robot.velocity` to compute predicted min-distance, so all robot pairs
appeared non-deadlocked at t=0 even if they were placed head-on. This could delay MGR
engagement by 1-2 steps.

**Fix:** On step 0, `robot.velocity` is initialised to `V_MAX × unit_vec_to_goal` so the
deadlock predictor has a realistic velocity estimate before the first QP solve.

### Fix 3 — Directional v-scaling in right-hand rule

**Problem:** The right-hand rule (RHR) overrides ω to steer clockwise around obstacles but
left v unchanged. When a robot was at or past an obstacle surface in the SDF sense (sdf < 0),
the QP would allow forward motion while the RHR steered away, causing the robot to advance
into the obstacle before the RHR could redirect it.

**Fix:** The RHR blending now also scales v proportionally when the robot is heading toward
the obstacle (forward component of motion toward obstacle surface is positive):

```python
toward_obs = max(0.0, -np.dot([cos(θ), sin(θ)], sdf_grad))
v = v * (1.0 - strength * toward_obs)
```

This preserves full speed when moving tangentially or away from obstacles, and reduces speed
toward zero when heading directly into one — matching the physical intent of the right-hand rule.
