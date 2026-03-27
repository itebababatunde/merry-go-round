# MGR Paper Reproducibility — Implementation Plan

## Context

This is a **Track A Reproducibility** term project for an AI Robotics course. The goal is to reimplement
**"Merry-Go-Round: Safe Control of Decentralized Multi-Robot Systems with Deadlock Prevention"**
(arXiv:2503.05848v1) from scratch in Python and reproduce all key experimental results (Table I, Table II,
Figs. 6 & 7).

**Deliverables:** Working simulation with visualization · Code repository · 6-page IEEE report · Final presentation (due April 27)

No existing code is present — everything is built from scratch.

---

## Workflow Rule

> **After completing each phase, Claude must:**
> 1. Explain in plain language what was implemented, how it works, and what decisions were made.
> 2. Show any key code snippets or results relevant to that phase.
> 3. **Wait for explicit user approval before proceeding to the next phase.**

---

## Module Structure

```
merry-go-round/
├── requirements.txt
├── README.md
├── src/
│   ├── robot.py                    # Robot state, unicycle kinematics, RobotMode enum
│   ├── environment.py              # Workspace, obstacle generation, free-space sampling
│   ├── controllers/
│   │   ├── clf_cbf_qp.py           # CVXOPT QP solver — core CLF-CBF formulation (Eq. 8)
│   │   ├── goal_controller.py      # Unicycle feedback law for GOAL mode
│   │   └── mgr_controller.py       # Orbital velocity computation for MGR mode (Eq. 9)
│   ├── mgr/
│   │   ├── roundabout.py           # Roundabout dataclass (center, radius, members)
│   │   ├── deadlock.py             # ISDEADLOCK_CANDIDATE, ISGOAL_CHECKING
│   │   ├── roundabout_mgr.py       # FIND_CENTER, ISMGRVALID, ADJUST_MGR, CREATE/JOIN_MGR
│   │   └── escape.py               # ISESCAPABLE — perpendicularity + sector clearance
│   ├── simulation/
│   │   ├── simulator.py            # Main loop: compute all controls, then apply simultaneously
│   │   └── metrics.py              # Success rate, arrival rate, makespan, mean time
│   ├── baselines/
│   │   ├── clf_cbf_only.py         # Same QP without MGR layer
│   │   └── orca_baseline.py        # rvo2-py wrapper
│   └── visualization/
│       ├── renderer.py             # matplotlib FuncAnimation
│       └── plotter.py              # Generate Tables I/II and Figs 6/7
├── experiments/
│   ├── config.py                   # ALL paper hyperparameters — single source of truth
│   ├── instance_generator.py       # Hash-seeded random instances (20 per (N, env))
│   ├── run_experiments.py          # Parallel batch runner (multiprocessing.Pool)
│   └── collect_results.py          # CSV → tables and figures
├── results/
└── tests/
    ├── test_qp_solver.py
    ├── test_deadlock_detection.py
    └── test_roundabout.py
```

---

## Phase 1 — Foundation

**Files:** `experiments/config.py`, `src/robot.py`, `src/environment.py`

### Step 1.1 — `experiments/config.py`

Write all paper constants in one place. Every other module imports from here — no magic numbers elsewhere.

```
ROBOT_RADIUS = 0.2 m      R_SAFE = 0.22 m       D_SAFE = 0.44 m
V_MAX = 0.8 m/s           W_MAX = π/2 rad/s     WORKSPACE = 16.0 m
H_DIAG = [2, 2, 1]        GAMMA_CLF = 1.0       ALPHA_CBF = 5.0
K_D = 1.0                 T_DEADLOCK = 2.0 s    DELTA_COMM = 1.0 m
DELTA_C = 2.0 m           MGR_RADIUS = 0.3 m    K_INCREMENT = 0.1 m
KP_RAD = 0.05             DELTA_THETA_OBS = π/6  DELTA_THETA_FREE = π/12
DT = 0.05 s               T_MAX = 120.0 s       N_INSTANCES = 20
```

> **Note for T_DEADLOCK:** The paper does not state T explicitly. Start with 2.0 s (1.6 m look-ahead at
> V_MAX). Reduce to 1.0 s if MGR triggers too often on routine passing. Expose as a CLI argument.

> **Note for DT = 0.05 s:** At V_MAX robots move 0.04 m/step = 9% of D_SAFE — sufficient resolution
> for CBF constraint continuity. Matches the implicit 20 Hz control rate in CBF simulation literature.

### Step 1.2 — `src/robot.py`

`Robot` class with:
- Fields: `pos [x,y]`, `theta`, `goal`, `velocity`, `mode (GOAL|MGR)`, `roundabout_id`, `arrived`, `arrival_time`
- `apply_control(v, w, dt)`: unicycle kinematics integration + angle wrapping to `[-π, π]`
- `RobotMode` enum: `GOAL`, `MGR`

### Step 1.3 — `src/environment.py`

- **`Circ15` / `Rect15`**: rejection-sample circular or rectangular obstacles (seeded RNG) until 15% of
  256 m² is covered (~38.4 m²).
- **`generate_starts_goals(N, rng)`**: rejection-sample N non-overlapping start/goal pairs
  (≥ 2·rsafe from obstacles and each other).
- **`swap` special case**: starts at x ≈ 0.5 m (left half), goals mirrored at x ≈ 15.5 m —
  forces head-on conflicts.
- `sdf(point)` on each obstacle: signed distance function used for CBF terms and validity checks.

> ### ✅ Phase 1 Completion Checklist — Claude must do all of the following before asking for approval:
> 1. Explain the module structure created, why `config.py` is first, and how the unicycle kinematics
>    equations are implemented.
> 2. Show the actual code written for each file in this phase.
> 3. Run a sanity check: instantiate a Robot, call `apply_control`, print the resulting state.
> 4. Show that obstacle coverage is approximately 15% for a sample Circ15 and Rect15 instance.
> 5. **Ask the user: "Phase 1 is complete. Shall I proceed to Phase 2?"**

---

## Phase 2 — Core QP Controller

**Files:** `src/controllers/clf_cbf_qp.py`, `src/controllers/goal_controller.py`, `src/controllers/mgr_controller.py`

### Step 2.1 — `src/controllers/clf_cbf_qp.py`

The mathematical core of the paper (Eq. 8). Decision variable: `u = [v, ω, δ]ᵀ` (linear vel, angular vel,
CLF slack).

**CVXOPT form** — minimise `½ xᵀ P x + qᵀ x`:
- `P = diag(2, 2, 1)` — directly matches paper's H (CVXOPT absorbs the ½ factor).
- `q = [-2·v_des, -2·ω_des, 0]ᵀ` — matches paper's F.

**Lie derivatives for unicycle** (`f=0`, `g(x) = [[cosθ,0],[sinθ,0],[0,1]]`):
- CLF `V = ‖p − g‖²`:  `Lg_V = [2(x−gx)cosθ + 2(y−gy)sinθ, 0]`
  Constraint: `Lg_V·[v,ω]ᵀ + λV ≤ δ`
- CBF `h_ij = ‖pi−pj‖² − D_SAFE²`:  `Lg_h = [2(xi−xj)cosθ + 2(yi−yj)sinθ, 0]`
  Constraint: `Lg_h·[v,ω]ᵀ + β·h_ij ≥ 0`

**Infeasibility fallback:** if CVXOPT returns status ≠ `'optimal'`, set `v=0`,
`ω = heading_gain·wrap(φ−θ)`.

**Guard for head-on degeneracy:** if `‖Lg_h_ij‖ < 1e-4` for any pair, trigger deadlock detection
regardless of predicted distance.

### Step 2.2 — `src/controllers/goal_controller.py`

Standard unicycle feedback toward goal → produces `(v_des, ω_des)` fed into QP:
```
ρ = ‖p − goal‖,  φ = atan2(Δy, Δx),  α = wrap(φ − θ)
v_des = min(k_ρ · ρ, V_MAX),  ω_des = clip(k_α · α, ±W_MAX)
```
Start with `k_ρ = 1.0`, `k_α = 2.0` — tune if oscillation occurs.

### Step 2.3 — `src/controllers/mgr_controller.py`

Orbital controller implementing Eq. 9:
```
θ_i = atan2(y − C.c_y, x − C.c_x)
v_tan = V_MAX · [−sin θ_i, cos θ_i]          ← counterclockwise
v_rad = (kp / C.n) · (‖xi − C.c‖ − C.r) · (C.c − xi)/‖C.c − xi‖ · V_MAX
v_des_2d = normalize(v_rad + v_tan, V_MAX)
```
Convert to unicycle: `v_scalar = v_des_2d · [cosθ, sinθ]`; `ω_des = wrap(atan2(v_des_2d) − θ)`.

> ### ✅ Phase 2 Completion Checklist — Claude must do all of the following before asking for approval:
> 1. Explain the QP formulation: what each variable represents, how the Lie derivatives are derived for
>    unicycle kinematics, and why the slack variable δ is necessary.
> 2. Show the actual code for all three controller files.
> 3. Run a 2-robot head-on test using only the CLF-CBF QP (no MGR yet): show the robots slow and stop
>    rather than collide, and print `h_ij` at each timestep to confirm it stays ≥ 0.
> 4. Print the QP solution `(v, ω, δ)` at a few key timesteps to show the solver working.
> 5. **Ask the user: "Phase 2 is complete. Shall I proceed to Phase 3?"**

---

## Phase 3 — MGR Algorithm

**Files:** `src/mgr/deadlock.py`, `src/mgr/roundabout.py`, `src/mgr/roundabout_mgr.py`, `src/mgr/escape.py`

### Step 3.1 — `src/mgr/deadlock.py`

`is_deadlock_candidate(ri, rj, params)`:
- **Condition (a):** `‖pi − pj‖ ≤ 2·R_SAFE + ε` — already at safety barrier.
- **Condition (b):** closed-form analytic minimum of `‖dp + dv·t‖²` over `t ∈ [0, T]`:
  `t* = clip(−dp·dv / ‖dv‖², 0, T)` → check if min dist ≤ `K_D · R_SAFE`.

`is_goal_checking(ri, rj, params)`: both robots within `ε < R_SAFE` of their goals and in GOAL mode →
skip deadlock.

### Step 3.2 — `src/mgr/roundabout.py`

```python
@dataclass
class Roundabout:
    id: int
    center: np.ndarray      # C.c
    radius: float           # C.r
    members: list           # robot IDs
    # n_members = len(members)
    # effective clearance needed = C.r + K_INCREMENT * n_members
```

### Step 3.3 — `src/mgr/roundabout_mgr.py`

- **`FIND_CENTER`**: midpoint of `ri.pos` and `rj.pos`.
- **`ISMGRVALID(C, obstacles)`**: for each obstacle, `sdf(C.center) ≥ C.r + K_INCREMENT · C.n`.
- **`ADJUST_MGR`**: 41×41 grid search over ±10·C.r around candidate center; pick valid cell with
  shortest distance to original candidate (matches paper's "lowest-index" rule).
- **`JOIN_MGR`**: append robot to `C.members`, set `robot.mode = MGR`, broadcast `C` to neighbors.

Processing order for multi-robot pairs: sort by ascending predicted min-distance (most urgent pair first)
to prevent inconsistent roundabout assignments.

### Step 3.4 — `src/mgr/escape.py`

`is_escapable(ri, C, all_robots, obstacles)`:
1. Compute `vic = C.center − ri.pos`, `vig = ri.goal − ri.pos`.
2. **Perpendicularity check:** `|dot(vic, vig)| / (‖vic‖·‖vig‖) < 0.15` — within ~81° of perpendicular.
   Require condition to hold for ≥ 2 consecutive timesteps (hysteresis).
3. **Sector clearance:** sweep `2·δθ` angular sector centred on outward direction,
   radius `‖xi−C.c‖ + δ_sensing` from `C.center`; no robot or obstacle within sector.

> ### ✅ Phase 3 Completion Checklist — Claude must do all of the following before asking for approval:
> 1. Explain Algorithm 1 step-by-step: how a deadlock is detected, how a roundabout is created or joined,
>    how the grid search works, and the geometric meaning of the escape condition.
> 2. Show the actual code for all four MGR module files.
> 3. Run a 2-robot head-on test with MGR active: show the robots forming a roundabout, orbiting,
>    then escaping and reaching their goals. Print the sequence of mode changes (`GOAL → MGR → GOAL`)
>    for each robot with timestamps.
> 4. Print the roundabout center coordinates and radius relative to the two robots.
> 5. **Ask the user: "Phase 3 is complete. Shall I proceed to Phase 4?"**

---

## Phase 4 — Simulation Loop

**Files:** `src/simulation/simulator.py`, `src/simulation/metrics.py`

### Step 4.1 — `src/simulation/simulator.py`

Main loop (runs until all robots arrive or `T_MAX`):

```
1. Build neighbor lists (within δ_comm) for all active robots
2. [MGR only] Run Algorithm 1 for all active pairs — MGR update
3. Compute QP control input for every robot using pre-step positions
4. Apply all controls simultaneously (double-buffer — never update mid-step)
5. Check arrivals: ‖pos − goal‖ < EPSILON_GOAL → mark arrived, record time
6. Prune empty roundabouts
7. Record state snapshot for visualization (every 5th frame)
```

Method flag: `'mgr'`, `'clf_cbf'`, `'orca'` — controls which update path runs.

### Step 4.2 — `src/simulation/metrics.py`

| Metric | Definition |
|--------|------------|
| **Success rate** | 1.0 if ALL N robots arrive within T_MAX, else 0.0 |
| **Arrival rate** | fraction of robots that arrive (n_arrived / N) |
| **Makespan** | time when last robot arrives (undefined if not all arrived) |
| **Mean time** | average arrival time of robots that did arrive |

> ### ✅ Phase 4 Completion Checklist — Claude must do all of the following before asking for approval:
> 1. Explain why all controls must be computed before any robot moves (decentralisation correctness),
>    and show how the double-buffer pattern is implemented in the code.
> 2. Show the actual code for `simulator.py` and `metrics.py`.
> 3. Run a demo: 5 robots in the Free environment with the MGR method. Print per-robot arrival times
>    and the four metrics.
> 4. Verify no collision occurred: print the minimum pairwise distance across all timesteps and confirm
>    it is ≥ D_SAFE (0.44 m).
> 5. **Ask the user: "Phase 4 is complete. Shall I proceed to Phase 5?"**

---

## Phase 5 — Baselines

**Files:** `src/baselines/clf_cbf_only.py`, `src/baselines/orca_baseline.py`

### Step 5.1 — CLF-CBF baseline

Reuse `Simulator` with `method='clf_cbf'` — identical to MGR setup but `_mgr_update` is never called.
No new files required; activated via method flag.

### Step 5.2 — ORCA baseline

Wrap `rvo2.PyRVOSimulator`:
- Convert circular obstacles to 16-gon polygon approximations; rectangular obstacles to 4-vertex polygons.
- Call `processObstacles()` once at setup.
- ORCA operates holonomically (no unicycle conversion) — matches the paper's treatment of ORCA as a
  holonomic baseline.

Install: `pip install rvo2` — if unavailable on macOS 26.2, build from `sybrenstuvel/Python-RVO2`.

> ### ✅ Phase 5 Completion Checklist — Claude must do all of the following before asking for approval:
> 1. Explain the difference between the MGR, CLF-CBF, and ORCA simulation paths and what each baseline
>    tests (what is switched off vs. MGR).
> 2. Show the ORCA wrapper code and how obstacles are converted.
> 3. Run the same 5-robot Free demo with all three methods and print their metrics side-by-side in a
>    table.
> 4. Confirm ORCA installation succeeded and that obstacles are passed correctly.
> 5. **Ask the user: "Phase 5 is complete. Shall I proceed to Phase 6?"**

---

## Phase 6 — Instance Generation & Batch Experiments

**Files:** `experiments/instance_generator.py`, `experiments/run_experiments.py`, `experiments/collect_results.py`

### Step 6.1 — `experiments/instance_generator.py`

Hash-based deterministic seed from `(env_type, N, instance_idx)` using `hashlib.md5`. Guarantees:
- Same 20 instances reproduced identically across all three methods.
- Adding/removing a config does not shift seeds for other configs.

### Step 6.2 — `experiments/run_experiments.py`

`multiprocessing.Pool` over all `(env, N, instance, method)` combinations → raw CSV.

**Total runs:** `(6+5+4+3) × 20 × 3 methods = 1,080 runs`

Performance notes:
- Each run: 2,400 timesteps (DT=0.05 s, T_MAX=120 s).
- CBF constraints limited to neighbors within δ_comm — typical 1–3 per robot at stated densities.
- Vectorise neighbor distance computation with numpy broadcasting.

### Step 6.3 — `experiments/collect_results.py`

Aggregate raw CSV → reproduce:
- **Table I**: success/arrival rate, averaged over 20 instances, per (env, N, method).
- **Table II**: makespan and mean time (with standard deviations).
- **Fig. 6**: success rate vs N, 4-subplot layout, 3 lines per subplot.
- **Fig. 7**: arrival rate vs N, same layout.

> ### ✅ Phase 6 Completion Checklist — Claude must do all of the following before asking for approval:
> 1. Explain the hash-based seeding strategy and why it ensures reproducibility across methods.
> 2. Show 5 sample rows from the raw CSV output.
> 3. Show the aggregated Table I for the Free environment and compare the Free/N=20 row against paper
>    Table I values (MGR: 100%, CLF-CBF: 95%). Note and explain any discrepancy.
> 4. Show Table II for at least one environment.
> 5. **Ask the user: "Phase 6 is complete. Shall I proceed to Phase 7?"**

---

## Phase 7 — Visualization & Results Figures

**Files:** `src/visualization/renderer.py`, `src/visualization/plotter.py`

### Step 7.1 — `src/visualization/renderer.py`

`matplotlib FuncAnimation` showing:
- Robots coloured by mode: **blue** = GOAL, **red** = MGR.
- Heading direction arrow per robot.
- Roundabout rings as orange dashed circles.
- Goal positions as green stars.
- Title showing current time `t` and method name.

Save every 5th simulation frame (0.25 s intervals) to limit memory. Provide `--viz` flag so visualization
only runs on demo instances, not all 1,080 experiments.

### Step 7.2 — `src/visualization/plotter.py`

Reproduce Figs. 6 and 7 with shaded standard-deviation bands and save as PDF/PNG for the report.

> ### ✅ Phase 7 Completion Checklist — Claude must do all of the following before asking for approval:
> 1. Explain what the animation shows and how to interpret robot colours and roundabout rings.
> 2. Save a demo animation of the Swap/N=20 scenario (all three methods) as `.mp4` or `.gif` files.
> 3. Show the reproduced Figs. 6 and 7 and compare them to the paper figures, commenting on
>    similarities and discrepancies.
> 4. Confirm all deliverables are ready: code repo, results CSV, figures, animations.
> 5. **Ask the user: "Phase 7 is complete — the implementation is done. Shall we move to the report?"**

---

## Key Implementation Challenges

| Challenge | Solution |
|-----------|----------|
| QP degeneracy: `Lg_h → 0` when robots face head-on | Trigger deadlock if `‖Lg_h_ij‖ < 1e-4` regardless of predicted distance |
| Multi-robot roundabout join ordering | Process pairs by ascending predicted min-distance |
| Perpendicularity threshold not stated in paper | Use `|cos_angle| < 0.15` + 2-step hysteresis; expose as CLI param |
| T_DEADLOCK not stated in paper | Start T=2.0 s; reduce if MGR over-triggers on passing manoeuvres |
| rvo2 build on macOS 26.2 | `pip install rvo2` or build from `sybrenstuvel/Python-RVO2` |
| Performance at N=120 | Limit CBF constraints to δ_comm neighbours; vectorise with numpy |

---

## Dependencies

```
numpy>=2.2        # already installed
scipy>=1.15       # already installed
matplotlib>=3.10  # already installed
pandas>=2.2       # already installed
cvxopt>=1.3       # already installed
tqdm>=4.0         # already installed
shapely>=2.0      # pip install shapely
rvo2              # pip install rvo2 (or build from source)
```

---

## Unit Tests

| Test | What it checks |
|------|----------------|
| 2-robot head-on (no MGR) | No collision; QP maintains `h_ij ≥ 0`; both stop safely |
| 2-robot head-on (with MGR) | MGR triggers; both reach goals; mode transitions logged |
| QP feasibility under crowding | 1 robot surrounded by 8 at D_SAFE → `v ≈ 0`, no crash |
| Roundabout capacity | 3 robots join → effective clearance = `C.r + K_INCREMENT × 3` |
| Escape geometry | Clear sector → `is_escapable=True`; blocked → `False` |
| Metrics correctness | 8/10 robots arrive → `arrival_rate=0.8`, `success_rate=0.0` |
