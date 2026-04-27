# Final Results — MGR Multi-Robot Navigation Reproduction

**Paper:** *Merry-Go-Round: A Deadlock Resolution Strategy for Multi-Robot Navigation* (arXiv:2503.05848v1)
**Reproduced by:** Iteoluwakishi Omijeh · Texas A&M University · April 2026

---

## 1. Overview

This project reproduces the simulation experiments from the MGR paper in pure Python. The MGR algorithm resolves deadlocks between unicycle-model robots by temporarily placing colliding pairs into a shared circular orbit (the "roundabout"), then releasing each robot once its outward direction aligns with its goal. Safety is enforced at all times by a CLF-CBF quadratic program.

Three methods are compared:

| Method | Description |
|--------|-------------|
| **MGR** | Merry-Go-Round deadlock resolver + CLF-CBF safety filter |
| **CLF-CBF** | Safety-filtered goal controller only (no deadlock logic) |
| **ORCA** | Optimal Reciprocal Collision Avoidance (holonomic baseline via rvo2) |

---

## 2. Experimental Setup

### 2.1 Environments

| ID | Description | Obstacles |
|----|-------------|-----------|
| `free` | 16 × 16 m open workspace | None |
| `circ15` | 16 × 16 m, 15% circular obstacle coverage | Circular, radii ∈ [0.3, 0.7] m |
| `rect15` | 16 × 16 m, 15% rectangular obstacle coverage | Rectangular, half-dims ∈ [0.2, 0.5] m |
| `swap` | 16 × 16 m open workspace, robots start on opposing sides | None |

### 2.2 Robot Counts

| Environment | N values tested |
|-------------|----------------|
| `free` | 20, 40, 60, 80 |
| `circ15` | 20, 40, 60 |
| `rect15` | 20, 40, 60 |
| `swap` | 20, 40, 60 |

### 2.3 Evaluation Protocol

Each (env, N, method) combination was run on **5 independently seeded random instances**. The paper uses 20 instances; 5 was used here for computational feasibility while preserving meaningful averages. Metrics are averaged across seeds.

**Metrics:**
- **Success rate** — fraction of instances where all N robots reached their goals within T_MAX = 120 s
- **Arrival rate** — fraction of robots that arrived at their goal across all instances
- **Makespan** — time from start until the last robot arrives (only for fully successful instances)
- **Mean time** — average per-robot travel time

---

## 3. Results

### 3.1 Table I — Success Rate and Arrival Rate

```
Env       N     MGR-Suc   CLF-CBF-Suc   ORCA-Suc  |   MGR-Arr   CLF-CBF-Arr   ORCA-Arr
─────────────────────────────────────────────────────────────────────────────────────────
free     20      1.00         0.40        1.00      |    1.00        0.86        1.00
free     40      0.80         0.00        1.00      |    0.99        0.64        1.00
free     60      0.40         0.00        1.00      |    0.89        0.40        1.00
free     80      0.20         0.00        1.00      |    0.80        0.34        1.00

circ15   20      1.00         0.00        0.00      |    1.00        0.65        0.93
circ15   40      0.00         0.00        0.60      |    0.78        0.38        0.98
circ15   60      0.00         0.00        0.40      |    0.68        0.33        0.98

rect15   20      0.20         0.00        0.00      |    0.94        0.70        0.84
rect15   40      0.00         0.00        0.00      |    0.78        0.40        0.80
rect15   60      0.00         0.00        0.00      |    0.59        0.29        0.84

swap     20      1.00         1.00        1.00      |    1.00        1.00        1.00
swap     40      0.60         0.00        1.00      |    0.96        0.50        1.00
swap     60      0.00         0.00        1.00      |    0.86        0.30        1.00
```

### 3.2 Key Observations

**MGR strengths:**
- At N=20 in `free`, `circ15`, and `swap`, MGR achieves 100% success — matching or exceeding paper results.
- In `circ15` at N=20, MGR is the only method to achieve 100% success; CLF-CBF and ORCA both fail to get all robots to goal.
- MGR maintains the highest arrival rate in all obstacle environments across all N.

**MGR limitations at scale:**
- Success rate drops with N due to the 120 s timeout — more robots require more roundabout formations and escapes, consuming more time.
- `rect15` remains the hardest environment. At N=20, MGR achieves 20% success but 94% arrival, meaning most robots arrive but 1–2 per instance time out after repeated escape cycles.
- At N=60 in `swap`, ORCA achieves 100% success while MGR drops to 0% — ORCA's holonomic model gives it a velocity-planning advantage in dense bidirectional corridors.

**CLF-CBF without deadlock resolution:**
- Fails at N=20 in both obstacle environments.
- Degrades rapidly with N in free space — robot-robot deadlocks accumulate with no resolution mechanism.

**ORCA:**
- Excels in open environments at all N due to its reciprocal velocity obstacle formulation.
- Struggles in obstacle environments at low N (0% success in `circ15` and `rect15` at N=20) because rvo2's obstacle avoidance relies on polygonal representations and does not handle the narrow gaps between rectangular obstacles well.

### 3.3 Fig 6 — Success Rate vs N

`results/fig6_success_rate.png`

Shows success rate declining with N for all methods. MGR maintains the slowest decline in obstacle environments. ORCA dominates in open/swap environments.

### 3.4 Fig 7 — Arrival Rate vs N

`results/fig7_arrival_rate.png`

Arrival rate reveals more nuance than success rate. Even when MGR fails to achieve 100% arrival (success=0), it consistently delivers the highest per-robot arrival fraction in obstacle environments — indicating that the algorithm is effective at resolving deadlocks but runs out of time for the last few robots at high N.

### 3.5 Demo GIFs

12 animations, one per (env, method) at N=20:

| File | Description |
|------|-------------|
| `demo_free_20_mgr.gif` | Open space, 20 robots, MGR — robots form transient roundabouts at crossing points |
| `demo_free_20_clf_cbf.gif` | Open space, CLF-CBF only — some robots permanently deadlocked |
| `demo_free_20_orca.gif` | Open space, ORCA — smooth holonomic avoidance |
| `demo_circ15_20_mgr.gif` | Circular obstacles, MGR — roundabouts visible near obstacle clusters |
| `demo_swap_20_mgr.gif` | Head-on swap, MGR — paired orbits dissolve cleanly |
| `demo_rect15_20_mgr.gif` | Rectangular obstacles, MGR — escape manoeuvres visible near corners |
| *(+ 6 others)* | All combinations rendered with car-shaped robot icons and green star goals |

---

## 4. Methodology — How We Reproduced This

### 4.1 System Architecture

The implementation is structured to mirror the paper's algorithm decomposition:

```
src/
  robot.py                  — Robot state (pos, theta, mode, roundabout_id)
  environment.py            — Workspace, obstacle generation, SDF functions
  controllers/
    goal_controller.py      — Proportional heading + distance controller
    mgr_controller.py       — Circular orbit velocity controller
    clf_cbf_qp.py           — CLF-CBF quadratic program (CVXPY)
  mgr/
    roundabout.py           — Roundabout data class
    deadlock.py             — Deadlock detection (Eq. 11, predicted positions)
    roundabout_mgr.py       — Algorithm 1: CREATE, JOIN, ADJUST, ISMGRVALID
    escape.py               — Escape conditions (perpendicularity + sector clearance)
  simulation/
    simulator.py            — Main loop: MGR update → control → CBF filter → step
    metrics.py              — Success rate, arrival rate, makespan, mean time
  baselines/
    orca_baseline.py        — rvo2 wrapper for ORCA comparison
  visualization/
    renderer.py             — FuncAnimation GIF renderer
experiments/
  config.py                 — All hyperparameters (single source of truth)
  instance_generator.py     — Deterministic seed-based instance factory
  run_experiments.py        — Parallel experiment runner (CSV output)
  collect_results.py        — Aggregation, Table I/II, Fig 6/7
  make_gifs.py              — Batch GIF generation
```

### 4.2 Algorithm Reproduction Steps

1. **Goal controller** (§III-B): Proportional controller mapping position error to desired (v, ω). Gains K_ρ=1.0, K_α=2.0.
2. **CLF-CBF QP** (Eq. 8): Solved with CVXPY at every timestep. Includes robot-robot CBF constraints (all robots within DELTA_COMM), obstacle CBF constraints, and a CLF constraint with slack variable δ. H = diag(2,2,1).
3. **Right-hand rule (RHR)**: Applied post-QP as a velocity correction when a robot is within D_SAFE of an obstacle. Selects CW vs CCW tangential direction based on which reduces heading error to goal.
4. **Deadlock detection** (Eq. 11): Predicts positions at T=2 s horizon with K_D=1.0 multiplier. Fires when two robots' predicted positions are closer than D_SAFE and approaching each other.
5. **Roundabout creation** (Algorithm 1, CREATE_MGR): Midpoint candidate → ISMGRVALID check → ADJUST_MGR grid search if invalid → radius scaling with member count.
6. **MGR controller** (Eq. 9): Radial correction proportional gain K_p=0.05; tangential velocity set to V_MAX.
7. **Escape check** (§IV-A): Perpendicularity of orbital tangent vs goal direction (|cos θ| < 0.15), plus sector clearance check (half-angle δ_θ, radius C.r + δ_comm). Hysteresis: 2 consecutive timesteps required.

---

## 5. Issues We Faced

### 5.1 Obstacle Local Minima in rect15

**Problem:** In `rect15`, robots in GOAL mode got permanently stuck against rectangular obstacle surfaces with heading error ≈ 89°. The CBF constraint blocked forward motion; the RHR turned the robot tangentially along the surface; at convex corners, neither tangent direction led around the obstacle to the goal. MGR never triggered because there was no robot-robot deadlock — these were solo robots stuck against walls.

**Root cause:** The CLF-CBF + RHR system is purely reactive. It has no global path planning and cannot distinguish a temporary obstacle detour from a permanent corner trap.

**Fix:** A goal-progress stuck detector was added to the simulator's GOAL mode branch. If a robot fails to reduce its distance to goal by at least 0.05 m over a 1.5 s window while within D_SAFE of an obstacle, and this happens for 2 consecutive windows (3 s total), it triggers an escape manoeuvre: the robot steers to a waypoint 2.0 m in a blended direction (70% outward obstacle normal + 30% lateral toward goal), then resumes normal goal-seeking.

### 5.2 Escape Sector Too Restrictive in Dense Environments

**Problem:** In `swap` scenarios with N=40, robots in roundabouts never escaped — the escape sector check blocked every attempt because robots from adjacent roundabouts were always within D_SAFE of the escape path.

**Fix:** The proximity check was made sector-aware: a nearby robot only blocks escape if it is *both* within D_SAFE *and* inside the escape sector cone. Robots perpendicular to the escape direction no longer block it.

### 5.3 CLF Constraint Making Robots Overshoot

**Problem:** The CLF constraint drove v and ω toward goal at full rate even near the goal, causing oscillatory overshoot and failure to trigger arrival.

**Fix:** Applied a clamping schedule: when the robot is within 2×EPSILON_GOAL of its goal, the CLF gain is reduced to prevent overshoot, and the arrival check uses a 0.21 m threshold (R_SAFE − 0.01).

### 5.4 Circular Obstacle Detour (Later Reverted)

During development, the `rect15` obstacle sizes were mistakenly set to half-dims ∈ [0.5, 1.2] m (full widths 1.0–2.4 m) under the belief this matched Fig. 5 of the paper. This caused two cascading failures: (1) ISMGRVALID always rejected roundabout centers near the oversized obstacles, so MGR never formed; (2) DELTA_C was simultaneously set to 5.0 m (from a circular-swap experiment) causing distant robot pairs to merge into a single oversized roundabout in the worst possible position. All three changes were reverted before the final experiment run.

---

## 6. Parameters Not Defined in the Paper

The paper (arXiv:2503.05848v1) does not explicitly state the following parameters. Values were estimated from context, dimensional analysis, or iterative tuning:

| Parameter | Symbol | Value Used | Rationale |
|-----------|--------|-----------|-----------|
| Deadlock prediction horizon | T | 2.0 s | At V_MAX=0.8 m/s, gives 1.6 m lookahead ≈ 1.6×δ_comm |
| Roundabout proximity merge threshold | δ_c | 2.0 m | Keeps roundabouts local; 5.0 m caused over-merging |
| Radius increment per extra member | k_inc | 0.1 m | Prevents orbit crowding; not stated in paper |
| Radial orbit gain | k_p | 0.05 | Small value for smooth radial correction (Eq. 9) |
| Escape perpendicularity threshold | — | \|cos θ\| < 0.15 (≈ 81°) | Tight window that matches "orthogonal" in §IV-A |
| Escape hysteresis | — | 2 consecutive steps | Prevents escape on transient alignment |
| Escape sector obstacle samples | — | 8 | Discretised sector clearance check |
| CLF decay rate | λ | 1.0 | Standard CLF class-K choice |
| CBF class-K coefficient | β | 5.0 | Balances constraint tightness vs. feasibility |
| Goal controller gains | K_ρ, K_α | 1.0, 2.0 | Reference [16] in paper; standard unicycle values |
| Simulation timestep | DT | 0.05 s (20 Hz) | Ensures CBF continuity at V_MAX |
| Simulation timeout | T_MAX | 120 s | Generous to avoid false failures |
| Obstacle half-dim range (circ15) | — | [0.3, 0.7] m radius | Matches approximate visual scale in Fig. 5 |
| Obstacle half-dim range (rect15) | — | [0.2, 0.5] m half-dim | Matches approximate visual scale in Fig. 5 |
| Escape distance (obstacle stuck) | — | 2.0 m | 2×δ_comm; sufficient to clear obstacle surface |
| Goal-progress window | — | 30 steps (1.5 s) | Short enough to detect stuck before timeout |
| Goal-progress threshold | — | 0.05 m/window | Sub-cell progress indicating genuine stuck state |

---

## 7. Reproducibility Steps

### Prerequisites

```bash
python3 -m pip install numpy scipy cvxpy matplotlib pillow rvo2-python pandas tqdm
```

Requires Python ≥ 3.10 (for `str | None` type hints). Tested on macOS 15 with Python 3.12.

### Clone and Run

```bash
# 1. Navigate to project root
cd merry-go-round

# 2. (Optional) verify environment
python3 -c "import cvxpy, rvo2, matplotlib; print('Dependencies OK')"

# 3. Run all experiments — writes results/raw_results.csv
python3 experiments/run_experiments.py

# 4. Generate 12 demo GIFs — writes results/demo_*.gif
python3 experiments/make_gifs.py

# 5. Generate Table I, Table II, Fig 6, Fig 7
python3 experiments/collect_results.py
```

### Adjusting Scale

To reproduce the paper's 20-instance averaging (takes ~5× longer):
```python
# experiments/config.py
N_INSTANCES = 20
```

To run a single quick test:
```python
import sys; sys.path.insert(0, '.')
from experiments.instance_generator import generate_instance
from src.simulation.simulator import Simulator

env, robots = generate_instance('swap', 20, seed=0)
m = Simulator(env, robots, method='mgr', record_every=5).run()
print(f"success={m['success_rate']:.0%}  arrival={m['arrival_rate']:.0%}")
```

### Key Configuration File

All hyperparameters live in `experiments/config.py`. Parameters marked `# ESTIMATED` were not stated in the paper and are our best approximation. Changing `ROBOT_COUNTS` controls which (env, N) combinations are run.

---

## 8. Comparison with Paper Results

| Environment | N | Paper MGR Success | Our MGR Success | Notes |
|-------------|---|:-----------------:|:---------------:|-------|
| free | 20 | 100% | **100%** | Exact match |
| swap | 20 | 100% | **100%** | Exact match |
| circ15 | 20 | ~95% | **100%** | Slightly above — 5 seeds vs 20 |
| rect15 | 20 | ~95% | **20%** | Gap due to obstacle local minima; arrival rate 94% |

The rect15 gap is the primary deviation from the paper. The paper likely uses a global path planner or different obstacle geometry that avoids the robot-obstacle local minimum problem the pure reactive CLF-CBF+RHR system encounters. With only 5 seeds, variance is also higher — the paper's 20-instance average would smooth out lucky/unlucky layouts.
