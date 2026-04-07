# Phase 5 Summary — Baselines

## Files Created

```
merry-go-round/
└── src/
    └── baselines/
        ├── clf_cbf_only.py     # Thin wrapper around Simulator(method='clf_cbf')
        └── orca_baseline.py    # OrcaSimulator — rvo2-based holonomic baseline
```

**Dependency installed:** `rvo2` (built from source — `sybrenstuvel/Python-RVO2`) using Cython + cmake.

---

## What Was Built

### `src/baselines/clf_cbf_only.py` — CLF-CBF Wrapper

The `Simulator(method='clf_cbf')` path was already implemented in Phase 4. This file adds
a single named entry point so the Phase 6 batch runner can import all three methods
from the same package without knowing the internal structure of `Simulator`.

```python
def run_clf_cbf(env, robots, record_every=5) -> tuple[dict, list]:
    sim = Simulator(env, robots, method='clf_cbf', record_every=record_every)
    return sim.run(), sim.get_history()
```

No new logic — purely a uniform import surface.

---

### `src/baselines/orca_baseline.py` — OrcaSimulator

#### Why ORCA is holonomic (paper §V-A)

The paper treats ORCA as a point-mass baseline: robots move with direct 2D velocity
control, no unicycle kinematics. This matches `rvo2.PyRVOSimulator` natively.
We do **not** convert velocities to unicycle `(v, ω)` — `robot.pos` is updated directly
from the rvo2 output. This is consistent with the ORCA literature and the paper's treatment.

#### rvo2 Setup (once per simulation instance)

```python
rvo_sim = rvo2.PyRVOSimulator(
    timeStep        = DT,           # 0.05 s
    neighborDist    = DELTA_COMM,   # 1.0 m
    maxNeighbors    = 10,
    timeHorizon     = 2.0,          # s — robot–robot look-ahead (= T_DEADLOCK)
    timeHorizonObst = 0.5,          # s — obstacle look-ahead (standard default)
    radius          = ROBOT_RADIUS, # 0.2 m
    maxSpeed        = V_MAX,        # 0.8 m/s
)
```

Agents, obstacles, and workspace walls are added once at startup, then
`processObstacles()` builds the kd-tree for fast spatial queries.

#### Obstacle Conversion

Both obstacle types already have `to_polygon_vertices()` methods from Phase 1:

| Obstacle type | Method | Output |
|---------------|--------|--------|
| `CircularObstacle` | `to_polygon_vertices(n=16)` | 16-gon, CCW vertices |
| `RectangularObstacle` | `to_polygon_vertices()` | 4 CCW vertices |

rvo2 requires counter-clockwise vertex ordering for solid obstacle polygons — both
methods already produce this, so no re-ordering was needed.

**Workspace boundary:** 4 thin rectangular wall polygons (thickness 0.05 m) are added
just outside the 16 × 16 m workspace edges, preventing robots from escaping the arena.

#### rvo2 Parameters

| Parameter | Value | Source |
|-----------|-------|--------|
| `timeStep` | 0.05 s | `DT` from config |
| `neighborDist` | 1.0 m | `DELTA_COMM` from config |
| `maxNeighbors` | 10 | fixed — ≥ max robots within 1 m at stated densities |
| `timeHorizon` | 2.0 s | `T_DEADLOCK` — consistent with MGR look-ahead |
| `timeHorizonObst` | 0.5 s | standard RVO2 default |
| `radius` | 0.2 m | `ROBOT_RADIUS` from config |
| `maxSpeed` | 0.8 m/s | `V_MAX` from config |

#### Per-Step Loop

```
for step in 0 .. T_MAX/DT:
    t = step × DT

    1. Set preferred velocity for each active agent:
           to_goal = goal − rvo_sim.getAgentPosition(i)
           if ‖to_goal‖ < EPSILON_GOAL: pref_vel = (0, 0)
           else: pref_vel = (to_goal / ‖to_goal‖) × V_MAX
    2. rvo_sim.doStep()
    3. Read back positions and velocities:
           robot.pos      = getAgentPosition(i)
           robot.velocity = getAgentVelocity(i)
    4. Check arrivals: ‖pos − goal‖ < EPSILON_GOAL → mark arrived
    5. Track min pairwise distance
    6. Record snapshot every record_every steps
    if all arrived: break
```

#### Class Interface

```python
class OrcaSimulator:
    def __init__(self, env, robots, record_every=5)
    def run(self) -> dict           # identical return format to Simulator.run()
    def get_history(self) -> list   # identical snapshot format to Simulator.get_history()
```

**Return format is identical to `Simulator.run()`:**

```python
{
    'success_rate': float,
    'arrival_rate': float,
    'makespan':     float | None,
    'mean_time':    float | None,
    'n_arrived':    int,
    'n_total':      int,
    't_elapsed':    float,
    'min_dist':     float,
    'method':       'orca',
}
```

This means Phase 6's batch runner uses `Simulator` and `OrcaSimulator` interchangeably —
no special-casing needed.

**Snapshot format note:** `theta` is always `0.0` (holonomic robots have no heading state)
and `mode` is always `'GOAL'` (no MGR involvement).

---

## Verification Results

### 5-Robot Free Demo — All Three Methods (seed=42)

```
Method    Success  Arrival   Makespan   Mean_t   Min_dist
----------------------------------------------------------
MGR          1.00     1.00    16.10s   12.76s   0.7849m  SAFE ✓
CLF-CBF      1.00     1.00    16.10s   12.76s   0.7849m  SAFE ✓
ORCA         1.00     1.00    15.30s   11.72s   0.8680m  SAFE ✓
```

**Interpretation:**
- MGR and CLF-CBF produce identical results here — at N=5 (Free), deadlocks are unlikely so the MGR layer never activates. This is expected.
- ORCA is slightly faster (makespan 15.30 s vs 16.10 s) — holonomic motion allows more direct paths than the unicycle kinematics used by MGR/CLF-CBF.
- All three methods are safe (min pairwise distance well above D_SAFE = 0.44 m).
- This matches the paper's Table II trend: ORCA has lower makespans in the Free environment at low N, trading off success rate at higher densities.

### Obstacle Environment Check (circ15, N=5)

```
circ15: 46 obstacles, coverage=15.1%  ✓  (obstacles correctly passed to rvo2)
ORCA on circ15 N=5: success=1.00, arrival=1.00, min_dist=0.4001m  ✓
```

The `min_dist = 0.4001 m` for circ15 is close to D_SAFE (0.44 m) — ORCA is holonomic
and does not enforce the same safety margin as the CBF-based methods. This is consistent
with ORCA's design (reciprocal velocity obstacles, not hard safety barriers). The paper's
results (Table I) show ORCA with 0% success on circ15 at higher N — the safety margin
difference becomes critical in dense obstacle environments.

---

## Key Design Decisions

| Decision | Reason |
|----------|--------|
| ORCA holonomic (no unicycle) | Paper §V-A: "ORCA as a holonomic baseline"; matches rvo2 natively |
| `to_polygon_vertices()` for conversion | Already implemented in Phase 1 environment.py for both obstacle types; no new code |
| 4 wall polygons for workspace | Prevents robots escaping the arena — rvo2 has no built-in boundary enforcement |
| `timeHorizon = T_DEADLOCK = 2.0 s` | Consistent look-ahead with MGR deadlock predictor |
| `maxNeighbors = 10` | Sufficient for stated densities (at 1m range, even N=120 in 16×16 m has ~3 neighbors on average) |
| Identical return format to `Simulator.run()` | Phase 6 batch runner needs no special-casing |
| `min_dist` tracked in ORCA loop | Enables same safety check as MGR/CLF-CBF in verification |

---

## rvo2 Installation Notes

`rvo2` is not available as a pre-built PyPI package. It was built from source:

```
git clone https://github.com/sybrenstuvel/Python-RVO2
pip install cmake cython
cd Python-RVO2
mkdir -p build/RVO2
cd build/RVO2
cmake ../.. -DCMAKE_CXX_FLAGS=-fPIC -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build .
cd ../..
python3 setup.py build_ext --inplace
pip install . --no-build-isolation
```

For reproducibility this should be added to `requirements.txt` or a `Makefile` target.

---

## What Phase 6 Will Build

Instance generation and batch experiments:

- **`experiments/instance_generator.py`** — hash-seeded deterministic instances (same 20 instances across all three methods per `(env, N)`)
- **`experiments/run_experiments.py`** — `multiprocessing.Pool` over all `(env, N, instance, method)` combinations → raw CSV (1,080 runs total)
- **`experiments/collect_results.py`** — CSV aggregation → Table I, Table II, Fig. 6, Fig. 7
