# Phase 7 Summary — Visualization & Results Figures

## Files Created

```
merry-go-round/
└── src/
    └── visualization/
        ├── renderer.py    # FuncAnimation replay of any simulation run
        └── plotter.py     # Publication-quality Figs. 6 & 7 (IEEE-sized)
```

---

## What Was Built

### `src/visualization/renderer.py` — Animated Simulation Replay

#### Public API

```python
from src.visualization.renderer import render_animation

anim = render_animation(
    history,          # list from sim.get_history()
    env,              # Environment object (for obstacles, workspace)
    method='mgr',     # 'mgr', 'clf_cbf', or 'orca' — shown in title
    output_path=None, # 'results/demo.gif' or 'results/demo.mp4'
    fps=10,
    show=False,       # True to open interactive window
)
```

#### Visual encoding

| Element | Colour / Style |
|---------|---------------|
| Robot — GOAL mode | Blue (#3a86ff) filled circle |
| Robot — MGR mode | Red (#e63946) filled circle |
| Robot — arrived | Gray (#adb5bd) filled circle |
| Heading arrow | Black arrow (not shown for ORCA) |
| Active roundabout | Orange dashed ring with translucent fill |
| Goal position | Green star marker |
| CircularObstacle | Dark gray filled circle |
| RectangularObstacle | Dark gray filled rectangle |
| Workspace boundary | Black rectangle (16 × 16 m) |
| Title | `METHOD  t = X.XXs  arrived = K / N` |

#### Key implementation decisions

- **In-place artist updates** (not `ax.cla()`) — each robot is a permanent `Circle`
  patch whose `.set_center()` and `.set_facecolor()` are updated per frame. This is
  ~10× faster than redrawing the whole axes.
- **Roundabout ring lifecycle** — patches are added when a new roundabout ID appears
  and hidden (`.set_visible(False)`) when it dissolves. No patch leaking.
- **`blit=False`** — avoids known macOS rendering issues with `FuncAnimation`.
- **ffmpeg → pillow fallback** — tries ffmpeg for `.mp4`; automatically writes `.gif`
  if ffmpeg is unavailable.

#### Verification results — Swap / N=20 / instance=0

```
Method      Success  Arrival  Frames
MGR          1.00     1.00     78     → demo_swap_20_mgr.gif     (473 KB)
CLF-CBF      1.00     1.00     78     → demo_swap_20_clf_cbf.gif (466 KB)
ORCA         1.00     1.00     76     → demo_swap_20_orca.gif    (376 KB)
```

---

### `src/visualization/plotter.py` — Publication-Quality Figures

Generates Figs. 6 & 7 at IEEE double-column width (7.16 in), with:
- Font sizes tuned for print (8 pt body, 7 pt axes labels, 6 pt tick labels)
- Dashed / dash-dot line styles per method (readable in B&W)
- ±1 std shading clipped to [0, 1]
- Empty-axes guard (no spurious legend warnings)
- `--fmt pdf` output for direct report embedding

---

## How to Run Everything

All commands assume you are in the project root: `cd /Users/iteoluwakishi/merry-go-round`

---

### 1. Run a Quick Single Simulation (MGR, 5 robots, Free env)

```bash
python3 -c "
import numpy as np
from src.environment import Environment
from src.robot import Robot
from src.simulation.simulator import Simulator

rng    = np.random.default_rng(42)
env    = Environment('free', rng)
s, g   = env.generate_starts_goals(5)
robots = [Robot(i, s[i], 0.0, g[i]) for i in range(5)]
sim    = Simulator(env, robots, method='mgr', record_every=5)
m      = sim.run()
print(f\"success={m['success_rate']}  arrival={m['arrival_rate']}  \"
      f\"makespan={m['makespan']}  min_dist={m['min_dist']:.4f}m\")
"
```

---

### 2. Run Demo Animations (3 methods, Swap env, N=20)

Saves three GIF files to `results/`:

```bash
python3 src/visualization/renderer.py
```

Output:
```
results/demo_swap_20_mgr.gif      # roundabout formation visible
results/demo_swap_20_clf_cbf.gif  # QP-only baseline
results/demo_swap_20_orca.gif     # holonomic baseline
```

**View the GIFs:** open them with any image viewer, browser, or QuickTime.

```bash
open results/demo_swap_20_mgr.gif
```

---

### 3. Animate Any Custom Scenario

```python
from experiments.instance_generator import generate_instance
from src.simulation.simulator import Simulator
from src.visualization.renderer import render_animation

env, robots = generate_instance('circ15', 40, 3)   # env, N, instance_idx
sim = Simulator(env, robots, method='mgr', record_every=5)
sim.run()

render_animation(
    sim.get_history(), env,
    method='mgr',
    output_path='results/my_demo.gif',
    fps=10,
)
```

For interactive display (opens a window):

```python
render_animation(sim.get_history(), env, method='mgr', show=True)
```

For Jupyter notebook display:

```python
from IPython.display import HTML
anim = render_animation(sim.get_history(), env, method='mgr')
HTML(anim.to_jshtml())
```

---

### 4. Run the Full Batch Experiment (1,080 runs)

**Estimated time:** 30–90 minutes depending on hardware.
Results are flushed to CSV after every completed run — safe to interrupt and restart.

```bash
python3 experiments/run_experiments.py
```

Progress bar shows completed runs. Output: `results/raw_results.csv` (1,080 rows).

To run a subset first (e.g. just `free` environment) and verify correctness before
the full run, edit `ROBOT_COUNTS` in `experiments/config.py` temporarily.

---

### 5. Print Tables I & II and Save Figures

```bash
python3 experiments/collect_results.py
```

Outputs:
- **Table I** (success & arrival rate) printed to terminal
- **Table II** (makespan ± std, mean time ± std) printed to terminal
- `results/fig6_success_rate.png`
- `results/fig7_arrival_rate.png`

To use a different CSV path:

```bash
python3 experiments/collect_results.py --csv results/raw_results.csv
```

---

### 6. Generate Publication-Quality Figures (PNG or PDF)

```bash
# PNG (default, 200 dpi)
python3 src/visualization/plotter.py

# PDF for IEEE report embedding
python3 src/visualization/plotter.py --fmt pdf

# Custom path and DPI
python3 src/visualization/plotter.py --csv results/raw_results.csv \
    --output-dir results --fmt pdf --dpi 300
```

Output:
```
results/fig6_success_rate.pdf
results/fig7_arrival_rate.pdf
```

From Python:

```python
from src.visualization.plotter import plot_figures
plot_figures(csv_path='results/raw_results.csv', fmt='pdf', dpi=300)
```

---

### 7. Full Pipeline (end-to-end)

```bash
# Step 1 — run all experiments
python3 experiments/run_experiments.py

# Step 2 — aggregate and print tables
python3 experiments/collect_results.py

# Step 3 — publication figures
python3 src/visualization/plotter.py --fmt pdf

# Step 4 — demo animations
python3 src/visualization/renderer.py

# Optional — open all outputs
open results/fig6_success_rate.pdf
open results/fig7_arrival_rate.pdf
open results/demo_swap_20_mgr.gif
```

---

## Complete File Tree (all phases)

```
merry-go-round/
├── experiments/
│   ├── config.py               # All paper hyperparameters (Phase 1)
│   ├── instance_generator.py   # Hash-seeded instances (Phase 6)
│   ├── run_experiments.py      # Parallel batch runner (Phase 6)
│   └── collect_results.py      # Tables + Figs 6 & 7 (Phase 6)
├── src/
│   ├── robot.py                # Robot state, unicycle kinematics (Phase 1)
│   ├── environment.py          # Workspace, obstacles, sampling (Phase 1)
│   ├── controllers/
│   │   ├── clf_cbf_qp.py       # CLF-CBF QP (CVXOPT) (Phase 2)
│   │   ├── goal_controller.py  # Unicycle feedback to goal (Phase 2)
│   │   └── mgr_controller.py   # Orbital velocity for MGR mode (Phase 2)
│   ├── mgr/
│   │   ├── roundabout.py       # Roundabout dataclass (Phase 3)
│   │   ├── deadlock.py         # ISDEADLOCK_CANDIDATE (Phase 3)
│   │   ├── roundabout_mgr.py   # FIND_CENTER, ADJUST_MGR, JOIN_MGR (Phase 3)
│   │   └── escape.py           # ISESCAPABLE (Phase 3)
│   ├── simulation/
│   │   ├── simulator.py        # Main loop + double-buffer + RHR (Phase 4)
│   │   └── metrics.py          # Success, arrival, makespan, mean time (Phase 4)
│   ├── baselines/
│   │   ├── clf_cbf_only.py     # CLF-CBF wrapper (Phase 5)
│   │   └── orca_baseline.py    # OrcaSimulator (rvo2) (Phase 5)
│   └── visualization/
│       ├── renderer.py         # FuncAnimation replay (Phase 7)
│       └── plotter.py          # IEEE-quality Figs. 6 & 7 (Phase 7)
├── results/                    # Created at runtime
│   ├── raw_results.csv
│   ├── fig6_success_rate.png / .pdf
│   ├── fig7_arrival_rate.png / .pdf
│   ├── demo_swap_20_mgr.gif
│   ├── demo_swap_20_clf_cbf.gif
│   └── demo_swap_20_orca.gif
├── PHASE_2_SUMMARY.md
├── PHASE_3_SUMMARY.md
├── PHASE_4_SUMMARY.md
├── PHASE_5_SUMMARY.md
├── PHASE_6_SUMMARY.md
└── PHASE_7_SUMMARY.md          ← this file
```

---

## Implementation Complete

All seven phases of the paper reproduction are done:

| Phase | What | Status |
|-------|------|--------|
| 1 | config, robot, environment | ✓ |
| 2 | CLF-CBF QP, goal controller, MGR controller | ✓ |
| 3 | MGR deadlock detection, roundabout management, escape | ✓ |
| 4 | Simulation loop (double-buffer, right-hand rule, metrics) | ✓ |
| 5 | CLF-CBF baseline, ORCA baseline (rvo2) | ✓ |
| 6 | Instance generator, batch runner, result aggregation | ✓ |
| 7 | Animation renderer, publication figures | ✓ |

**Remaining action:** run `python3 experiments/run_experiments.py` to generate
`raw_results.csv`, then produce the final figures and compare against the paper's
Table I, Table II, Figs. 6 and 7.
