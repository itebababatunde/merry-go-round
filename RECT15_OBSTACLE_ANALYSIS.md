# rect15 Failure Analysis: Robot-Obstacle Local Minima

## The Problem

After reverting the circular-swap parameter detour, `rect15` MGR results remained at **0% success / ~81% arrival** despite `free`, `circ15`, and `swap` all matching the paper. A targeted diagnostic revealed the failure has nothing to do with MGR or robot-robot deadlock — it is a fundamentally different failure mode.

---

## Diagnostic Findings

Running `experiments/rect15_debug.py` with N=4–8 robots across 15 seeds produced consistent results for every stuck robot:

| Field | Value | Meaning |
|-------|-------|---------|
| `mode` | `GOAL` | Never entered MGR mode |
| `obs_clearance` | `≈ 0.000 m` | Robot touching obstacle surface |
| `heading_err` | `≈ 89°` | Facing perpendicular to goal = tangentially hugging obstacle |
| `create_mgr` | `0 ok / 0 fail` | MGR never triggered |
| `path_mid_obs` | `< 0` | Direct path to goal passes through an obstacle |

**MGR escape was idle (`escape_ok: 0`) across all runs** — the escape mechanism never fired because the robots were never in MGR mode.

---

## Root Cause

The `goal_control` function always points the desired velocity directly at the robot's goal. When a rectangular obstacle sits between the robot and its goal (`path_mid_obs < 0`), the following sequence locks the robot against the obstacle surface:

1. **Goal controller** sets desired velocity pointing into the obstacle.
2. **CBF constraint** (`h = sdf(pos) − D_SAFE ≥ 0`) prevents forward motion once the robot is within `D_SAFE = 0.44 m` of the surface.
3. **Right-hand rule** turns the robot tangentially along the obstacle boundary — it picks CW vs CCW based on which direction reduces angular error toward the goal.
4. **Local minimum**: the robot ends up at a corner or convex section of the rectangle where neither tangent direction leads around the obstacle to the goal. The robot sits at the surface with `heading_err ≈ 89°`, speed ≈ 0, indefinitely.

This is the classic **reactive controller local minimum** problem: the CBF + goal controller system is purely reactive with no global path planning. It cannot distinguish between "obstacle in the way — go around" and "corner trap — need to back up and try a different side."

---

## Why MGR Does Not Help

MGR (Merry-Go-Round) is designed to resolve **robot-robot deadlocks**: two robots approaching head-on whose predicted future positions collide. Its deadlock detector (`is_deadlocked`) checks for pairs of robots whose velocities are approaching each other, not for solo robots stuck against walls.

A robot pinned alone against an obstacle:
- Has no approaching robot partner → deadlock detector never fires
- Never enters `MGR` mode → `is_escapable` never runs
- Orbits no roundabout → the whole MGR machinery is bypassed

The paper (§III-D) describes the escape check and roundabout lifetime mechanisms, but these only apply to robots already in MGR mode. A robot in GOAL mode stuck against an obstacle is outside the scope of what Algorithm 1 addresses.

---

## The Fix: Obstacle-Stuck Escape in the Simulator

**Location:** `src/simulation/simulator.py`

A lightweight stuck detector was added to the GOAL mode control branch. It tracks per-robot consecutive steps where:
- Nearest obstacle SDF < `D_SAFE` (within safety margin)
- World-frame speed < `0.05 × V_MAX` (≈ 0.04 m/s, nearly stationary)

**Parameters:**

| Constant | Value | Meaning |
|----------|-------|---------|
| `_STUCK_OBS_THRESH` | `D_SAFE = 0.44 m` | Obstacle proximity threshold |
| `_STUCK_SPEED_THRESH` | `0.05 × V_MAX = 0.04 m/s` | Near-zero speed threshold |
| `_STUCK_STEPS` | `60 steps = 3 s` | Consecutive steps before declaring stuck |
| `_ESCAPE_STEPS` | `80 steps = 4 s` | Duration of escape maneuver |
| `_ESCAPE_DIST` | `2 × DELTA_COMM = 2.0 m` | Outward displacement for escape waypoint |

**Escape maneuver:**
1. Compute the outward surface normal at the robot's position via SDF gradient.
2. Set a temporary waypoint `robot.pos + 2.0 m × n_hat`, clamped to workspace bounds.
3. For the next 4 seconds, steer toward the waypoint instead of the goal.
4. After reaching the waypoint (or after 4 s), resume normal goal-seeking from the new position.

The robot emerges from behind the obstacle from a different angle, allowing the goal controller to find an unblocked approach to the goal.

**What is not changed:**
- MGR logic is untouched — this fix only affects GOAL mode robots.
- CBF and RHR remain active during the escape maneuver (the escape waypoint is still safety-filtered).
- The fix only triggers if a robot has been stuck for ≥ 3 consecutive seconds — normal obstacle navigation (turning, brief slowdowns) does not trigger it.

---

## Expected Impact

- `rect15` stuck robots should be rescued and resume goal-seeking from clear positions.
- `free`, `circ15`, `swap` are unaffected — robots in open environments never hit the `_STUCK_STEPS` threshold.
- `min_dist` safety metric should remain above 0.05 m (the escape waypoint aims outward from the obstacle, away from other robots).
