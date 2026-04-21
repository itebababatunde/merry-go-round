"""
rect15_debug.py — Minimal replication of rect15 MGR failure.

Runs rect15 with N=4 robots across multiple seeds and instruments:
  - create_mgr success/failure rate (how often center search fails)
  - is_escapable failure reason (perp condition vs sector blocked)
  - Per-robot final state (mode, roundabout, dist-to-goal, obs clearance)
  - Roundabout lifecycle (open at end vs dissolved)

Usage:
    python3 experiments/rect15_debug.py
"""

import sys, os, math
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from experiments.config import DT, T_MAX, DELTA_COMM, D_SAFE, V_MAX, DELTA_THETA_OBS
from experiments.instance_generator import generate_instance
from src.robot import RobotMode
from src.controllers.goal_controller import goal_control
from src.controllers.mgr_controller import mgr_control
from src.controllers.clf_cbf_qp import clf_cbf_qp
from src.mgr.roundabout_mgr import run_mgr_update
import src.mgr.roundabout_mgr as _rmgr_mod
import src.mgr.escape as _escape_mod
from src.mgr.escape import is_escapable, escape_robot
from src.simulation.simulator import _right_hand_rule
from src.simulation.metrics import compute_metrics


# -----------------------------------------------------------------------
# Monkey-patch counters
# -----------------------------------------------------------------------

_create_ok   = [0]
_create_fail = [0]
_esc_fail_perp   = [0]
_esc_fail_sector = [0]
_esc_ok          = [0]

_orig_create = _rmgr_mod.create_mgr

def _patched_create(ri, rj, obstacles, next_id):
    result = _orig_create(ri, rj, obstacles, next_id)
    if result is None:
        _create_fail[0] += 1
    else:
        _create_ok[0] += 1
    return result

_rmgr_mod.create_mgr = _patched_create

_orig_escapable = _escape_mod.is_escapable

def _patched_escapable(robot, C, all_robots, obstacles, env_type):
    result = _orig_escapable(robot, C, all_robots, obstacles, env_type)
    if result:
        _esc_ok[0] += 1
    return result

_escape_mod.is_escapable = _patched_escapable


# -----------------------------------------------------------------------
# Instrumented run
# -----------------------------------------------------------------------

def run_debug(env_type, N, seed):
    _create_ok[0] = _create_fail[0] = 0
    _esc_fail_perp[0] = _esc_fail_sector[0] = _esc_ok[0] = 0

    env, robots = generate_instance(env_type, N, seed)
    obstacles   = env.obstacles

    roundabouts  = {}
    qp_info_map  = {}
    next_id      = 0
    n_steps      = int(T_MAX / DT)
    rt_created   = {}
    rt_dissolved = 0

    # Per-robot MGR-step tracking
    mgr_steps = [0] * N

    for step in range(n_steps):
        t = step * DT
        active = [r for r in robots if not r.arrived]
        if not active:
            break

        next_id = run_mgr_update(active, roundabouts, obstacles, qp_info_map, next_id)

        for rid, C in roundabouts.items():
            if rid not in rt_created:
                rt_created[rid] = t

        for r in active:
            if r.mode == RobotMode.MGR:
                C = roundabouts.get(r.roundabout_id)
                if C and is_escapable(r, C, active, obstacles, env_type):
                    escape_robot(r, C)

        for r in active:
            if r.mode == RobotMode.MGR:
                mgr_steps[r.id] += 1

        pending = {}
        new_qp  = {}
        for r in active:
            nbs = [
                n for n in active
                if n.id != r.id
                and np.linalg.norm(r.pos - n.pos) <= DELTA_COMM
                and not (r.mode == RobotMode.MGR
                         and n.mode == RobotMode.MGR
                         and r.roundabout_id == n.roundabout_id
                         and r.roundabout_id is not None)
            ]
            if r.mode == RobotMode.GOAL:
                v_des, w_des = goal_control(r)
            else:
                C = roundabouts.get(r.roundabout_id)
                if C is None:
                    r.mode = RobotMode.GOAL
                    r.roundabout_id = None
                    v_des, w_des = goal_control(r)
                else:
                    v_des, w_des = mgr_control(r, C)

            v, w, info = clf_cbf_qp(r, nbs, obstacles, v_des, w_des)
            v, w = _right_hand_rule(r, v, w, obstacles)
            pending[r.id] = (v, w)
            new_qp[r.id]  = info

        qp_info_map = new_qp
        for r in active:
            r.apply_control(*pending[r.id], DT)
        for r in active:
            r.check_arrival(t)

        gone = [rid for rid, C in roundabouts.items() if len(C.members) == 0]
        for rid in gone:
            rt_dissolved += 1
            del roundabouts[rid]
            rt_created.pop(rid, None)

    metrics = compute_metrics(robots, t)

    # Track min pairwise dist manually
    all_pos = [r.pos for r in robots]
    min_dist = float('inf')
    for i in range(len(all_pos)):
        for j in range(i+1, len(all_pos)):
            d = float(np.linalg.norm(all_pos[i] - all_pos[j]))
            if d < min_dist:
                min_dist = d

    # ---- Report ----
    print(f"\n{'='*64}")
    print(f"  {env_type}/N={N}/seed={seed}   "
          f"arrived={metrics['n_arrived']}/{N}  "
          f"success={metrics['success_rate']:.0%}  "
          f"min_dist={min_dist:.3f}m")
    print(f"  Obstacles: {len(obstacles)}  "
          f"create_mgr: {_create_ok[0]} ok / {_create_fail[0]} fail  "
          f"escape_ok: {_esc_ok[0]}  "
          f"roundabouts dissolved: {rt_dissolved}  still_open: {len(roundabouts)}")
    print(f"{'='*64}")

    stuck = [r for r in robots if not r.arrived]
    if not stuck:
        print("  All robots arrived.")
        return

    for r in stuck:
        C     = roundabouts.get(r.roundabout_id)
        obs_d = min((obs.sdf(r.pos) for obs in obstacles), default=999.0)
        g_d   = float(np.linalg.norm(r.pos - r.goal))
        c_str = (f"roundabout={r.roundabout_id}  C.r={C.radius:.2f}  "
                 f"members={len(C.members)}"
                 if C else "no roundabout")
        print(f"  STUCK robot {r.id:2d}  mode={r.mode.name:4s}  "
              f"dist_goal={g_d:.2f}m  obs_clearance={obs_d:.3f}m  "
              f"mgr_steps={mgr_steps[r.id]}  cooldown={r.escape_cooldown}  "
              f"{c_str}")

    if roundabouts:
        print(f"\n  Open roundabouts at end:")
        for rid, C in roundabouts.items():
            age = t - rt_created.get(rid, t)
            print(f"    C{rid}: center=({C.center[0]:.1f},{C.center[1]:.1f})  "
                  f"r={C.radius:.2f}  members={C.members}  age={age:.1f}s  "
                  f"obs_clearance={min(obs.sdf(C.center) for obs in obstacles):.3f}m")

    # Nearest obstacle to each stuck robot's path midpoint
    print(f"\n  Goal directions for stuck robots:")
    for r in stuck:
        mid   = (r.pos + r.goal) / 2
        obs_d_mid = min((obs.sdf(mid) for obs in obstacles), default=999.0)
        heading_to_goal = math.atan2(r.goal[1]-r.pos[1], r.goal[0]-r.pos[0])
        heading_err     = abs((heading_to_goal - r.theta + math.pi) % (2*math.pi) - math.pi)
        print(f"    robot {r.id:2d}: pos=({r.pos[0]:.1f},{r.pos[1]:.1f})  "
              f"goal=({r.goal[0]:.1f},{r.goal[1]:.1f})  "
              f"path_mid_obs={obs_d_mid:.3f}m  heading_err={math.degrees(heading_err):.0f}°")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

if __name__ == '__main__':
    # Small N to isolate individual failures
    for N in [4, 6, 8]:
        for seed in range(5):
            run_debug('rect15', N, seed)
    print("\n\nDone.")
