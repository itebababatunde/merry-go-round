"""
diagnose.py — Diagnostic run to find why MGR underperforms CLF-CBF.

Runs the same instance with both methods and tracks per-step internals:
  - QP infeasibility rate
  - How often deadlock detection fires
  - Time each robot spends in MGR mode vs GOAL mode
  - Roundabout lifecycle (creation, member count, dissolution)
  - Whether robots are making progress toward goals or stuck circling

Usage:
    python3 experiments/diagnose.py
"""

import sys, os
import math
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from experiments.config import (
    DT, T_MAX, DELTA_COMM, D_SAFE, W_MAX, V_MAX
)
from experiments.instance_generator import generate_instance
from src.robot import RobotMode
from src.controllers.goal_controller import goal_control
from src.controllers.mgr_controller import mgr_control
from src.controllers.clf_cbf_qp import clf_cbf_qp
from src.mgr.roundabout_mgr import run_mgr_update
from src.mgr.escape import is_escapable, escape_robot
from src.simulation.metrics import compute_metrics

# -----------------------------------------------------------------------
# Instrumented simulation loop
# -----------------------------------------------------------------------

def run_instrumented(env, robots, method='mgr'):
    """Run one simulation and return detailed per-step diagnostics."""
    from src.simulation.simulator import _right_hand_rule

    obstacles  = env.obstacles
    env_type   = env.env_type
    use_mgr    = (method == 'mgr')
    n_steps    = int(T_MAX / DT)
    N          = len(robots)

    # Counters
    qp_infeasible_count = 0
    qp_total_count      = 0
    deadlock_pair_count = 0   # cumulative pairs flagged per step
    roundabout_events   = []  # (t_create, t_dissolve, peak_members)

    # Per-robot tracking
    steps_in_mgr  = [0] * N   # steps each robot spent in MGR mode
    steps_active  = [0] * N   # steps each robot was active (not arrived)
    dist_snapshots = []        # (t, mean_dist_to_goal) for active robots

    roundabouts   = {}
    qp_info_map   = {}
    next_id       = 0

    # Track roundabout lifetimes
    rt_created  = {}   # id → t_created
    rt_peak     = {}   # id → peak member count

    t = 0.0
    for step in range(n_steps):
        t = step * DT
        active = [r for r in robots if not r.arrived]
        if not active:
            break

        # Track active steps per robot
        for r in active:
            steps_active[r.id] += 1

        # ---- MGR update ----
        if use_mgr:
            prev_rids = set(roundabouts.keys())
            next_id = run_mgr_update(
                active, roundabouts, obstacles, qp_info_map, next_id
            )
            # Detect newly created roundabouts
            for rid, C in roundabouts.items():
                if rid not in prev_rids:
                    rt_created[rid] = t
                    rt_peak[rid]    = len(C.members)
                else:
                    rt_peak[rid] = max(rt_peak.get(rid, 0), len(C.members))

        # ---- Escape ----
        if use_mgr:
            for r in active:
                if r.mode != RobotMode.MGR:
                    continue
                C = roundabouts.get(r.roundabout_id)
                if C and is_escapable(r, C, active, obstacles, env_type):
                    escape_robot(r, C)

        # ---- Count MGR-mode robots ----
        for r in active:
            if r.mode == RobotMode.MGR:
                steps_in_mgr[r.id] += 1

        # ---- QP solve (double-buffer) ----
        pending    = {}
        new_qp_info = {}

        for r in active:
            neighbors = [
                n for n in active
                if n.id != r.id
                and np.linalg.norm(r.pos - n.pos) <= DELTA_COMM
                and not (r.mode == RobotMode.MGR
                         and n.mode == RobotMode.MGR
                         and r.roundabout_id is not None
                         and r.roundabout_id == n.roundabout_id)
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

            v, w, info = clf_cbf_qp(r, neighbors, obstacles, v_des, w_des)
            new_qp_info[r.id] = info
            w = _right_hand_rule(r, w, obstacles)
            pending[r.id] = (v, w)

            qp_total_count += 1
            if not info['feasible']:
                qp_infeasible_count += 1
            # Count deadlock pairs flagged this step (from QP head-on guard)
            deadlock_pair_count += len(info.get('deadlock_flags', set()))

        qp_info_map = new_qp_info

        # ---- Apply controls ----
        for r in active:
            r.apply_control(*pending[r.id], DT)

        # ---- Arrival checks ----
        for r in active:
            r.check_arrival(t)

        # ---- Prune dissolved roundabouts and record their lifetimes ----
        if use_mgr:
            dissolved = [rid for rid, C in roundabouts.items() if len(C.members) == 0]
            for rid in dissolved:
                t_c = rt_created.get(rid, t)
                roundabout_events.append((t_c, t, rt_peak.get(rid, 0)))
                del roundabouts[rid]
                rt_created.pop(rid, None)
                rt_peak.pop(rid, None)

        # ---- Distance-to-goal snapshot every 1 s ----
        if step % 20 == 0:
            active_now = [r for r in robots if not r.arrived]
            if active_now:
                mean_d = np.mean([r.dist_to_goal() for r in active_now])
                dist_snapshots.append((t, float(mean_d)))

    # Close any roundabouts still open at end
    if use_mgr:
        for rid, C in roundabouts.items():
            t_c = rt_created.get(rid, t)
            roundabout_events.append((t_c, t, rt_peak.get(rid, 0)))

    metrics = compute_metrics(robots, t)

    return {
        'metrics':              metrics,
        'qp_infeasible_count':  qp_infeasible_count,
        'qp_total_count':       qp_total_count,
        'deadlock_pair_count':  deadlock_pair_count,
        'steps_in_mgr':         steps_in_mgr,
        'steps_active':         steps_active,
        'roundabout_events':    roundabout_events,
        'dist_snapshots':       dist_snapshots,
        'n_roundabouts_formed': len(roundabout_events),
    }


# -----------------------------------------------------------------------
# Report printer
# -----------------------------------------------------------------------

def print_report(label, N, diag):
    m   = diag['metrics']
    inf = diag['qp_infeasible_count']
    tot = diag['qp_total_count']
    inf_pct = 100 * inf / tot if tot > 0 else 0

    print(f"\n{'='*60}")
    print(f"  {label}  |  N={N}")
    print(f"{'='*60}")
    print(f"  Outcome:   success={m['success_rate']:.2f}  "
          f"arrival={m['arrival_rate']:.2f}  "
          f"n_arrived={m['n_arrived']}/{m['n_total']}")
    print(f"  QP:        {inf}/{tot} infeasible  ({inf_pct:.1f}%)")
    print(f"  Deadlock pairs flagged (cumulative): {diag['deadlock_pair_count']}")

    # MGR-specific
    if diag['n_roundabouts_formed'] is not None:
        n_rt = diag['n_roundabouts_formed']
        events = diag['roundabout_events']
        if events:
            durations = [t2 - t1 for t1, t2, _ in events]
            peaks     = [pk for _, _, pk in events]
            print(f"  Roundabouts formed: {n_rt}  "
                  f"avg_duration={np.mean(durations):.1f}s  "
                  f"max_duration={np.max(durations):.1f}s  "
                  f"avg_peak_members={np.mean(peaks):.1f}")
        else:
            print(f"  Roundabouts formed: 0")

    # Per-robot MGR fraction
    sa = diag['steps_active']
    sm = diag['steps_in_mgr']
    mgr_fracs = [
        sm[i] / sa[i] if sa[i] > 0 else 0.0
        for i in range(len(sa))
    ]
    if any(f > 0 for f in mgr_fracs):
        print(f"  MGR mode fraction per robot (non-zero only):")
        for i, f in enumerate(mgr_fracs):
            if f > 0.01:
                arrived = "ARRIVED" if diag['metrics']['n_arrived'] > i else ""
                print(f"    robot {i:3d}: {f*100:5.1f}% in MGR  {arrived}")
    else:
        print(f"  MGR mode fraction: 0% (no roundabouts triggered)")

    # Distance-to-goal progress
    snaps = diag['dist_snapshots']
    if snaps:
        t0, d0 = snaps[0]
        t_mid, d_mid = snaps[len(snaps)//2]
        t_end, d_end = snaps[-1]
        print(f"  Dist-to-goal (mean active):  "
              f"t={t0:.0f}s → {d0:.2f}m  |  "
              f"t={t_mid:.0f}s → {d_mid:.2f}m  |  "
              f"t={t_end:.0f}s → {d_end:.2f}m")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

if __name__ == '__main__':
    CONFIGS = [
        ('swap', 20),
        ('swap', 40),
        ('swap', 60),
    ]
    INSTANCE_IDX = 0

    for env_type, N in CONFIGS:
        print(f"\n{'#'*60}")
        print(f"# Diagnosing: env={env_type}  N={N}  instance={INSTANCE_IDX}")
        print(f"{'#'*60}")

        env, robots = generate_instance(env_type, N, INSTANCE_IDX)
        diag = run_instrumented(env, robots, method='mgr')
        print_report('MGR', N, diag)

        print()
