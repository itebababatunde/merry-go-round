"""
clf_cbf_only.py — CLF-CBF baseline (no MGR deadlock prevention).

This is a thin convenience wrapper around Simulator(method='clf_cbf').
All robots stay in GOAL mode throughout; the CLF-CBF QP provides safety
but no deadlock prevention mechanism is active.

Used by the Phase 6 batch runner to import all three methods uniformly:

    from src.baselines.clf_cbf_only import run_clf_cbf
    from src.baselines.orca_baseline import OrcaSimulator
    from src.simulation.simulator   import Simulator   # MGR
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.simulation.simulator import Simulator


def run_clf_cbf(env, robots: list, record_every: int = 5) -> tuple:
    """
    Run the CLF-CBF baseline on a pre-initialised scenario.

    Parameters
    ----------
    env : Environment
        Workspace with obstacles and env_type.
    robots : list of Robot
        Pre-initialised robots (positions, goals, theta already set).
    record_every : int
        Snapshot interval in timesteps (default 5 = 0.25 s).

    Returns
    -------
    (metrics_dict, history_list)
        metrics_dict — same format as Simulator.run()
        history_list — state snapshots for visualization
    """
    sim = Simulator(env, robots, method='clf_cbf', record_every=record_every)
    metrics = sim.run()
    return metrics, sim.get_history()
