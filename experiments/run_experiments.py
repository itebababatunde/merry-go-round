"""
run_experiments.py — Parallel batch runner for all 1,080 paper experiments.

Produces results/raw_results.csv with one row per (env, N, instance, method) run.
Uses multiprocessing.Pool so all CPU cores are utilised.

Usage
-----
    cd /path/to/merry-go-round
    python experiments/run_experiments.py

Runtime estimate
----------------
Depends heavily on hardware. Each run takes up to T_MAX=120 s in the worst case
(all robots fail to arrive). Typical runs are much faster. With 8 cores:
  ~1,080 runs / 7 workers ≈ 155 runs per worker
  Expect 30–90 minutes depending on N values and success rates.

The CSV is flushed after every completed row so partial results survive
if the process is interrupted. Re-running regenerates all rows deterministically.
"""

import csv
import multiprocessing
import pathlib
import sys
import os

import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from experiments.config import ROBOT_COUNTS, N_INSTANCES
from experiments.instance_generator import generate_instance

# ------------------------------------------------------------------
# CSV field names (order matches paper tables)
# ------------------------------------------------------------------
FIELDS = [
    'env', 'N', 'instance_idx', 'method',
    'success_rate', 'arrival_rate',
    'makespan', 'mean_time',
    'n_arrived', 'n_total', 't_elapsed', 'min_dist',
]

METHODS = ['mgr', 'clf_cbf', 'orca']

# record_every=9999 skips snapshot recording in batch mode — no history needed.
_RECORD_EVERY = 9999


# ------------------------------------------------------------------
# Worker function — must be top-level for multiprocessing pickle
# ------------------------------------------------------------------

def run_one(args):
    """
    Execute a single (env, N, instance_idx, method) simulation run.

    Parameters
    ----------
    args : tuple
        (env_type: str, N: int, instance_idx: int, method: str)

    Returns
    -------
    dict — one CSV row. 'makespan' and 'mean_time' are None when not all
           robots arrived; csv.DictWriter writes them as empty strings.
    """
    env_type, N, instance_idx, method = args

    # Generate a fresh instance — no shared state between workers.
    env, robots = generate_instance(env_type, N, instance_idx)

    if method == 'orca':
        from src.baselines.orca_baseline import OrcaSimulator
        metrics = OrcaSimulator(env, robots, record_every=_RECORD_EVERY).run()
    else:
        from src.simulation.simulator import Simulator
        metrics = Simulator(env, robots, method=method,
                            record_every=_RECORD_EVERY).run()

    return {
        'env':          env_type,
        'N':            N,
        'instance_idx': instance_idx,
        'method':       method,
        **metrics,
    }


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

if __name__ == '__main__':
    # Build the full task list: 18 (env, N) configs × 20 instances × 3 methods
    tasks = [
        (env_type, N, idx, method)
        for env_type, ns in ROBOT_COUNTS.items()
        for N in ns
        for idx in range(N_INSTANCES)
        for method in METHODS
    ]
    total = len(tasks)  # should be 1,080

    out_path = pathlib.Path('results/raw_results.csv')
    out_path.parent.mkdir(exist_ok=True)

    n_workers = max(1, multiprocessing.cpu_count() - 1)
    print(f"Starting {total} runs on {n_workers} workers → {out_path}")
    print(f"Environments: {list(ROBOT_COUNTS.keys())}")
    print(f"Methods: {METHODS}")
    print()

    completed = 0
    with (
        multiprocessing.Pool(n_workers) as pool,
        open(out_path, 'w', newline='') as f,
    ):
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction='ignore')
        writer.writeheader()

        for row in tqdm.tqdm(
            pool.imap_unordered(run_one, tasks),
            total=total,
            desc='Running experiments',
            unit='run',
        ):
            writer.writerow(row)
            f.flush()
            completed += 1

    print(f"\nDone. {completed}/{total} runs written to {out_path}")
