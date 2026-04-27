"""
make_gifs.py — Generate one GIF per (env, method) combination.

Produces 12 GIFs (4 envs × 3 methods), each from instance_idx=0, N=20.
Output: results/demo_{env}_20_{method}.gif

Usage
-----
    python experiments/make_gifs.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from experiments.instance_generator import generate_instance
from experiments.config import N_INSTANCES
from src.simulation.simulator import Simulator
from src.baselines.orca_baseline import OrcaSimulator
from src.visualization.renderer import render_animation
import pathlib

ENVS    = ['free', 'circ15', 'rect15', 'swap']
METHODS = ['mgr', 'clf_cbf', 'orca']
N       = 20
IDX     = 0          # use instance 0 for all GIFs

out_dir = pathlib.Path('results')
out_dir.mkdir(exist_ok=True)

for env in ENVS:
    for method in METHODS:
        out_path = out_dir / f'demo_{env}_{N}_{method}.gif'
        print(f'Rendering {env}/{method} → {out_path} ...', end=' ', flush=True)

        env_obj, robots = generate_instance(env, N, IDX)
        goals = [r.goal.tolist() for r in robots]

        if method == 'orca':
            sim = OrcaSimulator(env_obj, robots, record_every=5)
        else:
            sim = Simulator(env_obj, robots, method=method, record_every=5)

        sim.run()
        render_animation(sim.get_history(), env_obj, method=method,
                         goals=goals, output_path=str(out_path), fps=10)
        print('done')

print('\nAll 12 GIFs written to results/')
