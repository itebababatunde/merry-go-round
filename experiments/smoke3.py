import sys, numpy as np
sys.path.insert(0, '.')
from experiments.instance_generator import generate_instance
from src.simulation.simulator import Simulator

lines = []
for env in ['free', 'swap', 'circ15', 'rect15']:
    successes = 0; arrivals = []; min_dists = []
    for seed in range(3):
        env_obj, robots = generate_instance(env, 20, seed)
        sim = Simulator(env_obj, robots, method='mgr', record_every=5)
        m = sim.run()
        if m['success_rate'] == 1.0: successes += 1
        arrivals.append(m['arrival_rate'])
        min_dists.append(m['min_dist'])
    line = f"{env:8s}: {successes}/3  arr={[f'{a:.0%}' for a in arrivals]}  min_d={[f'{d:.3f}' for d in min_dists]}"
    lines.append(line)
    print(line, flush=True)

with open('/tmp/smoke3.out', 'w') as f:
    f.write('\n'.join(lines) + '\n')
