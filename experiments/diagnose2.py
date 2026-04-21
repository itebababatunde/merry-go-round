import sys, numpy as np
sys.path.insert(0, '.')
from experiments.instance_generator import generate_instance
from src.simulation.simulator import Simulator
from experiments.config import D_SAFE, DELTA_COMM

lines = []
for env, seed in [('circ15', 2), ('rect15', 1), ('rect15', 0), ('rect15', 2)]:
    env_obj, robots = generate_instance(env, 20, seed)
    sim = Simulator(env_obj, robots, method='mgr', record_every=5)
    m = sim.run()
    stuck = [r for r in robots if not r.arrived]
    lines.append(f"\n{env}/seed{seed}: {m['n_arrived']}/{m['n_total']} arrived  min_dist={m['min_dist']:.3f}")
    for r in stuck:
        nbs = [n for n in robots if not n.arrived and n.id != r.id and np.linalg.norm(r.pos - n.pos) <= DELTA_COMM]
        near_obs = min((obs.sdf(r.pos) for obs in env_obj.obstacles), default=99)
        close_nbs = [(round(float(np.linalg.norm(r.pos-n.pos)),3), n.id, n.mode.name)
                     for n in nbs if np.linalg.norm(r.pos-n.pos) < D_SAFE*2]
        lines.append(f"  r{r.id} {r.mode.name}: obs={near_obs:.3f}  mgrsteps={r.mgr_step_count}  cooldown={r.escape_cooldown}  nb_close={close_nbs}")

out = "\n".join(lines)
print(out)
with open("experiments/diagnose2.out", "w") as f:
    f.write(out + "\n")
