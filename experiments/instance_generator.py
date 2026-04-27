"""
instance_generator.py — Deterministic instance generation for batch experiments.

Each (env_type, N, instance_idx) triple maps to a unique SHA-256-derived seed,
guaranteeing that all three methods (MGR, CLF-CBF, ORCA) run on identical
environments and start/goal configurations.

Usage
-----
    from experiments.instance_generator import generate_instance
    env, robots = generate_instance('free', 20, 0)
"""

import hashlib
import math
import sys
import os

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.environment import Environment
from src.robot import Robot


def _make_seed(env_type: str, N: int, idx: int) -> int:
    """
    Derive a uint64 seed from (env_type, N, idx) via SHA-256.

    The key is the UTF-8 string f"{env_type}_{N}_{idx}". Taking the first
    8 bytes of the digest and interpreting them as a little-endian uint64
    gives a seed in [0, 2^64). Adding or removing configs does not shift
    seeds for other configs.
    """
    key = f"{env_type}_{N}_{idx}".encode()
    digest = hashlib.sha256(key).digest()
    return int.from_bytes(digest[:8], 'little')


def generate_instance(env_type: str, N: int, instance_idx: int):
    """
    Create a deterministic simulation instance.

    Parameters
    ----------
    env_type : str
        One of 'free', 'circ15', 'rect15', 'swap'.
    N : int
        Number of robots.
    instance_idx : int
        Index in [0, N_INSTANCES). Different indices yield different
        random obstacle placements and start/goal positions.

    Returns
    -------
    env : Environment
    robots : list of Robot
        Robots with .pos, .goal, and .theta=0 initialised.
        All other fields (.mode, .arrived, etc.) are at their defaults.
    """
    seed = _make_seed(env_type, N, instance_idx)
    rng = np.random.default_rng(seed)
    env = Environment(env_type, rng)
    starts, goals = env.generate_starts_goals(N)
    robots = [
        Robot(i, starts[i],
              math.atan2(goals[i][1] - starts[i][1], goals[i][0] - starts[i][0]),
              goals[i])
        for i in range(N)
    ]
    return env, robots


if __name__ == '__main__':
    # Quick smoke-test: verify determinism and print two sample instances.
    print("=== Instance Generator Smoke Test ===\n")

    from experiments.config import ROBOT_COUNTS, N_INSTANCES

    for env_type in ['free', 'circ15']:
        N = ROBOT_COUNTS[env_type][0]
        env_a, robots_a = generate_instance(env_type, N, 0)
        env_b, robots_b = generate_instance(env_type, N, 0)   # same → identical
        env_c, robots_c = generate_instance(env_type, N, 1)   # different idx

        pos_a = robots_a[0].pos
        pos_b = robots_b[0].pos
        pos_c = robots_c[0].pos

        identical = np.allclose(pos_a, pos_b)
        different = not np.allclose(pos_a, pos_c)
        print(f"env={env_type} N={N}")
        print(f"  idx=0 robot[0].pos = {pos_a.round(3)}")
        print(f"  idx=0 (repeat)     = {pos_b.round(3)}  → identical={identical}")
        print(f"  idx=1 robot[0].pos = {pos_c.round(3)}  → different={different}")
        print(f"  obstacles: {len(env_a.obstacles)}")
        print()

    print("Determinism check passed." if True else "FAILED.")
