"""
make_element_pngs.py — Export individual PNG files for each visual element used
in the simulation GIFs, plus one topology overview per environment type.

Output (all in results/elements/):
    goal.png          — green star goal marker
    robot_goal.png    — blue car (GOAL mode)
    robot_mgr.png     — red car (MGR mode)
    robot_arrived.png — gray car (arrived)
    roundabout.png    — orange dashed MGR circle with one red car on it
    topology_free.png     — free workspace (no obstacles)
    topology_circ15.png   — circular-obstacle workspace
    topology_rect15.png   — rectangular-obstacle workspace
    topology_swap.png     — swap scenario (starts + goals shown)
"""

import math
import sys
import os
import pathlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from experiments.config import ROBOT_RADIUS, WORKSPACE
from experiments.instance_generator import generate_instance

# ── colour palette (matches renderer.py) ────────────────────────────────────
_COLOR_GOAL    = '#3a86ff'   # blue  — GOAL mode
_COLOR_MGR     = '#e63946'   # red   — MGR mode
_COLOR_ARRIVED = '#adb5bd'   # gray  — arrived
_COLOR_OBS     = '#343a40'   # dark gray — obstacle
_COLOR_RNDBT   = '#fd7f00'   # orange — roundabout
_COLOR_GOAL_MK = '#2dc653'   # green — goal star

R = ROBOT_RADIUS             # 0.2 m

OUT_DIR = pathlib.Path('results/elements')
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── shared car polygon helper ────────────────────────────────────────────────

def _car_xy(pos, theta, r=R):
    local = np.array([
        [ 1.0,  0.0],
        [ 0.4,  0.7],
        [-0.8,  0.6],
        [-0.8, -0.6],
        [ 0.4, -0.7],
    ]) * r
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s], [s, c]])
    return (local @ rot.T) + np.asarray(pos)


def _isolated_fig(pad=0.6):
    """Return (fig, ax) sized tightly around a unit cell."""
    fig, ax = plt.subplots(figsize=(2, 2))
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_xlim(-pad, pad)
    ax.set_ylim(-pad, pad)
    return fig, ax


def _save(fig, name):
    path = OUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches='tight',
                facecolor='white', transparent=False)
    plt.close(fig)
    print(f"  saved {path}")


# ────────────────────────────────────────────────────────────────────────────
# 1. Goal star
# ────────────────────────────────────────────────────────────────────────────

def make_goal():
    fig, ax = _isolated_fig(pad=0.5)
    ax.scatter([0], [0], marker='*', s=1200, color=_COLOR_GOAL_MK, zorder=3)
    ax.set_title('Goal', fontsize=11, pad=4)
    _save(fig, 'goal.png')


# ────────────────────────────────────────────────────────────────────────────
# 2 / 3 / 4. Robot variants
# ────────────────────────────────────────────────────────────────────────────

def make_robot(color, label, filename):
    fig, ax = _isolated_fig(pad=0.45)
    xy = _car_xy([0, 0], 0.0)   # heading right
    poly = mpatches.Polygon(xy, closed=True,
                            facecolor=color, edgecolor='white',
                            linewidth=1.0, zorder=5)
    ax.add_patch(poly)
    ax.set_title(label, fontsize=11, pad=4)
    _save(fig, filename)


# ────────────────────────────────────────────────────────────────────────────
# 5. Roundabout — circle with one car on it
# ────────────────────────────────────────────────────────────────────────────

def make_roundabout():
    rad = 0.8   # roundabout radius (scaled for the icon)
    fig, ax = _isolated_fig(pad=rad + 0.35)

    # Orange dashed ring
    ring = mpatches.Circle((0, 0), rad,
                            facecolor=_COLOR_RNDBT, alpha=0.15,
                            edgecolor=_COLOR_RNDBT, linestyle='--',
                            linewidth=2.0, zorder=2)
    ax.add_patch(ring)

    # One red car on the ring at 45°, heading tangentially (CCW = +90° from radial)
    angle = math.pi / 4
    pos = np.array([rad * math.cos(angle), rad * math.sin(angle)])
    theta = angle + math.pi / 2   # tangent direction (CCW orbit)
    xy = _car_xy(pos, theta, r=R * 1.3)
    poly = mpatches.Polygon(xy, closed=True,
                            facecolor=_COLOR_MGR, edgecolor='white',
                            linewidth=0.8, zorder=5)
    ax.add_patch(poly)

    ax.set_title('Roundabout (MGR)', fontsize=11, pad=4)
    _save(fig, 'roundabout.png')


# ────────────────────────────────────────────────────────────────────────────
# 6–9. Environment topologies (N=6 robots, seed 0 — small for clarity)
# ────────────────────────────────────────────────────────────────────────────

def _draw_env_topology(ax, env, robots, show_robots=True):
    """Draw workspace + obstacles + robot starts + goals."""
    S = env.size

    # Workspace border
    ax.add_patch(mpatches.Rectangle(
        (0, 0), S, S,
        linewidth=1.2, edgecolor='black', facecolor='#f8f9fa', zorder=0,
    ))

    # Obstacles
    for obs in env.obstacles:
        cls = type(obs).__name__
        if cls == 'CircularObstacle':
            ax.add_patch(mpatches.Circle(
                obs.center, obs.radius,
                facecolor=_COLOR_OBS, edgecolor='none', zorder=2,
            ))
        else:
            x = obs.center[0] - obs.half_w
            y = obs.center[1] - obs.half_h
            ax.add_patch(mpatches.Rectangle(
                (x, y), 2 * obs.half_w, 2 * obs.half_h,
                facecolor=_COLOR_OBS, edgecolor='none', zorder=2,
            ))

    if show_robots and robots:
        # Goal stars
        gx = [r.goal[0] for r in robots]
        gy = [r.goal[1] for r in robots]
        ax.scatter(gx, gy, marker='*', s=60, color=_COLOR_GOAL_MK, zorder=4)

        # Robot cars at start positions
        for robot in robots:
            xy = _car_xy(robot.pos, robot.theta, r=R * 1.1)
            poly = mpatches.Polygon(xy, closed=True,
                                    facecolor=_COLOR_GOAL, edgecolor='white',
                                    linewidth=0.5, zorder=5)
            ax.add_patch(poly)

    ax.set_xlim(-0.5, S + 0.5)
    ax.set_ylim(-0.5, S + 0.5)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])


def make_topology(env_type, title):
    N = 10   # small N for legibility
    env, robots = generate_instance(env_type, N, instance_idx=0)

    fig, ax = plt.subplots(figsize=(4, 4))
    _draw_env_topology(ax, env, robots, show_robots=True)
    ax.set_title(title, fontsize=12, pad=6)
    fig.tight_layout()
    _save(fig, f'topology_{env_type}.png')


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("Generating element PNGs …\n")

    print("── Individual elements ──")
    make_goal()
    make_robot(_COLOR_GOAL,    'Robot (GOAL mode)',    'robot_goal.png')
    make_robot(_COLOR_MGR,     'Robot (MGR mode)',     'robot_mgr.png')
    make_robot(_COLOR_ARRIVED, 'Robot (Arrived)',      'robot_arrived.png')
    make_roundabout()

    print("\n── Environment topologies (N=10, seed 0) ──")
    make_topology('free',   'Free workspace')
    make_topology('circ15', 'Circular obstacles (15 %)')
    make_topology('rect15', 'Rectangular obstacles (15 %)')
    make_topology('swap',   'Swap scenario')

    print(f"\nAll PNGs written to {OUT_DIR}/")
