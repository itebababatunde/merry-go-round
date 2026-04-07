"""
renderer.py — Animated replay of a single simulation run.

Creates a matplotlib FuncAnimation from the history list returned by
Simulator.get_history() or OrcaSimulator.get_history().

Visual encoding
---------------
  Blue  circle   — robot in GOAL mode
  Red   circle   — robot in MGR mode
  Gray  circle   — arrived robot (stationary)
  Black arrow    — heading direction θ (skipped for ORCA: theta always 0)
  Orange dashed  — active roundabout ring
  Green star     — goal position
  Dark gray      — obstacle (circle or rectangle)
  Black rect     — workspace boundary

Usage
-----
    from src.visualization.renderer import render_animation
    from experiments.instance_generator import generate_instance
    from src.simulation.simulator import Simulator

    env, robots = generate_instance('swap', 20, 0)
    sim = Simulator(env, robots, method='mgr', record_every=5)
    sim.run()
    render_animation(sim.get_history(), env, method='mgr',
                     output_path='results/demo_swap_20_mgr.gif', fps=10)
"""

import pathlib
import sys
import os

import matplotlib
matplotlib.use('Agg')   # non-interactive backend; safe for scripts and multiprocessing
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from experiments.config import ROBOT_RADIUS, WORKSPACE

# ------------------------------------------------------------------
# Colour constants
# ------------------------------------------------------------------
_COLOR_GOAL    = '#3a86ff'   # blue
_COLOR_MGR     = '#e63946'   # red
_COLOR_ARRIVED = '#adb5bd'   # gray
_COLOR_OBS     = '#343a40'   # dark gray
_COLOR_RNDBT   = '#fd7f00'   # orange
_COLOR_GOAL_MK = '#2dc653'   # green (goal star markers)
_COLOR_ARROW   = '#000000'   # black

_ROBOT_DRAW_R  = ROBOT_RADIUS
_ARROW_LEN     = ROBOT_RADIUS * 1.8   # length of heading arrow


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _robot_color(r: dict) -> str:
    if r['arrived']:
        return _COLOR_ARRIVED
    return _COLOR_MGR if r['mode'] == 'MGR' else _COLOR_GOAL


def _draw_static(ax, env, goals: list) -> None:
    """Draw workspace boundary, obstacles, and goal markers (static elements)."""
    # Workspace boundary
    ax.add_patch(mpatches.Rectangle(
        (0, 0), env.size, env.size,
        linewidth=1.0, edgecolor='black', facecolor='none', zorder=1,
    ))

    # Obstacles
    for obs in env.obstacles:
        cls = type(obs).__name__
        if cls == 'CircularObstacle':
            ax.add_patch(mpatches.Circle(
                obs.center, obs.radius,
                facecolor=_COLOR_OBS, edgecolor='none', zorder=2,
            ))
        elif cls == 'RectangularObstacle':
            x = obs.center[0] - obs.half_w
            y = obs.center[1] - obs.half_h
            ax.add_patch(mpatches.Rectangle(
                (x, y), 2 * obs.half_w, 2 * obs.half_h,
                facecolor=_COLOR_OBS, edgecolor='none', zorder=2,
            ))

    # Goal markers (green stars)
    if goals:
        gx = [g[0] for g in goals]
        gy = [g[1] for g in goals]
        ax.scatter(gx, gy, marker='*', s=80, color=_COLOR_GOAL_MK,
                   zorder=3, label='Goal')


# ------------------------------------------------------------------
# Main public function
# ------------------------------------------------------------------

def render_animation(
    history: list,
    env,
    method: str = 'mgr',
    output_path: str | None = None,
    fps: int = 10,
    show: bool = False,
) -> FuncAnimation:
    """
    Build a FuncAnimation from a Simulator / OrcaSimulator history list.

    Parameters
    ----------
    history : list
        Snapshots from sim.get_history(). Each snapshot has keys:
        't', 'robots' (list of dicts), 'roundabouts' (list of dicts).
    env : Environment
        Used to draw obstacles and workspace boundary.
    method : str
        'mgr', 'clf_cbf', or 'orca' — shown in the animation title.
    output_path : str or None
        If given, save the animation.  Extension determines format:
          .gif → pillow writer (no extra dependency)
          .mp4 → ffmpeg writer
    fps : int
        Frames per second for saved file.
    show : bool
        If True, call plt.show() for interactive display.

    Returns
    -------
    FuncAnimation
        Can be displayed in Jupyter via HTML(anim.to_jshtml()).
    """
    if not history:
        raise ValueError("history list is empty — run sim.run() first.")

    # ----------------------------------------------------------------
    # Extract initial state for setup
    # ----------------------------------------------------------------
    snap0  = history[0]
    N      = len(snap0['robots'])
    # Goals: read from robot positions at t=0? No — goals are fixed.
    # We can't reconstruct goals from history alone; pass None if unavailable.
    # The demo entry point passes goals separately; API accepts history only.
    # Goals are drawn from a separate argument if provided.
    goals  = []   # populated below if goal positions are embedded

    # ----------------------------------------------------------------
    # Figure / axes setup
    # ----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(-0.5, env.size + 0.5)
    ax.set_ylim(-0.5, env.size + 0.5)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])

    _draw_static(ax, env, goals)

    # ----------------------------------------------------------------
    # Dynamic artists — robots
    # ----------------------------------------------------------------
    robot_circles = []
    robot_arrows  = []   # FancyArrow patches

    for r in snap0['robots']:
        pos   = r['pos']
        color = _robot_color(r)

        circle = mpatches.Circle(pos, _ROBOT_DRAW_R,
                                 facecolor=color, edgecolor='white',
                                 linewidth=0.5, zorder=5)
        ax.add_patch(circle)
        robot_circles.append(circle)

        # Heading arrow (not shown for ORCA since theta is meaningless)
        theta = r['theta']
        dx    = _ARROW_LEN * np.cos(theta)
        dy    = _ARROW_LEN * np.sin(theta)
        arrow = ax.annotate(
            '', xy=(pos[0] + dx, pos[1] + dy), xytext=(pos[0], pos[1]),
            arrowprops=dict(arrowstyle='->', color=_COLOR_ARROW,
                            lw=0.8, mutation_scale=8),
            zorder=6,
        )
        robot_arrows.append(arrow)

    # ----------------------------------------------------------------
    # Dynamic artists — roundabout rings
    # (dict: roundabout_id → Circle patch)
    # ----------------------------------------------------------------
    roundabout_patches: dict = {}

    # ----------------------------------------------------------------
    # Title
    # ----------------------------------------------------------------
    title = ax.set_title('', fontsize=10)

    # Legend proxy artists
    legend_elements = [
        mpatches.Patch(facecolor=_COLOR_GOAL,    label='GOAL mode'),
        mpatches.Patch(facecolor=_COLOR_MGR,     label='MGR mode'),
        mpatches.Patch(facecolor=_COLOR_ARRIVED, label='Arrived'),
    ]
    if method == 'mgr':
        legend_elements.append(
            mpatches.Patch(facecolor=_COLOR_RNDBT, alpha=0.3, label='Roundabout')
        )
    ax.legend(handles=legend_elements, loc='upper right', fontsize=7,
              framealpha=0.8)

    # ----------------------------------------------------------------
    # Update function
    # ----------------------------------------------------------------
    def update(frame_idx: int):
        snap = history[frame_idx]
        t    = snap['t']
        robots_snap = snap['robots']
        rounds_snap = snap['roundabouts']

        n_arrived = sum(1 for r in robots_snap if r['arrived'])

        # — Update robot circles and arrows —
        for i, r in enumerate(robots_snap):
            pos   = r['pos']
            color = _robot_color(r)
            theta = r['theta']

            robot_circles[i].set_center(pos)
            robot_circles[i].set_facecolor(color)

            # Update arrow: reposition by mutating xyann (tail) and xy (head)
            dx = _ARROW_LEN * np.cos(theta)
            dy = _ARROW_LEN * np.sin(theta)
            robot_arrows[i].xy       = (pos[0] + dx, pos[1] + dy)
            robot_arrows[i].xyann   = (pos[0], pos[1])
            # Hide arrow for arrived robots or ORCA (theta always 0 → not meaningful)
            visible = (not r['arrived']) and (method != 'orca')
            robot_arrows[i].set_visible(visible)

        # — Sync roundabout rings —
        active_ids = {rd['id'] for rd in rounds_snap}

        # Add new roundabouts
        for rd in rounds_snap:
            rid = rd['id']
            if rid not in roundabout_patches:
                patch = mpatches.Circle(
                    rd['center'], rd['radius'],
                    facecolor=_COLOR_RNDBT, alpha=0.15,
                    edgecolor=_COLOR_RNDBT, linestyle='--', linewidth=1.5,
                    zorder=4,
                )
                ax.add_patch(patch)
                roundabout_patches[rid] = patch
            else:
                # Update center/radius in case it changed
                roundabout_patches[rid].set_center(rd['center'])
                roundabout_patches[rid].set_radius(rd['radius'])
                roundabout_patches[rid].set_visible(True)

        # Hide departed roundabouts
        for rid, patch in roundabout_patches.items():
            if rid not in active_ids:
                patch.set_visible(False)

        # — Update title —
        title.set_text(
            f"{method.upper()}    t = {t:.2f} s    "
            f"arrived = {n_arrived} / {N}"
        )

        return (robot_circles + robot_arrows +
                list(roundabout_patches.values()) + [title])

    # ----------------------------------------------------------------
    # Build FuncAnimation
    # ----------------------------------------------------------------
    anim = FuncAnimation(
        fig,
        update,
        frames=len(history),
        interval=max(1, int(1000 / fps)),
        blit=False,   # blit=True has macOS issues with some backends
        repeat=False,
    )

    # ----------------------------------------------------------------
    # Save / show
    # ----------------------------------------------------------------
    if output_path:
        _save_animation(anim, output_path, fps)

    if show:
        matplotlib.use('TkAgg')   # switch to interactive backend
        plt.show()

    return anim


def _save_animation(anim: FuncAnimation, output_path: str, fps: int) -> None:
    """Save animation, falling back from mp4 to gif if ffmpeg is unavailable."""
    out = pathlib.Path(output_path)
    out.parent.mkdir(exist_ok=True)
    ext = out.suffix.lower()

    if ext == '.mp4':
        try:
            anim.save(str(out), writer='ffmpeg', fps=fps,
                      extra_args=['-vcodec', 'libx264'])
            print(f"Saved: {out}")
            return
        except Exception as e:
            gif_path = out.with_suffix('.gif')
            print(f"ffmpeg unavailable ({e}); falling back to GIF → {gif_path}")
            out = gif_path

    # gif via pillow
    anim.save(str(out), writer='pillow', fps=fps)
    print(f"Saved: {out}")


# ------------------------------------------------------------------
# Entry point — demo animations
# ------------------------------------------------------------------

if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from experiments.instance_generator import generate_instance
    from src.simulation.simulator import Simulator
    from src.baselines.orca_baseline import OrcaSimulator

    pathlib.Path('results').mkdir(exist_ok=True)

    DEMO_ENV  = 'swap'
    DEMO_N    = 20
    DEMO_IDX  = 0

    demos = [
        ('mgr',     Simulator),
        ('clf_cbf', Simulator),
        ('orca',    None),
    ]

    for method, SimClass in demos:
        print(f"Running {method.upper()} on {DEMO_ENV} N={DEMO_N} …")
        env, robots = generate_instance(DEMO_ENV, DEMO_N, DEMO_IDX)

        if method == 'orca':
            sim = OrcaSimulator(env, robots, record_every=5)
        else:
            sim = SimClass(env, robots, method=method, record_every=5)

        metrics = sim.run()
        history = sim.get_history()
        print(f"  success={metrics['success_rate']:.2f}  "
              f"arrival={metrics['arrival_rate']:.2f}  "
              f"frames={len(history)}")

        out = f"results/demo_{DEMO_ENV}_{DEMO_N}_{method}.gif"
        render_animation(history, env, method=method, output_path=out, fps=10)

    print("\nAll demo animations saved to results/")
