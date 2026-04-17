"""
environment.py — Workspace, obstacle generation, and start/goal sampling.

Supports four environment types from the paper (Section V-A):
    free    — 16 m × 16 m, no obstacles
    circ15  — 15 % obstacle coverage, circular obstacles
    rect15  — 15 % obstacle coverage, rectangular obstacles
    swap    — no obstacles; robots start on opposite sides (forced head-on)

Obstacle coverage target:
    WORKSPACE² × 0.15 = 256 × 0.15 = 38.4 m²

Signed-distance functions (sdf) are provided for each obstacle type and are
used by:
    • CBF obstacle constraint construction (Phase 2)
    • MGR roundabout validity checks (Phase 3)
    • Escape sector clearance checks (Phase 3)
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from experiments.config import (
    WORKSPACE, ROBOT_RADIUS, R_SAFE, D_SAFE,
    EPSILON_GOAL,
)

# Target obstacle coverage fraction
COVERAGE_TARGET = 0.15
TOTAL_AREA      = WORKSPACE ** 2          # 256 m²
TARGET_OBS_AREA = TOTAL_AREA * COVERAGE_TARGET  # 38.4 m²


# ---------------------------------------------------------------------------
# Obstacle types
# ---------------------------------------------------------------------------

@dataclass
class CircularObstacle:
    """A filled circle obstacle."""
    center: np.ndarray   # [x, y]
    radius: float

    def area(self) -> float:
        return math.pi * self.radius ** 2

    def sdf(self, point: np.ndarray) -> float:
        """
        Signed distance from `point` to the obstacle surface.
        Negative  → inside obstacle (collision).
        Positive  → outside (free space); value = distance to nearest surface.
        """
        return float(np.linalg.norm(point - self.center)) - self.radius

    def cbf_h(self, robot_pos: np.ndarray) -> float:
        """
        CBF barrier value for a robot vs. this obstacle.
        h = (‖p − c‖ − radius)² − D_SAFE²
        We use a simpler linear form: h = ‖p − c‖ − radius − D_SAFE
        (positive when the robot is at least D_SAFE away from the surface).
        """
        return self.sdf(robot_pos) - D_SAFE

    def contains(self, point: np.ndarray, margin: float = 0.0) -> bool:
        """True if point is inside obstacle + margin."""
        return self.sdf(point) < margin

    def to_polygon_vertices(self, n: int = 16) -> List[Tuple[float, float]]:
        """Approximate as n-gon (used for rvo2 obstacle input)."""
        angles = np.linspace(0, 2 * math.pi, n, endpoint=False)
        return [(self.center[0] + self.radius * math.cos(a),
                 self.center[1] + self.radius * math.sin(a))
                for a in angles]


@dataclass
class RectangularObstacle:
    """An axis-aligned rectangular obstacle."""
    center: np.ndarray   # [x, y]  (geometric centre)
    half_w: float        # half-width  (x-direction)
    half_h: float        # half-height (y-direction)

    def area(self) -> float:
        return 4 * self.half_w * self.half_h

    def sdf(self, point: np.ndarray) -> float:
        """
        Exact SDF for an axis-aligned rectangle.
        Returns distance to nearest surface (negative if inside).
        """
        dx = abs(point[0] - self.center[0]) - self.half_w
        dy = abs(point[1] - self.center[1]) - self.half_h
        outside_dist = math.sqrt(max(dx, 0.0) ** 2 + max(dy, 0.0) ** 2)
        inside_dist  = min(max(dx, dy), 0.0)
        return outside_dist + inside_dist

    def cbf_h(self, robot_pos: np.ndarray) -> float:
        return self.sdf(robot_pos) - D_SAFE

    def contains(self, point: np.ndarray, margin: float = 0.0) -> bool:
        return self.sdf(point) < margin

    def to_polygon_vertices(self) -> List[Tuple[float, float]]:
        """4 corners in CCW order (used for rvo2 obstacle input)."""
        cx, cy = self.center
        hw, hh = self.half_w, self.half_h
        return [
            (cx - hw, cy - hh),
            (cx + hw, cy - hh),
            (cx + hw, cy + hh),
            (cx - hw, cy + hh),
        ]


# Type alias
Obstacle = (CircularObstacle, RectangularObstacle)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class Environment:
    """
    Simulation workspace with obstacles, start positions, and goal positions.

    Parameters
    ----------
    env_type : str
        One of 'free', 'circ15', 'rect15', 'swap'.
    rng : np.random.Generator
        Seeded random number generator — all randomness flows through this.
    """

    def __init__(self, env_type: str, rng: np.random.Generator):
        self.env_type  = env_type.lower()
        self.rng       = rng
        self.size      = WORKSPACE
        self.obstacles: List = []

        if self.env_type == "circ15":
            self._generate_circular_obstacles()
        elif self.env_type == "rect15":
            self._generate_rectangular_obstacles()
        elif self.env_type in ("free", "swap"):
            pass   # no obstacles
        else:
            raise ValueError(f"Unknown environment type: {env_type!r}. "
                             f"Choose from 'free', 'circ15', 'rect15', 'swap'.")

    # ------------------------------------------------------------------
    # Obstacle generation
    # ------------------------------------------------------------------

    def _generate_circular_obstacles(self) -> None:
        """
        Rejection-sample circular obstacles until coverage ≥ 15 %.
        Obstacle radii are drawn uniformly from [0.3, 0.7] m.
        Obstacles must not overlap each other and must stay ≥ D_SAFE from
        workspace boundaries.
        """
        covered = 0.0
        max_attempts = 50_000
        attempt = 0

        while covered < TARGET_OBS_AREA and attempt < max_attempts:
            attempt += 1
            r  = float(self.rng.uniform(0.3, 0.7))
            margin = r + D_SAFE
            x  = float(self.rng.uniform(margin, self.size - margin))
            y  = float(self.rng.uniform(margin, self.size - margin))
            c  = np.array([x, y])

            # Check against existing obstacles (no overlap + clearance)
            if any(np.linalg.norm(c - obs.center) < r + obs.radius + D_SAFE
                   for obs in self.obstacles):
                continue

            self.obstacles.append(CircularObstacle(c, r))
            covered += math.pi * r ** 2

    def _generate_rectangular_obstacles(self) -> None:
        """
        Rejection-sample rectangular obstacles until coverage ≥ 15 %.
        Half-widths and half-heights drawn from [0.2, 0.5] m each.
        """
        covered = 0.0
        max_attempts = 50_000
        attempt = 0

        while covered < TARGET_OBS_AREA and attempt < max_attempts:
            attempt += 1
            hw = float(self.rng.uniform(0.2, 0.5))
            hh = float(self.rng.uniform(0.2, 0.5))
            margin = max(hw, hh) + D_SAFE
            x  = float(self.rng.uniform(margin, self.size - margin))
            y  = float(self.rng.uniform(margin, self.size - margin))
            c  = np.array([x, y])
            cand = RectangularObstacle(c, hw, hh)

            # Check against existing obstacles
            if any(_rect_overlap_with_margin(cand, obs, D_SAFE)
                   for obs in self.obstacles):
                continue

            self.obstacles.append(cand)
            covered += 4 * hw * hh

    # ------------------------------------------------------------------
    # Start / goal sampling
    # ------------------------------------------------------------------

    def generate_starts_goals(
        self, n_robots: int
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Sample N start positions and N goal positions.

        Rules (from paper Section III-E):
            • Each position must be ≥ 2·rsafe from all obstacles.
            • Each position must be ≥ 2·rsafe from workspace boundaries.
            • Any two start positions separated by ≥ D_SAFE.
            • Any two goal positions separated by ≥ D_SAFE.
            • Each goal separated from every obstacle by ≥ 2·rsafe (so robots
              can converge without violating barrier constraints).

        For the 'swap' environment:
            • Starts are sampled in the left half  (x ∈ [margin, size/2]).
            • Goals  are the mirror positions in the right half.
            This creates guaranteed head-on conflicts as in the paper.
        """
        if self.env_type == "swap":
            return self._generate_swap_starts_goals(n_robots)

        min_sep = D_SAFE          # minimum separation between any two robots/goals
        boundary = 2 * R_SAFE     # keep positions this far from walls

        starts = self._sample_positions(n_robots, min_sep, boundary)
        goals  = self._sample_positions(n_robots, min_sep, boundary)
        return starts, goals

    def _sample_positions(
        self,
        n: int,
        min_sep: float,
        boundary: float,
        x_range: Optional[Tuple[float, float]] = None,
        y_range: Optional[Tuple[float, float]] = None,
    ) -> List[np.ndarray]:
        """
        Rejection-sample `n` positions that are:
            • Inside the workspace (respecting boundary margin).
            • ≥ min_sep from every already-sampled position.
            • ≥ D_SAFE from every obstacle surface.
        """
        x_lo = boundary if x_range is None else x_range[0]
        x_hi = (self.size - boundary) if x_range is None else x_range[1]
        y_lo = boundary if y_range is None else y_range[0]
        y_hi = (self.size - boundary) if y_range is None else y_range[1]

        positions = []
        max_attempts = 100_000
        attempt = 0

        while len(positions) < n and attempt < max_attempts:
            attempt += 1
            p = np.array([
                float(self.rng.uniform(x_lo, x_hi)),
                float(self.rng.uniform(y_lo, y_hi)),
            ])

            # Check obstacle clearance
            if any(obs.sdf(p) < D_SAFE for obs in self.obstacles):
                continue

            # Check separation from already-placed positions
            if any(np.linalg.norm(p - q) < min_sep for q in positions):
                continue

            positions.append(p)

        if len(positions) < n:
            raise RuntimeError(
                f"Could only place {len(positions)}/{n} positions "
                f"in '{self.env_type}' environment after {max_attempts} attempts. "
                f"Try reducing N or obstacle density."
            )
        return positions

    def _generate_swap_starts_goals(
        self, n_robots: int
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Swap scenario: N/2 robots start on the left heading right; N/2 start
        on the right heading left. Each pair shares the same y-coordinate so
        robots travel on directly opposing paths — the canonical head-on
        deadlock test from paper Section V-A.

        Placement is grid-based (equally spaced in y) so packing always
        succeeds regardless of N. The RNG shuffles which y-slots are assigned
        to which robot index, preserving per-instance variety.
        """
        boundary = 2 * R_SAFE       # ≈ 0.44 m
        x_left  = boundary
        x_right = self.size - boundary

        # How many y-slots do we need?  ceil(N/2) pairs.
        n_pairs = math.ceil(n_robots / 2)
        y_lo = boundary
        y_hi = self.size - boundary
        # Space pairs evenly; y-gap = (y_hi - y_lo) / n_pairs ≥ D_SAFE for
        # N ≤ 2 * floor((y_hi-y_lo)/D_SAFE) ≈ 2 * 34 = 68 — well above our max.
        y_positions = [
            y_lo + (i + 0.5) * (y_hi - y_lo) / n_pairs
            for i in range(n_pairs)
        ]

        # Shuffle y-slot assignments so each instance is distinct
        slot_order = self.rng.permutation(n_pairs)

        starts: List[np.ndarray] = []
        goals:  List[np.ndarray] = []

        for k in range(n_robots):
            y = float(y_positions[slot_order[k % n_pairs]])
            if k < n_robots // 2:
                # First half: left → right
                starts.append(np.array([x_left,  y]))
                goals.append( np.array([x_right, y]))
            else:
                # Second half: right → left
                starts.append(np.array([x_right, y]))
                goals.append( np.array([x_left,  y]))

        return starts, goals

    # ------------------------------------------------------------------
    # Utility / queries
    # ------------------------------------------------------------------

    def is_in_workspace(self, point: np.ndarray, margin: float = 0.0) -> bool:
        """True if point is inside the workspace (with optional margin)."""
        lo, hi = margin, self.size - margin
        return bool(lo <= point[0] <= hi and lo <= point[1] <= hi)

    def is_collision_free(self, pos: np.ndarray, radius: float = ROBOT_RADIUS) -> bool:
        """
        True if a circle of `radius` at `pos` does not intersect any obstacle
        and stays within the workspace.
        """
        if not self.is_in_workspace(pos, margin=radius):
            return False
        return all(obs.sdf(pos) >= radius for obs in self.obstacles)

    def get_obstacle_coverage(self) -> float:
        """Return the total obstacle area as a fraction of workspace area."""
        total = sum(
            (math.pi * obs.radius ** 2 if isinstance(obs, CircularObstacle)
             else 4 * obs.half_w * obs.half_h)
            for obs in self.obstacles
        )
        return total / TOTAL_AREA

    def __repr__(self) -> str:
        cov = self.get_obstacle_coverage()
        return (f"Environment(type={self.env_type!r}, "
                f"size={self.size}×{self.size}, "
                f"n_obstacles={len(self.obstacles)}, "
                f"coverage={cov:.1%})")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _rect_overlap_with_margin(
    a: RectangularObstacle,
    b,
    margin: float,
) -> bool:
    """
    Conservative overlap check between a rectangle and any obstacle type.
    Returns True if they are within `margin` of each other.
    """
    if isinstance(b, RectangularObstacle):
        # AABB check with margin
        sep_x = abs(a.center[0] - b.center[0]) - (a.half_w + b.half_w) - margin
        sep_y = abs(a.center[1] - b.center[1]) - (a.half_h + b.half_h) - margin
        return sep_x < 0 and sep_y < 0
    elif isinstance(b, CircularObstacle):
        # Distance from circle center to nearest point of rectangle
        return b.sdf(a.center) < a.half_w + a.half_h + b.radius + margin
    return False
