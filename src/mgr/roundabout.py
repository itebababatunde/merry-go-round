"""
roundabout.py — Roundabout dataclass.

A roundabout C is a circular reference path for robots facing potential deadlock.
Each robot maintains a distance of C.r from the center C.c and travels
counterclockwise until the escape condition is met.

Attributes match the paper notation exactly:
    C.c   → center
    C.r   → radius
    C.n   → n_members (property)
"""

from dataclasses import dataclass, field
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from experiments.config import K_INCREMENT


@dataclass
class Roundabout:
    """
    Circular reference path for deadlock prevention.

    Parameters
    ----------
    id : int
        Unique roundabout identifier.
    center : np.ndarray, shape (2,)
        Roundabout center position C.c in metres.
    radius : float
        Target orbit radius C.r in metres.
    members : list of int
        IDs of robots currently orbiting this roundabout (C.n = len(members)).
    """
    id: int
    center: np.ndarray
    radius: float
    members: list = field(default_factory=list)

    @property
    def n_members(self) -> int:
        """Number of robots currently in this roundabout (C.n)."""
        return len(self.members)

    def effective_clearance(self) -> float:
        """
        Minimum distance from C.c to any obstacle required for validity.

        Paper ISMGRVALID condition: dC ≥ C.r + k·C.n
        where k = K_INCREMENT = 0.1 m (radius increment per member).
        """
        return self.radius + K_INCREMENT * self.n_members

    def __repr__(self) -> str:
        return (f"Roundabout(id={self.id}, center={np.round(self.center, 3)}, "
                f"r={self.radius:.3f}, members={self.members})")
