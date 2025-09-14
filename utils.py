# =============================
# utils.py
# =============================
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional

# Map encoding
FREE = 0
UNKNOWN = 1
OCCUPIED = 2
GOAL = 3
START = 4
FRONTIER = 5

@dataclass
class GridSpec:
    meters_h: float = 1.0
    meters_w: float = 5.0
    res_m: float = 0.01  # 1 cm

    @property
    def shape(self) -> Tuple[int, int]:
        H = int(round(self.meters_h / self.res_m))
        W = int(round(self.meters_w / self.res_m))
        return H, W

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        H, W = self.shape
        col = int(np.clip(x / self.res_m, 0, W - 1))
        row_from_bottom = int(np.clip(y / self.res_m, 0, H - 1))
        row = (H - 1) - row_from_bottom
        return row, col

    def grid_to_world(self, row: int, col: int) -> Tuple[float, float]:
        H, W = self.shape
        y_from_bottom = (H - 1 - row) * self.res_m
        x = col * self.res_m
        y = y_from_bottom
        return x, y

class Maps:
    def __init__(self, spec: GridSpec):
        H, W = spec.shape
        self.gt = np.full((H, W), FREE, dtype=np.int8)
        self.belief = np.full((H, W), UNKNOWN, dtype=np.int8)
        self.spec = spec

    def reset_belief(self):
        self.belief.fill(UNKNOWN)

    def place_start_goal(self, start_xy=(0.1, 0.5), goal_xy=(4.9, 0.5)):
        rs, cs = self.spec.world_to_grid(*start_xy)
        rg, cg = self.spec.world_to_grid(*goal_xy)
        self.gt[rs, cs] = START
        self.gt[rg, cg] = GOAL
        self.belief[rs, cs] = START

    def add_random_obstacles(self, n: int = 5, min_w: float = 0.1, min_h: float = 0.1, seed: Optional[int] = None):
        rng = np.random.default_rng(seed)
        for _ in range(n):
            w = min_w + rng.random() * 0.3
            h = min_h + rng.random() * 0.3
            x0 = rng.random() * (self.spec.meters_w - w)
            y0 = rng.random() * (self.spec.meters_h - h)
            self.add_rect_obstacle(x0, y0, x0+w, y0+h)

    def add_rect_obstacle(self, xmin: float, ymin: float, xmax: float, ymax: float):
        H, W = self.gt.shape
        r1, c1 = self.spec.world_to_grid(max(0.0, xmin), max(0.0, ymin))
        r2, c2 = self.spec.world_to_grid(min(self.spec.meters_w, xmax), min(self.spec.meters_h, ymax))
        r_lo, r_hi = sorted((r1, r2))
        c_lo, c_hi = sorted((c1, c2))
        self.gt[r_lo:r_hi+1, c_lo:c_hi+1] = OCCUPIED

    def add_random_rect_obstacles(self, n: int, min_w_m: float = 0.05, min_h_m: float = 0.05,
                                  max_w_m: float = 0.30, max_h_m: float = 0.30,
                                  seed: Optional[int] = None):
        """Place N rectangular obstacles of random size with min/max dimensions."""
        rng = np.random.default_rng(seed)
        for _ in range(max(0, int(n))):
            w = rng.uniform(min_w_m, max_w_m)
            h = rng.uniform(min_h_m, max_h_m)
            x = rng.uniform(0.0, max(1e-6, self.spec.meters_w - w))
            y = rng.uniform(0.0, max(1e-6, self.spec.meters_h - h))
            self.add_rect_obstacle(x, y, x+w, y+h)

    def add_border_walls(self, thickness_m: float = 0.05):
        H, W = self.gt.shape
        t = max(1, int(round(thickness_m / self.spec.res_m)))
        self.gt[:t, :] = OCCUPIED
        self.gt[-t:, :] = OCCUPIED
        self.gt[:, :t] = OCCUPIED
        self.gt[:, -t:] = OCCUPIED