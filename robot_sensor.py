# robot_sensor.py
from __future__ import annotations
import math
from typing import List, Tuple, Optional, Set
import numpy as np
from dataclasses import dataclass
from utils import GridSpec, Maps, FREE, UNKNOWN, OCCUPIED, START, GOAL, FRONTIER

@dataclass
class Robot:
    x: float
    y: float
    yaw: float  # radians
    v_max: float = 0.3
    yaw_rate_max: float = 1.0

    def pose(self):
        return self.x, self.y, self.yaw

    def step(self, v: float, yaw_rate: float, dt: float):
        v = float(np.clip(v, 0.0, self.v_max))
        yaw_rate = float(np.clip(yaw_rate, -self.yaw_rate_max, self.yaw_rate_max))
        self.x += v * math.cos(self.yaw) * dt
        self.y += v * math.sin(self.yaw) * dt
        self.yaw = ((self.yaw + yaw_rate * dt + math.pi) % (2 * math.pi)) - math.pi

@dataclass
class RaySensor:
    fov_deg: float = 120.0
    max_range_m: float = 0.3
    num_rays: int = 61

    def sense_and_update(self, maps: Maps, robot: Robot
                         ) -> Tuple[List[Tuple[float,float]], List[Tuple[int,int]], List[Tuple[float,float]]]:
        """
        Raycast in FOV, update belief FREE/OCCUPIED only, and return:
          - frontier_local:  [(lx, ly), ...]   (<= per-ray 최대 1개)
          - frontier_rc:     [(row, col), ...] (<= per-ray 최대 1개)
          - obs_local:       [(lx, ly), ...]   (<= per-ray 최대 1개)
        NOTE: FRONTIER 표기는 여기서 하지 않음 (belief에는 FREE/OCCUPIED만 기록)
        """
        spec = maps.spec
        H, W = spec.shape
        half = math.radians(self.fov_deg / 2.0)
        angles = np.linspace(-half, half, self.num_rays)
    
        frontier_local: List[Tuple[float, float]] = []
        frontier_rc: List[Tuple[int, int]] = []
        obs_local: List[Tuple[float, float]] = []
    
        for a in angles:
            ang = robot.yaw + a
            step = spec.res_m
            L = int(self.max_range_m / step)
    
            last_rc = None
            hit_recorded = False          # per-ray: obs 최대 1개
            frontier_candidate_rc = None  # per-ray: frontier 후보(마지막 FREE∧UNKNOWN-인접)
    
            for i in range(1, L + 1):
                x = robot.x + i * step * math.cos(ang)
                y = robot.y + i * step * math.sin(ang)
                if x < 0 or y < 0 or x > spec.meters_w or y > spec.meters_h:
                    break
    
                r, c = spec.world_to_grid(x, y)
                if last_rc == (r, c):
                    continue
                last_rc = (r, c)
    
                if maps.gt[r, c] == OCCUPIED:
                    # 첫 OCC 히트만 기록
                    maps.belief[r, c] = OCCUPIED
                    if not hit_recorded:
                        dx = x - robot.x; dy = y - robot.y
                        cth = math.cos(-robot.yaw); sth = math.sin(-robot.yaw)
                        lx = cth*dx - sth*dy; ly = sth*dx + cth*dy
                        obs_local.append((lx, ly))
                        hit_recorded = True
                    break  # 이 ray 종료 (더 이상 진행 X)
    
                else:
                    # 관측된 FREE 갱신
                    if maps.belief[r, c] != START:
                        maps.belief[r, c] = FREE
    
                    # 이 셀의 8-이웃 중 UNKNOWN이 있으면 'frontier 후보'
                    found_unknown = False
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            if dr == 0 and dc == 0:
                                continue
                            rr = r + dr; cc = c + dc
                            if 0 <= rr < H and 0 <= cc < W and maps.belief[rr, cc] == UNKNOWN:
                                found_unknown = True
                                break
                        if found_unknown:
                            break
    
                    # frontier 후보는 ray를 따라 '마지막으로' 갱신하여, 경계에 가장 가까운 FREE를 선택
                    if found_unknown:
                        frontier_candidate_rc = (r, c)
    
            # ray가 끝난 뒤, 후보가 있으면 frontier를 1개만 최종 채택
            if frontier_candidate_rc is not None:
                r, c = frontier_candidate_rc
                wx, wy = spec.grid_to_world(r, c)
                dx = wx - robot.x; dy = wy - robot.y
                cth = math.cos(-robot.yaw); sth = math.sin(-robot.yaw)
                lx = cth*dx - sth*dy; ly = sth*dx + cth*dy
                frontier_local.append((lx, ly))
                frontier_rc.append((r, c))
    
        return frontier_local, frontier_rc, obs_local
    
