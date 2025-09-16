# =============================
# simulation.py
# =============================
from __future__ import annotations
import math
import numpy as np
from typing import List, Tuple, Optional, Dict
from utils import GridSpec, Maps, FRONTIER, START, OCCUPIED, FREE
from robot_sensor import Robot, RaySensor
from controller_cbf import DecentralizedCBFController
from controller_hocbf import DecentralizedHOCBFController

# ======================================================================================
# Target Selection Logic
# ======================================================================================
class TargetSelector:
    def __init__(self, v_max: float, cluster_radius_m: float):
        self.v_max = v_max
        self.cluster_radius_m = cluster_radius_m

    def _split_clusters_by_radius(self, F: np.ndarray) -> List[np.ndarray]:
        if F.shape[0] == 0: return []
        r = self.cluster_radius_m
        N = F.shape[0]
        adj = [[] for _ in range(N)]
        for i in range(N):
            for j in range(i + 1, N):
                if np.linalg.norm(F[i] - F[j]) <= r:
                    adj[i].append(j)
                    adj[j].append(i)
        seen = np.zeros(N, dtype=bool)
        clusters: List[np.ndarray] = []
        for i in range(N):
            if seen[i]: continue
            stack = [i]
            seen[i] = True
            comp_idx = []
            while stack:
                u = stack.pop()
                comp_idx.append(u)
                for v in adj[u]:
                    if not seen[v]:
                        seen[v] = True
                        stack.append(v)
            clusters.append(F[np.array(comp_idx)])
        return clusters

    @staticmethod
    def _polyline_length_by_angle(cluster: np.ndarray) -> float:
        if cluster.shape[0] < 2: return 0.0
        ang = np.arctan2(cluster[:, 1], cluster[:, 0])
        idx = np.argsort(ang)
        P = cluster[idx]
        diffs = np.diff(P, axis=0)
        return float(np.linalg.norm(diffs, axis=1).sum())

    def _choose_cluster(self, F: np.ndarray) -> np.ndarray:
        clusters = self._split_clusters_by_radius(F)
        if not clusters: return F
        lengths = [self._polyline_length_by_angle(c) for c in clusters]
        Lmax = np.max(lengths)
        cand = [i for i, L in enumerate(lengths) if np.isclose(L, Lmax)]
        if len(cand) == 1: return clusters[cand[0]]
        ymeans = [clusters[i][:, 1].mean() for i in cand]
        j = cand[int(np.argmin(ymeans))]
        return clusters[j]

    def pick_target(self, frontier_local: List[Tuple[float, float]]) -> Tuple[Tuple[float, float], Dict]:
        if not frontier_local:
            return (0.0, 0.0), {}

        F = np.array(frontier_local, dtype=float).reshape(-1, 2)
        C = self._choose_cluster(F)
        if C.shape[0] == 0:
            return (self.v_max, 0.0), {} # Fallback target

        c = C.mean(axis=0)
        p_tgt = (float(c[0]), float(c[1]))
        viz = {"target_local": np.asarray(p_tgt, dtype=float), "cluster_pts_local": C.astype(float)}
        return p_tgt, viz

# ======================================================================================
# Main Simulator Class
# ======================================================================================
class Simulator:
    def __init__(self, maps: Maps, robots: List[Robot], sensor: RaySensor, dt: float = 0.1, control_mode: str = 'decentralized', leader_idx: int = 0):
        self.maps = maps
        self.robots = robots
        self.sensor = sensor
        self.dt = dt
        self.control_mode = control_mode
        self.num_agents = len(self.robots)
        self.leader_idx = leader_idx

        self.paths = [[(r.x, r.y)] for r in self.robots]
        self.last_cmds = [(0.0, 0.0)] * self.num_agents
        self.crashed = [False] * self.num_agents
        self.crash_msgs = [""] * self.num_agents
        self.stopped = [False] * self.num_agents
        self.stop_msgs = [""] * self.num_agents

        if self.control_mode == 'decentralized':
            # self.controllers = [DecentralizedCBFController(
            #     v_max=r.v_max, w_max=r.yaw_rate_max, d_safe=0.05, d_max=0.3, gamma_avoid=2.0, gamma_conn=2.0
            # ) for r in self.robots]
            self.controllers = [DecentralizedHOCBFController(
                v_max=r.v_max, w_max=r.yaw_rate_max, d_safe=0.1, d_max=0.3) for r in self.robots]
        
        self.target_selectors = [TargetSelector(
            v_max=r.v_max, cluster_radius_m=3 * self.maps.spec.res_m
        ) for r in self.robots]

        self.last_targets_world = [None] * self.num_agents
        self.last_clusters_world = [None] * self.num_agents
        self.fov_edge_margin_deg = 5.0

    def _point_local_to_world(self, robot: Robot, px: float, py: float) -> Tuple[float, float]:
        c, s = math.cos(robot.yaw), math.sin(robot.yaw)
        return (robot.x + c*px - s*py, robot.y + s*px + c*py)

    def _vector_world_to_local(self, robot: Robot, vx: float, vy: float) -> Tuple[float, float]:
        c, s = math.cos(-robot.yaw), math.sin(-robot.yaw)
        return (vx * c - vy * s, vx * s + vy * c)

    def _poly_local_to_world(self, robot: Robot, Pnx2) -> List[Tuple[float, float]]:
        return [self._point_local_to_world(robot, px, py) for px, py in Pnx2]

    def step(self):
        if all(self.crashed) or all(self.stopped):
            return

        commands = []
        all_viz_data = [{} for _ in range(self.num_agents)]

        # --- 1. First, determine the leader's target in world coordinates ---
        leader_robot = self.robots[self.leader_idx]
        leader_frontier_local, _, _ = self.sensor.sense_and_update(self.maps, leader_robot)
        
        half = math.radians(self.sensor.fov_deg * 0.5)
        margin = math.radians(self.fov_edge_margin_deg)
        leader_filtered_local = [p for p in leader_frontier_local if abs(math.atan2(p[1], p[0])) <= (half - margin) and p[0] > 0.0]
        
        leader_p_target_local, leader_viz_data = self.target_selectors[self.leader_idx].pick_target(leader_filtered_local)
        all_viz_data[self.leader_idx] = leader_viz_data

        p_target_world = None
        if leader_p_target_local != (0.0, 0.0):
            p_target_world = self._point_local_to_world(leader_robot, *leader_p_target_local)

        if p_target_world is None:
            self.stopped = [True] * self.num_agents
            self.stop_msgs = ["No target from leader."] * self.num_agents
            return

        # --- 2. Calculate control for all agents ---
        for i in range(self.num_agents):
            if self.crashed[i] or self.stopped[i]:
                commands.append((0.0, 0.0))
                continue

            robot = self.robots[i]
            
            # All agents need to sense for obstacles
            _, frontier_rc, obs_local = self.sensor.sense_and_update(self.maps, robot)

            # Update belief map
            bel = self.maps.belief
            bel[bel == FRONTIER] = FREE
            for r, c in frontier_rc:
                if bel[r, c] == FREE: bel[r, c] = FRONTIER

            # ---  Set p_target for agent i based on its role ---
            if i == self.leader_idx:
                # Leader's target is the frontier point, already in its local frame.
                p_target = leader_p_target_local
            else:
                # Follower's target is the leader's current position.
                # Convert leader's world position to the follower's local frame.
                p_target = robot.world_to_local(leader_robot.x, leader_robot.y)
                all_viz_data[i]["target_local"] = np.asarray(p_target, dtype=float)
            
            # Get other robots' local positions and velocities
            other_robots_local, other_robots_vel_local = [], []
            for j, other_robot in enumerate(self.robots):
                if i == j: continue
                lx, ly = robot.world_to_local(other_robot.x, other_robot.y)
                # Now use the corrected robot.vx, vy attributes
                lvx, lvy = self._vector_world_to_local(robot, other_robot.vx, other_robot.vy)
                other_robots_local.append((lx, ly))
                other_robots_vel_local.append((lvx, lvy))

            # Compute control command
            if self.control_mode == 'decentralized':
                # v_cmd, w_cmd = self.controllers[i].compute_control(i, self.leader_idx, p_target, obs_local, other_robots_local, other_robots_vel_local)
                robot_vel = np.sqrt(robot.vx**2 + robot.vy**2)
                v_cmd, w_cmd = self.controllers[i].compute_control(p_target, robot_vel, obs_local, other_robots_local, other_robots_vel_local)
                commands.append((v_cmd, w_cmd))
            else:
                commands.append((0.0, 0.0))

        # --- 3. Update visualization & state for all agents ---
        for i in range(self.num_agents):
            self._update_viz_from_local(i, all_viz_data[i])

        for i in range(self.num_agents):
            if self.crashed[i] or self.stopped[i]:
                continue
            
            v_cmd, w_cmd = commands[i]
            self.robots[i].step(v_cmd, w_cmd, self.dt)
            self.last_cmds[i] = (v_cmd, w_cmd)
            self.paths[i].append((self.robots[i].x, self.robots[i].y))

            row, col = self.maps.spec.world_to_grid(self.robots[i].x, self.robots[i].y)
            if self.maps.gt[row, col] == OCCUPIED:
                self.crashed[i] = True
                self.crash_msgs[i] = "Collision detected."

    def _clear_viz(self, agent_idx: int):
        self.last_targets_world[agent_idx] = None
        self.last_clusters_world[agent_idx] = None

    def _update_viz_from_local(self, agent_idx: int, viz: dict):
        robot = self.robots[agent_idx]
        if not viz or "target_local" not in viz:
            self._clear_viz(agent_idx)
            return

        tx_l, ty_l = viz["target_local"]
        self.last_targets_world[agent_idx] = self._point_local_to_world(robot, tx_l, ty_l)

        C_local = viz.get("cluster_pts_local")
        if C_local is not None and len(C_local) > 0:
            self.last_clusters_world[agent_idx] = self._poly_local_to_world(robot, C_local)
        else:
            self.last_clusters_world[agent_idx] = None

    def get_viz(self) -> dict:
        d_safe = self.controllers[0].d_safe if self.controllers else 0
        d_max = self.controllers[0].d_max if self.controllers else 0

        return {"paths": self.paths, "last_cmds": self.last_cmds, "crashed": self.crashed, "crash_msgs": self.crash_msgs, "stopped": self.stopped, "stop_msgs": self.stop_msgs, "targets_world": self.last_targets_world, "clusters_world": self.last_clusters_world, "d_safe": d_safe, "d_max": d_max}

# ======================================================================================
# Environment Builder
# ======================================================================================
def build_minimal_env(num_agents: int = 3):
    spec = GridSpec()
    maps = Maps(spec)
    maps.add_border_walls(thickness_m=0.05)
    # maps.gt[20:80, 50:55] = OCCUPIED # Removed by user request

    robots = []
    start_positions = [(0.2, 0.4), (0.2, 0.5), (0.2, 0.6)]
    for i in range(num_agents):
        x0, y0 = start_positions[i] if i < len(start_positions) else (0.1, 0.2 + i * 0.2)
        rs, cs = spec.world_to_grid(x0, y0)
        wx, wy = spec.grid_to_world(rs, cs)
        robots.append(Robot(x=wx, y=wy, yaw=0.0))

    sensor = RaySensor(fov_deg=80.0, max_range_m=0.5, num_rays=41)
    sim = Simulator(maps, robots, sensor, dt=0.1, control_mode='decentralized', leader_idx=1)
    return maps, sim
