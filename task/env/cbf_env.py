import numpy as np
from typing import Tuple

from .cbf_env_cfg import CBFEnvCfg
from task.base.env.env import Env
from task.utils import *


from sklearn.cluster import DBSCAN
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


class CBFEnv(Env):

    def __init__(self, episode_index: int | np.ndarray, cfg: dict):
        self.cfg = CBFEnvCfg(cfg)
        super().__init__(self.cfg)
        # Simulation Parameters
        self.dt = self.cfg.physics_dt
        self.decimation = self.cfg.decimation
        self.max_episode_steps = self.cfg.max_episode_steps

        # 핵심 Planning State
        self.num_obstacles = np.zeros(self.num_agent, dtype=np.long)
        self.num_neighbors = (self.num_agent-1) * np.ones(self.num_agent, dtype=np.long)
        self.num_frontiers = np.zeros(self.num_agent, dtype=np.long)

        self.active_obstacles = np.zeros((self.num_agent, self.cfg.max_obs), dtype=np.long)
        self.active_agents = np.zeros((self.num_agent, self.cfg.max_agents-1), dtype=np.long)

        self.virtual_ray = np.zeros((self.num_agent, self.cfg.num_virtual_rays), dtype=np.float32)
        self.local_frontiers = np.zeros((self.num_agent, self.cfg.num_rays, 2), dtype=np.float32)

        self.cfvr_r = np.zeros(self.num_agent, dtype=np.float32)
        self.cfvr_d = np.zeros(self.num_agent, dtype=np.float32)
        self.cfvr_pos = np.zeros((self.num_agent, 2), dtype=np.float32)

        # [agent_dim, max_dim, specific_dim]
        self.obsacle_states = np.zeros((self.num_agent, self.cfg.max_obs, 2), dtype=np.float32)
        self.neighbor_states = np.zeros((self.num_agent, self.cfg.max_agents-1, 4), dtype=np.float32)
        self.neighbor_ids = np.zeros((self.num_agent, self.cfg.max_agents-1), dtype=np.long)

        # Done flags
        self.is_collided_obstacle = np.zeros((self.num_agent, 1), dtype=np.bool_)
        self.is_collided_drone = np.zeros((self.num_agent, 1), dtype=np.bool_)
        self.is_reached_goal = np.zeros((self.num_agent, 1), dtype=np.bool_)
        self.is_first_reached = np.ones((self.num_agent, 1), dtype=np.bool_)

        # Additional Info
        self.infos["safety"] = {}
        self.infos["next_safety"] = {}


    def reset(self, episode_index: int = None):
        # 나머지 플래그는 사용하기 전 계산 되므로 초기화 X
        self.actions = np.zeros((self.num_agent, self.cfg.num_act), dtype=np.float32)
        self.is_collided_obstacle = np.zeros((self.num_agent, 1), dtype=np.bool_)
        self.is_collided_drone = np.zeros((self.num_agent, 1), dtype=np.bool_)
        self.is_reached_goal = np.zeros((self.num_agent, 1), dtype=np.bool_)
        self.is_first_reached = np.ones((self.num_agent, 1), dtype=np.bool_)
        
        return super().reset(episode_index)
    

    def _set_init_state(self,
                        max_attempts: int = 1000
                        ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            max_attempts (int): 유효한 위치를 찾기 위한 최대 시도 횟수.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (world_x, world_y) 각 에이전트의 월드 좌표.
        """
        H, W = self.map_info.H, self.map_info.W
        d_max, d_min = self.cfg.d_max, self.cfg.d_safe 

        start_rows, start_cols = np.where(self.map_info.gt == self.map_info.map_mask["start"])
        start_cell_candidates = np.stack([start_rows, start_cols], axis=1)

        if len(start_cell_candidates) < self.num_agent:
            raise ValueError(f"Number of start cells ({len(start_cell_candidates)}) is less than "
                            f"the number of agents ({self.num_agent}).")

        for attempt in range(max_attempts):
            # 랜덤 샘플링
            # 후보 셀 중에서 num_agent 개수만큼 비복원 추출합니다.
            indices = np.random.choice(len(start_cell_candidates), self.num_agent, replace=False)
            selected_cells = start_cell_candidates[indices] # shape: (num_agent, 2)

            # 월드 좌표로 변환 (거리 계산을 위해)
            selected_world_coords = self.map_info.grid_to_world_np(selected_cells)

            diff = selected_world_coords[:, np.newaxis, :] - selected_world_coords[np.newaxis, :, :]
            dist_matrix = np.linalg.norm(diff, axis=-1)

            np.fill_diagonal(dist_matrix, np.inf)

            # d_safe 제약조건 검사 (전체)
            # 모든 에이전트 쌍 간의 거리가 d_safe보다 커야 합니다.
            is_safe = np.min(dist_matrix) > d_min
            if not is_safe:
                continue # 조건을 만족하지 않으면 다시 샘플링

            # d_max 제약조건 검사 (개별)
            # 각 에이전트에 대해, 가장 가까운 이웃과의 거리가 d_max보다 작거나 같아야 합니다.
            min_distances_to_neighbors = np.min(dist_matrix, axis=1)
            is_connected = np.all(min_distances_to_neighbors <= d_max)

            if is_safe and is_connected:
                print(f"Valid start positions found after {attempt + 1} attempts.")
                return selected_world_coords[:, 0], selected_world_coords[:, 1]

        raise RuntimeError(f"No valid starting positions found within {max_attempts}")
        
    

    def _pre_apply_action(self, actions: np.ndarray) -> None:
        self.actions = actions.copy()
        self.preprocessed_actions = actions.copy()
        # Acceleration & Angular Velocity 생성
        self.preprocessed_actions[:, 0] *= self.max_lin_acc
        self.preprocessed_actions[:, 1] *= self.max_ang_vel
    

    def _apply_action(self, agent_id):
        # Acceleration을 바탕으로 속도 업데이트
        self.robot_velocities[:, 0] += self.preprocessed_actions[:, 0] * np.cos(self.robot_angles) * self.dt
        self.robot_velocities[:, 1] += self.preprocessed_actions[:, 1] * np.sin(self.robot_angles) * self.dt
        self.robot_velocities[:, 0] = np.clip(self.robot_velocities[:, 0], -self.max_lin_vel, self.max_lin_vel)
        self.robot_velocities[:, 1] = np.clip(self.robot_velocities[:, 1], -self.max_ang_vel, self.max_lin_vel)
        # Non-Holodemic Model 특성에 의해 Position 먼저 업데이트
        self.robot_locations[:, 0] += self.robot_velocities[:, 0] * self.dt
        self.robot_locations[:, 1] += self.robot_velocities[:, 1] * self.dt
        # Yaw rate를 바탕으로 각도 업데이트
        self.robot_yaw_rate = np.clip(self.preprocessed_actions[:, 1], -self.max_ang_vel, self.max_ang_vel)
        self.robot_angles = ((self.robot_angles + self.robot_yaw_rate * self.dt + np.pi) % (2 * np.pi)) - np.pi
    

    def _compute_intermediate_values(self):
        """
            업데이트된 state값들을 바탕으로, obs값에 들어가는 planning state 계산
        """
        drone_pos = np.hstack((self.robot_locations, self.robot_angles.reshape(-1, 1)))
        
        # 중복 없는 프론티어 탐지를 위해 임시 belief map 생성
        belief_copy = self.map_info.belief.copy()

        # Active Mask도 매 스텝에서 초기화
        self.active_agents[:] = 0
        self.active_obstacles[:] = 0
        
        for i in range(self.num_agent):
            # Virtual Ray
            drone_pos_i = drone_pos[i]
            self.virtual_ray[i] = compute_virtual_rays(map_info=self.map_info,
                                                       num_rays=self.cfg.num_virtual_rays, 
                                                       drone_pos=drone_pos_i, 
                                                       drone_cell=self.map_info.world_to_grid_np(drone_pos_i[:2].reshape(1, -1))[0])
            # Relative States
            rel_pos = world_to_local(drone_pos_i[:2], drone_pos[:, :2], drone_pos_i[2])
            rel_vel = world_to_local(self.robot_velocities[i], self.robot_velocities, drone_pos_i[2])
            distance = np.linalg.norm(rel_pos, axis=1)
            active_agent_ids = np.where(np.logical_and(distance < self.cfg.d_max, distance > 1e-5))[0]

            self.neighbor_states[i, :len(active_agent_ids), :2] = rel_pos[active_agent_ids]
            self.neighbor_states[i, :len(active_agent_ids), 2:] = rel_vel[active_agent_ids]
            self.num_neighbors[i] = len(active_agent_ids)
            self.neighbor_ids[i, :self.num_neighbors[i]] = active_agent_ids

            # Obstacles & Frontiers Sensing
            local_frontiers, local_obstacles = self._sense_local_environment(
                agent_id=i, belief_map_t=belief_copy
            )
            # Store obstacle information
            num_obs = min(len(local_obstacles), self.cfg.max_obs)
            if num_obs > 0:
                self.obsacle_states[i, :num_obs] = local_obstacles[:num_obs]
            self.num_obstacles[i] = num_obs
            # Store frontier information
            self.num_frontiers[i] = len(local_frontiers)
            self.local_frontiers[i, :self.num_frontiers[i]] = local_frontiers

            # CFVR (Continuous Frontier Visibility Region)
            self.cfvr_d[i], self.cfvr_r[i], self.cfvr_pos[i] = self.get_CFVR(drone_pos_i[:2], local_frontiers)

            # Active Mask
            self.active_agents[i, :self.num_neighbors[i]] = 1
            self.active_obstacles[i, :self.num_obstacles[i]] = 1
        

    
    def _get_observations(self) -> np.ndarray | dict | list[dict]:
        """
            Observation Config for Actor Network [n, obs_dim]
        """
        all_observations = []
        for i in range(self.num_agent):
            local_edge_index = create_fully_connected_edges(num_nodes=self.num_neighbors[i]+1)
            node_features = self.get_node_features(i)
            obs = {
                "graph_features": node_features,
                "edge_index": local_edge_index       
            }
            all_observations.append(obs)

        return all_observations
    

    def _get_states(self) -> np.ndarray | dict:
        """
            State Config for Critic Network [agent_dim, state_dim]
                1. agent_dim : Centralized Critic 특성 상, 모든 에이전트를 Node화
                2. state_dim : 각 Node가 가져야 할 고유 Features
                3. edge_dim  : 모든 에이전트를 연결시켜야 하므로 Fully Connected Edge를 agent_dim으로 생성
        """

        self.graph_edge_index = create_fully_connected_edges(self.num_agent)
        global_node_features = self.get_global_features()
        state = {
            "graph_features": global_node_features,
            "edge_index": self.graph_edge_index        
        }

        return state


    def _get_dones(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
            특정 종료조건 및 타임아웃 계산
            Return :
                1. terminated : 
                    1-1. 벽에 충돌
                    1-2. 드론끼리 충돌
                    1-3. 골 지점 도달
                2. truncated :
                    2-1. 타임아웃

        """
        # Planning State 업데이트
        self._compute_intermediate_values()

        # ============== Done 계산 로직 ===================

        # ---- Truncated 계산 -----
        timeout = self.num_step >= self.max_episode_steps - 1
        truncated = np.full((self.num_agent, 1), timeout, dtype=np.bool_)

        # ---- Terminated 계산 ----
        terminated = np.zeros((self.num_agent, 1), dtype=np.bool_)

        # 로봇 셀 좌표 변환
        cells = self.map_info.world_to_grid_np(self.robot_locations)
        rows, cols = cells[:, 1], cells[:, 0]

        # 목표 도달 유무 체크
        self.is_reached_goal = self.map_info.gt[rows, cols] == self.map_info.map_mask["goal"]

        # 맵 경계 체크
        H, W = self.map_info.H, self.map_info.W
        out_of_bounds = (rows < 0) | (rows >= H) | (cols < 0) | (cols >= W)

        # 유효한 셀에 대해서만 값 확인
        valid_indices = ~out_of_bounds
        valid_rows, valid_cols = rows[valid_indices], cols[valid_indices]

        # 장애물 충돌 (맵 밖 포함)
        hit_obstacle = np.zeros_like(out_of_bounds, dtype=np.bool_)
        hit_obstacle[valid_indices] = self.map_info.gt[valid_rows, valid_cols] == self.map_info.map_mask["occupied"]
        self.is_collided_obstacle = (hit_obstacle | out_of_bounds)[:, np.newaxis]

        # 드론 간 충돌 (점유 셀이 겹치면 충돌 판단)
        flat_indices = rows * W + cols
        unique_indices, counts = np.unique(flat_indices, return_counts=True)
        collided_indices = unique_indices[counts > 1]
        
        self.is_collided_drone.fill(False)
        for idx in collided_indices:
            colliding_agents = np.where(flat_indices == idx)[0]
            for agent_idx in colliding_agents:
                self.is_collided_drone[agent_idx] = True

        # 목표 지점 도달
        all_reached_goal = np.all(self.reached_goal)
        any_reached_goal = np.any(self.reached_goal)

        # 최종 Terminated 조건
        # 개별 드론이 충돌하거나 모든 드론이 목표에 도달하면 종료
        terminated = self.is_collided_obstacle | self.is_collided_drone | any_reached_goal

        return terminated, truncated, self.is_reached_goal
    

    def _get_rewards(self):
        reward =  -1 * np.ones(1)

        return reward
    

    def _update_infos(self):
        self.infos["safety"]["v_current"] = np.sqrt(self.robot_velocities[:, 0]**2 + self.robot_velocities[:, 1]**2) # (A,)
        self.infos["safety"]["p_obs"] = self.obsacle_states # (A, M, 2)
        self.infos["safety"]["p_agents"] = self.neighbor_states[:, :, :2] # (A, M, 2)
        self.infos["safety"]["v_agents_local"] = self.neighbor_states[:, :, 2:] # (A, M, 2)

        # Boolean 자료형으로.. [0, 1]
        self.infos["safety"]["agent_active"] = self.active_agents # (A, M)
        self.infos["safety"]["obs_active"] = self.active_obstacles # (A, M)



    # ============= Auxilary Methods ==============
    def _sense_local_environment(self, 
                                 agent_id: int, 
                                 belief_map_t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        에이전트의 시점에서 레이캐스팅을 수행하여 프론티어와 장애물의 로컬 좌표를 탐지합니다.
        중복 탐지를 피하기 위해 임시 belief map을 사용합니다.

        Args:
            agent_id (int): 에이전트의 인덱스
            belief_map_t (np.ndarray): 탐지 및 마킹에 사용될 임시 belief map

        Returns:
            Tuple[np.ndarray, np.ndarray]:
            - local_frontiers (np.ndarray): 에이전트의 로컬 좌표계 기준 프론티어 점 (Nx2)
            - local_obstacles (np.ndarray): 에이전트의 로컬 좌표계 기준 장애물 점 (Mx2)
        """
        drone_pose = np.hstack((self.robot_locations[agent_id], self.robot_angles[agent_id]))
        
        max_range = 5.0
        map_info = self.map_info
        H, W = belief_map_t.shape
        
        drone_x_world, drone_y_world, yaw_rad = drone_pose

        local_frontiers = []
        local_obstacles = []
        
        start_angle = yaw_rad - np.deg2rad(self.fov / 2)
        end_angle = yaw_rad + np.deg2rad(self.fov / 2)
        angles = np.linspace(start_angle, end_angle, self.cfg.num_rays)

        start_c, start_r = self.map_info.world_to_grid_np(np.array([[drone_x_world, drone_y_world]]))[0]

        for angle in angles:
            end_x_world = drone_x_world + max_range * np.cos(angle)
            end_y_world = drone_y_world + max_range * np.sin(angle)
            end_c, end_r = self.map_info.world_to_grid_np(np.array([[end_x_world, end_y_world]]))[0]

            prev_cell_val = belief_map_t[start_r, start_c]

            for r, c in bresenham_line(start_c, start_r, end_c, end_r):
                if r == start_r and c == start_c:
                    continue

                if not (0 <= r < H and 0 <= c < W):
                    break
                
                current_cell_val = belief_map_t[r, c]
                hit_point_world = self.map_info.grid_to_world_np(np.array([[c, r]]))[0]

                if current_cell_val == self.map_info.map_mask["occupied"]:
                    local_pt = world_to_local(drone_pose[:2], np.array([hit_point_world]), drone_pose[2])[0]
                    local_obstacles.append(local_pt)
                    break
                
                if current_cell_val == self.map_info.map_mask["frontier"]:
                    break

                if prev_cell_val == self.map_info.map_mask["free"] and current_cell_val == self.map_info.map_mask["unknown"]:
                    local_pt = world_to_local(drone_pose[:2], np.array([hit_point_world]), drone_pose[2])[0]
                    local_frontiers.append(local_pt)
                    belief_map_t[r, c] = self.map_info.map_mask["frontier"]
                    break
                
                prev_cell_val = current_cell_val

        return np.array(local_frontiers), np.array(local_obstacles)
    
    def get_CFVR(self, pos: np.ndarray, frontiers: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        dx = pos[0] - frontiers[:, 0]
        dy = pos[1] - frontiers[:, 1]
        alpha = np.atan2(dy, dx)
        angle_diff = np.max(alpha) - np.min(alpha)

        psi = np.atan2(np.sin(angle_diff), np.cos(angle_diff))
        d = self.sensor_range / (1+np.sin(psi/2))
        r = self.sensor_range * np.sin(psi/2) / (1+np.sin(psi/2))

        alpha_mid = np.min(alpha) + psi / 2
        center_x = pos[0] + d * np.cos(alpha_mid)
        center_y = pos[1] + d * np.sin(alpha_mid)
        center = np.array([center_x, center_y])

        return d, r, center
        

    def get_node_features(self, agent_id: int):
        num_neighbors = self.num_neighbors[agent_id]
        total_ids = np.hstack((agent_id, self.neighbor_ids[agent_id, :num_neighbors]))
        
        pos = self.robot_locations[total_ids]                       # (N, 2)
        vel = self.robot_velocities[total_ids]                      # (N, 2)
        yaw = self.robot_angles[total_ids].reshape(-1, 1)           # (N, 1)
        yaw_rate = self.robot_yaw_rate[total_ids].reshape(-1, 1)    # (N, 1)
        virtual_ray = self.virtual_ray[total_ids]                   # (N, R)
        num_frontier = self.num_frontiers[total_ids].reshape(-1, 1) # (N, 1)
        cfvr_r, cfvr_d, cfvr_pos = self.cfvr_r[total_ids], self.cfvr_d[total_ids], self.cfvr_pos[total_ids] # (N, 4)

        return np.hstack((pos,
                          vel,
                          yaw,
                          yaw_rate,
                          virtual_ray,
                          num_frontier,
                          cfvr_r.reshape(-1, 1),
                          cfvr_d.reshape(-1, 1),
                          cfvr_pos)) # (N, 11+R)
    

    def get_global_features(self):
        pos = self.robot_locations
        vel = self.robot_velocities
        yaw = self.robot_angles.reshape(-1, 1)
        yaw_rate = self.robot_yaw_rate.reshape(-1, 1)
        virtual_ray = self.virtual_ray
        num_frontier = self.num_frontiers.reshape(-1, 1)
        cfvr_r, cfvr_d, cfvr_pos = self.cfvr_r, self.cfvr_d, self.cfvr_pos
        action = self.actions

        return np.hstack((pos,
                          vel,
                          yaw,
                          yaw_rate,
                          virtual_ray,
                          num_frontier,
                          cfvr_r.reshape(-1, 1),
                          cfvr_d.reshape(-1, 1),
                          cfvr_pos,
                          action)) # (N, 13+R)

        

        
        
