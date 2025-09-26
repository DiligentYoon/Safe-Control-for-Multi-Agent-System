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

        self.dt = self.cfg.physics_dt
        self.decimation = self.cfg.decimation
        self.max_episode_steps = self.cfg.max_episode_steps
        self.patch_size = self.cfg.patch_size

        # 핵심 Planning State
        self.local_patches = np.zeros((self.num_agent, self.patch_size, self.patch_size), dtype=np.float32)
        self.pos_patches = np.zeros((self.patch_size, self.patch_size), dtype=np.float32)
        self.virtual_ray = np.zeros((self.num_agent, self.cfg.num_rays), dtype=np.float32)
        self.frontier_features = np.zeros((self.num_agent, self.cfg.num_cluster_state * self.cfg.num_valid_cluster), dtype=np.float32)

        self.num_frontiers = np.zeros((self.num_agent, 1), dtype=np.float32)
        self.robot_target_angle = np.zeros((self.num_agent, 1), dtype=np.float32)
        self.neighbor_states = np.zeros((self.num_agent, self.num_agent-1, 4), dtype=np.float32)

        # Done flags
        self.is_collided_obstacle = np.zeros((self.num_agent, 1), dtype=np.bool_)
        self.is_collided_drone = np.zeros((self.num_agent, 1), dtype=np.bool_)
        self.is_reached_goal = np.zeros((self.num_agent, 1), dtype=np.bool_)
        self.is_first_reached = np.ones((self.num_agent, 1), dtype=np.bool_)


    def reset(self, episode_index: int = None):
        # 나머지 플래그는 사용하기 전 계산 되므로 초기화 X
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
        
    

    def _pre_apply_action(self, actions: dict) -> None:
        self.actions = actions.copy()
    
    

    def _apply_action(self, agent_id, action):
        pass
        


    def _compute_intermediate_values(self):
        """
            업데이트된 state값들을 바탕으로, obs값에 들어가는 planning state 계산
        """
        # patch_center_coords = np.array([self.patch_size / 2.0, self.patch_size / 2.0])
        # self.cur_dists = [np.linalg.norm(self.robot_locations[i] - self.goal_locations) for i in range(self.num_agent)]

        # # ================= Planning State for GNN ======================
        self.graph_edge_index = create_fully_connected_edges(self.num_agent)
        # self.centroids = np.mean(self.robot_locations, axis=0)
        # drone_pos = np.hstack((self.robot_locations, self.angles.reshape(-1, 1)))
        # drone_cell = self.map_info.world_to_grid_np(drone_pos[:, :2])
        # for i in range(self.num_agent):
        #     drone_pos_i = drone_pos[i]
        #     drone_cell_i = drone_cell[i]

        #     local_patch = self.compute_local_patch(drone_cell_i, self.map_info.belief)
        #     local_patch, frontier_clusters = self.marking_frontier_pixels(local_patch=local_patch,
        #                                                                   drone_poses=drone_pos_i, 
        #                                                                   drone_cells=drone_cell_i,
        #                                                                   agent_id=i)
        #     self.local_patches[i] = local_patch
        #     self.virtual_ray[i] = self._compute_virtual_rays(drone_pos=drone_pos_i, drone_cell=drone_cell_i)

        #     for j in range(self.cfg.num_valid_cluster):
        #         cluster = frontier_clusters[j]
        #         centroid_vec = (np.flip(cluster['centroid']) - patch_center_coords) * self.cell_size
        #         centroid_vec[1] *= -1 
                
        #         idx = j * self.cfg.num_cluster_state
        #         self.frontier_features[i, idx : idx+2] = centroid_vec
        #         self.frontier_features[i, idx+2] = cluster['size'] 

    
    def _get_observations(self) -> np.ndarray | dict:
        """
            Observation Config for Actor Network [n, obs_dim]
        """
        # obs = {
        #     "graph_features": np.hstack([self.robot_pose [x,y,yaw],       # (3)
        #                                  self.robot_velocities [vx,vy,w], # (3)
        #                                  self.virtual_ray,                # (36)
        #                                  self.frontier_features]),        # (9)

        #     "edge_index": self.graph_edge_index        

        # }
        obs = {
            "graph_features": np.zeros((self.cfg.num_obs, ), dtype=np.float32),
            "edge_index": self.graph_edge_index        
        }

        return obs
    

    def _get_states(self) -> np.ndarray | dict:
        """
            State Config for Critic Network [n, state_dim]
        """

        state = {
            "graph_features": np.zeros((self.cfg.num_state, ), dtype=np.float32),
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
        pass


    # ============= Auxilary Methods ==============
    # def compute_local_patch(
    #     self,
    #     drone_cell: np.ndarray,        # [col, row] in cell indices
    #     belief_map: np.ndarray,        # 2D grid of ints (0=free, 1=unknown, 2=obstacle, etc.)
    #     ) -> Tuple[np.ndarray, np.ndarray]:
    #     """
    #         Extract a patch around the drone and compute APF.

    #         Returns:
    #         - apf_vec: 2D APF vector [vx, vy]
    #         - patch:   the extracted patch (patch_size x patch_size)
    #     """
    #     H, W = self.map_info.belief.map.shape
    #     half = self.patch_size // 2
        
    #     c, r = int(drone_cell[0]), int(drone_cell[1])
        
    #     # 1) 패치 범위: [r-half, r+half), [c-half, c+half)
    #     r0 = r - half
    #     r1 = r + half
    #     c0 = c - half
    #     c1 = c + half

    #     # 2) 맵 바깥 클램핑
    #     r0_clip, r1_clip = max(r0, 0), min(r1, H)
    #     c0_clip, c1_clip = max(c0, 0), min(c1, W)

    #     # 3) 패치 내 복사 시작 인덱스
    #     pr0 = r0_clip - r0    # if r0<0, pr0>0
    #     pc0 = c0_clip - c0

    #     # 4) 추출 & 복사
    #     patch = np.ones((self.patch_size, self.patch_size), dtype=belief_map.dtype)  # UNKNOWN=1
    #     h_src = r1_clip - r0_clip  # always <= patch_size
    #     w_src = c1_clip - c0_clip
    #     patch[pr0:pr0 + h_src, pc0:pc0 + w_src] = belief_map[r0_clip:r1_clip, c0_clip:c1_clip]

    #     return patch


    # def marking_frontier_pixels(self,
    #                             drone_poses: np.ndarray,
    #                             drone_cells: np.ndarray,
    #                             local_patch: np.ndarray = None,
    #                             agent_id: int = None,
    #                             fov_deg = 120.0,
    #                             num_rays = 40) -> tuple[np.ndarray, list[dict]]:
    #     half = self.patch_size // 2
    #     c, r = int(drone_cells[0]), int(drone_cells[1])
    #     r0, r1 = r - half, r + half
    #     c0, c1 = c - half, c + half

    #     frontiers = self.extract_frontier_pixels(drone_pose=drone_poses, 
    #                                                 fov_deg=fov_deg, 
    #                                                 num_rays=num_rays)
    #     self.num_frontiers[agent_id] = frontiers.shape[0]
    
    #     if self.num_frontiers[agent_id] > 0:
    #         for row, col in frontiers:
    #             # patch에 frontier 표시
    #             if r0 <= row < r1 and c0 <= col < c1:
    #                 pr = row - r0
    #                 pc = col - c0
    #                 if 0 <= pr < self.patch_size and 0 <= pc < self.patch_size:
    #                     local_patch[pr, pc] = self.map_info.map_mask["frontier"]

    #     clustered_frontier = self.marking_frontier_utility(local_patch, max_cluster=self.cfg.num_valid_cluster)

    #     return local_patch, clustered_frontier


    # def extract_frontier_pixels(self,
    #                             drone_pose, 
    #                             fov_deg=120.0,
    #                             num_rays=40) -> np.ndarray:
    #     """
    #     업데이트 된 Belief Map 내부에서
    #     bresenham_line을 사용하여 raycasting 방식으로 frontier pixel을 추출
    #     """
    #     max_range = 5.0  # meter
    #     map_info = self.map_info.belief
    #     belief_map = self.map_info.belief
    #     H, W = belief_map.shape
        
    #     # 드론 월드 좌표 (meter)
    #     try:
    #         drone_x_world, drone_y_world, yaw_rad = drone_pose
    #     except:
    #         drone_x_world, drone_y_world, yaw_rad = drone_pose[0]
        
    #     # 드론 셀 좌표
    #     drone_c = int((drone_x_world - map_info.map_origin_x) / map_info.cell_size)
    #     drone_r = int(H - 1 - (drone_y_world - map_info.map_origin_y) / map_info.cell_size)

    #     frontier_coords = set() # 중복 제거를 위해 set 사용
        
    #     # 시야각에 맞춰 레이를 쏠 각도 계산
    #     start_angle = yaw_rad - np.deg2rad(fov_deg / 2)
    #     end_angle = yaw_rad + np.deg2rad(fov_deg / 2)
    #     angles = np.linspace(start_angle, end_angle, num_rays)

    #     for angle in angles:
    #         # 레이의 끝점 계산 (월드 좌표)
    #         end_x_world = drone_x_world + max_range * np.cos(angle)
    #         end_y_world = drone_y_world + max_range * np.sin(angle)
            
    #         # 끝점 셀 좌표
    #         end_c = int((end_x_world - map_info.map_origin_x) / map_info.cell_size)
    #         end_r = int(H - 1 - (end_y_world - map_info.map_origin_y) / map_info.cell_size)

    #         # 레이 캐스팅 실행
    #         prev_cell_val = None
    #         for r, c in bresenham_line(drone_c, drone_r, end_c, end_r):
    #             if not (0 <= r < H and 0 <= c < W):
    #                 break

    #             current_cell_val = belief_map[r, c]

    #             # free(0) 셀을 지나 unknown(1) 셀을 만나면 프론티어로 간주
    #             if prev_cell_val == self.map_info.map_mask["free"] and current_cell_val == self.map_info.map_mask["unknown"]:
    #                 frontier_coords.add((r, c))
    #                 break
                
    #             # 장애물(2)을 만나면 이 레이는 더 이상 진행하지 않음
    #             if current_cell_val == self.map_info.map_mask["occupied"]:
    #                 break
                
    #             prev_cell_val = current_cell_val

    #     return np.array(list(frontier_coords))


    # def marking_frontier_utility(self,
    #                             frontier_patch: np.ndarray,
    #                             min_samples: int = 5,
    #                             max_cluster: int = 3
    #                             ) -> list[dict]:
    #     frontier_coords = np.argwhere(frontier_patch == self.map_info.map_mask["frontier"])
    #     num_frontiers = frontier_coords.shape[0]
        
    #     clusters = []
    #     if num_frontiers >= min_samples:
    #         dbscan = DBSCAN(eps=5.0, min_samples=min_samples).fit(frontier_coords)
    #         labels = dbscan.labels_
    #         unique_labels = set(labels)
    #         if -1 in unique_labels:
    #             unique_labels.remove(-1) 
            
    #         for label in unique_labels:
    #             points_in_cluster = frontier_coords[labels == label]
    #             clusters.append({
    #                 'size': len(points_in_cluster),
    #                 'centroid': np.mean(points_in_cluster, axis=0)
    #             })
    #         clusters.sort(key=lambda c: c['size'], reverse=True)
        
    #     elif num_frontiers > 0:
    #         # 각 점을 크기 1의 클러스터로 취급
    #         for coord in frontier_coords:
    #             clusters.append({'size': 1, 'centroid': coord})

    #     slicing = min(max_cluster, len(clusters))
    #     final_clusters = clusters[:slicing]

    #     delta_num = max_cluster - slicing
    #     for _ in range(delta_num):
    #         final_clusters.append({'size': 0, 'centroid': np.zeros(2)})
                
    #     return final_clusters