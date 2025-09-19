import os
import numpy as np
from PIL import Image
from abc import abstractmethod
from typing import Tuple, Optional

from .env_cfg import EnvCfg
from utils import *


# Map encoding
FREE = 0
UNKNOWN = 1
OCCUPIED = 2
GOAL = 3
START = 4
FRONTIER = 5

class MapInfo:
    def __init__(self, cfg: dict):
        self.cfg = cfg

        self.meters_h = cfg.get("height", 5.0)
        self.meters_w = cfg.get("width", 1.0)
        self.res_m = cfg.get("resolution", 0.01)
        self.map_mask = self.cfg["map_representation"]

        self.H = int(round(self.meters_h / self.res_m))
        self.W = int(round(self.meters_w / self.res_m))

        self.gt = np.full((self.H, self.W), self.map_mask["free"], dtype=np.int8)
        self.belief = np.full((self.H, self.W), self.map_mask["unknown"], dtype=np.int8)

        self.belief_origin_x = 0.0
        self.belief_origin_y = 0.0

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        H, W = self.H, self.W
        col = int(np.clip(x / self.res_m, 0, W - 1))
        row_from_bottom = int(np.clip(y / self.res_m, 0, H - 1))
        row = (H - 1) - row_from_bottom
        return row, col

    def grid_to_world(self, row: int, col: int) -> Tuple[float, float]:
        H, W = self.H, self.W
        y_from_bottom = (H - 1 - row) * self.res_m
        x = col * self.res_m
        y = y_from_bottom
        return x, y
    
    def world_to_grid_np(self, world: np.ndarray) -> np.ndarray:
        H, W = self.H, self.W
        x = world[:, 0]
        y = world[:, 1]
        col = int(np.clip(x / self.res_m, 0, W - 1))
        row = (H - 1) - int(np.clip(y / self.res_m, 0, H - 1))

        grid_position = np.floor(
            np.stack((col, row), axis=-1)
        ).astype(int)

        return grid_position

    def grid_to_world_np(self, grid: np.ndarray) -> np.ndarray:
        H, W = self.H, self.W
        col = grid[:, 0]
        row = grid[:, 1]
        x = col * self.res_m
        y = (H - 1 - row) * self.res_m    

        world = np.stack((x, y), axis=-1)

        return  world

    def reset_gt_and_belief(self):
        self.gt.fill(self.map_mask["free"])
        self.belief.fill(self.map_mask["unknown"])

    def place_start_goal(self, start_xy=(0.1, 0.5), goal_xy=(4.9, 0.5)):
        rs, cs = self.world_to_grid(*start_xy)
        rg, cg = self.world_to_grid(*goal_xy)
        self.gt[rs, cs] = self.map_mask["start"]
        self.gt[rg, cg] = self.map_mask["goal"]
        self.belief[rs, cs] = self.map_mask["start"]

    def add_rect_obstacle(self, xmin: float, ymin: float, xmax: float, ymax: float):
        r1, c1 = self.world_to_grid(max(0.0, xmin), max(0.0, ymin))
        r2, c2 = self.world_to_grid(min(self.meters_w, xmax), min(self.meters_h, ymax))
        r_lo, r_hi = sorted((r1, r2))
        c_lo, c_hi = sorted((c1, c2))
        self.gt[r_lo:r_hi+1, c_lo:c_hi+1] = self.map_mask["occupied"]

    def add_random_rect_obstacles(self, n: int = 5, min_w_m: float = 0.05, min_h_m: float = 0.05,
                                  max_w_m: float = 0.30, max_h_m: float = 0.30,
                                  seed: Optional[int] = None):
        """Place N rectangular obstacles of random size with min/max dimensions."""
        rng = np.random.default_rng(seed)
        for _ in range(max(0, int(n))):
            w = rng.uniform(min_w_m, max_w_m)
            h = rng.uniform(min_h_m, max_h_m)
            x = rng.uniform(0.0, max(1e-6, self.meters_w - w))
            y = rng.uniform(0.0, max(1e-6, self.meters_h - h))
            self.add_rect_obstacle(x, y, x+w, y+h)



class Env():
    def __init__(self, cfg: EnvCfg) ->None:
        self.cfg = cfg

        self.device = self.cfg.device
        self.seed = self.cfg.seed
        self.dt = self.cfg.physics_dt

        self.fov = self.cfg.fov
        self.sensor_range = self.cfg.sensor_range
        self.num_agent = self.cfg.num_agent
        self.max_lin_vel = self.cfg.max_velocity
        self.max_ang_vel = self.cfg.max_yaw_rate

        self.map_info = MapInfo(cfg=cfg["map"])
        
        # Location은 2D, Velocity는 스칼라 커맨드
        self.robot_locations = np.zeros((self.num_agent, 2), dtype=np.float32)
        self.robot_global_velocities = np.zeros((self.num_agent, 2), dtype=np.float32)
        self.robot_yaw_rate = np.zeros((self.num_agent, 1), dtype=np.float32)

        self.num_step = 0
        self.reached_goal = np.zeros((self.cfg.num_agent, 1), dtype=np.bool_)

        self.infos = {}


    def reset(self, episode_seed: int = None) -> Tuple[np.ndarray, np.ndarray]:
        if episode_seed is not None:
            self.seed = episode_seed
        # Load ground truth map and initial cell
        self.reached_goal = np.zeros((self.cfg.num_agent, 1), dtype=np.bool_)
        self.num_step = 0
        self.map_info.reset_gt_and_belief()
        self.map_info.add_random_rect_obstacles(seed=self.seed)

        # Randomly place N_AGENTS in start zone 
        world_x, world_y = self._set_init_state()
        self.robot_locations = np.stack([world_x, world_y], axis=1)

        # Compute goal coordinates (average over all goal cells)
        goal_world = self._set_goal_state()
        self.goal_locations = goal_world

        # Initialize headings
        self.angles = 0 * np.random.uniform(0, 2*np.pi, size=self.num_agent)
        # Perform initial sensing update for each agent
        for i in range(self.num_agent):
            cell = self.map_info.world_to_grid_np(self.robot_locations[i])


            self.map_info.belief = sensor_work_heading(
                                                        cell,
                                                        round(self.sensor_range / self.map_info.res_m),
                                                        self.self.map_info.belief,
                                                        self.map_info.gt,
                                                        np.rad2deg(self.angles[i]),
                                                        360,
                                                        self.map_info.map_mask)

        self.prev_dists = [np.linalg.norm(self.robot_locations[i] - self.goal_locations) for i in range(self.num_agent)]
        
        self._compute_intermediate_values()
        self.obs_buf = self._get_observations()
        self.state_buf = self._get_states()
        self._update_infos()

        return self.obs_buf, self.state_buf, self.infos


    def _set_goal_state(self) -> np.ndarray:
        goal_cells = np.column_stack(np.nonzero(self.map_info.gt == self.map_info.map_mask["goal"]))
        num_samples = min(self.num_agent, len(goal_cells))

        goal_indices = np.random.choice(len(goal_cells), size=num_samples)
        sampled_cells = goal_cells[goal_indices]

        rows = sampled_cells[:, 0]
        cols = sampled_cells[:, 1]

        x_coords, y_coords = self.map_info.grid_to_world(rows, cols)

        sampled_goal_world = np.column_stack((x_coords, y_coords))

        return np.mean(sampled_goal_world, axis=0)

    def _set_init_state(self) -> Tuple[np.ndarray, np.ndarray]:
        map_info = self.map_info
        H = map_info.H
        start_cells = np.column_stack((np.nonzero(map_info.gt == map_info.map_mask["start"])[0], 
                                       np.nonzero(map_info.gt == map_info.map_mask["start"])[1]))
        idx = np.random.choice(len(start_cells), self.num_agent, replace=False)
        chosen = start_cells[idx]
        rows, cols = chosen[:, 0], chosen[:, 1]

        world_x, world_y = map_info.grid_to_world(rows, cols)

        return world_x, world_y

    def update_robot_belief(self, robot_cell, heading) -> None:
        self.map_info.belief = sensor_work_heading(robot_cell, 
                                                   round(self.sensor_range / self.map_info.res_m), 
                                                   self.map_info.belief,
                                                   self.map_info.gt, 
                                                   heading, 
                                                   self.fov, 
                                                   self.map_info.map_mask)


    def step(self, actions) -> Tuple[np.ndarray,
                                     np.ndarray,
                                     np.ndarray,
                                     np.ndarray,
                                     np.ndarray,
                                     np.ndarray,
                                     dict[str, np.ndarray]]:
        """
            actions
                [n, 0] : linear acceleration command of n'th agent
                [n, 1] : angular velocity command of n'th agent

            Return :
                obs_buf -> [n, obs_dim]         : t+1 observation
                state_buf -> [n, state_dim]     : t+1 state
                action_buf -> [n, act_dim]      : t action
                reward_buf -> [n, 1]            : t+1 reward
                termination_buf -> [n, 1]       : t+1 terminated
                truncation_buf  -> [n, 1]       : t+1 truncated
                info -> dict[str, [n, dim]]     : additional metric 

        """
        
        # RL Action 전처리 단계
        self._pre_apply_action(actions)

        for i in range(self.cfg.decimation):
            for j in range(self.num_agent):
                # 이미 도달한 에이전트는 상태 업데이트 X
                if self.reached_goal[j]:
                    continue
                # ============== Step Numerical Simulation ================

                # action을 적용하여 robot state (위치 및 각도) 업데이트
                self._apply_action(j)

                # ========================================================

                # Belief 업데이트
                cell = self.map_info.world_to_grid(self.robot_locations[j])
                self.update_robot_belief(cell, np.rad2deg(self.angles[j]))

        # Done 신호 생성
        self.num_step += 1
        self.termination_buf, self.truncation_buf, self.reached_goal = self._get_dones()

        # 보상 계산
        self.reward_buf = self._get_rewards()
        
        # Next Observation 세팅
        self.obs_buf = self._get_observations()
        self.state_buf = self._get_states()

         # ======== 추가 정보 infos 업데이트 ===========
        self._update_infos()

        return self.obs_buf, self.state_buf, self.reward_buf, self.termination_buf, self.truncation_buf, self.infos


    # =============== Env-Specific Abstract Methods =================
    
    @abstractmethod
    def _pre_apply_action(self, actions):
        raise NotImplementedError(f"Please implement the '_pre_apply_action' method for {self.__class__.__name__}.") 

    @abstractmethod
    def _apply_action(self, agent_id: int) -> np.ndarray:
        raise NotImplementedError(f"Please implement the '_apply_action' method for {self.__class__.__name__}.") 
    

    @abstractmethod
    def _get_observations(self) -> np.ndarray:
        raise NotImplementedError(f"Please implement the '_get_observations' method for {self.__class__.__name__}.")
    

    @abstractmethod
    def _get_states(self) -> np.ndarray:
        raise NotImplementedError(f"Please implement the '_get_states' method for {self.__class__.__name__}.")
    

    @abstractmethod
    def _get_dones(self) -> np.ndarray:
        raise NotImplementedError(f"Please implement the '_get_dones' method for {self.__class__.__name__}.")


    @abstractmethod
    def _compute_intermediate_values(self) -> None:
        raise NotImplementedError(f"Please implement the '_compute_intermediate_values' method for {self.__class__.__name__}.")


    @abstractmethod
    def _get_rewards(self) -> np.ndarray:
        raise NotImplementedError(f"Please implement the '_get_rewards' method for {self.__class__.__name__}.")
    
    @abstractmethod
    def _update_infos(self):
        raise NotImplementedError(f"Please implement the '_update_infos' method for {self.__class__.__name__}.")