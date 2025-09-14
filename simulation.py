# =============================
# simulation.py
# =============================
from __future__ import annotations
import math
from typing import List, Tuple, Optional
from utils import GridSpec, Maps, FRONTIER, START, OCCUPIED, FREE
from robot_sensor import Robot, RaySensor
from controller2 import ShortHorizonMPC
from controller_cbf import CBFController

class Simulator:

    """
    - 센서(FOV 기반)로부터 frontier/obs를 받고,
    - controller(MPC)로 로컬 경로를 계획/실행하며,
    - 항상 시각화용 마커를 최신 상태로 유지합니다.
      (target point, heading ref arrow, 군집 중심/폴리라인, 주축/법선 등)
    """
    def __init__(self, maps: Maps, robots: List[Robot], sensor: RaySensor, cbf: CBFController, dt: float = 0.1):
        self.maps = maps
        self.robots = robots
        self.sensor = sensor
        self.cbf = cbf
        self.dt = dt

        # 실행/경로 기록
        r = robots[0]
        self.path = [(r.x, r.y)]
        self.last_cmd = (0.0, 0.0)
        self.last_world_traj = None  # List[(x,y,theta)]

        # 상태 플래그
        self.crashed: bool = False
        self.crash_msg: str = ""
        self.stopped: bool = False
        self.stop_msg: str = ""

        # FOV 옆면(사이드 엣지) 필터 마진 (도)
        self.fov_edge_margin_deg = 5.0

        # 시각화 상태 (항상 갱신)
        self.last_target_world: Optional[Tuple[float, float]] = None
        self.last_heading_arrow: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None
        self.last_centroid_world: Optional[Tuple[float, float]] = None
        self.last_cluster_poly_world: Optional[List[Tuple[float, float]]] = None
        self.last_axes_world: Optional[dict] = None  # {"t":((x0,y0),(x1,y1)), "n":((x0,y0),(x1,y1))}

    # ------------ 좌표 변환 유틸 ------------
    def _traj_local_to_world(self, robot: Robot, xs_local):
        c = math.cos(robot.yaw); s = math.sin(robot.yaw)
        out = []
        for px, py, th in xs_local:
            wx = robot.x + c*px - s*py
            wy = robot.y + s*px + c*py
            wth = ((robot.yaw + th + math.pi) % (2*math.pi)) - math.pi
            out.append((wx, wy, wth))
        return out

    def _point_local_to_world(self, robot: Robot, px: float, py: float) -> Tuple[float, float]:
        c = math.cos(robot.yaw); s = math.sin(robot.yaw)
        wx = robot.x + c*px - s*py
        wy = robot.y + s*px + c*py
        return wx, wy

    def _vec_local_to_world(self, robot: Robot, vx: float, vy: float) -> Tuple[float, float]:
        c = math.cos(robot.yaw); s = math.sin(robot.yaw)
        wx = c*vx - s*vy
        wy = s*vx + c*vy
        return wx, wy

    def _poly_local_to_world(self, robot: Robot, Pnx2) -> List[Tuple[float, float]]:
        out: List[Tuple[float, float]] = []
        for px, py in Pnx2:
            out.append(self._point_local_to_world(robot, px, py))
        return out

    # ------------ 시뮬레이션 한 스텝 ------------
    def step(self):
        if self.crashed or self.stopped:
            return

        r = self.robots[0]

        # 1) 센싱: frontier_local, frontier_rc, obs_local
        frontier_local, frontier_rc, obs_local = self.sensor.sense_and_update(self.maps, r)

        # 2) FOV 사이드 엣지 필터 (부채꼴 옆면 제거 + 전방성 유지)
        half = math.radians(self.sensor.fov_deg * 0.5)
        margin = math.radians(self.fov_edge_margin_deg)
        filtered_local = []
        filtered_rc = []
        for (p, rc) in zip(frontier_local, frontier_rc):
            lx, ly = p
            ang = math.atan2(ly, lx)
            if abs(ang) <= (half - margin) and lx > 0.0:
                filtered_local.append((lx, ly))
                filtered_rc.append(rc)

        # 3) belief에서 FRONTIER 초기화 후, 계획에 쓰이는 것만 FRONTIER로 표기
        bel = self.maps.belief
        bel[bel == FRONTIER] = FREE
        for (rr, cc) in filtered_rc:
            if bel[rr, cc] == FREE:
                bel[rr, cc] = FRONTIER

        # 4) 프론티어 없으면 정지
        if not filtered_local:
            self.stopped = True
            self.stop_msg = "No frontier in FOV. Exploration stopped."
            # viz 초기화(표시는 유지하거나 None으로 비울 수 있음)
            self._clear_viz()
            return

        # 5) 계획/실행 (컨트롤러는 항상 viz를 반환하도록 설계됨.
        plan = self.cbf.plan(filtered_local, obs_local)
        
        if plan is None:
            self.stopped = True
            self.stop_msg = "No frontier in FOV. Exploration stopped."
            self._clear_viz()
            return

        v_cmd, w_cmd, xs_local, viz = plan
        
        # 6) 로컬 경로를 월드로 변환 (CBF는 예측 경로가 없으므로 None 체크)
        if xs_local is not None:
            self.last_world_traj = self._traj_local_to_world(r, xs_local)
        else:
            self.last_world_traj = None

        # 7) 시각화 상태 업데이트 (항상)
        self._update_viz_from_local(r, viz)

        # 8) 실제 이동
        r.step(v_cmd, w_cmd, self.dt)
        self.last_cmd = (v_cmd, w_cmd)
        self.path.append((r.x, r.y))

        # 9) 충돌 체크 (GT 맵 기준)
        row, col = self.maps.spec.world_to_grid(r.x, r.y)
        if self.maps.gt[row, col] == OCCUPIED:
            self.crashed = True
            self.crash_msg = "Collision with ground-truth obstacle detected. Simulation stopped."

    # ------------ viz 도우미들 ------------
    def _clear_viz(self):
        self.last_target_world = None
        self.last_heading_arrow = None
        self.last_centroid_world = None
        self.last_cluster_poly_world = None
        self.last_axes_world = None

    def _update_viz_from_local(self, robot: Robot, viz: dict):
        """
        컨트롤러가 반환한 viz(로컬)를 월드 좌표로 변환해 내부 상태 변수에 저장.
        CBF 컨트롤러는 MPC와 달리 heading/axis 정보가 없으므로, 해당 부분을 비활성화.
        """
        # Clear viz data that CBF doesn't provide
        self.last_heading_arrow = None
        self.last_axes_world = None

        if not viz or "target_local" not in viz:
            self._clear_viz()
            return

        # 타깃 점
        tx_l, ty_l = viz["target_local"]
        tx_w, ty_w = self._point_local_to_world(robot, tx_l, ty_l)
        self.last_target_world = (tx_w, ty_w)

        # 군집 중심 (CBF에서는 타겟을 군집의 중심으로 사용)
        if "centroid_local" in viz:
            cx_l, cy_l = viz["centroid_local"]
            cx_w, cy_w = self._point_local_to_world(robot, cx_l, cy_l)
            self.last_centroid_world = (cx_w, cy_w)
        else:
            # Fallback to target if centroid is not explicitly provided
            self.last_centroid_world = (tx_w, ty_w)

        # 군집 폴리라인
        C_local = viz.get("cluster_pts_local", None)
        if C_local is not None and len(C_local) > 0:
            self.last_cluster_poly_world = self._poly_local_to_world(robot, C_local)
        else:
            self.last_cluster_poly_world = None

    # ------------ 선택: viz 딕셔너리로 한번에 꺼내기 ------------
    def get_viz(self) -> dict:
        """
        렌더 코드에서 바로 쓰기 좋은 형태로 반환.
        """
        return {
            "target_world": self.last_target_world,                 # (x,y)
            "heading_arrow": self.last_heading_arrow,               # ((x0,y0),(x1,y1))
            "centroid_world": self.last_centroid_world,             # (x,y) or None
            "cluster_poly_world": self.last_cluster_poly_world,     # [(x,y),...] or None
            "axes_world": self.last_axes_world,                     # {"t": ((x0,y0),(x1,y1)), "n": ...} or None
            "world_traj": self.last_world_traj,                     # [(x,y,theta), ...]
            "path": self.path,                                      # [(x,y), ...]
            "last_cmd": self.last_cmd,                              # (v,w)
            "crashed": self.crashed,
            "crash_msg": self.crash_msg,
            "stopped": self.stopped,
            "stop_msg": self.stop_msg,
        }


# ------------ 편의: 최소 환경 구성 ------------
def build_minimal_env():
    spec = GridSpec()
    maps = Maps(spec)
    maps.add_border_walls(thickness_m=0.05)
    maps.place_start_goal(start_xy=(0.1, 0.5), goal_xy=(4.9, 0.5))

    # 로봇 초기화 (start 셀의 세계좌표로 맞춤)
    rs, cs = spec.world_to_grid(0.1, 0.5)
    x0, y0 = spec.grid_to_world(rs, cs)
    robot = Robot(x=x0, y=y0, yaw=0.0)

    # 센서 (필요시 레이 수를 늘리면 좁은 통로에서 안정적)
    sensor = RaySensor(fov_deg=80.0, max_range_m=0.3, num_rays=41)

    # CBF 컨트롤러
    cbf = CBFController(
        v_max=robot.v_max,
        w_max=robot.yaw_rate_max,
        d_safe=0.05,
        gamma=5.0,
        k_v=1.5,
        k_w=2.5,
        cluster_radius_m=3 * spec.res_m
    )

    sim = Simulator(maps, [robot], sensor, cbf, dt=0.1)
    return maps, sim