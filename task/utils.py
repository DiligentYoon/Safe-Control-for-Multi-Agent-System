import numpy as np
import torch

from torch_geometric.data import Data, Batch

def obs_to_graph(self, obs, device):
    graph_features = torch.tensor(obs['graph_features'], dtype=torch.float, device=self.device)
    edge_index = torch.tensor(obs['edge_index'], dtype=torch.long, device=self.device)
    data = Batch.from_data_list([Data(x=graph_features, edge_index=edge_index)])
    data = data.to(device)
    return data


def create_fully_connected_edges(num_agent:int) -> np.ndarray:
    """
        완전 연결 그래프의 edge_index를 NumPy 배열로 생성
        PyTorch Geometric 형식 [2, num_edges]
    """
    adj = ~np.eye(num_agent, dtype=bool)
    edge_index = np.array(np.where(adj))
    return edge_index

def compute_virtual_rays(map_info, 
                          drone_pos: np.ndarray = None,
                          drone_cell: np.ndarray = None,
                          num_rays: int = 36, 
                          max_range: float = 5.0) -> np.ndarray:
    """중심점에서 360도로 레이를 쏘아 장애물까지의 거리를 측정합니다."""
    # Shared Belief Map을 기준으로 하여, centroid에서부터 map에 대한 전반적인 정보를 함축
    # 가상의 Ray 추출 -> 어차피 업데이트가 안된 곳은 ground truth로 obstacle이여도 unknown으로 뜰 것

    # 장애물만 감지
    map_info = map_info.belief
    H, W = map_info.H, map_info.W

    # 중심 월드 좌표 -> 셀 좌표 변환
    c0 = drone_cell[0]
    r0 = drone_cell[1]

    distances = np.full(num_rays, max_range, dtype=np.float32)
    angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)

    for i, angle in enumerate(angles):
        # 레이 끝점 계산 (월드 좌표 기준)
        end_x = drone_pos[0] + max_range * np.cos(angle)
        end_y = drone_pos[1] + max_range * np.sin(angle)
        
        # 끝점 셀 좌표 변환
        c1 = int(end_x / map_info.cell_size)
        r1 = int(H - 1 - end_y / map_info.cell_size)

        # Bresenham's line 알고리즘으로 레이 경로 상의 셀들을 순회
        for r, c in bresenham_line(r0, c0, r1, c1):
            if not (0 <= r < H and 0 <= c < W):
                break
            
            # 장애물을 만나면 거리를 계산하고 이 레이는 종료
            if map_info.belief[r, c] == map_info.map_mask["occupied"]:
                dist = np.sqrt(((r - r0)**2 + (c - c0)**2)) * map_info.cell_size
                distances[i] = dist
                break
    
    # 거리를 최대값으로 나눠 0~1 사이로 정규화
    return distances / max_range



def collision_check(x0, y0, x1, y1, ground_truth, robot_belief, map_mask):
    """
    Ray-cast from (x0,y0) to (x1,y1) in cell coordinates.
    Update robot_belief cell-by-cell:
      - FREE 영역은 FREE로,
      - 첫 번째 OCCUPIED 셀은 OCCUPIED로 표시한 뒤 중단,
      - 그 이후는 UNKNOWN(기존 상태 유지).
    """
    # 1) 정수 셀 인덱스로 변환
    x0, y0 = int(round(x0)), int(round(y0))
    x1, y1 = int(round(x1)), int(round(y1))

    # 2) Bresenham 준비
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = 1 if x1 > x0 else -1
    sy = 1 if y1 > y0 else -1
    err = dx - dy

    # 3) 레이 캐스팅 루프
    while True:
        # 3.1) 맵 범위 체크
        if not (0 <= x < ground_truth.shape[1] and 0 <= y < ground_truth.shape[0]):
            break

        # 3.2) 셀 클래스 읽기
        gt = ground_truth[y, x]

        if gt == map_mask["occupied"]:
            # 충돌 지점만 OCCUPIED로 업데이트하고 종료
            robot_belief[y, x] = map_mask["occupied"]
            break
        else:
            # FREE 또는 기타(UNKNOWN) 영역은 FREE로 업데이트
            robot_belief[y, x] = map_mask["free"]

        # 3.3) 종료 조건: 끝점 도달
        if x == x1 and y == y1:
            break

        # 3.4) Bresenham step
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x   += sx
        if e2 <  dx:
            err += dx
            y   += sy

    return robot_belief

def bresenham_line(x0, y0, x1, y1):
    """Bresenham's line algorithm을 사용하여 (x0, y0)에서 (x1, y1)까지의 모든 셀 좌표를 반환"""
    x0, y0 = int(round(x0)), int(round(y0))
    x1, y1 = int(round(x1)), int(round(y1))
    
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = 1 if x1 > x0 else -1
    sy = 1 if y1 > y0 else -1
    err = dx - dy

    while True:
        yield (y, x)  # (row, col) 순서로 반환

        if x == x1 and y == y1:
            break
        
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy


def normalize_angle(angle):
    """Normalize an angle to be within [0, 360) degrees."""
    return angle % 360


def calculate_fov_boundaries(center_angle, fov):
    """Calculate the start and end angles of the field of vision (FOV).
    
    Args:
        center_angle (float): The central angle of the FOV in degrees.
        fov (float): The total field of vision in degrees.
        
    Returns:
        (float, float): The start and end angles of the FOV.
    """
    half_fov = fov / 2
    
    start_angle = center_angle - half_fov
    end_angle = center_angle + half_fov
    
    start_angle = normalize_angle(start_angle)
    end_angle = normalize_angle(end_angle)
    
    return start_angle, end_angle

def fov_sweep(start_angle, end_angle, increment):
    """Generate the correct sequence of angles to sweep the FOV from start to end with a specified increment.
    
    Args:
        start_angle (float): The starting angle of the FOV in degrees.
        end_angle (float): The ending angle of the FOV in degrees.
        increment (float): The angle increment in degrees.
        
    Returns:
        list: The sequence of angles representing the FOV sweep.
    """
    angles = []
    
    if start_angle < end_angle:
        angles = list(np.arange(start_angle, end_angle + increment, increment))
    else:
        angles = list(np.arange(start_angle, 360, increment)) + list(np.arange(0, end_angle + increment, increment))
    
    angles = [angle % 360 for angle in angles]
    
    angles_in_radians = np.radians(angles)

    return angles_in_radians

def sensor_work_heading(robot_position, 
                        sensor_range, 
                        robot_belief, 
                        ground_truth, 
                        heading, 
                        fov,
                        map_mask):

    sensor_angle_inc = 2.0
    x0 = robot_position[0]
    y0 = robot_position[1]
    start_angle, end_angle = calculate_fov_boundaries(heading, fov)
    sweep_angles = fov_sweep(start_angle, end_angle, sensor_angle_inc)

    x1_values = []
    y1_values = []
    
    for angle in sweep_angles:
        x1 = x0 + np.cos(angle) * sensor_range    
        y1 = y0 + np.sin(-angle) * sensor_range
        x1_values.append(x1)
        y1_values.append(y1)    
        
        robot_belief = collision_check(x0, y0, x1, y1, ground_truth, robot_belief, map_mask)

    return robot_belief