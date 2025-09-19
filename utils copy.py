import numpy as np
import imageio
import os


def get_cell_position_from_coords(coords, map_info, check_negative=True):
    """
        coords: array-like of shape (2,) or (N,2) in world [x, y] coordinates
        map_info: MapInfo 객체 (map, map_origin_x, map_origin_y, cell_size)
        check_negative: True일 경우, boundary 밖으로 나간 인덱스를 자동으로 클램핑
    """
    # 1) 입력 형태 정리
    single_cell = False
    coords = np.asarray(coords)
    if coords.ndim == 1 and coords.size == 2:
        single_cell = True
        coords = coords.reshape(1, 2)
    else:
        coords = coords.reshape(-1, 2)

    # 2) world → fractional cell 좌표
    coords_x = coords[:, 0]
    coords_y = coords[:, 1]
    H, W = map_info.map.shape

    # y 축 뒤집기: 맵 row 0이 하단(origin_y)에 대응하도록
    frac_x = (coords_x - map_info.map_origin_x) / map_info.cell_size
    frac_y = H - 1 - ((coords_y - map_info.map_origin_y) / map_info.cell_size)

    # 3) 내림 처리하여 정수 인덱스로 변환
    cell_position = np.floor(
        np.stack((frac_x, frac_y), axis=-1)
    ).astype(int)

    # 4) boundary 클램핑 (optional)
    if check_negative:
        cell_position[:, 0] = np.clip(cell_position[:, 0], 0, W - 1)
        cell_position[:, 1] = np.clip(cell_position[:, 1], 0, H - 1)

    # 5) 반환 형태 맞추기
    if single_cell:
        return cell_position[0]
    return cell_position


def get_coords_from_cell_position(cell_position, map_info):
    cell_position = cell_position.reshape(-1, 2)
    cell_x = cell_position[:, 0]
    H = map_info.map.shape[0]
    cell_y = H - 1 - cell_position[:, 1]

    coords_x = cell_x * map_info.cell_size + map_info.map_origin_x
    coords_y = cell_y * map_info.cell_size + map_info.map_origin_y
    coords = np.stack((coords_x, coords_y), axis=-1)
    coords = np.around(coords, 1)
    if coords.shape[0] == map_info.map_mask["occupied"]:

        return coords[0]
    else:
        return coords


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


def sensor_work(robot_position, sensor_range, robot_belief, ground_truth):
    sensor_angle_inc = 0.5 / 180 * np.pi
    sensor_angle = 0
    x0 = robot_position[0]
    y0 = robot_position[1]
    while sensor_angle < 2 * np.pi:
        x1 = x0 + np.cos(sensor_angle) * sensor_range
        y1 = y0 + np.sin(sensor_angle) * sensor_range
        robot_belief = collision_check(x0, y0, x1, y1, ground_truth, robot_belief)
        sensor_angle += sensor_angle_inc
    return robot_belief

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

    sensor_angle_inc = 1.0
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






def make_gif(path, n, frame_files, rate):
    with imageio.get_writer('{}/{}_explored_rate_{:.4g}.gif'.format(path, n, rate), mode='I', duration=1) as writer:
        for frame in frame_files:
            image = imageio.imread(frame)
            writer.append_data(image)
    print('gif complete\n')

    for filename in frame_files[:-1]:
        os.remove(filename)

def make_gif_test(path, n, frame_files, rate, n_agents, fov, sensor_range):
    with imageio.get_writer('{}/{}_{}_{}_{}_explored_rate_{:.4g}.gif'.format(path, n, n_agents, fov, sensor_range, rate), mode='I', duration=1) as writer:
        for frame in frame_files:
            image = imageio.imread(frame)
            writer.append_data(image)
    print('gif complete\n')
    for filename in frame_files[:-1]:
        os.remove(filename)