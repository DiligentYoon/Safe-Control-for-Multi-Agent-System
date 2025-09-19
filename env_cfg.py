

class EnvCfg():
    seed: int
    physics_dt: float
    device: str
    num_agent: int
    max_velocity: float
    max_yaw_rate: float
    fov: int
    sensor_range: float
    map: dict

    def __init__(self, cfg: dict) -> None:
        for key, value in cfg.items():
            setattr(self, key, value)
