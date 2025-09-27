
from ..base.env.env_cfg import EnvCfg

class CBFEnvCfg(EnvCfg):
    num_obs: int
    num_state: int
    decimation: int
    max_episode_steps: int
    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)

        # Space Information
        self.num_virtual_rays = 37
        self.num_obs = 48
        self.num_state = 50
        self.num_act = 2

        # Episode Information
        self.decimation = 2
        self.max_episode_steps = 300

        # Controller Cfg
        self.d_safe = 0.1
        self.d_max = 0.3
        self.d_obs = 0.05
        self.max_obs = self.num_rays
        self.max_agents = self.num_agent


