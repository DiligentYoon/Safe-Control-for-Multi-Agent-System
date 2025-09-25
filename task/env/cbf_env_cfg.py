
from ..base.env.env_cfg import EnvCfg

class CBFEnvCfg(EnvCfg):
    num_channel: int
    num_cont_act: int
    num_disc_act: int
    num_state: int
    decimation: int
    max_episode_steps: int
    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)

        # Local Patch Information
        self.patch_size = 100
        self.num_channel = 3

        # Space Information
        self.num_valid_cluster = 3
        self.num_cluster_state = 3
        self.num_obs = 49
        self.num_state = 51
        self.num_act = 2

        # Episode Information
        self.decimation = 2
        self.max_episode_steps = 300

        # Action Scale
        self.max_delta = 0.5

        # Controller Cfg


