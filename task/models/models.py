import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cvxpy as cp

from cvxpylayers.torch import CvxpyLayer
from torch.distributions import Categorical, Normal
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data
from typing import Dict

import warnings
warnings.filterwarnings("ignore", message="Converting G to a CSC matrix; may take a while.")


LOG_STD_MIN = -20
LOG_STD_MAX = 2

class DifferentiableCBFLayer(nn.Module):
    def __init__(self, cfg: Dict):
        super().__init__()
        # 설정값(hyperparameters) 저장
        self.cfg = cfg
        self.max_obs = cfg['max_obs']
        self.max_agents = cfg['max_agents']
        
        self.num_hocbf_constraints = self.max_obs + 2 * (self.max_agents-1)
        self.num_box_constraints = 5
        self.num_constraints = self.num_hocbf_constraints

        # Constraints Info
        self.a_max = cfg['a_max']
        self.w_max = cfg['w_max']

        self.d_max =  cfg['d_max']
        self.d_obs = cfg['d_obs']
        self.d_safe = cfg['d_safe']

        self.damping = cfg['damping']
        self.stiffness = cfg['stiffness']

        # Objective Info
        self.w_slack = cfg['w_slack']

        # Action Scaler
        self.action_scale = torch.tensor([self.a_max, self.w_max])

        # --- 1. 최적화 변수 정의 ---
        u = cp.Variable(2, name='u')
        delta_avoid = cp.Variable(1, name='delta_avoid')
        x_vars_for_constraints = cp.hstack([u, delta_avoid])

        # --- 2. 파라미터 정의 ---
        # 목적 함수용 파라미터
        u_ref = cp.Parameter(2, name='u_ref') 
        # 제약 조건용 파라미터
        G = cp.Parameter((self.num_constraints, 3), name='G')
        h = cp.Parameter(self.num_constraints, name='h')

        # --- 3. 목적 함수 정의 ---
        objective = cp.Minimize(
            cp.sum_squares(u - u_ref) + self.cfg['w_slack'] * cp.sum_squares(delta_avoid)
        )

        # --- 4. 제약 조건 정의 (Gx <= h 행렬 형태 유지) ---
        constraints = [ G @ x_vars_for_constraints <= h ]
        constraints += [
            u[0] >= -self.cfg['a_max'], u[0] <= self.cfg['a_max'],
            u[1] >= -self.cfg['w_max'], u[1] <= self.cfg['w_max'],
            delta_avoid >= 0.0
        ]
        
        # --- 5. CvxpyLayer 생성 ---
        problem = cp.Problem(objective, constraints)
        self.layer = CvxpyLayer(
            problem,
            # 파라미터 리스트를 목적 함수와 제약 조건에 맞게 수정
            parameters=[u_ref, G, h],
            # 변수 리스트도 명확히 지정
            variables=[u, delta_avoid]
        )
        # self.cfg = cfg
        # self.max_obs = cfg['max_obs']
        # self.max_agents = cfg['max_agents']

        # # Constraints Info
        # self.a_max = cfg['a_max']
        # self.w_max = cfg['w_max']

        # self.d_max =  cfg['d_max']
        # self.d_obs = cfg['d_obs']
        # self.d_safe = cfg['d_safe']

        # self.damping = cfg['damping']
        # self.stiffness = cfg['stiffness']

        # # Objective Info
        # self.w_slack = cfg['w_slack']

        # # Action Scaler
        # self.action_scale = torch.tensor([self.a_max, self.w_max])

        # # Total Constraints (Static Obstacle + Dynamic Obstacle + Connectivity)
        # self.num_constraints = self.max_obs + 2 * (self.max_agents-1)
        
        # # 1. 최적화 변수 (Decision Variables) 정의
        # x_vars = cp.Variable(3, name='x_vars')             # x = [a, w, delta]

        # # 2. 목적 함수 (Objective Function) 정의
        # # Q와 P 행렬을 사용하여 목적 함수를 명시적으로 정의 (1/2 * x^T * Q * x + p^T * x)
        # Q_val = np.zeros((3, 3))
        # Q_val[0, 0] = 2.0                       # a^2 계수
        # Q_val[1, 1] = 2.0                       # w^2 계수
        # Q_val[2, 2] = 2.0 * self.cfg['w_slack'] # delta^2 계수
        # p = cp.Parameter(3, name='p')                # Linear part
        # objective = cp.Minimize(0.5 * cp.quad_form(x_vars, Q_val) + p.T @ x_vars)

        # # 3. 파라미터 (Inputs to the Layer) 정의 - 이제 G, h, u_ref가 파라미터가 됨
        # # CBF의 부등식 제약조건을 Gx <= h로 표현하기 위해 G와 h를 파라미터로 정의
        # G = cp.Parameter((self.num_constraints, 3), name='G')
        # h = cp.Parameter(self.num_constraints, name='h')

        # # 4. 제약 조건 (Constraints) 정의
        # constraints = [ G @ x_vars <= h ]
        # constraints += [
        #     x_vars[0] >= -self.cfg['a_max'], x_vars[0] <= self.cfg['a_max'],
        #     x_vars[1] >= -self.cfg['w_max'], x_vars[1] <= self.cfg['w_max'],
        #     x_vars[2] >= 0.0
        # ]

        # # 5. CvxpyLayer 생성
        # problem = cp.Problem(objective, constraints)
        # self.layer = CvxpyLayer(
        #     problem,
        #     parameters=[Q, p, G, h],
        #     variables=[x_vars]
        # )

    def forward(self, u_nominal: torch.Tensor, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        :param u_nominal: 제안된 공칭 제어 입력 (B & N, 2)
        :param state: 현재 상태 정보를 담은 딕셔너리
        :return: 안전 필터를 거친 최종 제어 입력 (B & N, 2)
        """
        # state 딕셔너리에서 텐서 추출 및 디바이스 맞춤
        batch_size = u_nominal.shape[0]
        device = u_nominal.device

        # [-1, 1]범위로 정규화 되어있는 u_nominal을 실제 제어 입력 스케일로 전환
        u_nominal_scaled = u_nominal * self.action_scale.to(u_nominal.device)

        # --- 목적 함수 행렬 Q, p 계산 ---
        # Form : 1/2 * x^T * Q * x + p^T * x
        # Q는 파라미터가 아니므로 계산 X
        p = torch.zeros(batch_size, 3, device=device)
        p[:, 0] = -2.0 * u_nominal_scaled[:, 0] # -2*a_ref*a 항
        p[:, 1] = -2.0 * u_nominal_scaled[:, 1] # -2*w_ref*w 항

        # --- 제약 조건 행렬 G, h 계산 ---
        # 제약 조건 담을 때, 감지된 obs와 agent수 만큼만 유효하도록 마스킹 해야 함.
        G = torch.zeros(batch_size, self.num_constraints, 3, device=device)
        h = torch.zeros(batch_size, self.num_constraints, device=device)

        # state 딕셔너리에서 데이터 추출
        if not isinstance(state['v_current'], torch.Tensor):
            v_current = torch.tensor(state['v_current'], device=device).unsqueeze(1)             # (B, 1)
            p_obs = torch.tensor(state['p_obs'], device=device)                                  # (B, max_obs, 2)
            p_agents = torch.tensor(state['p_agents'], device=device)                            # (B, max_agents, 2)
            v_agents_local = torch.tensor(state['v_agents_local'], device=device)                # (B, max_agents, 2)
            agent_active = torch.tensor(state['agent_active'], device=device)       # (B, max_agent)
            obs_active = torch.tensor(state['obs_active'], device=device)           # (B, max_obs)


        # --- 정적 장애물 제약 (G, h의 첫 max_obs개 행) ---
        lx_obs, ly_obs = p_obs[..., 0], p_obs[..., 1] # (B, max_obs)
        h_obs = lx_obs**2 + ly_obs**2 - self.cfg['d_obs']**2
        h_dot_obs = -2 * lx_obs * v_current
        # Form : 2l_x * a + 2l_y * v * w - \delta <= 2v^2 - k_1 * \dot{h} + k_2 * h
        # TODO: 가변 Obstacle 수에 대응할 수 있는 zero-padding 데이터로 저장하기 (worker쪽에서 수행)
        G[:, :self.max_obs, 0] = 2 * lx_obs                          # a의 계수
        G[:, :self.max_obs, 1] = 2 * ly_obs * v_current              # w의 계수
        G[:, :self.max_obs, 2] = -1.0                                # delta의 계수
        # h의 값들
        h[:, :self.max_obs] = ((2 * v_current**2) + \
                              self.cfg['damping'] * h_dot_obs + \
                              self.cfg['stiffness'] * h_obs) * obs_active
    

        # --- 동적 에이전트 제약 ---
        lx_ag, ly_ag = p_agents[..., 0], p_agents[..., 1]
        v_jx, v_jy = v_agents_local[..., 0], v_agents_local[..., 1]
        
        # 1. 충돌 회피
        # Form : 2l_x * a + 2l_y * v * w - 2l_y * v_{jx} * w + 2l_x * v_{jy} * w <= ...
        h_avoid = lx_ag**2 + ly_ag**2 - self.cfg['d_safe']**2
        h_dot_avoid = -2*lx_ag*v_current + 2*(lx_ag*v_jx + ly_ag*v_jy)
        
        G_avoid_a = 2 * lx_ag
        G_avoid_w = 2*ly_ag*v_current - 2*ly_ag*v_jx + 2*lx_ag*v_jy
        G[:, self.max_obs:(self.max_obs+self.max_agents-1), 0] = G_avoid_a * agent_active
        G[:, self.max_obs:(self.max_obs+self.max_agents-1), 1] = G_avoid_w * agent_active
        
        # TODO: 가변 Agent 수에 대응할 수 있는 zero-padding 데이터로 저장하기 (worker쪽에서 수행)
        h_dot_dot_const = 2*v_current**2 - 2*v_current*v_jx + 2*(-v_current*v_jx + v_jx**2 + v_jy**2)
        h[:, self.max_obs:(self.max_obs+self.max_agents-1)] = (h_dot_dot_const + self.cfg['damping']*h_dot_avoid + self.cfg['stiffness']*h_avoid) * agent_active

        # 2. 연결 유지
        G[:, (self.max_obs+self.max_agents-1):, 0] = -G_avoid_a * agent_active
        G[:, (self.max_obs+self.max_agents-1):, 1] = -G_avoid_w * agent_active
        h_conn = self.cfg['d_max']**2 - (lx_ag**2 + ly_ag**2)
        h_dot_conn = -h_dot_avoid
        # TODO: 가변 Agent 수에 대응할 수 있는 zero-padding 데이터로 저장하기 (worker쪽에서 수행)
        h[:, (self.max_obs+self.max_agents-1):] = (-h_dot_dot_const + self.cfg['damping']*h_dot_conn + self.cfg['stiffness']*h_conn) * agent_active

        # CvxpyLayer에 계산된 텐서들 전달
        solution, _ = self.layer(u_nominal_scaled, G, h, solver_args={'solve_method': 'ECOS'})

        # 솔루션에서 안전한 제어 입력 u_safe만 추출
        u_safe = solution[:, :2]
        
        # 제어 입력 스케일로 나온 Safe Control Input을 다시 정규화
        u_safe_normalized = u_safe / self.action_scale.to(u_nominal.device)
        
        return u_safe_normalized


class GNN_Feature_Extractor(nn.Module):
    def __init__(self, 
                 num_node_features: int, 
                 cfg: dict,
                 role: str):
        super().__init__()

        gnn_hidden_dim = cfg['hidden']
        self.role = role
        self.in_features = num_node_features
        self.gnn_conv1 = GATConv(num_node_features, gnn_hidden_dim, heads=4)
        self.gnn_conv2 = GATConv(gnn_hidden_dim * 4, gnn_hidden_dim, heads=1)
    

    def forward(self, data: Data):
        x, edge_index, batch, ptr = data.x, data.edge_index, data.batch, data.ptr
        x = F.relu(self.gnn_conv1(x, edge_index))
        x = self.gnn_conv2(x, edge_index)

        if self.role == "global":
            graph_vector = global_mean_pool(x, batch)
        elif self.role == "local":
            graph_vector = x[ptr[:-1]]
        else:
            ValueError("Not supported role type.")

        return graph_vector
    
    
class ActorGaussianNet(nn.Module):
    def __init__(self, obs_dim, action_dim, device, cfg):
        super(ActorGaussianNet, self).__init__()
        self.cfg = cfg
        self.device = device
        self.out_features = action_dim
        in_dim = obs_dim
        hidden_sizes = self.cfg["hidden"]
        layers = []
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU(inplace=True))
            in_dim = h
        self.net = nn.Sequential(*layers)
        self.mu_layer    = nn.Linear(hidden_sizes[-1], action_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], action_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            mu:    (batch, action_dim)
            log_std: (batch, action_dim), clipped to [LOG_STD_MIN, LOG_STD_MAX]
        """
        x = self.net(obs)
        mu = self.mu_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std
    

    def compute(self, obs: torch.Tensor) -> torch.Tensor:
        """
            Samples an action from the policy, using reparameterization trick.
            Returns:
                action: (batch, action_dim)
                logp:   (batch, 1) log probability of sampled action
        """
        mu, log_std = self.forward(obs)
        std = log_std.exp()

        dist = torch.distributions.Normal(mu, std)
        z = dist.rsample()
        action = torch.tanh(z)
        # correction for Tanh squashing
        logp = dist.log_prob(z).sum(-1, keepdim=True) \
            - (2*(math.log(2) - z - F.softplus(-2*z))).sum(-1, keepdim=True)
        
        return action, logp


class CriticDeterministicNet(nn.Module):
    """
    Centralized Q‐value network.
    Inputs: joint_obs = [agent1_obs, ..., agentN_obs] 
            joint_act = [agent1_act, ..., agentN_act]
    Output: scalar Q
    """
    def __init__(self, obs_dim, action_dim, device, cfg):
        super(CriticDeterministicNet, self ).__init__()
        self.input_dim = obs_dim
        self.device = device
        self.out_features = action_dim
        in_dim = self.input_dim
        hidden_sizes = cfg["hidden"]
        layers = []
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU(inplace=True))
            in_dim = h
        self.net = nn.Sequential(*layers)
        self.q_out = nn.Linear(hidden_sizes[-1], action_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.net(obs)
        return self.q_out(x)
    

    def compute(self, obs: torch.Tensor) -> torch.Tensor:
        return self.forward(obs)