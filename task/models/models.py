import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data

LOG_STD_MIN = -20
LOG_STD_MAX = 2


class GNN_Feature_Extractor(nn.Module):
    def __init__(self, 
                 num_node_features: int, 
                 cfg):
        super().__init__()

        gnn_hidden_dim = cfg['hidden'] 
        self.gnn_conv1 = GATConv(num_node_features, gnn_hidden_dim, heads=4)
        self.gnn_conv2 = GATConv(gnn_hidden_dim * 4, gnn_hidden_dim, heads=1)
    

    def forward(self, data: Data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Actor GNN 순전파
        x = F.relu(self.gnn_conv1(x, edge_index))
        x = self.gnn_conv2(x, edge_index)
        graph_vector = global_mean_pool(x, batch)

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